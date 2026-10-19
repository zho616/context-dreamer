import math
import pathlib

import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = pathlib.Path(logdir)
        self.save_best = bool(getattr(config, "save_best", False))
        self.save_every = int(getattr(config, "save_every", 20000))
        self._last_checkpoint_step = 0
        self._best_eval_key = None
        self._last_eval_step = None
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        if hasattr(envs, "begin_evaluation"):
            envs.begin_evaluation()
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        # Assign episode quotas so statistics include exactly eval_episode_num
        # episodes even when workers finish at different times.
        quotas = self.eval_episode_num // envs.env_num + (
            torch.arange(envs.env_num, device=agent.device) < self.eval_episode_num % envs.env_num
        ).to(torch.int32)
        completed = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        episode_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while (completed < quotas).any():
            active = completed < quotas
            steps += ~done * active
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * active
            finished = done & active
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                        episode_metrics[key] = torch.zeros_like(returns)
                    if key.startswith("log_chassis_"):
                        # Chassis metadata is constant per episode, not a cost.
                        metric = value[:, 0]
                    else:
                        episode_metrics[key] += value[:, 0] * active
                        metric = episode_metrics[key]
                    if key == "log_success":
                        metric = metric.clamp(max=1.0)
                    log_metrics[key] += metric * finished
                    episode_metrics[key] = torch.where(finished, 0.0, episode_metrics[key])
            completed += finished.to(torch.int32)
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        metrics = {
            "episode/eval_score": float(returns.sum() / self.eval_episode_num),
            "episode/eval_length": float(steps.sum().to(torch.float32) / self.eval_episode_num),
            "episode/eval_episodes": int(completed.sum()),
        }
        for key, value in log_metrics.items():
            metrics[f"episode/eval_{key[4:]}"] = float(value.sum() / self.eval_episode_num)
        for key, value in metrics.items():
            self.logger.scalar(key, value)
        if self.save_best:
            self._save_best(agent, train_step, metrics)
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        self._last_eval_step = train_step
        agent.train()
        return metrics

    def _save_best(self, agent, step, metrics):
        """Lexicographic selection: success, route completion, then episode reward."""
        names = ("episode/eval_success", "episode/eval_route_completion", "episode/eval_score")
        key = tuple(float(metrics.get(name, 0.0)) for name in names)
        if not all(math.isfinite(value) for value in key):
            print("Skipping best checkpoint: non-finite evaluation metrics.")
            return False
        if self._best_eval_key is not None and key <= self._best_eval_key:
            return False
        self.logdir.mkdir(parents=True, exist_ok=True)
        temporary = self.logdir / "best.pt.tmp"
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "step": int(step),
                "eval_metrics": dict(metrics),
                "selection_order": names,
            },
            temporary,
        )
        temporary.replace(self.logdir / "best.pt")
        self._best_eval_key = key
        print(f"Saved best.pt at step {step}: success={key[0]:.3f}, route={key[1]:.3f}, reward={key[2]:.3f}")
        self.logger.scalar("checkpoint/best_step", step)
        return True

    def _save_latest(self, agent, step, update_count):
        """Atomically replace the recovery checkpoint independently of best selection."""
        self.logdir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": int(step),
            "update_count": int(update_count),
            "best_eval_key": self._best_eval_key,
        }
        for attribute, key in (("_scheduler", "scheduler_state_dict"), ("_scaler", "scaler_state_dict")):
            if hasattr(agent, attribute):
                checkpoint[key] = getattr(agent, attribute).state_dict()
        temporary = self.logdir / "latest.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(self.logdir / "latest.pt")
        self._last_checkpoint_step = int(step)
        print(f"Saved latest.pt at step {step}", flush=True)

    def _maybe_save_latest(self, agent, step, update_count):
        if self.save_every > 0 and step // self.save_every > self._last_checkpoint_step // self.save_every:
            self._save_latest(agent, step, update_count)

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        self._last_checkpoint_step = step
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        successes = torch.zeros_like(returns)
        has_success = False
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        def log_finished_episodes():
            nonlocal video_cache
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        if has_success:
                            self.logger.scalar("episode/success", successes[i])
                            # Keep the explicitly requested spelling as an alias.
                            self.logger.scalar("episode/sucess", successes[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
                        successes[i] = 0

        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            log_finished_episodes()
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            if "log_success" in trans:
                has_success = True
                successes = torch.maximum(successes, trans["log_success"][:, 0].clamp(0, 1))
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
            self._maybe_save_latest(agent, step, update_count)
        log_finished_episodes()
        # Include the final trained policy in best-model selection.
        if self.save_best and self.eval_envs is not None and self.eval_episode_num > 0 and self._last_eval_step != step:
            self.eval(agent, step)
        self._save_latest(agent, step, update_count)
