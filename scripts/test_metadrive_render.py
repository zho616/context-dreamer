#!/usr/bin/env python3
"""Interactive LiDAR policy evaluation; see docs/metadrive_render.md."""
import argparse
import importlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from envs.metadrive import MetaDrive


class DreamerPolicy:
    """Model-loading and inference interface shared by R2Dreamer and DreamerV3."""

    def __init__(self, config, obs_space, action_space, checkpoint, device):
        self.device = torch.device(device)
        self.model = self.load_model(config, obs_space, action_space, checkpoint)
        self.state = None

    def load_model(self, config, obs_space, action_space, checkpoint):
        # Change the import/construction here if your model architecture changes.
        from dreamer import Dreamer

        model = Dreamer(config.model, obs_space, action_space).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        weights = payload.get("agent_state_dict", payload)
        model.load_state_dict(weights, strict=True)
        # act() uses frozen inference modules; refresh them after loading weights.
        model.clone_and_freeze()
        model.eval()
        print(f"Loaded {checkpoint}; checkpoint step={payload.get('step', 'unrecorded')}")
        return model

    def reset(self, seed):
        # Reset recurrent memory at EVERY episode boundary.
        self.state = self.model.get_initial_state(1)

    @torch.inference_mode()
    def act(self, obs):
        tensors = {}
        for key, value in obs.items():
            value = torch.as_tensor(np.asarray(value), device=self.device)
            tensors[key] = value.reshape(1, 1) if value.ndim == 0 else value.unsqueeze(0)
        batch = TensorDict(tensors, batch_size=(1,), device=self.device)
        action, self.state = self.model.act(batch, self.state, eval=True)
        return action[0].detach().cpu().numpy()


class RandomPolicy:
    """Only for checking the rendering pipeline, not driving quality."""

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, seed):
        self.action_space.seed(seed)

    def act(self, obs):
        return self.action_space.sample()


def load_config(args):
    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    config_path = pathlib.Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None and checkpoint is not None:
        candidate = checkpoint.parent / '.hydra' / 'config.yaml'
        if candidate.is_file():
            config_path = candidate
    if config_path is not None:
        config = OmegaConf.load(config_path)
        print(f"Configuration: {config_path}")
    else:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
            config = compose(config_name='metadrive')
        if args.policy == 'dreamer':
            raise ValueError('Checkpoint has no .hydra/config.yaml; supply --config with its training config.')
    # Materialize training interpolations before applying testing-only settings.
    OmegaConf.resolve(config)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    if args.device is None:
        args.device = str(config.device)
    if args.episodes is None:
        args.episodes = max(1, int(config.env.eval_episode_num))
    if args.seed_start is None:
        args.seed_start = int(config.env.seed) + int(config.env.eval_start_seed)
    config.device = args.device
    config.model.device = args.device
    config.env.device = args.device
    config.model.compile = False
    # Older additive experiments predate the explicit reward-mode field.
    if 'mode' not in config.env.reward:
        config.env.reward.mode = 'additive'
    if args.reward_mode:
        config.env.reward.mode = args.reward_mode
    # Single-env testing covers a consecutive user-selected seed range.
    config.env.eval_start_seed = args.seed_start - int(config.env.seed)
    config.env.eval_num_scenarios = args.episodes
    config.env.eval_env_num = 1
    return config, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='best.pt, latest.pt, or a custom checkpoint')
    parser.add_argument('--config', help='Full training YAML; auto-read beside checkpoint if omitted')
    parser.add_argument('--policy', default='dreamer', help='dreamer | random | module:factory')
    parser.add_argument('--render', choices=['3d', 'topdown', 'both', 'none'], default='both')
    parser.add_argument('--no-window', action='store_true', help='Offscreen topdown (use --save-topdown); also valid with none')
    parser.add_argument('--device', help='Default: training device')
    parser.add_argument('--episodes', type=int, help='Default: training evaluation episode count')
    parser.add_argument('--seed-start', type=int, help='Default: training evaluation start seed')
    parser.add_argument('--fps', type=float, default=10, help='Display pacing; 0 runs without delay')
    parser.add_argument('--max-steps', type=int, help='Optional cap in policy decisions per episode')
    parser.add_argument('--size', type=int, default=800, help='Topdown window width and height')
    parser.add_argument('--reward-mode', choices=['additive', 'metadrive', 'paper'], help='Default: checkpoint training reward')
    parser.add_argument('--override', action='append', default=[], help='Repeatable YAML override, e.g. model.rep_loss=dreamer')
    parser.add_argument('--output', help='Save episode metrics to JSON (including on Ctrl+C)')
    parser.add_argument('--save-topdown', help='Save first and final topdown PNGs for each episode')
    args = parser.parse_args()
    if (args.episodes is not None and args.episodes < 1) or args.size < 1 or args.fps < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('episodes/size/max-steps must be positive; fps must be nonnegative')
    if args.no_window and args.render in ('3d', 'both'):
        parser.error('--no-window requires --render topdown or none')
    if args.save_topdown and args.render not in ('topdown', 'both'):
        parser.error('--save-topdown requires --render topdown or both')
    if args.policy == 'dreamer' and not args.checkpoint:
        parser.error('--policy dreamer requires --checkpoint; use --policy random to check rendering')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config, checkpoint = load_config(args)
    env = MetaDrive(config.env, evaluation=True, use_render=args.render in ('3d', 'both'))
    rows = []
    interrupted = False
    frames_dir = pathlib.Path(args.save_topdown).expanduser().resolve() if args.save_topdown else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    def display(text, seed, tag=None):
        if args.render in ('3d', 'both'):
            env.render(text=text)
        if args.render in ('topdown', 'both'):
            frame = env.render(mode='topdown', text=text, window=not args.no_window,
                               screen_size=(args.size, args.size), draw_center_line=True,
                               draw_target_vehicle_trajectory=True)
            if frames_dir and tag:
                from PIL import Image
                Image.fromarray(frame).save(frames_dir / f'seed_{seed}_{tag}.png')

    try:
        if args.policy == 'dreamer':
            policy = DreamerPolicy(config, env.observation_space, env.action_space, checkpoint, args.device)
        elif args.policy == 'random':
            policy = RandomPolicy(env.action_space)
        else:
            module_name, factory_name = args.policy.rsplit(':', 1)
            factory = getattr(importlib.import_module(module_name), factory_name)
            policy = factory(config, env.observation_space, env.action_space, checkpoint, args.device)
        print(f'Render={args.render}; reward={config.env.reward.mode}; '
              f'LiDAR state={env.observation_space["state"].shape}; Ctrl+C to exit')
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            torch.manual_seed(seed + 12345)
            np.random.seed(seed + 12345)
            obs = env.reset(seed=seed)
            policy.reset(seed)
            score, step = 0.0, 0
            speeds_km_h = []
            info = {}
            display({'seed': seed, 'step': 0}, seed, 'start')
            limit = args.max_steps or int(config.env.time_limit)
            while step < limit:
                start = time.monotonic()
                action = np.asarray(policy.act(obs), dtype=np.float32)
                obs, reward, done, info = env.step(action)
                speeds_km_h.append(float(env._env.vehicle.speed_km_h))
                step += 1
                score += reward
                capped = step >= limit and not done
                display({'seed': seed, 'step': step, 'reward': round(score, 2),
                         'route': round(float(info.get('route_completion', 0)), 3)},
                        seed, 'end' if done or capped else None)
                if done or capped:
                    break
                if args.fps:
                    time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - start)))
            row = dict(seed=seed, steps=step, reward=score,
                       mean_speed_km_h=float(np.mean(speeds_km_h)),
                       success=bool(info.get('arrive_dest', False)),
                       crash=bool(info.get('crash', False)),
                       out_of_road=bool(info.get('out_of_road', False)),
                       yellow_line=bool(info.get('yellow_line', False)),
                       white_line=bool(info.get('white_line', False)),
                       off_lane=bool(info.get('off_lane', False)),
                       timeout=bool(obs['is_last'] and not obs['is_terminal']),
                       test_step_cap=capped, route_completion=float(info.get('route_completion', 0)))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        print('Stopped by user.')
    finally:
        env.close()
        summary = {key: float(np.mean([row[key] for row in rows])) for key in
                   ('success', 'crash', 'out_of_road', 'reward', 'route_completion',
                    'mean_speed_km_h')} if rows else {}
        if rows:
            summary['step_weighted_speed_km_h'] = float(
                sum(row['mean_speed_km_h'] * row['steps'] for row in rows)
                / sum(row['steps'] for row in rows))
        result = dict(completed_episodes=len(rows), interrupted=interrupted,
                      checkpoint=str(checkpoint) if checkpoint else None,
                      render=args.render, config=OmegaConf.to_container(config, resolve=True),
                      summary=summary, episodes=rows)
        print('Summary:', json.dumps(summary, ensure_ascii=False))
        if args.output:
            output = pathlib.Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(f'Metrics: {output}')


if __name__ == '__main__':
    main()
