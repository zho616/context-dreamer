"""Prepare an isolated, auditable partial-state recovery of the interrupted run."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
from zoneinfo import ZoneInfo


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    root = Path(__file__).resolve().parents[1]
    experiment = root / 'logdir/context_injection_500k_20261016_020630'
    parent = experiment / 'r2dreamer_recurrent_actor_critic'
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    recovery = parent / ('recovery_' + stamp)
    recovery.mkdir()
    source = recovery / 'source'
    shutil.copytree(experiment / 'source', source, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(parent / 'best.pt', recovery / 'resume_origin.pt')
    shutil.copy2(parent / 'best.pt', recovery / 'best.pt')
    # Load the exact saved config rather than rebuilding overrides from defaults.
    shutil.copy2(parent / '.hydra/config.yaml', source / 'configs/recovery.yaml')
    path = source / 'train.py'
    text = path.read_text()
    text = replace_once(text, '    policy_trainer.begin(agent)', '''    checkpoint = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    # The old file omitted scheduler/scaler/RNG states. Warmup was completed:
    # recover its saturated LR from saved optimizer steps, not a fresh warmup.
    optim_steps = max(int(s.get("step", 0)) for s in agent._optimizer.state.values())
    agent._scheduler.last_epoch = optim_steps
    agent._scheduler._step_count = optim_steps + 1
    agent._scheduler._last_lr = [g["lr"] for g in agent._optimizer.param_groups]
    agent.clone_and_freeze()
    policy_trainer.resume_step = int(checkpoint["step"])
    names = checkpoint["selection_order"]
    policy_trainer._best_eval_key = tuple(float(checkpoint["eval_metrics"][n]) for n in names)
    policy_trainer.resume_updates = optim_steps
    print(f"RECOVERY loaded step={checkpoint['step']} optimizer_steps={optim_steps}; replay/environment/RNG/AMP scaler reset", flush=True)
    del checkpoint
    policy_trainer.begin(agent)''')
    text = replace_once(text, '        "agent_state_dict": agent.state_dict(),', '        "step": policy_trainer.final_step,\n        "agent_state_dict": agent.state_dict(),')
    path.write_text(text)
    path = source / 'trainer.py'
    text = path.read_text()
    text = replace_once(text, '        step = self.replay_buffer.count() * self._action_repeat\n        update_count = 0',
        '        start_step = getattr(self, "resume_step", 0)\n        step = start_step + self.replay_buffer.count() * self._action_repeat\n        update_count = getattr(self, "resume_updates", 0)')
    text = replace_once(text, '            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:',
        '            if (step - start_step) // (envs.env_num * self._action_repeat) > self.batch_length + 1:')
    text = replace_once(text, '        # Include the final trained policy in best-model selection.',
        '        self.final_step = step\n        # Include the final trained policy in best-model selection.')
    text = replace_once(text, '        self._last_eval_step = train_step', '''        # Preserve the most recent weights even if validation did not improve.
        temporary = self.logdir / "recovery_latest.pt.tmp"
        torch.save({"agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                    "step": int(train_step), "eval_metrics": dict(metrics),
                    "selection_order": ("episode/eval_success", "episode/eval_route_completion", "episode/eval_score"),
                    "scheduler_state_dict": agent._scheduler.state_dict(),
                    "scaler_state_dict": agent._scaler.state_dict()}, temporary)
        temporary.replace(self.logdir / "recovery_latest.pt")
        self._last_eval_step = train_step''')
    path.write_text(text)
    metadata = {'created_at': stamp, 'source': str(source), 'run': str(recovery),
        'resume_checkpoint': str(recovery / 'resume_origin.pt'), 'target_step': 500000,
        'origin_step': 140005, 'discarded_log_tail_step': 302629,
        'missing_state': ['replay', 'environment', 'RNG', 'AMP scaler', 'exact scheduler counter'],
        'changes': ['strict model and optimizer restoration', 'saturated LR retained',
                    'environment counter restored to checkpoint step', 'warmup measured on newly collected replay',
                    'previous best selection preserved', 'latest weights saved every validation'],
        'checkpoint_sha256': hashlib.sha256((recovery / 'resume_origin.pt').read_bytes()).hexdigest()}
    (recovery / 'recovery.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (recovery / 'source_sha256.json').write_text(json.dumps({str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()}, indent=2) + '\n')
    print(recovery)


if __name__ == '__main__':
    main()
