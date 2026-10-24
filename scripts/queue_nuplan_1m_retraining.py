"""Prepare the five-method nuPlan rerun after the current managed queue."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from omegaconf import OmegaConf
from run_nuplan_roundabout_queue import (
    fingerprint, freeze, job_config, now, run, write_json,
)


ORDER = ('r2dreamer_context', 'r2dreamer', 'dreamerv3', 'ppo', 'sac')


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    controls = output / 'controls'
    configs = output / 'configs'
    controls.mkdir()
    configs.mkdir()
    for name in ('run_nuplan_roundabout_queue.py', 'train_metadrive_sb3.py',
                 'evaluate_metadrive.py', Path(__file__).name):
        shutil.copy2(root / 'scripts' / name, controls / name)
    reference_dir = args.reference.resolve()
    source = output / 'source'
    freeze(reference_dir / 'nuplan_source', source)
    reference_path = reference_dir / 'nuplan_r2dreamer_context/.hydra/config.yaml'
    reference = OmegaConf.load(reference_path)
    shutil.copy2(reference_path, output / 'reference_config.yaml')
    assert reference.env.task == 'metadrive_nuplan'
    assert Path(args.after_status).is_file()
    for split in ('train', 'eval', 'test'):
        assert (Path(reference.env[f'scenario_{split}_dir']) / 'dataset_summary.pkl').is_file()
    jobs = []
    for method in ORDER:
        name = 'nuplan_' + method
        context = method == 'r2dreamer_context'
        sb3 = method in ('ppo', 'sac')
        destination = output / name
        config = job_config(reference, destination, 1_000_000,
                            'r2dreamer' if method.startswith('r2dreamer') else 'dreamer', context)
        config.trainer.eval_every = 5000
        assert config.env.eval_episode_num == 100
        left = OmegaConf.to_container(reference.env, resolve=True)
        right = OmegaConf.to_container(config.env, resolve=True)
        for env in (left, right):
            env.pop('steps'); env.pop('context_enabled')
        assert left == right, 'Unexpected change to the nuPlan environment protocol'
        config_path = configs / (name + '.yaml')
        OmegaConf.save(config, config_path)
        base = [sys.executable, '-u']
        if sb3:
            command = base + [str(controls / 'train_metadrive_sb3.py'), '--source', str(source),
                '--config', str(config_path), '--output', str(destination),
                '--method', method, '--steps', '1000000', '--eval-every', '5000']
            preflight = command.copy()
            preflight[preflight.index('--output') + 1] = str(output / 'validation' / name)
            preflight.append('--smoke')
        else:
            command = base + [str(source / 'train.py'), '--config-path', str(configs),
                              '--config-name', name]
            preflight = command + [
                'device=cpu', 'env.steps=128', 'env.env_num=2', 'env.eval_env_num=1',
                'env.eval_episode_num=2', 'env.time_limit=32', 'env.train_ratio=8',
                'buffer.max_size=512', 'batch_size=4', 'batch_length=8', 'model.compile=false',
                'model.deter=32', 'model.hidden=16', 'model.units=16', 'model.discrete=4',
                'model.rssm.stoch=4', 'model.rssm.blocks=4', 'model.imag_horizon=3',
                'trainer.eval_every=64', 'trainer.update_log_every=32',
                'env.chassis.models=[default]', 'env.chassis.mass_scales=[1.0]',
                'env.chassis.steering_scales=[1.0]', 'env.chassis.friction_scales=[1.0]',
                f'logdir={output / "validation" / name}']
        # Retain the previous nuPlan best-checkpoint ID test procedure.
        evaluations = []
        if not sb3:
            for i in range(3):
                split = f'test_seed{i}'
                evaluations.append(dict(name=split, command=base + [
                    str(controls / 'evaluate_metadrive.py'), '--checkpoint', str(destination / 'best.pt'),
                    '--source-root', str(source), '--output', str(destination / 'evaluation' / split),
                    '--episodes', '100', '--seed-start', '0',
                    '--scenario-dir', str(config.env.scenario_test_dir),
                    '--inference-seed-offset', str(12345 + 100000 * i)]))
        jobs.append(dict(name=name, source=str(source), config=str(config_path),
                         logdir=str(destination), context=context, sb3=sb3, steps=1000000,
                         command=command, preflight=preflight, evaluations=evaluations))
    plan = dict(created_at=now(), after_status=str(args.after_status.resolve()), jobs=jobs,
                sources={str(p):fingerprint(p) for p in (source, configs, controls)},
                protocol=dict(task='nuplan_pittsburgh', steps=1000000, eval_every=5000,
                    validation_episodes=100, training_seed=int(reference.seed),
                    from_scratch=True, order=list(ORDER),
                    note='Original nuPlan splits, observation, reward, chassis mass/friction protocol retained'))
    write_json(output / 'plan.json', plan)
    write_json(output / 'status.json', dict(phase='prepared', created_at=now(),
               after_status=plan['after_status'],
               jobs={j['name']:dict(state='queued', logdir=j['logdir']) for j in jobs}))
    (output / 'README.md').write_text(
        '# nuPlan 1M retraining\n\n'
        'Starts automatically after the prerequisite queue completes successfully.\n'
        'Order: R2Dreamer+Context, R2Dreamer, DreamerV3, PPO, SAC.\n'
        'Each starts from scratch: seed 0, 1,000,000 environment steps; '
        '100 validation episodes every 5,000 steps. Original nuPlan input/reward/physics/splits retained.\n'
        'Prior 500K results are preserved. Check status.json and runner.log for live progress.\n'
        'Each job runs a small preflight immediately before formal training. '
        'A failed prerequisite or job stops the queue.\n')
    print(json.dumps(dict(output=str(output), order=list(ORDER), steps=1000000,
                         eval_every=5000), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-status', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        return run(args.output.resolve())
    if not args.after_status or not args.reference:
        parser.error('--after-status and --reference are required for preparation')
    prepare(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
