"""Controlled-action context PCA; matched chassis probes and random-encoder control."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    from omegaconf import OmegaConf
    from context import HistoryEncoder
    from envs.metadrive import MetaDrive

    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')
    cfg.device = 'cpu'
    OmegaConf.resolve(cfg)
    cfg.env.traffic_density = 0.
    cfg.env.spawn.randomize = False
    cfg.env.eval_start_seed = 20000
    cfg.env.eval_num_scenarios = 2
    cfg.env.time_limit = 200
    cfg.env.termination.out_of_road_done = True
    cfg.env.termination.on_continuous_line_done = False
    OmegaConf.save(cfg, args.output / 'sampling_config.yaml')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    learned = HistoryEncoder()
    prefix = 'context_encoder.'
    learned.load_state_dict({k[len(prefix):]: v for k, v in payload['agent_state_dict'].items()
                             if k.startswith(prefix)}, strict=True)
    learned.eval()
    torch.manual_seed(20261015)
    random_encoder = HistoryEncoder().eval()
    step = payload['step']
    del payload

    models = ['default', 'm', 'l', 's', 'xl']
    scales = [.8, .9, 1., 1.1, 1.2]
    conditions = list(itertools.product(models, scales, scales, [20000, 20001], [5., 10.], [0, 1]))
    if args.pilot:
        conditions = [(m, 1., 1., 20000, 5., 0) for m in models]
    env = MetaDrive(cfg.env, evaluation=True)
    # Separate diagnostic road, not an evaluation of the learned driving policy.
    env._env.config['map_config']['config'] = 'SSSS'
    records, histories = [], []
    rejected = []
    started = time.monotonic()
    try:
        for rollout, (model, ms, ss, map_seed, speed, protocol) in enumerate(conditions):
            spec = env._env.config['chassis_spec']
            spec['models'] = [model]
            spec['mass_scales'] = [ms]
            spec['steering_scales'] = [ss]
            env.reset(seed=map_seed)
            # Settle the rigid body before setting the prescribed initial speed.
            for _ in range(5):
                _, _, done, _ = env.step(np.zeros(2, np.float32))
                if done:
                    raise RuntimeError('Unexpected termination during body settling')
            vehicle = env._env.vehicle
            vehicle.set_velocity([1., 0.], value=speed, in_local_frame=True)
            env._context_history.observe(vehicle)
            actual = env.chassis_parameters
            for t in range(25):
                if protocol == 0:
                    steering = .07 * np.sin(2 * np.pi * t / 20)
                    throttle = .30 + .12 * np.sin(2 * np.pi * t / 15)
                else:
                    steering = .07 if (t // 5) % 2 == 0 else -.07
                    throttle = .35 if t < 12 else -.20
                obs, _, done, info = env.step(np.array([steering, throttle], np.float32))
                if done or bool(obs['ctx_collision']):
                    rejected.append(dict(rollout=rollout, time=t + 1, chassis=actual,
                                         reason='terminal_or_collision'))
                    break
                if t + 1 in (10, 15, 20, 25):
                    assert obs['ctx_mask'].all()
                    histories.append(obs['ctx_history'].copy())
                    records.append(dict(rollout=rollout, step=t + 1, model=model,
                        mass=actual['mass'], max_steering=actual['max_steering'],
                        mass_scale=ms, steering_scale=ss, map_seed=map_seed,
                        initial_speed_m_s=speed, protocol=protocol,
                        speed_m_s=float(np.linalg.norm(vehicle.velocity)),
                        acceleration=float(obs['ctx_ego_state'][0]),
                        yaw_rate=float(obs['ctx_ego_state'][1]),
                        training_model=model in ('default', 'm', 'l')))
            if (rollout + 1) % 50 == 0 or args.pilot:
                print(f'{rollout + 1}/{len(conditions)} rollouts; {len(records)} windows; '
                      f'{len(rejected)} truncated; {time.monotonic()-started:.1f}s', flush=True)
    finally:
        env.close()
    history = np.stack(histories)
    mask = torch.ones(len(history), 5, dtype=torch.bool)
    with torch.inference_mode():
        learned_c = learned(torch.from_numpy(history), mask).numpy()
        random_c = random_encoder(torch.from_numpy(history), mask).numpy()
    np.savez_compressed(args.output / 'representations.npz', history=history,
                        learned_context=learned_c, random_context=random_c)
    with (args.output / 'windows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    (args.output / 'sampling_metadata.json').write_text(json.dumps(dict(
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_step=step, rollouts=len(conditions), windows=len(records),
        rejected=rejected, elapsed_seconds=time.monotonic()-started,
        protocols='Two shared open-loop steering/throttle sequences; no learned policy',
        initial_speeds_m_s=[5.,10.], history_steps=5,
        diagnostic='Traffic-free straight road; sampled every 5 steps after step 10; no overlapping windows',
        random_encoder_seed=20261015), indent=2) + '\n')
    print('COLLECTION_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
