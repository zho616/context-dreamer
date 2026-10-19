"""Paired short physical probes at matched speeds; frozen encoder, no driving policy."""
import argparse
import csv
import hashlib
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
    p.add_argument('--groups', type=int, default=100)
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
    cfg.env.eval_num_scenarios = 1
    cfg.env.time_limit = 200
    cfg.env.termination.out_of_road_done = True
    cfg.env.termination.on_continuous_line_done = False
    OmegaConf.save(cfg, args.output / 'sampling_config.yaml')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder = HistoryEncoder().eval()
    prefix = 'context_encoder.'
    encoder.load_state_dict({k[len(prefix):]: v for k,v in payload['agent_state_dict'].items() if k.startswith(prefix)}, strict=True)
    torch.manual_seed(20261015)
    random_encoder = HistoryEncoder().eval()
    ckpt_step = payload['step']
    del payload
    env = MetaDrive(cfg.env, evaluation=True)
    env._env.config['map_config']['config'] = 'SSSS'
    models = ['default','m','l','s','xl']
    rows, histories, speed_paths, actions_kept, rejects = [], [], [], [], []
    target = 10.
    tolerance_target, tolerance_between = .4, .25
    t0 = time.monotonic()
    accepted = 0
    try:
        for candidate in range(args.groups * 10):
            seed = 710000 + candidate
            rng = np.random.default_rng(seed)
            # New smooth action sequence per independent rollout group, shared by all cars.
            knots = rng.uniform(-1,1,(3,2)) * np.array([.10,.12])
            actions = np.column_stack([np.interp(np.arange(5),[0,2,4],knots[:,j]) for j in range(2)]).astype(np.float32)
            group_rows, group_hist, group_speeds = [], [], []
            invalid = None
            for model in models:
                spec = env._env.config['chassis_spec']
                spec['models'], spec['mass_scales'], spec['steering_scales'] = [model], [1.], [1.]
                env.reset(seed=20000)
                for _ in range(5):
                    _,_,done,_ = env.step(np.zeros(2,np.float32))
                    if done:
                        raise RuntimeError('Unexpected termination during settling')
                vehicle = env._env.vehicle
                vehicle.set_velocity([1.,0.],value=target,in_local_frame=True)
                env._context_history.observe(vehicle)
                speeds = [float(np.linalg.norm(vehicle.velocity))]
                for action in actions:
                    obs,_,done,_ = env.step(action)
                    speeds.append(float(np.linalg.norm(vehicle.velocity)))
                    if done or bool(obs['ctx_collision']):
                        invalid = 'termination_or_collision'
                        break
                if invalid:
                    break
                assert obs['ctx_mask'].all()
                group_hist.append(obs['ctx_history'].copy())
                group_speeds.append(speeds)
                actual = env.chassis_parameters
                group_rows.append(dict(group=accepted,candidate=candidate,action_seed=seed,model=model,
                    mass=actual['mass'],max_steering=actual['max_steering'],map_seed=20000,
                    target_speed_m_s=target,initial_speed_m_s=speeds[0],speed_m_s=speeds[-1],
                    mean_speed_m_s=float(np.mean(speeds)),min_speed_m_s=min(speeds),max_speed_m_s=max(speeds),
                    acceleration=float(obs['ctx_ego_state'][0]),yaw_rate=float(obs['ctx_ego_state'][1])))
            if not invalid:
                speeds = np.array(group_speeds)
                if np.max(np.abs(speeds-target)) > tolerance_target:
                    invalid = 'outside_target_speed_tolerance'
                elif np.ptp(speeds,axis=0).max() > tolerance_between:
                    invalid = 'between_model_speed_spread'
            if invalid:
                rejects.append(dict(candidate=candidate,action_seed=seed,reason=invalid))
                continue
            rows.extend(group_rows);histories.extend(group_hist);speed_paths.extend(group_speeds)
            actions_kept.append(actions)
            accepted += 1
            if accepted % 20 == 0 or accepted == args.groups:
                print(f'{accepted}/{args.groups} paired groups; {len(rejects)} rejected; {time.monotonic()-t0:.1f}s',flush=True)
            if accepted == args.groups:
                break
    finally:
        env.close()
    if accepted != args.groups:
        raise RuntimeError(f'Only {accepted} valid groups; do not relax speed constraints silently')
    h = np.stack(histories)
    mask = torch.ones(len(h),5,dtype=torch.bool)
    with torch.inference_mode():
        c = encoder(torch.from_numpy(h),mask).numpy()
        random_c = random_encoder(torch.from_numpy(h),mask).numpy()
    np.savez_compressed(args.output/'representations.npz',history=h,learned_context=c,
                        random_context=random_c,speed_paths=np.array(speed_paths),actions=np.array(actions_kept))
    with (args.output/'windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta=dict(checkpoint_step=ckpt_step,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_root=str(args.source_root.resolve()),groups=accepted,windows=len(rows),models=models,
        target_speed_m_s=target,target_speed_tolerance_m_s=tolerance_target,
        between_model_speed_tolerance_m_s=tolerance_between,
        actual_speed_range_m_s=[float(np.min(speed_paths)),float(np.max(speed_paths))],
        max_between_model_speed_spread_m_s=float(np.ptp(np.array(speed_paths).reshape(-1,5,6),axis=1).max()),
        rejection_log=rejects,control_interval_seconds=env._context_history.dt,
        protocol='One natural five-step window per reset; velocity initialized once, never clamped during window; smooth seeded open-loop actions shared by five models; all-or-none group acceptance.',
        duration_seconds=time.monotonic()-t0,random_encoder_seed=20261015)
    (args.output/'sampling_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == '__main__':
    main()
