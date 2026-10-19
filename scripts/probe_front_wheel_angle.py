"""Physical wheel-angle sweep with directly measured angles and paired throttle probes."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--metadrive-package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=12)
    p.add_argument('--settle-steps',type=int,default=30)
    p.add_argument('--angle-degrees',type=float,nargs='+',
                   help='Optional shared absolute front-wheel angles, rather than fractions of each chassis maximum')
    p.add_argument('--future-steps',type=int,choices=(0,5),default=0,
                   help='Optionally continue five more steps after encoding the support history')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root.resolve()))
    import torch
    from omegaconf import OmegaConf
    from context import HistoryEncoder
    # Load only this simulator package; retain the current Python environment's
    # torch, numpy, gymnasium and remaining dependencies.
    spec=importlib.util.spec_from_file_location('metadrive',args.metadrive_package/'__init__.py',
        submodule_search_locations=[str(args.metadrive_package)])
    module=importlib.util.module_from_spec(spec);sys.modules['metadrive']=module;spec.loader.exec_module(module)
    from envs.metadrive import MetaDrive
    torch.set_num_threads(1)
    cfg=OmegaConf.load(args.checkpoint.parent/'.hydra/config.yaml');cfg.device='cpu';OmegaConf.resolve(cfg)
    cfg.env.traffic_density=0.;cfg.env.spawn.randomize=False
    cfg.env.eval_start_seed=20000;cfg.env.eval_num_scenarios=1;cfg.env.time_limit=200
    cfg.env.termination.out_of_road_done=False;cfg.env.termination.on_continuous_line_done=False
    OmegaConf.save(cfg,args.output/'sampling_config.yaml')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    E=HistoryEncoder().eval();prefix='context_encoder.'
    E.load_state_dict({k[len(prefix):]:v for k,v in payload['agent_state_dict'].items()if k.startswith(prefix)},strict=True)
    ckpt_step=payload['step'];del payload
    torch.manual_seed(20261015);random_E=HistoryEncoder().eval()
    env=MetaDrive(cfg.env,evaluation=True)
    # Only the initial straight block is needed. A wide physical road prevents
    # curb contacts from masquerading as ordinary steering response.
    env._env.config['map_config']['config']=''
    env._env.config['map_config']['lane_width']=20.
    fractions=[0.,.2,.4,.6,.8,1.];models=['default','m','l','s','xl']
    settings=args.angle_degrees if args.angle_degrees is not None else fractions
    conditions=list(itertools.product([5.,10.],range(args.repeats),models,settings))
    rows=[];histories=[];paths=[];angle_paths=[];future_tokens=[];rejects=[];started=time.monotonic()
    print(f'START {len(conditions)} rollouts',flush=True)
    try:
        for rollout,(speed,repeat,model,setting)in enumerate(conditions):
            rng=np.random.default_rng(870000+repeat)
            knots=rng.uniform(.15,.4,5 if args.future_steps else 3)
            times=[0,2,4,7,9]if args.future_steps else[0,2,4]
            throttle=np.interp(np.arange(5+args.future_steps),times,knots).astype(np.float32)
            chassis=env._env.config['chassis_spec'];chassis['models']=[model]
            chassis['mass_scales']=[1.];chassis['steering_scales']=[1.]
            env.reset(seed=20000)
            for _ in range(args.settle_steps):
                _,_,done,_=env.step(np.zeros(2,np.float32))
                if done:raise RuntimeError('Termination during settling')
            vehicle=env._env.vehicle
            actual=env.chassis_parameters
            fraction=setting/actual['max_steering'] if args.angle_degrees is not None else setting
            if not -1<=fraction<=1:raise ValueError(f'Angle {setting} exceeds {model} steering range')
            vehicle.set_velocity([1.,0.],value=speed,in_local_frame=True)
            # Current wheel angle is already set before the first probe step.
            vehicle._set_action(np.array([fraction,throttle[0]],np.float32))
            env._context_history.observe(vehicle)
            initial_angles=[float(vehicle.system.getSteeringValue(j))for j in (0,1)]
            initial_speed=float(np.linalg.norm(vehicle.velocity))
            initial_yaw=-float(vehicle.body.getAngularVelocity()[2])
            physical=[];angles=[];invalid=None
            for t in range(5+args.future_steps):
                obs,_,done,_=env.step(np.array([fraction,throttle[t]],np.float32))
                theta=vehicle.heading_theta;vel=np.asarray(vehicle.velocity)
                vx=vel@np.array([np.cos(theta),np.sin(theta)])
                vy=vel@np.array([-np.sin(theta),np.cos(theta)])
                physical.append([float(obs['ctx_ego_state'][0]),float(obs['ctx_ego_state'][1]),float(np.linalg.norm(vel)),float(vx),float(vy)])
                angles.append([float(vehicle.system.getSteeringValue(j))for j in (0,1)])
                if t==4:support_history=obs['ctx_history'].copy()
                if done or bool(obs['ctx_collision']):invalid='termination_or_collision';break
            if invalid:
                rejects.append(dict(rollout=rollout,model=model,initial_speed=speed,repeat=repeat,fraction=fraction,reason=invalid));continue
            assert obs['ctx_mask'].all()
            assert np.allclose(angles,np.array([fraction*actual['max_steering']]*2),atol=1e-4)
            assert abs(initial_speed-speed)<1e-4, (initial_speed,speed)
            histories.append(support_history);paths.append(physical);angle_paths.append(angles)
            if args.future_steps:future_tokens.append(obs['ctx_history'].copy())
            rows.append(dict(rollout=rollout,model=model,repeat=repeat,action_seed=870000+repeat,
                initial_speed_m_s=initial_speed,initial_yaw_rate=initial_yaw,angle_fraction=fraction,
                max_steering_degrees=actual['max_steering'],front_left_angle_degrees=initial_angles[0],
                front_right_angle_degrees=initial_angles[1],mass=actual['mass'],
                first_acceleration=physical[0][0],first_yaw_rate=physical[0][1],first_speed_m_s=physical[0][2],
                final_acceleration=physical[-1][0],final_yaw_rate=physical[-1][1],final_speed_m_s=physical[-1][2],
                mean_speed_m_s=float(np.mean(np.array(physical)[:,2])),mean_throttle=float(throttle.mean())))
            if (rollout+1)%60==0:print(f'{rollout+1}/{len(conditions)}; valid {len(rows)}; {len(rejects)} rejected; {time.monotonic()-started:.1f}s',flush=True)
    finally:env.close()
    h=np.stack(histories);mask=torch.ones(len(h),5,dtype=torch.bool)
    with torch.inference_mode():c=E(torch.from_numpy(h),mask).numpy();rc=random_E(torch.from_numpy(h),mask).numpy()
    extra={}
    if args.future_steps:
        targets=np.stack(future_tokens)
        extra=dict(future_states=targets[...,:4],future_actions=targets[...,4:])
    np.savez_compressed(args.output/'representations.npz',history=h,learned_context=c,random_context=rc,
                        physical_paths=np.array(paths),front_wheel_angle_paths=np.array(angle_paths),**extra)
    with(args.output/'windows.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta=dict(checkpoint_step=ckpt_step,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        simulator_package=str(args.metadrive_package),source_root=str(args.source_root),
        requested_rollouts=len(conditions),valid_windows=len(rows),rejected=rejects,
        fractions=fractions,models=models,initial_speeds_m_s=[5.,10.],repeats=args.repeats,
        common_requested_angles_degrees=args.angle_degrees,
        history_length=5,control_dt_seconds=env._context_history.dt,
        future_steps=args.future_steps,
        settle_steps=args.settle_steps,
        physical_path_columns=['longitudinal_acceleration','right_positive_yaw_rate','speed','body_vx','body_vy'],
        angle_sign='Left positive; directly read BulletVehicle.getSteeringValue for both front wheels, in degrees.',
        protocol='Fixed native chassis; wide traffic-free flat initial straight block; five-step support and optional five-step future; initial wheel angles set before probe; same throttle profiles across every model, angle and speed; no velocity clamping after initialization.',
        elapsed_seconds=time.monotonic()-started)
    (args.output/'sampling_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('COMPLETE',len(rows),len(rejects),flush=True)


if __name__=='__main__':main()
