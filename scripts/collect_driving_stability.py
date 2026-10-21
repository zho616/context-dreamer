"""Frozen-policy paired driving traces; no decoder, optimization or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=20);p.add_argument('--device',default='cuda:0');args=p.parse_args()
 sys.path.insert(0,str(args.source.resolve()))
 import torch
 from omegaconf import OmegaConf
 from tensordict import TensorDict
 from dreamer import Dreamer
 from envs.metadrive import MetaDrive
 torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
 cfg=OmegaConf.load(args.model/'.hydra/config.yaml');cfg.device=args.device;OmegaConf.resolve(cfg);cfg.model.compile=False
 checkpoint=args.model/'best.pt';payload=torch.load(checkpoint,map_location=args.device,weights_only=False)
 args.output.mkdir(parents=True,exist_ok=False);(args.output/'traces').mkdir();(args.output/'maps').mkdir()
 provenance={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'step':payload['step'],'source':str(args.source.resolve()),'collection_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes_per_split':args.episodes,'inference_seed_offset':12345,'policy':'actor mode, stochastic RSSM posterior, fixed per-episode seed; float32 high matmul precision','splits':{'new_maps':10020,'s':10320,'xl':10420}}
 (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');model=None;rows=[];started=time.monotonic()
 def status(phase):
  (args.output/'status.json').write_text(json.dumps({'phase':phase,'episodes':len(rows),'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
 status('running')
 for split,seed_start in provenance['splits'].items():
  c=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));c.env.eval_start_seed=seed_start-int(c.env.seed);c.env.eval_num_scenarios=args.episodes;c.env.eval_episode_num=args.episodes;c.env.eval_env_num=1
  if split!='new_maps':c.env.chassis.eval_models=[split]
  env=MetaDrive(c.env,evaluation=True)
  try:
   if model is None:
    model=Dreamer(c.model,env.observation_space,env.action_space).to(args.device);model.load_state_dict(payload['agent_state_dict'],strict=True);model.clone_and_freeze();model.eval();del payload
   with torch.inference_mode():
    for seed in range(seed_start,seed_start+args.episodes):
     torch.manual_seed(seed+12345);np.random.seed(seed+12345);obs=env.reset(seed=seed);state=model.get_initial_state(1);car=env._env.vehicle
     dt=float(env._env.config['physics_world_step_size']*env._env.config['decision_repeat']);chassis=env.chassis_parameters
     previous_velocity=None
     def snapshot(observation):
      nonlocal previous_velocity
      car=env._env.vehicle;velocity=np.array(car.velocity,dtype=np.float64);theta=float(car.heading_theta);forward=np.array([np.cos(theta),np.sin(theta)]);right=np.array([np.sin(theta),-np.cos(theta)])
      acc=np.zeros(2) if previous_velocity is None else (velocity-previous_velocity)/dt;previous_velocity=velocity.copy()
      longitudinal,lateral=car.lane.local_coordinates(car.position)
      heading_error=float((theta-car.lane.heading_theta_at(longitudinal)+np.pi)%(2*np.pi)-np.pi)
      return dict(position=np.array(car.position,dtype=np.float64),velocity=velocity,heading=theta,speed_mps=float(np.linalg.norm(velocity)),wheel_rad=-np.deg2rad(np.mean([car.system.getSteeringValue(i) for i in (0,1)])),yaw_rate_right=-float(car.body.getAngularVelocity()[2]),accel_long=float(acc@forward),accel_right=float(acc@right),lateral_offset=float(lateral),heading_error=heading_error,is_last=bool(observation['is_last']),is_terminal=bool(observation['is_terminal']),collision=bool(any(getattr(car,k,False) for k in ['crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'])))
     geometry=[]
     for frm,edges in env._env.current_map.road_network.graph.items():
      for to,lanes in edges.items():
       for lane in lanes:
        ss=np.linspace(0,lane.length,max(3,int(lane.length/2)+1))
        geometry.append({'center':np.array([lane.position(float(s),0) for s in ss]).tolist(),'left':np.array([lane.position(float(s),lane.width/2) for s in ss]).tolist(),'right':np.array([lane.position(float(s),-lane.width/2) for s in ss]).tolist()})
     (args.output/'maps'/f'{split}_{seed}.json').write_text(json.dumps(geometry)+'\n')
     snapshots=[snapshot(obs)];actions=[];score=0.;info={}
     for step in range(int(c.env.time_limit)):
      td=TensorDict({k:torch.as_tensor(np.asarray(v),device=args.device).reshape(1,*np.asarray(v).shape) for k,v in obs.items()},batch_size=[1])
      action,state=model.act(td,state,eval=True);action=action[0].cpu().numpy();actions.append(action.copy());obs,reward,done,info=env.step(action);score+=float(reward);snapshots.append(snapshot(obs))
      if done:break
     assert done,'Horizon was not enforced'
     arrays={k:np.array([s[k] for s in snapshots]) for k in snapshots[0]};arrays['action']=np.array(actions);arrays['time']=np.arange(len(snapshots))*dt
     np.savez_compressed(args.output/'traces'/f'{split}_{seed}.npz',**arrays)
     row={'split':split,'seed':seed,'chassis':chassis,'dt':dt,'steps':len(actions),'success':bool(info.get('arrive_dest',False)),'crash':bool(info.get('crash',False)),'out_of_road':bool(info.get('out_of_road',False)),'timeout':bool(obs['is_last'] and not obs['is_terminal']),'reward':score,'route_completion':float(info.get('route_completion',0.)),'mean_speed_km_h':float(arrays['speed_mps'][1:].mean()*3.6),'initial_position':arrays['position'][0].tolist(),'initial_heading':float(arrays['heading'][0]),'trace':f'traces/{split}_{seed}.npz'}
     rows.append(row)
     with (args.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     status('running');print(split,seed,'success',row['success'],'steps',row['steps'],flush=True)
  finally:env.close()
 status('completed');print('COMPLETED',args.output,flush=True)


if __name__=='__main__':main()
