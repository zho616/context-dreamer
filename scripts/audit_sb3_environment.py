"""Compare native and SB3-adapted physics under identical evaluation actions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['native','sb3'],required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.source.resolve()));sys.path.insert(1,str((a.source/'scripts').resolve()))
 c=OmegaConf.load(a.config);OmegaConf.resolve(c)
 if a.mode=='native':
  from envs.metadrive import MetaDrive
  env=MetaDrive(c.env,evaluation=True);native=env
 else:
  from train_metadrive_sb3 import StateEnv
  env=StateEnv(OmegaConf.to_container(c.env,resolve=True),evaluation=True);native=env.native
 rows=[]
 try:
  for episode in range(3):
   obs=env.reset();state=obs['state'] if a.mode=='native'else obs[0]
   rows.append(dict(episode=episode,reset=True,state=state.tolist(),chassis=native.chassis_parameters))
   for t in range(80):
    action=np.array([.12*np.sin(t*.11),.4],np.float32)
    if a.mode=='native':
     obs,reward,done,info=env.step(action);state=obs['state'];term=bool(obs['is_terminal']);trunc=bool(done and not term)
    else:state,reward,term,trunc,info=env.step(action);done=term or trunc
    v=native._env.vehicle
    rows.append(dict(episode=episode,t=t,state=state.tolist(),reward=float(reward),term=term,trunc=trunc,position=np.asarray(v.position).tolist(),speed=float(v.speed_km_h),wheel=float(v.system.getSteeringValue(0))))
    if done:break
 finally:env.close()
 a.output.write_text(json.dumps(rows))
if __name__=='__main__':main()
