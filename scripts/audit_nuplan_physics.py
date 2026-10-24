"""Validate physical ego actuation, chassis and context alignment on real nuPlan scenes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import hydra
from omegaconf import OmegaConf
from envs.nuplan import NuPlanDrive

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with hydra.initialize(config_path='../configs',version_base=None):c=hydra.compose(config_name='metadrive_nuplan_shallow',overrides=['device=cpu','model.compile=false'])
 c.env.scenario_train_dir=str(a.data.resolve()/'train');c.env.scenario_eval_dir=str(a.data.resolve()/'val')
 env=NuPlanDrive(c.env,evaluation=True);rows=[]
 try:
  for seed in range(min(8,env._num_scenarios)):
   obs=env.reset(seed=seed);v=env._env.vehicle;params=env.chassis_parameters
   assert env.observation_space.contains(obs)
   assert not obs['ctx_mask'].any()
   np.testing.assert_allclose(v.body.getMass(),params['mass'])
   np.testing.assert_allclose([v.system.getWheel(i).getFrictionSlip()for i in range(4)],params['wheel_friction'],rtol=1e-6)
   for t in range(6):
    action=np.array([.15,.3],np.float32);obs,reward,done,info=env.step(action)
    assert np.isfinite(reward)and env.observation_space.contains(obs)
    assert np.isclose(reward,sum(info['reward_'+k]for k in ['progress','speed','lateral','terminal']))
    np.testing.assert_allclose([v.system.getSteeringValue(i)for i in [0,1]],action[0]*params['max_steering'],atol=1e-4)
    np.testing.assert_allclose(obs['ctx_prev_action'],[action[1],-action[0]],atol=1e-6)
    np.testing.assert_allclose(obs['ctx_ego_state'][3],-np.deg2rad(action[0]*params['max_steering']),atol=1e-5)
    if done:break
   rows.append(dict(scenario_index=seed,scenario_id=env._env.engine.data_manager.current_scenario_id,steps=t+1,obs_dim=len(obs['state']),chassis=params,reward=float(reward),context_mask_count=int(obs['ctx_mask'].sum()),ego_policy=type(env._env.engine.get_policy(v.id)).__name__))
 finally:env.close()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'passed':True,'scenarios':rows},indent=2));print('PHYSICS_AUDIT_PASSED',len(rows),flush=True)
if __name__=='__main__':main()
