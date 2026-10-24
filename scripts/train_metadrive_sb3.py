"""PPO/SAC baselines using exactly the repository's MetaDrive physics/reward adapter."""
import argparse
import functools
import json
from pathlib import Path
import sys
import numpy as np
import gymnasium as gym
from omegaconf import OmegaConf

class StateEnv(gym.Env):
    """Expose only LiDAR/state; preserve termination vs time-limit truncation."""
    def __init__(self, config, env_id=0, evaluation=False):
        from envs.metadrive import MetaDrive
        config=OmegaConf.create(config)
        if config.task=='metadrive_nuplan':
            from envs.nuplan import NuPlanDrive
            MetaDrive=NuPlanDrive
        self.native=MetaDrive(config,env_id=env_id,evaluation=evaluation)
        self.observation_space=self.native.observation_space['state']
        self.action_space=self.native.action_space
        self.total=0.;self.length=0;self.speeds=[]
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Native adapter owns seeded per-worker map/chassis RNGs. SB3's worker
        # seeds must not override its map split or force explicit-spawn mode.
        obs=self.native.reset();self.total=0.;self.length=0;self.speeds=[]
        return obs['state'],{}
    def begin_evaluation(self):self.native.begin_evaluation()
    def step(self,action):
        obs,reward,done,info=self.native.step(action)
        self.total+=reward;self.length+=1;self.speeds.append(float(self.native._env.vehicle.speed_km_h))
        terminated=bool(obs['is_terminal']);truncated=bool(done and not terminated)
        if done:
            info['episode_result']=dict(reward=self.total,steps=self.length,
                success=bool(info.get('arrive_dest',False)),crash=bool(info.get('crash',False)),
                out_of_road=bool(info.get('out_of_road',False)),timeout=truncated,
                route_completion=float(info.get('route_completion',0)),
                mean_speed_km_h=float(np.mean(self.speeds)),chassis=self.native.chassis_parameters,
                seed=int(self.native._env.current_seed))
        return obs['state'],reward,terminated,truncated,info
    def close(self):self.native.close()

def make_vec(config, n, evaluation=False):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    plain=OmegaConf.to_container(config,resolve=True)
    return SubprocVecEnv([functools.partial(StateEnv,plain,i,evaluation) for i in range(n)],start_method='spawn')

def evaluate(model,env,episodes):
    env.env_method('begin_evaluation');obs=env.reset();rows=[]
    n=env.num_envs;counts=np.zeros(n,int);quotas=np.array([episodes//n+(i<episodes%n) for i in range(n)])
    while (counts<quotas).any():
        actions,_=model.predict(obs,deterministic=True)
        obs,_,done,infos=env.step(actions)
        for i in range(n):
            if done[i] and counts[i]<quotas[i]:
                rows.append(infos[i]['episode_result']);counts[i]+=1
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['ppo','sac'],required=True)
    p.add_argument('--steps',type=int,default=500000);p.add_argument('--smoke',action='store_true')
    p.add_argument('--eval-every',type=int,help='Validation interval in environment steps')
    a=p.parse_args();sys.path.insert(0,str(a.source.resolve()))
    import torch,stable_baselines3
    from stable_baselines3 import PPO,SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure, KVWriter
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1)
    c=OmegaConf.load(a.config);OmegaConf.resolve(c);c.env.context_enabled=False
    nuplan=c.env.task=='metadrive_nuplan'
    if not nuplan:c.env.eval_env_num=2;c.env.eval_episode_num=20
    if a.smoke:c.env.env_num=2;c.env.eval_env_num=1;c.env.eval_episode_num=1;c.env.time_limit=16;a.steps=32
    if a.steps%int(c.env.env_num):raise ValueError('Budget must divide env count')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(c,out/'config.yaml')
    interval=16 if a.smoke else (a.eval_every or 10000)
    if interval<1:raise ValueError('Validation interval must be positive')
    settings=dict(policy='MlpPolicy',learning_rate=3e-4,gamma=.99,seed=int(c.seed),policy_kwargs=dict(net_arch=[256,256]),verbose=0)
    if a.method=='ppo':settings.update(n_steps=8 if a.smoke else 1250,batch_size=8 if a.smoke else 250,n_epochs=2 if a.smoke else 10,device='cpu')
    else:settings.update(buffer_size=256 if a.smoke else 500000,learning_starts=8 if a.smoke else 10000,batch_size=8 if a.smoke else 256,train_freq=1,gradient_steps=int(c.env.env_num),device='cpu' if a.smoke else str(c.device))
    (out/'hyperparameters.json').write_text(json.dumps(dict(method=a.method,steps=a.steps,stable_baselines3=stable_baselines3.__version__,settings=settings,observation='Native state only, no context or chassis metadata',normalization='none',eval_every=interval,eval_episodes=int(c.env.eval_episode_num)),indent=2))
    training=validation=None;writer=SummaryWriter(str(out));metrics=(out/'metrics.jsonl').open('w',buffering=1)
    def log(step,values):
        metrics.write(json.dumps(dict(step=step,**values))+'\n')
        for k,v in values.items():writer.add_scalar(k,v,step)
        writer.flush()
    class Callback(BaseCallback):
        def __init__(self):super().__init__();self.best=None;self.last_eval=-1
        def assess(self):
            rows=evaluate(self.model,validation,int(c.env.eval_episode_num))
            values={f'episode/eval_{tag}':float(np.mean([r[key] for r in rows])) for tag,key in [('score','reward'),('length','steps'),('success','success'),('crash','crash'),('out_of_road','out_of_road'),('route_completion','route_completion')]}
            values['episode/eval_episodes']=len(rows)
            log(self.num_timesteps,values);key=tuple(values['episode/eval_'+k] for k in ['success','route_completion','score'])
            if self.best is None or key>self.best:
                self.best=key;self.model.save(out/'best')
                (out/'best.json').write_text(json.dumps(dict(step=self.num_timesteps,metrics=values)))
            self.last_eval=self.num_timesteps
            print(json.dumps(dict(step=self.num_timesteps,**values)),flush=True)
        def _on_training_start(self):self.assess()
        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode_result' in info:
                    r=info['episode_result'];log(self.num_timesteps,{'episode/score':r['reward'],'episode/length':r['steps'],'episode/success':float(r['success'])})
            if self.num_timesteps-self.last_eval>=interval:self.assess()
            return True
    try:
        training=make_vec(c.env,int(c.env.env_num));validation=make_vec(c.env,int(c.env.eval_env_num),True)
        cls=PPO if a.method=='ppo' else SAC
        class SharedScalars(KVWriter):
            # One writer for both evaluation and SB3 updates avoids concurrent
            # event streams being interpreted as restarted runs by TensorBoard.
            def write(self, key_values, key_excluded, step=0):
                for key,value in key_values.items():
                    if 'tensorboard' not in key_excluded.get(key, ()) and isinstance(value,(int,float,np.number)):
                        writer.add_scalar(key,value,step)
                writer.flush()
            def close(self):pass
        model=cls(env=training,**settings)
        sb3_logger=configure(str(out),['csv']);sb3_logger.output_formats.append(SharedScalars());model.set_logger(sb3_logger)
        callback=Callback();model.learn(total_timesteps=a.steps,callback=callback)
        # Last PPO rollout updates happen after the step callback, so reevaluate.
        callback.num_timesteps=model.num_timesteps;callback.assess();model.save(out/'latest')
        assert model.num_timesteps==a.steps,(model.num_timesteps,a.steps)
        assert model._n_updates>0
        (out/'training_summary.json').write_text(json.dumps(dict(steps=model.num_timesteps,updates=model._n_updates,finite_parameters=all(bool(torch.isfinite(x).all()) for x in model.policy.parameters()))))
        if not all(bool(torch.isfinite(x).all()) for x in model.policy.parameters()):raise RuntimeError('Nonfinite weights')
        validation.close();validation=None;training.close();training=None
        model=cls.load(out/'best',device='cpu')
        tests=([('test',0,None),('s',0,['s']),('xl',0,['xl'])] if nuplan else
               [('new_maps',10020,None),('s',10320,['s']),('xl',10420,['xl'])])
        for name,start,pool in tests:
            ec=OmegaConf.create(OmegaConf.to_container(c.env,resolve=True));count=1 if a.smoke else 100
            ec.eval_start_seed=start;ec.eval_num_scenarios=count;ec.eval_env_num=1;ec.eval_episode_num=count
            if nuplan:ec.scenario_eval_dir=c.env.scenario_test_dir
            if pool:ec.chassis.eval_models=pool
            validation=make_vec(ec,1,True);rows=evaluate(model,validation,count)
            folder=out/'evaluation'/name;folder.mkdir(parents=True)
            (folder/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary={k:float(np.mean([r[k] for r in rows])) for k in ['reward','steps','success','crash','out_of_road','timeout','route_completion','mean_speed_km_h']}
            (folder/'summary.json').write_text(json.dumps(dict(summary=summary,episodes=count,seed_start=start),indent=2))
            validation.close();validation=None
    finally:
        if training is not None:training.close()
        if validation is not None:validation.close()
        writer.close();metrics.close()

if __name__=='__main__':main()
