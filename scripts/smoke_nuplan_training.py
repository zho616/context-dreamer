"""End-to-end baseline/old11 validation on converted nuPlan scenes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

def optimizer_steps(value):
 """Read actual Adam steps; trainer update_count also counts AMP-skipped attempts."""
 if isinstance(value,dict):
  for key,item in value.items():
   if key=='step'and(isinstance(item,(int,float))or isinstance(item,torch.Tensor)and item.numel()==1):yield float(item)
   else:yield from optimizer_steps(item)
 elif isinstance(value,(list,tuple)):
  for item in value:yield from optimizer_steps(item)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--gpu-short',action='store_true',help='Full default network, batch16x64, random chassis, 1024 steps per method on cuda:0; ratio64 and compile=false for bounded validation')
 p.add_argument('--methods',nargs='+',choices=['baseline','11_shallow'],default=['baseline','11_shallow'])
 a=p.parse_args()
 if a.gpu_short and not torch.cuda.is_available():p.error('--gpu-short requires CUDA')
 root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
 for f in root.glob('*.py'):shutil.copy2(f,source/f.name)
 for folder in ['configs','envs','optim']:shutil.copytree(root/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
 status={'phase':'starting','profile':'gpu-short'if a.gpu_short else'cpu-smoke','runner_pid':os.getpid(),'methods':{}};(out/'status.json').write_text(json.dumps(status))
 for method,conf in [('baseline','metadrive_nuplan'),('11_shallow','metadrive_nuplan_shallow')]:
  if method not in a.methods:continue
  run=out/method
  cmd=[sys.executable,'-u',str(source/'train.py'),'--config-name',conf,'device=cpu','env.steps=128','env.env_num=2','env.eval_env_num=1','env.eval_episode_num=2','env.time_limit=32','env.train_ratio=8','buffer.max_size=512','batch_size=4','batch_length=8','model.compile=false','model.deter=32','model.hidden=16','model.units=16','model.discrete=4','model.rssm.stoch=4','model.rssm.blocks=4','model.imag_horizon=3','trainer.eval_every=64','trainer.update_log_every=32',f'env.scenario_train_dir={a.data.resolve()/"train"}',f'env.scenario_eval_dir={a.data.resolve()/"val"}',
  'env.chassis.models=[default]','env.chassis.mass_scales=[1.0]','env.chassis.steering_scales=[1.0]','env.chassis.friction_scales=[1.0]',f'logdir={run}']
  if a.gpu_short:
   cmd=[sys.executable,'-u',str(source/'train.py'),'--config-name',conf,'device=cuda:0','env.steps=1024','env.env_num=8','env.eval_env_num=2','env.eval_episode_num=8','env.train_ratio=64','buffer.max_size=4096','model.compile=false','trainer.eval_every=1024','trainer.update_log_every=128',f'env.scenario_train_dir={a.data.resolve()/"train"}',f'env.scenario_eval_dir={a.data.resolve()/"val"}',f'logdir={run}']
  status['phase']=method;status['methods'][method]={'state':'running','command':cmd};(out/'status.json').write_text(json.dumps(status,indent=2))
  env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
  if not a.gpu_short:env['CUDA_VISIBLE_DEVICES']=''
  started=time.monotonic()
  with(out/f'{method}.log').open('w')as f:ret=subprocess.run(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT)
  if ret.returncode:
   status['phase']='failed';status['methods'][method].update(state='failed',returncode=ret.returncode);(out/'status.json').write_text(json.dumps(status,indent=2))
   raise RuntimeError(f'{method} failed: see {out}/{method}.log')
  payload=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
  assert payload['step'] >= (1024 if a.gpu_short else 128)
  assert payload['update_count'] > 0
  actual_steps=list(optimizer_steps(payload['optims_state_dict']))
  assert actual_steps and min(actual_steps)>0,'Optimizer did not update parameters'
  assert all(bool(torch.isfinite(p).all())for p in payload['agent_state_dict'].values()if p.is_floating_point())
  rows=[json.loads(s)for s in(run/'metrics.jsonl').read_text().splitlines()];updates=[r for r in rows if r.get('train/opt/updates',0)>0];assert updates
  if method=='11_shallow':
   if not a.gpu_short:assert any(r.get('train/context/swap_pairs',0)>0 for r in updates),'Exchange path was not exercised'
   assert all(r.get('train/context/wm_grad_scale',0)==0 for r in updates)
  assert (run/'best.pt').exists()
  for row in rows:
   for key,value in row.items():
    if key.startswith('train/')and isinstance(value,(int,float)):assert bool(torch.isfinite(torch.tensor(value))),f'Nonfinite metric: {key}'
  status['methods'][method].update(state='completed',step=payload['step'],finite_parameters=True,updates=payload['update_count'],optimizer_step_range=[min(actual_steps),max(actual_steps)],swap_exercised=any(r.get('train/context/swap_pairs',0)>0 for r in updates),elapsed_seconds=round(time.monotonic()-started,2),state_dict_elements=sum(v.numel()for v in payload['agent_state_dict'].values()),final_evaluation=next((r for r in reversed(rows)if'episode/eval_score'in r),None))
  (out/'status.json').write_text(json.dumps(status,indent=2))
  print(method,status['methods'][method]['step'],'completed',flush=True)
 status['phase']='completed';(out/'status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
