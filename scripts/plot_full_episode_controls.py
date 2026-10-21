"""Full-episode baseline/11 control and common geometric-route reference comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logdir/driving_stability_20261021'


def reference_routes():
 source=ROOT/'logdir/chassis_baselines_500k_20261015_020330/source';sys.path.insert(0,str(source))
 from omegaconf import OmegaConf
 from envs.metadrive import MetaDrive
 rows={name:{(r['split'],r['seed']):r for r in map(json.loads,(OUT/name/'episodes.jsonl').read_text().splitlines())} for name in ['baseline','11_shallow']}
 folder=OUT/'full_episode';folder.mkdir(exist_ok=True);refs={};scores=[]
 cfg=OmegaConf.load(source.parent/'r2dreamer/.hydra/config.yaml');cfg.device='cpu';OmegaConf.resolve(cfg)
 for split,seed_start in [('new_maps',10020),('s',10320),('xl',10420)]:
  cfg.env.eval_start_seed=seed_start;cfg.env.eval_num_scenarios=20;cfg.env.eval_episode_num=20;cfg.env.eval_env_num=1;cfg.env.chassis.eval_models=['default','m','l'] if split=='new_maps' else [split]
  env=MetaDrive(cfg.env,evaluation=True)
  try:
   for key,r in rows['baseline'].items():
    if key[0]!=split or not(r['success'] and rows['11_shallow'][key]['success']):continue
    obs=env.reset(seed=r['seed']);car=env._env.vehicle
    np.testing.assert_allclose(car.position,r['initial_position'],atol=1e-6);assert env.chassis_parameters==r['chassis']
    checkpoints=car.navigation.checkpoints;graph=env._env.current_map.road_network.graph
    lane=car.lane;lanes=[lane];edges=list(zip(checkpoints[:-1],checkpoints[1:]));selected=[list(lane.index)]
    for frm,to in edges[1:]:
     candidates=graph[frm][to];end=np.array(lane.position(lane.length,0));heading=lane.heading_theta_at(lane.length)
     def cost(candidate):
      gap=np.linalg.norm(np.array(candidate.position(0,0))-end);turn=abs((candidate.heading_theta_at(0)-heading+np.pi)%(2*np.pi)-np.pi)
      return gap+turn*.1
     lane=min(candidates,key=cost);lanes.append(lane);selected.append(list(lane.index))
    pieces=[]
    for j,lane in enumerate(lanes):
     start=float(lane.local_coordinates(car.position)[0]) if j==0 else 0.
     points=np.array([lane.position(float(s),0) for s in np.linspace(max(0,start),lane.length,max(2,int((lane.length-start)/.25)+1))])
     pieces.append(points)
    ref=np.concatenate(pieces)
    refs[f'{split}_{r["seed"]}']={'points':ref.tolist(),'checkpoints':checkpoints,'lane_indices':selected,'construction':'Route shortest-path edges from navigation; start at spawn lane center, then choose next lane with closest geometric endpoint (heading tie-break). Independent of either policy trajectory; no fitted smoothing. Not an explicit policy tracking target.'}
    vals={}
    for name in rows:
     row=rows[name][key];d=dict(np.load(OUT/name/row['trace']));pos=d['position'];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
     # Spatial rather than temporal weighting prevents waiting from dominating error.
     keep=np.r_[True,np.diff(arc)>1e-8];ss=np.arange(0,arc[-1],.25);sample=np.column_stack([np.interp(ss,arc[keep],pos[keep,k]) for k in [0,1]])
     err=polyline_distance(sample,ref);rates=np.abs(np.diff(d['action'],axis=0)).mean(0)/row['dt']
     vals[name]={'route_rmse_m':float(np.sqrt(np.mean(err**2))),'route_mae_m':float(err.mean()),'route_p95_m':float(np.quantile(err,.95)),'steering_rate':float(rates[0]),'throttle_rate':float(rates[1]),'mean_speed_km_h':row['mean_speed_km_h'],'duration_s':float(d['time'][-1]),'success':row['success'],'reward':row['reward']}
    base,ours=vals['baseline'],vals['11_shallow'];ratio=np.array([ours[k]/base[k] for k in ['route_rmse_m','steering_rate','throttle_rate']])
    scores.append({'split':split,'seed':r['seed'],'metrics':vals,'ratios':ratio.tolist(),'score':float(np.log(ratio).sum()),'chassis':r['chassis']})
  finally:env.close()
 (folder/'reference_routes.json').write_text(json.dumps(refs,indent=2)+'\n');(folder/'candidate_scores.json').write_text(json.dumps(sorted(scores,key=lambda r:r['score']),indent=2)+'\n')
 print(json.dumps(sorted(scores,key=lambda r:r['score'])[:8],indent=2),flush=True)


def polyline_distance(points,reference):
 start=reference[:-1];vec=np.diff(reference,axis=0);den=(vec*vec).sum(-1);good=den>1e-12;start=start[good];vec=vec[good];den=den[good]
 result=[]
 for i in range(0,len(points),128):
  delta=points[i:i+128,None]-start[None];t=np.clip((delta*vec[None]).sum(-1)/den[None],0,1);dist=np.linalg.norm(delta-t[...,None]*vec[None],axis=-1);result.append(dist.min(-1))
 return np.concatenate(result)


def plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 folder=OUT/'full_episode';candidates=json.loads((folder/'candidate_scores.json').read_text());refs=json.loads((folder/'reference_routes.json').read_text())
 # Select an episode with better full-episode control smoothness AND route error,
 # both successes and no large mean-speed difference, not a cropped segment.
 eligible=[r for r in candidates if max(r['ratios'])<1 and abs(r['metrics']['baseline']['mean_speed_km_h']-r['metrics']['11_shallow']['mean_speed_km_h'])<=5]
 assert eligible,'No qualifying favorable full episode'
 selected=min(eligible,key=lambda r:max(r['ratios']));split,seed=selected['split'],selected['seed'];key=f'{split}_{seed}';ref=np.array(refs[key]['points']);geom=json.loads((OUT/'baseline/maps'/f'{key}.json').read_text())
 data={name:dict(np.load(OUT/name/'traces'/f'{key}.npz')) for name in ['baseline','11_shallow']}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig=plt.figure(figsize=(10,5.6),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.05,1.5]);ax=fig.add_subplot(grid[:,0]);steer=fig.add_subplot(grid[0,1]);throttle=fig.add_subplot(grid[1,1],sharex=steer)
 for lane in geom:
  for k in ['left','right']:
   xy=np.array(lane[k]);ax.plot(xy[:,0],xy[:,1],color='#DCE0E3',lw=.65,zorder=0)
 ax.plot(ref[:,0],ref[:,1],color='#1F3933',ls='--',lw=1.4,label='Reference (lane center)',zorder=2)
 styles={'baseline':('Baseline','#697580'),'11_shallow':('Ours (11)','#D48A42')}
 allpos=[]
 for name,(label,color) in styles.items():
  d=data[name];pos=d['position'];allpos.append(pos);ax.plot(pos[:,0],pos[:,1],color=color,lw=1.6,label=label,zorder=3);ax.scatter(*pos[-1],marker='o',s=25,color=color,zorder=4)
  steer.plot(d['time'][:-1],d['action'][:,0],color=color,lw=1.1,label=label)
  throttle.plot(d['time'][:-1],d['action'][:,1],color=color,lw=1.1)
 ax.scatter(*data['baseline']['position'][0],marker='s',s=24,color='black',label='Start',zorder=4)
 ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_title('a  Full-episode trajectories',loc='left',fontweight='bold');ax.legend(loc='best',fontsize=8)
 for axis,label,title in [(steer,'Steering','b  Steering'),(throttle,'Throttle / brake','c  Throttle / brake')]:
  axis.set_ylabel(label);axis.set_ylim(-1.08,1.08);axis.set_yticks([-1,0,1]);axis.grid(axis='y',color='#E6E8EB',lw=.5);axis.set_title(title,loc='left',fontweight='bold');axis.set_xlim(0,max(d['time'][-1] for d in data.values()));axis.set_xlabel('Time (s)')
 steer.legend(loc='upper right',fontsize=8,ncol=2)
 chassis=selected['chassis'];fig.suptitle(f'Full episode · {split.upper()} · seed {seed} · {chassis["model"]} · {chassis["mass"]} kg · max steer {chassis["max_steering"]}°',fontsize=11)
 for ext in ['png','pdf','svg']:fig.savefig(folder/f'baseline_vs_11_full_{key}.{ext}',dpi=230,bbox_inches='tight')
 plt.close(fig)
 # Separate full-route and full-episode action figures, matching the requested
 # trajectory/control presentation without omitting any part of either rollout.
 fig,axis=plt.subplots(figsize=(7.2,3.35),layout='constrained')
 for lane in geom:
  for border in ['left','right']:
   xy=np.array(lane[border]);axis.plot(xy[:,0],xy[:,1],color='#DCE0E3',lw=.7,zorder=0)
 axis.plot(ref[:,0],ref[:,1],color='#1F3933',ls='--',lw=1.45,label='Reference (lane center)')
 for name,(label,color) in styles.items():
  pos=data[name]['position'];axis.plot(pos[:,0],pos[:,1],color=color,lw=1.6,label=label);axis.scatter(*pos[-1],s=25,color=color,zorder=5)
 axis.scatter(*data['baseline']['position'][0],s=25,marker='s',color='black',label='Start',zorder=5)
 bounds=np.concatenate([ref]+[d['position'] for d in data.values()]);axis.set_xlim(bounds[:,0].min()-6,bounds[:,0].max()+6);axis.set_ylim(bounds[:,1].min()-7,bounds[:,1].max()+12)
 axis.set_aspect('equal');axis.set_xlabel('World x (m)');axis.set_ylabel('World y (m)');axis.legend(loc='upper center',ncol=4,fontsize=8);axis.set_title(f'Full-episode trajectories · seed {seed}',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(folder/f'trajectories_{key}.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(7.2,3.65),sharex=True,layout='constrained')
 for j,(axis,label) in enumerate(zip(axes,['Steering','Throttle / brake'])):
  for name,(method,color) in styles.items():axis.plot(data[name]['time'][:-1],data[name]['action'][:,j],label=method,color=color,lw=1.2 if name=='11_shallow' else 1.05)
  axis.set_ylabel(label);axis.set_ylim(-1.08,1.08);axis.set_yticks([-1,0,1]);axis.grid(axis='y',color='#E6E8EB',lw=.5)
 axes[0].legend(loc='upper right',ncol=2,fontsize=8);axes[1].set_xlabel('Time (s)');axes[1].set_xlim(0,max(d['time'][-1] for d in data.values()));fig.suptitle(f'Full-episode controls · seed {seed}',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(folder/f'controls_{key}.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
 selected.update(reference=refs[key],selection='Post-hoc favorable full episode requested by user. Both policies successful. Full-episode route RMSE and steering/throttle mean absolute change rates all lower for11; mean speeds differ<=5km/h. Among eligible episodes minimize the worst of the three ratios, favoring balanced full-episode improvements. No temporal cropping.',metric='Route errors are distance-weighted distances from actual trajectory to shared geometric polyline, sampled every0.25m. Not measured error to an explicit controller target.')
 (folder/'selected_episode.json').write_text(json.dumps(selected,indent=2)+'\n')
 for name,d in data.items():
  with (folder/f'{name}_{key}_source.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['time_s','x_m','y_m','steering','throttle_brake'])
   for i in range(len(d['time'])):w.writerow([d['time'][i],*d['position'][i],*(d['action'][i] if i<len(d['action']) else [None,None])])
 np.savetxt(folder/f'reference_{key}.csv',ref,delimiter=',',header='x_m,y_m',comments='')
 print(json.dumps({k:v for k,v in selected.items() if k!='reference'},indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reference','plot'],required=True);a=p.parse_args()
 reference_routes() if a.phase=='reference' else plot()
