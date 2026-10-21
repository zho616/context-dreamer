"""Paired real driving trajectories, control curves and speed-aware diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logdir/driving_stability_20261021'
METHODS=['baseline','09_deep','11_shallow']
LABELS={'baseline':'Baseline','09_deep':'09 Deep','11_shallow':'11 Shallow'}
COLORS={'baseline':'#697580','09_deep':'#3977A8','11_shallow':'#D48A42'}
METRICS=['steer_rate','throttle_rate','wheel_rate_deg','jerk_abs','accel_abs','lateral_rms','yaw_accel_abs']


def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
 rows={name:[json.loads(x) for x in (OUT/name/'episodes.jsonl').read_text().splitlines()] for name in METHODS}
 for name in METHODS:
  assert json.loads((OUT/name/'status.json').read_text())['phase']=='completed'
  assert len(rows[name])==60
 ident=lambda r:(r['split'],r['seed'],r['chassis']['model'],r['chassis']['mass'],r['chassis']['max_steering'])
 assert all(list(map(ident,rows[name]))==list(map(ident,rows['baseline'])) for name in METHODS)
 for i in range(60):
  for name in METHODS[1:]:
   np.testing.assert_allclose(rows[name][i]['initial_position'],rows['baseline'][i]['initial_position'],atol=1e-6)
   np.testing.assert_allclose(rows[name][i]['initial_heading'],rows['baseline'][i]['initial_heading'],atol=1e-6)
 data={name:[dict(np.load(OUT/name/r['trace'])) for r in rows[name]] for name in METHODS}
 for traces in data.values():
  for d in traces:
   assert all(np.isfinite(v).all() for v in d.values());assert len(d['time'])==len(d['action'])+1
   np.testing.assert_allclose(np.diff(d['time']),np.diff(d['time'])[0],atol=1e-9)
 common_end=[min(10.,min(data[name][i]['time'][-1] for name in METHODS)) for i in range(60)]
 output_rows=[]
 for name in METHODS:
  for i,(r,d) in enumerate(zip(rows[name],data[name])):
   dt=r['dt'];T=len(d['action']);t=d['time'][1:];speed=d['speed_mps'][1:]*3.6
   nan=np.array([np.nan]); values={
    'steer_rate':np.abs(np.r_[nan,np.diff(d['action'][:,0])/dt]),
    'throttle_rate':np.abs(np.r_[nan,np.diff(d['action'][:,1])/dt]),
    'wheel_rate_deg':np.rad2deg(np.abs(np.diff(d['wheel_rad'])/dt)),
    'jerk_abs':np.abs(np.r_[nan,np.diff(d['accel_long'][1:])/dt]),
    'accel_abs':np.abs(d['accel_long'][1:]),'lateral_rms':d['lateral_offset'][1:]**2,
    'yaw_accel_abs':np.abs(np.diff(d['yaw_rate_right'])/dt)}
   safe=(t>=1.)&~d['is_last'][1:]&~d['collision'][1:]&~d['collision'][:-1]
   scopes={'all_safe':safe,'common_prefix':safe&(t<=common_end[i]-dt+1e-8),
    'speed_5_15':safe&(speed>=5)&(speed<15),'speed_15_25':safe&(speed>=15)&(speed<25),'speed_25_35':safe&(speed>=25)&(speed<35)}
   for scope,mask in scopes.items():
    row={'method':name,'split':r['split'],'seed':r['seed'],'scope':scope,'success':int(r['success']),'samples':int(mask.sum()),'duration_s':float(mask.sum()*dt),'mean_speed_km_h':float(speed[mask].mean()) if mask.any() else None,
      'reward':r['reward'],'route_completion':r['route_completion'],'crash':int(r['crash']),'out_of_road':int(r['out_of_road'])}
    for k,value in values.items():
     v=value[mask&np.isfinite(value)];row[k]=(float(np.sqrt(v.mean())) if k=='lateral_rms' else float(v.mean())) if len(v) else None
    output_rows.append(row)
 with (OUT/'episode_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(output_rows[0]));w.writeheader();w.writerows(output_rows)
 rng=np.random.default_rng(20261021);summary={};paired={}
 def aggregate(selected):
  result={'episodes':len(selected),'success':float(np.mean([r['success'] for r in selected])),'mean_speed_km_h':float(np.mean([r['mean_speed_km_h'] for r in selected if r['mean_speed_km_h'] is not None]))}
  for k in METRICS:
   values=[r[k] for r in selected if r[k] is not None];result[k]=float(np.mean(values)) if values else None
  return result
 for split in ['new_maps','s','xl']:
  summary[split]={};paired[split]={}
  for scope in ['all_safe','common_prefix','speed_5_15','speed_15_25','speed_25_35']:
   summary[split][scope]={name:aggregate([r for r in output_rows if r['method']==name and r['split']==split and r['scope']==scope and r['samples']>=5]) for name in METHODS}
   paired[split][scope]={}
   for name in METHODS[1:]:
    paired[split][scope][name]={}
    aa={r['seed']:r for r in output_rows if r['method']=='baseline' and r['split']==split and r['scope']==scope and r['samples']>=5}
    bb={r['seed']:r for r in output_rows if r['method']==name and r['split']==split and r['scope']==scope and r['samples']>=5}
    for k in METRICS+['mean_speed_km_h']:
     ids=[seed for seed in sorted(set(aa)&set(bb)) if aa[seed][k] is not None and bb[seed][k] is not None]
     if not ids:continue
     x=np.array([bb[seed][k]-aa[seed][k] for seed in ids]);samples=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
     paired[split][scope][name][k]={'n':len(x),'difference':float(x.mean()),'paired_bootstrap95':np.quantile(samples,[.025,.975]).tolist()}
 driving={split:{name:{k:float(np.mean([r[k] for r in rows[name] if r['split']==split])) for k in ['success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h']} for name in METHODS} for split in ['new_maps','s','xl']}
 result={'driving':driving,'summary':summary,'paired':paired,'protocol':{'episodes_per_method':60,'episodes_per_split':20,'warmup_excluded_seconds':1,'dt_seconds':rows['baseline'][0]['dt'],'precision':'Matches prior driving evaluation: float32 high matmul','averaging':'Equal episode means. Stability excludes terminal/collision target frames; exclude episode/scope with <5 samples. Jerk is control-rate finite difference of projected longitudinal acceleration; no smoothing/filtering.',
 'scope':'One training seed. New evaluation is the first20-map subset of earlier100-map tests; do not add as independent episodes. Policy alters subsequent traffic interactions.',
 'common_prefix':'Per scenario all models truncated at min(10s, earliest termination), with 1s warmup and endpoint margin.',
 'speed_bins':'5-15,15-25,25-35 km/h, descriptive speed-stratified diagnostic, not a causal speed-matched experiment.',
 'example_selection':'First seed in each split, fixed before outcomes. Also export every episode example.'}}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 # Export fixed first-seed illustrations, and all paired scenario traces for audit.
 examples=OUT/'examples';examples.mkdir(exist_ok=True)
 def save(fig,path):
  fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)
 def plot_case(index,full_exports=True):
  ref=rows['baseline'][index];split=ref['split'];seed=ref['seed'];geom=json.loads((OUT/'baseline/maps'/f'{split}_{seed}.json').read_text())
  fig=plt.figure(figsize=(11.2,7.0),layout='constrained');grid=fig.add_gridspec(3,3,width_ratios=[1.15,1,1]);ax=fig.add_subplot(grid[:,0]);axes=[fig.add_subplot(grid[r,c]) for r in range(3) for c in [1,2]]
  for lane in geom:
   for key in ['left','right']:
    pts=np.array(lane[key]);ax.plot(pts[:,0],pts[:,1],color='#D4D7DA',lw=.55,zorder=0)
  for name in METHODS:
   d=data[name][index];r=rows[name][index];color=COLORS[name];pos=d['position'];time=d['time'];action=d['action'];dt=r['dt']
   outcome='success' if r['success'] else ('crash' if r['crash'] else ('out' if r['out_of_road'] else 'timeout'))
   ax.plot(pos[:,0],pos[:,1],color=color,lw=1.4,label=LABELS[name]+' · '+outcome);ax.scatter(*pos[-1],s=28,marker='o' if r['success'] else 'x',color=color,zorder=5)
   items=[(time[:-1],action[:,0],'Steering action (left +)'),(time[:-1],action[:,1],'Throttle / brake'),(time,np.rad2deg(d['wheel_rad']),'Front-wheel angle (deg, right +)'),(time,d['speed_mps']*3.6,'Speed (km/h)'),(time[1:],d['accel_long'][1:],'Long. acceleration (m/s²)'),(time[2:],np.diff(d['accel_long'][1:])/dt,'Long. jerk (m/s³)')]
   for sub,(tt,yy,label) in zip(axes,items):sub.plot(tt,yy,color=color,lw=.85,alpha=.9);sub.set_ylabel(label);sub.grid(axis='y',color='#E8EAED',lw=.45)
  ax.scatter(*data['baseline'][index]['position'][0],marker='s',s=22,color='black',label='Start');ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.legend(loc='best',fontsize=7);ax.set_title('a  Actual trajectories',loc='left',fontweight='bold')
  for j,sub in enumerate(axes):sub.set_title(chr(98+j),loc='left',fontweight='bold');sub.set_xlabel('Time (s)')
  for sub in axes[:2]:sub.set_ylim(-1.08,1.08)
  chassis=ref['chassis'];fig.suptitle(f'{split.upper()} · seed {seed} · {chassis["model"]} · mass {chassis["mass"]} kg · max steer {chassis["max_steering"]}°',fontsize=11)
  path=examples/f'{split}_{seed}'
  if full_exports:save(fig,path)
  else:fig.savefig(path.with_suffix('.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 for index in range(60):plot_case(index,index in [0,20,40])
 # Overview: every raw episode and mean/95% bootstrap, columns share split categories.
 figure,axs=plt.subplots(2,3,figsize=(10.6,6.1),layout='constrained')
 keys=['success','mean_speed_km_h','steer_rate','throttle_rate','wheel_rate_deg','jerk_abs'];titles=['Success rate','Speed (km/h)','Steering action rate (1/s)','Throttle / brake rate (1/s)','Front-wheel rate (deg/s)','Longitudinal jerk (m/s³)']
 splits=['new_maps','s','xl']
 for ax,key,title in zip(axs.flat,keys,titles):
  for mi,name in enumerate(METHODS):
   for si,split in enumerate(splits):
    vals=np.array([r[key] for r in output_rows if r['method']==name and r['split']==split and r['scope']=='all_safe' and r['samples']>=5 and r[key] is not None]);x=si+(mi-1)*.23
    if key!='success':ax.scatter(x+np.linspace(-.04,.04,len(vals)),vals,s=8,alpha=.3,color=COLORS[name])
    mean=vals.mean();bs=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(1);lo,hi=np.quantile(bs,[.025,.975]);ax.errorbar(x,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',ms=4,color=COLORS[name],capsize=2,lw=1.1)
  ax.set_xticks(range(3),['New maps','S','XL']);ax.set_title(title,loc='left');ax.grid(axis='y',color='#E8EAED',lw=.5)
 axs.flat[0].set_ylim(-.03,1.05)
 figure.legend(handles=[Line2D([0],[0],marker='o',color=COLORS[n],label=LABELS[n],lw=1) for n in METHODS],loc='outside upper center',ncol=3,title='20 paired scenarios per split · means with 95% episode bootstrap',title_fontsize=10)
 save(figure,OUT/'stability_overview')
 lines=['# 实际轨迹、控制曲线与稳定性评估','','冻结基线、09、11的best，严格加载各自训练源码；新地图/S/XL各20回合，每组60，共180回合。实际地图种子、底盘车型/质量/最大转角、初始位置和航向均逐回合核验一致。该样本是先前100回合评估的前20个种子，不能与旧结果叠加计算独立样本数。','','## 驾驶表现','','|测试池|方法|成功率|碰撞率|出界率|速度km/h|路线完成率|','|---|---|---:|---:|---:|---:|---:|']
 for split in splits:
  for name in METHODS:
   r=driving[split][name];lines.append(f'|{split}|{name}|{r["success"]:.0%}|{r["crash"]:.0%}|{r["out_of_road"]:.0%}|{r["mean_speed_km_h"]:.2f}|{r["route_completion"]:.1%}|')
 lines+=['','## 每回合等权的稳定性指标','','排除初始1秒、结束/碰撞帧；原始决策频率信号，不作平滑。转向/油门变化率为mean(|Δu|/Δt)；实际轮角变化率单位deg/s；jerk为纵向投影加速度差分，单位m/s³；加速度单位m/s²。轮角/横摆角速度均右正；转向动作保留MetaDrive左正，两者符号不同。更小的控制变化率不能单独代表更安全。','','|测试池|方法|转向动作变化率|油门变化率|前轮角变化率|平均绝对jerk|平均绝对加速度|车道横偏RMS m|','|---|---|---:|---:|---:|---:|---:|---:|']
 for split in splits:
  for name in METHODS:
   r=summary[split]['all_safe'][name];lines.append(f'|{split}|{name}|'+'|'.join(f'{r[k]:.3f}' for k in ['steer_rate','throttle_rate','wheel_rate_deg','jerk_abs','accel_abs','lateral_rms'])+'|')
 lines+=['','## 速度分层与相同时间窗配对差','','以下差值为方法减基线，负值表示变化率/jerk更小。速度分层不是严格的同状态控制，子集回合数可少于20。common_prefix使用每个场景三方法共同可用的前≤10秒，排除初始1秒；样本不足5帧的回合不进入该指标。95%区间按配对地图种子bootstrap10000次；一个训练seed。','','|测试池|范围|方法|指标|配对回合数|差值|95%区间|','|---|---|---|---|---:|---:|---|']
 for split in splits:
  for scope in ['all_safe','common_prefix','speed_15_25','speed_25_35']:
   for name in METHODS[1:]:
    for k in ['steer_rate','throttle_rate','wheel_rate_deg','jerk_abs']:
     if k not in paired[split][scope][name]:continue
     r=paired[split][scope][name][k];lo,hi=r['paired_bootstrap95'];lines.append(f'|{split}|{scope}|{name}|{k}|{r["n"]}|{r["difference"]:+.3f}|[{lo:+.3f}, {hi:+.3f}]|')
 lines+=['','## 图与解释边界','','- stability_overview.png/pdf/svg：全部回合分布、均值和bootstrap区间。','- examples/new_maps_10020、s_10320、xl_10420：预先固定的首个种子，均导出PNG/PDF/SVG；examples内还有全部60个种子的PNG，未按优势挑选案例。','- 每张案例图包含实际位置轨迹和原始控制、轮角、速度、加速度、jerk曲线，碰撞/出界结束点也保留用于展示。统计时事件帧被排除。静态道路背景来自实际生成的同一地图；不是世界模型生成。交通车辆省略，策略交互后其他车辆可出现不同响应。','- 当前lane的横偏会受车道切换影响，不能单独作为稳定性证据。异常帧和结束造成的生存偏差需结合common_prefix、速度分层和失败率阅读。','- 这些图不涉及预测解码器，也不能证明RSSM未来重建更准，结论仅限闭环驾驶与控制。','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 print(OUT/'REPORT.md')


if __name__=='__main__':main()
