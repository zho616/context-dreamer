"""Explicitly selected favorable control segment; not an aggregate result."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 root=Path(__file__).resolve().parents[1]/'logdir/driving_stability_20261021'
 out=root/'selected_controls';out.mkdir(exist_ok=True)
 split,seed,start,stop='xl',10431,70,150
 traces={name:dict(np.load(root/name/'traces'/f'{split}_{seed}.npz')) for name in ['baseline','11_shallow']}
 rows={name:next(json.loads(x) for x in (root/name/'episodes.jsonl').read_text().splitlines() if json.loads(x)['seed']==seed and json.loads(x)['split']==split) for name in traces}
 assert all(r['success'] for r in rows.values())
 assert rows['baseline']['chassis']==rows['11_shallow']['chassis']
 dt=rows['baseline']['dt'];time=traces['baseline']['time'][start:stop]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,axes=plt.subplots(2,1,figsize=(7.2,3.65),sharex=True,layout='constrained')
 styles={'baseline':('Baseline','#697580'),'11_shallow':('Ours (11)','#D48A42')}
 for j,(ax,label) in enumerate(zip(axes,['Steering','Throttle / brake'])):
  for name,(method,color) in styles.items():ax.plot(time,traces[name]['action'][start:stop,j],label=method,color=color,lw=1.35 if name=='11_shallow' else 1.05)
  ax.set_ylabel(label);ax.set_ylim(-1.08,1.08);ax.set_yticks([-1,0,1]);ax.grid(axis='y',color='#E6E8EB',lw=.55)
 axes[0].legend(loc='upper right',ncol=2,fontsize=9)
 axes[1].set_xlabel('Time (s)');axes[1].set_xlim(start*dt,stop*dt)
 fig.suptitle('Selected control segment · XL · seed 10431 · 7–15 s',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'baseline_vs_11_XL_10431_7_15s.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
 with (out/'source_data.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['time_s','baseline_steering','ours11_steering','baseline_throttle_brake','ours11_throttle_brake'])
  for i,t in enumerate(time):writer.writerow([t,traces['baseline']['action'][start+i,0],traces['11_shallow']['action'][start+i,0],traces['baseline']['action'][start+i,1],traces['11_shallow']['action'][start+i,1]])
 metrics={}
 for name,d in traces.items():
  rates=np.abs(np.diff(d['action'][start:stop],axis=0)).mean(0)/dt
  metrics[name]={'mean_abs_steering_rate_per_s':float(rates[0]),'mean_abs_throttle_brake_rate_per_s':float(rates[1]),'mean_speed_km_h':float(d['speed_mps'][start:stop].mean()*3.6),'episode_success':rows[name]['success']}
 metadata={'split':split,'seed':seed,'time_interval_s':[7,15],'samples':stop-start,'sampling_interval_s':dt,'chassis':rows['baseline']['chassis'],'metrics':metrics,
 'selection':'Post-hoc favorable illustration explicitly requested by user. Among both-success episodes, scan8s windows at1s increments, starting>=2s; require both mean speeds>=15km/h, speed difference<=5km/h, and both steering/throttle rates at least15% below baseline. Select the minimum sum of log rate ratios. Not representative of all episodes or evidence of average improvement.',
 'signals':'Recorded raw actor-mode steering (left positive) and throttle/brake, no filtering/smoothing or resampling.',
 'figure_contract':{'conclusion':'Illustrate a favorable segment in which11 has lower steering and throttle/brake variability than baseline.','archetype':'quantitative grid','backend':'python','dimensions_in':[7.2,3.65],'exports':['png','pdf','svg','csv'],'risk':'Outcome-selected example, not aggregate inference.'}}
 (out/'selection.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
