"""Render the already-audited fixed-history random/best context PCA pair only."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parents[1];out=root/'logdir/fixed_vehicle_random_vs_trained_20261020'
a=np.load(out/'representations.npz');r=json.loads((out/'results.json').read_text());rows=list(csv.DictReader((out/'source_data.csv').open()))
types=['default','m','l','s','xl'];colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2'];y=np.array([types.index(x['vehicle']) for x in rows]);order=np.random.default_rng(13).permutation(len(y))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
fig,axs=plt.subplots(1,2,figsize=(10,4.7));fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.78,wspace=.3)
for ax,key,title,label in zip(axs,['random','trained'],['Untrained encoder output: c_random','Best encoder output: c_best'],['a','b']):
 z=a[key+'_pca'];ratio=r['pca_variance_ratios'][key]
 ax.scatter(*z[order].T,c=np.array(colors)[y[order]],s=29,alpha=.65,edgecolors='none')
 ax.set(title=title,xlabel=f'PC1 ({ratio[0]:.1%})',ylabel=f'PC2 ({ratio[1]:.1%})');ax.grid(alpha=.12);ax.set_axisbelow(True)
 ax.text(-.13,1.08,label,transform=ax.transAxes,fontweight='bold',fontsize=13)
fig.suptitle('Identical histories → two encoders → 8D context → PCA\nInitial speed 10 m/s; actual front-wheel angle 15°; 24 windows per type',fontsize=11,y=.97)
handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=m.upper(),markersize=6) for m,c in zip(types,colors)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=5)
fig.text(.5,.025,f'Random seed {r["random_seeds"][0]}; best step {r["checkpoint_step"]:,}. Separate PCA bases; neither panel shows raw inputs or independent noise.',ha='center',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(out/f'encoder_pair_only.{ext}',dpi=300)
plt.close(fig)
# Audit response similarity in the cached fixed-condition probes; no new simulation.
data=root/'logdir/context_kinematics_500k_20261015_180340/analysis/vehicle_context_pca'
source=list(csv.DictReader((data/'windows.csv').open()));d=np.load(data/'representations.npz')
result={}
for typ in ['m','l']:
 ii=[i for i,x in enumerate(source) if x['model']==typ and round(float(x['initial_speed_m_s']))==10 and round(float(x['front_left_angle_degrees']))==15]
 h=d['history'][ii]
 result[typ]={'mean_history_acceleration':float(h[:,:,0].mean()),'mean_history_yaw_rate':float(h[:,:,1].mean()),'mean_final_speed':float(np.mean([float(source[i]['final_speed_m_s']) for i in ii]))}
(out/'M_L_probe_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
