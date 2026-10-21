"""Reference-style training-progress validation curves for baseline01 and shallow11."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MultipleLocator


def tensorboard_ema(values, weight=.99):
 means=[];last=0.
 for index,value in enumerate(values):
  last=weight*last+(1-weight)*value
  means.append(last/(1-weight**(index+1)))
 return np.array(means),np.zeros(len(values))


def main():
 root=Path(__file__).resolve().parents[1];out=root/'logdir/training_curves_01_vs_11_20261021/tensorboard_ema099';out.mkdir(parents=True,exist_ok=True)
 specs={
 '01_500K_baseline':('chassis_baselines_500k_20261015_020330/r2dreamer','R2Dreamer (01)','#6B7280'),
 '11_500K_state4_DALI_shallow_swap1':('context_swap_dali_shallow_500k_20261021/training/r2dreamer_dali_shallow','Ours (11)','#E5474D')}
 metrics=[('episode/eval_score','Episode score',1.),('episode/eval_success','Success rate (%)',100.),('episode/eval_route_completion','Route completion (%)',100.),('episode/eval_crash','Collision rate (%)',100.)]
 data={};provenance={};export=[]
 for name,(folder,label,color) in specs.items():
  source=root/'logdir'/folder/'metrics.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines()];ev=sorted([r for r in rows if 'episode/eval_score' in r],key=lambda r:r['step'])
  assert len(ev)==51 and len({r['step'] for r in ev})==51
  assert all(int(r['episode/eval_episodes'])==20 for r in ev)
  step=np.array([r['step'] for r in ev],dtype=float);assert np.all(np.diff(step)>0)
  data[name]={'step':step,'metrics':{}}
  provenance[name]={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'evaluation_points':len(ev),'first_step':int(step[0]),'last_step':int(step[-1]),'legend_label':label}
  for key,ylabel,scale in metrics:
   raw=np.array([r[key] for r in ev])*scale;mean,std=tensorboard_ema(raw)
   assert np.isfinite(raw).all()
   lo,hi=mean-std,mean+std
   if scale==100:lo=np.clip(lo,0,100);hi=np.clip(hi,0,100)
   data[name]['metrics'][key]=(raw,mean,std,lo,hi)
   for x,y,m,s,l,h in zip(step,raw,mean,std,lo,hi):export.append({'run':name,'step':int(x),'metric':key,'raw_display_units':float(y),'tensorboard_ema099':float(m)})
 with (out/'source_data.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'axes.edgecolor':'#3F3F3F','axes.linewidth':.8,'axes.spines.top':True,'axes.spines.right':True,'legend.frameon':True,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
 fig,axs=plt.subplots(2,2,figsize=(183/25.4,190/25.4))
 fig.subplots_adjust(left=.13,right=.975,bottom=.155,top=.885,wspace=.34,hspace=.43)
 for index,(ax,(key,ylabel,scale)) in enumerate(zip(axs.flat,metrics)):
  for name,(_,label,color) in specs.items():
   step=data[name]['step'];raw,mean,std,lo,hi=data[name]['metrics'][key]
   ax.plot(step,mean,color=color,lw=1.65,label=label,zorder=3)
  ax.set_xlim(0,500010);ax.xaxis.set_major_locator(MultipleLocator(100000));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:'0' if x==0 else f'{x/1000:.0f}k'));ax.tick_params(length=3.2,width=.75,direction='out');ax.set_ylabel(ylabel);ax.grid(False)
  if key in ['episode/eval_success','episode/eval_route_completion']:
   upper=max(float(data[n]['metrics'][key][1].max()) for n in specs);ax.set_ylim(0,min(100,max(20,np.ceil(upper/10)*10)));ax.yaxis.set_major_locator(MultipleLocator(10 if upper<60 else 20))
  elif key=='episode/eval_crash':
   upper=max(float(data[n]['metrics'][key][4].max()) for n in specs);ax.set_ylim(0,max(40,np.ceil(upper/10)*10));ax.yaxis.set_major_locator(MultipleLocator(10))
  else:
   lower=min(float(data[n]['metrics'][key][3].min()) for n in specs);upper=max(float(data[n]['metrics'][key][4].max()) for n in specs);ax.set_ylim(min(-5,np.floor(lower/20)*20),np.ceil(upper/20)*20);ax.yaxis.set_major_locator(MultipleLocator(20))
  if index>=2:ax.set_xlabel('Step')
  ax.text(.5,-.19 if index<2 else -.29,f'({chr(97+index)})',transform=ax.transAxes,ha='center',va='top',fontsize=11)
 handles=[Line2D([0],[0],color=color,lw=1.8,label=label) for _,label,color in specs.values()]
 legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=2,fontsize=11,handlelength=2.,columnspacing=2.5,borderpad=.45)
 legend.get_frame().set_edgecolor('#D7D7D7');legend.get_frame().set_linewidth(.65);legend.get_frame().set_facecolor('white')
 fig.text(.5,.012,'Single seed · TensorBoard EMA smoothing = 0.99 (bias corrected)',ha='center',va='bottom',fontsize=7.1,color='#555555')
 for ext in ['png','pdf','svg']:fig.savefig(out/f'training_curves_01_vs_11.{ext}',dpi=300)
 plt.close(fig)
 (out/'README.md').write_text("# 对齐TensorBoard的EMA0.99训练曲线\n\n原始51个固定20回合验证点，与TensorBoard逐点核验一致。沿时间顺序按TensorBoard偏差修正EMA算法计算：acc_i=0.99*acc_(i-1)+0.01*y_i，初始acc=0；smooth_i=acc_i/(1-0.99^(i+1))。两组和四项指标使用相同权重。百分比面板乘100显示。没有额外高斯平滑、插值、标准差阴影或原始细线。原始数据及平滑数据保留在CSV。单训练seed。\n")

 print(out/'training_curves_01_vs_11.png')


if __name__=='__main__':main()
