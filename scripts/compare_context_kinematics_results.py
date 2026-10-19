"""Compare three frozen R2Dreamer runs, learning curves and paired tests."""
import argparse,csv,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,FuncFormatter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--training-only',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True);root=Path(__file__).resolve().parents[1]
    base=root/'logdir/chassis_baselines_500k_20261015_020330'
    old=root/'logdir/context_chassis_500k_20261015_103141'
    new=root/'logdir/context_kinematics_500k_20261015_180340'
    runs={'baseline':base/'r2dreamer','previous_context':old/'r2dreamer_context','new_context':new/'r2dreamer_context'}
    metrics=['episode/eval_score','episode/eval_success','episode/eval_crash','episode/eval_route_completion','episode/eval_out_of_road']
    training={};curves={};csv_rows=[];hashes={}
    for name,path in runs.items():
        raw=(path/'metrics.jsonl').read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        all_rows=[json.loads(line)for line in raw.decode().splitlines()if line.strip()]
        vals=sorted({r['step']:r for r in all_rows if all(k in r for k in metrics)}.values(),key=lambda r:r['step'])
        assert all(r['episode/eval_episodes']==20 for r in vals)
        losses=sorted({r['step']:r for r in all_rows if 'train/loss/dyn'in r}.values(),key=lambda r:r['step'])
        state=json.loads((path.parent/'status.json').read_text())['methods'][path.name]
        best=state['best_step'];last=vals[-1]
        training[name]={'best_step':best,'best_eval':state['best_eval_metrics'],'last_eval':{k:last[k]for k in metrics},'last_step':last['step'],
                        'eval_points':len(vals),'elapsed_hours':None,
                        **{f'last{n}':{k:float(np.mean([r[k]for r in vals[-n:]]))for k in metrics}for n in [10,20]},
                        'last20_shared_losses':{k:float(np.mean([r[k]for r in losses[-20:]]))for k in ['train/loss/dyn','train/loss/rep','train/loss/barlow','train/loss/rew','train/loss/con']}}
        from datetime import datetime
        training[name]['elapsed_hours']=(datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['started_at'])).total_seconds()/3600
        steps=np.array([r['step']for r in vals]);values={}
        for k in metrics:
            y=np.array([r[k]for r in vals],float);sm=y.copy()
            for i in range(1,len(y)):sm[i]=.7*sm[i-1]+.3*y[i]
            values[k]=(y,sm)
        curves[name]=(steps,values)
        for i,r in enumerate(vals):csv_rows.append({'method':name,'step':r['step'],**{k:r[k]for k in metrics},**{k+'_ema':values[k][1][i]for k in metrics}})
    (out/'training_comparison.json').write_text(json.dumps(training,indent=2)+'\n')
    with(out/'learning_curve_source.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    # Figure contract: compare learning trajectory and driving metrics, without
    # treating one run as a distribution of training seeds. Raw + EMA, no fake CI.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    names=['baseline','previous_context','new_context'];labels=['R2 baseline','Previous context','Speed + wheel-angle context'];colors=['#648840','#9D91B5','#6A3D9A']
    fig,axs=plt.subplots(2,2,figsize=(8,5.7),layout='constrained')
    for ax,k,title,letter in zip(axs.flat,metrics[:4],['Evaluation reward','Success rate','Collision rate','Route completion'],'abcd'):
        for name,label,col in zip(names,labels,colors):
            steps,y=curves[name];ax.plot(steps,y[k][0],color=col,alpha=.16,lw=.6)
            ax.plot(steps,y[k][1],color=col,label=label,lw=1.8,ls='--'if name=='baseline'else'-')
        ax.set(title=title,xlabel='Environment steps',xlim=(0,500010));ax.grid(alpha=.15)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
        if 'score'not in k:ax.set_ylim(-.02,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('R2Dreamer: baseline and two context versions\nEMA weight 0.7; faint lines = raw 20-episode evaluations; one training seed',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'three_way_learning_curves.{ext}',dpi=300)
    plt.close(fig)
    (out/'figure_metadata.json').write_text(json.dumps({'source_sha256':hashes,'ema_weight':.7,'training_seeds':1,'evaluation_episodes_per_point':20,'uncertainty':'None; faint lines are raw evaluations, not confidence intervals.'},indent=2)+'\n')
    print('TRAINING',json.dumps({n:{'best_step':v['best_step'],'best_success':v['best_eval']['episode/eval_success'],'last_success':v['last_eval']['episode/eval_success'],'last10_success':v['last10']['episode/eval_success'],'last20_success':v['last20']['episode/eval_success'],'last20_losses':v['last20_shared_losses']}for n,v in training.items()},indent=2),flush=True)
    if args.training_only:return
    old_per=old/'ood_per_model100_20261015_154849'
    paths={'new_maps':{'baseline':base/'best_eval100_20261015_095447/r2dreamer','previous_context':old/'comparison_r2_baseline_20261015/context_best_eval100','new_context':out/'new_maps'},
           's':{'baseline':old_per/'s_r2_baseline','previous_context':old_per/'s_r2_context','new_context':out/'s'},
           'xl':{'baseline':old_per/'xl_r2_baseline','previous_context':old_per/'xl_r2_context','new_context':out/'xl'}}
    results={};all_comparisons=[]
    for condition,dirs in paths.items():
        summaries={name:json.loads((p/'summary.json').read_text())for name,p in dirs.items()}
        assert all(s['status']=='completed'and s['completed_episodes']==100 for s in summaries.values())
        traces={name:[json.loads(x)for x in(p/'episodes.jsonl').read_text().splitlines()]for name,p in dirs.items()}
        for trace in traces.values():assert len(trace)==100
        ref=traces['new_context']
        for name,trace in traces.items():
            assert all(a['seed']==b['seed']and a['chassis']==b['chassis']for a,b in zip(trace,ref)),(condition,name)
            assert summaries[name]['inference_seed_offset']==12345
        paired={};rng=np.random.default_rng(20261016);resample=rng.integers(0,100,(10000,100))
        for reference in ['baseline','previous_context']:
            a,b=traces[reference],ref
            only_a=sum(x['success']and not y['success']for x,y in zip(a,b));only_b=sum(not x['success']and y['success']for x,y in zip(a,b));n=only_a+only_b
            pval=min(1.,2*sum(math.comb(n,j)for j in range(min(only_a,only_b)+1))/2**n)if n else 1.
            deltas={}
            for key in ['success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h']:
                d=np.array([float(y[key])-float(x[key])for x,y in zip(a,b)])
                deltas[key]={'new_minus_reference':float(d.mean()),'paired_bootstrap95':np.quantile(d[resample].mean(1),[.025,.975]).tolist()}
            paired[reference]={'reference_only_success':only_a,'new_only_success':only_b,'mcnemar_raw_p':pval,'deltas':deltas}
            all_comparisons.append((condition,reference,pval))
        results[condition]={'summaries':{n:s['summary']for n,s in summaries.items()},'per_model':{n:s['per_model']for n,s in summaries.items()},
                            'paired':paired,'pairing_verified':True,'data_paths':{n:str(p)for n,p in dirs.items()},
                            'data_provenance':{n:{'sha256':hashlib.sha256((p/'episodes.jsonl').read_bytes()).hexdigest(),'reused_previous_test':n!='new_context'}for n,p in dirs.items()}}
        with(out/f'{condition}_paired.csv').open('w')as f:
            fields=['seed','model','mass','max_steering']+[f'{n}_{k}'for n in names for k in ['success','reward','route_completion','crash','mean_speed_km_h']]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i,r in enumerate(ref):w.writerow({'seed':r['seed'],'model':r['chassis']['model'],'mass':r['chassis']['mass'],'max_steering':r['chassis']['max_steering'],**{f'{n}_{k}':traces[n][i][k]for n in names for k in ['success','reward','route_completion','crash','mean_speed_km_h']}})
    previous=0
    for i,(condition,reference,pval)in enumerate(sorted(all_comparisons,key=lambda x:x[2])):
        adjusted=min(1.,max(previous,pval*(len(all_comparisons)-i)));previous=adjusted
        results[condition]['paired'][reference]['mcnemar_holm6_p']=adjusted
    (out/'evaluation_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    # The test figure pairs success with driving speed, so a policy's change
    # of driving behavior is visible alongside the success-rate comparison.
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.8),layout='constrained')
    for ax,key,title,letter in zip(axs,['success','mean_speed_km_h'],['Success rate','Mean episode speed'],'ab'):
        for j,(name,label,col)in enumerate(zip(names,labels,colors)):
            y=[results[c]['summaries'][name][key]for c in ['new_maps','s','xl']]
            pos=np.arange(3)+(j-1)*.24
            ax.bar(pos,y,width=.22,color=col,label=label)
            for x,v in zip(pos,y):ax.text(x,v+(.015 if key=='success'else.35),f'{v:.0%}'if key=='success'else f'{v:.1f}',ha='center',fontsize=7)
        ax.set(xticks=np.arange(3),xticklabels=['New maps','Unseen S','Unseen XL'],title=title)
        if key=='success':ax.set_ylim(0,1.08);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Success rate')
        else:ax.set_ylim(0,29);ax.set_ylabel('Mean speed (km/h)')
        ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('Best-checkpoint tests: 100 paired episodes per condition\nSame seeds and chassis; baseline/previous results reused; one training seed per method',fontsize=10)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'best_checkpoint_comparison.{ext}',dpi=300)
    plt.close(fig)
    lines=['# 新版 Context R2Dreamer 对比结果','','新版使用每步车速与实际前轮角，且辅助预测/监督从两维扩为四维。不是仅更换编码器输入的单因素消融。',
        '三者均为 seed0、500K 预算，各自按训练验证集选择 best，本次测试不重新选模型。',
        '基线及旧版复用此前完成的逐局结果；新版重新测试。评估脚本 SHA256 与此前完全相同，使用各自冻结源码。逐局核对种子和完整实际底盘参数一致。',
        '新地图种子10020–10119；S种子10320–10419；XL种子10420–10519，各100局。质量和转角scale仍为训练档位，不属于scale域外测试。',
        '','## 训练结果','','| 方法 | best步数 | best验证成功率 | 最终验证成功率 | 最后10次均值 | 最后20次均值 | 耗时h |','|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        t=training[n];lines.append(f"| {n} | {t['best_step']} | {t['best_eval']['episode/eval_success']:.1%} | {t['last_eval']['episode/eval_success']:.1%} | {t['last10']['episode/eval_success']:.1%} | {t['last20']['episode/eval_success']:.1%} | {t['elapsed_hours']:.3f} |")
    lines+=['','## 独立测试','','| 条件 | 方法 | 成功率 | 碰撞率 | 出界率 | 超时率 | 得分 | 路线完成率 | 速度km/h |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition,res in results.items():
        for name,s in res['summaries'].items():lines.append(f"| {condition} | {name} | {s['success']:.0%} | {s['crash']:.0%} | {s['out_of_road']:.0%} | {s['timeout']:.0%} | {s['reward']:.2f} | {s['route_completion']:.1%} | {s['mean_speed_km_h']:.2f} |")
    lines+=['','## 配对成功率差','','| 条件 | 参照 | 新版减参照 | 地图配对bootstrap95%区间 | 仅参照成功 | 仅新版成功 | 精确McNemar p | Holm六比较p |','|---|---|---:|---|---:|---:|---:|---:|']
    for condition,res in results.items():
        for ref,pair in res['paired'].items():
            delta=pair['deltas']['success'];lo,hi=delta['paired_bootstrap95']
            lines.append(f"| {condition} | {ref} | {100*delta['new_minus_reference']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {pair['reference_only_success']} | {pair['new_only_success']} | {pair['mcnemar_raw_p']:.5g} | {pair['mcnemar_holm6_p']:.5g} |")
    lines+=['','## 解释边界','',
        '每个方法只有一个训练seed；地图级不确定性不代表跨训练seed稳定性。训练评估每次20局，best高点不等于收敛。',
        '所有成功率/碰撞率来自不同策略自己采集的轨迹。世界模型训练loss也受不同轨迹分布影响；不能由训练loss直接断言预测能力优劣。',
        '旧版辅助loss求和两维，新版求和四维，不直接比较两个aux总量。shared loss结果见training_comparison.json，仍只是训练日志统计。',
        '尚未做新版context跨工况参数探针、context错配预测或相同速度控制，不能由驾驶结果证明底盘表征解耦。',
        '平滑曲线仅为EMA展示，浅线为原始点，没有伪造多seed阴影或置信区间。','']
    lines+=['## 本次判断','',
        '新版 best 在三组测试中均低于旧版：新地图 -24、S -15、XL -29 个百分点。相对基线的差为 -9、-6、-21 个百分点；前两组相对基线的配对区间跨零，XL 退步更明确。',
        '新版速度更高，同时成功率及路线完成率下降。速度差可能参与行为差异，但未进行速度匹配的干预实验，不能将退步直接归因于速度。',
        '结果不支持把当前新版作为旧版的性能替代，也不能推出加入车速/转角信息本身无用。编码器输入和辅助预测目标同时改变，需通过单独消融区分影响。',
        '末20条训练loss均值：dyn 旧版2.5818、新版2.5366；Barlow旧版4.1801、新版4.4261；reward旧版0.5189、新版0.5078；continuation旧版0.01216、新版0.01238。没有整体一致下降，且训练数据分布不同，不等于共同测试集预测误差。','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print('TESTS',json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
