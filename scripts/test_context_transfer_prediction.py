"""Frozen E+D context transfer: donor chassis x donor operating condition."""
import argparse,csv,hashlib,importlib.util,itertools,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_modules(source,checkpoint,name):
    spec=importlib.util.spec_from_file_location(name,source/'context.py');module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    encoder=module.HistoryEncoder().eval();predictor=module.ResidualDynamics().eval()
    for obj,prefix in [(encoder,'context_encoder.'),(predictor,'context_dynamics.')]:
        obj.load_state_dict({k[len(prefix):]:v for k,v in payload['agent_state_dict'].items()if k.startswith(prefix)},strict=True)
    step=payload['step'];del payload
    return module,encoder,predictor,step


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    out=args.directory;root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
    rows=list(csv.DictReader((out/'windows.csv').open()));d=np.load(out/'representations.npz')
    H=d['history'];truth=d['future_states'];actions=d['future_actions']
    assert H.shape==(1680,5,6)and truth.shape==(1680,5,4)and actions.shape==(1680,5,2)
    assert np.isfinite(H).all()and np.isfinite(truth).all()
    np.testing.assert_allclose(truth[:,:,:2],d['physical_paths'][:,5:,:2],rtol=1e-6,atol=1e-6)
    # Verify the protocol against the prior collection, allowing measured physics
    # to vary on an independent simulator run. Both encoders below use H exactly.
    previous=np.load(out.parent/'vehicle_context_pca/representations.npz')['history']
    support_max_diff=float(np.abs(previous-H).max())
    np.testing.assert_array_equal(previous[...,3:],H[...,3:])
    support_channel_max_diff=np.abs(previous-H).max(axis=(0,1)).tolist()
    models=['default','m','l','s','xl'];model=np.array([r['model']for r in rows]);profile=np.array([int(r['repeat'])for r in rows])
    speed=np.rint([float(r['initial_speed_m_s'])for r in rows]).astype(int)
    angle=np.rint([float(r['front_left_angle_degrees'])for r in rows]).astype(int);angles=sorted(set(angle))
    lookup={(model[i],speed[i],angle[i],profile[i]):i for i in range(len(rows))}
    donors={}
    for condition in ['matched','cross_speed','cross_angle','cross_both']:
        for chassis in ['same','other']:
            indices=[]
            for i in range(len(rows)):
                ms=[model[i]]if chassis=='same'else[m for m in models if m!=model[i]]
                vs=[15-speed[i]]if condition in ['cross_speed','cross_both']else[speed[i]]
                aa=[a for a in angles if a!=angle[i]]if condition in ['cross_angle','cross_both']else[angle[i]]
                indices.append([lookup[(m,v,a,profile[i])]for m,v,a in itertools.product(ms,vs,aa)])
            donors[chassis+'/'+condition]=np.array(indices,int)
    donors['same/other_profile']=np.array([[lookup[(model[i],speed[i],angle[i],(profile[i]+1)%24)]]for i in range(len(rows))])
    assert np.array_equal(donors['same/matched'][:,0],np.arange(len(rows)))
    target=torch.from_numpy(truth);a=torch.from_numpy(actions)
    def predict_errors(module,predictor,c,state0,indices=None,zero=False,teacher=False):
        k=1 if zero else indices.shape[1]
        donor_c=torch.zeros(len(rows),1,8)if zero else c[torch.from_numpy(indices)]
        context=donor_c.reshape(-1,8);state=state0[:,None,:].expand(-1,k,-1).reshape(-1,state0.shape[-1])
        errors=[];predictions=[]
        for horizon in range(5):
            if teacher and horizon:
                state=target[:,horizon-1,:state0.shape[-1]][:,None,:].expand(-1,k,-1).reshape(-1,state0.shape[-1])
            act=a[:,horizon,None,:].expand(-1,k,-1).reshape(-1,2)
            if hasattr(module,'predict_next_state'):state=module.predict_next_state(predictor,context,state,act)
            else:state=state+predictor(context,state,act)
            predicted=state.reshape(len(rows),k,-1)
            error=(predicted[...,:2]-target[:,horizon,None,:2]).abs().mean(1)
            errors.append(error.numpy());predictions.append(predicted.mean(1).numpy())
        return np.stack(errors,1),np.stack(predictions,1)
    run_roots={'old':root/'logdir/context_chassis_500k_20261015_103141','new':root/'logdir/context_kinematics_500k_20261015_180340'}
    errors={};teacher_errors={};own_predictions={};checkpoints={}
    with torch.inference_mode():
        for name,run in run_roots.items():
            checkpoint=run/'r2dreamer_context/best.pt'
            module,E,D,step=load_modules(run/'source',checkpoint,'frozen_prediction_'+name)
            history=H if name=='new'else H[...,[0,1,4,5]]
            c=E(torch.from_numpy(history),torch.ones(len(rows),5,dtype=torch.bool));state0=torch.from_numpy(history[:,-1,:-2].copy())
            errors[name]={};teacher_errors[name]={}
            for variant,indices in donors.items():
                errors[name][variant],pred=predict_errors(module,D,c,state0,indices=indices)
                teacher_errors[name][variant],_=predict_errors(module,D,c,state0,indices=indices,teacher=True)
                if variant=='same/matched':own_predictions[name]=pred
            errors[name]['zero'],_=predict_errors(module,D,c,state0,zero=True)
            teacher_errors[name]['zero'],_=predict_errors(module,D,c,state0,zero=True,teacher=True)
            errors[name]['persistence']=np.abs(history[:,-1,None,:2]-truth[:,:,:2])
            true_previous=np.concatenate([history[:,-1,None,:2],truth[:,:-1,:2]],1)
            teacher_errors[name]['persistence']=np.abs(true_previous-truth[:,:,:2])
            checkpoints[name]={'step':step,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source':str(run/'source')}
    def score(e):return (e/np.array([10.,1.])).sum(-1)
    def stats(e):
        s=score(e)
        return {'acceleration_mae_m_s2':float(e[:,:,0].mean()),'yaw_rate_mae_rad_s':float(e[:,:,1].mean()),
            'normalized_l1_mean_h1_h5':float(s.mean()),'normalized_l1_h1':float(s[:,0].mean()),'normalized_l1_h5':float(s[:,-1].mean()),
            'normalized_l1_by_horizon':s.mean(0).tolist()}
    summary={n:{v:stats(e)for v,e in modes.items()}for n,modes in errors.items()}
    teacher_summary={n:{v:stats(e)for v,e in modes.items()}for n,modes in teacher_errors.items()}
    comparisons={};rng=np.random.default_rng(20261016);resample=rng.integers(0,24,(10000,24))
    for name in errors:
        comparisons[name]={}
        for condition in ['matched','cross_speed','cross_angle','cross_both']:
            correct=score(errors[name]['same/'+condition]).mean(1);wrong=score(errors[name]['other/'+condition]).mean(1)
            delta=wrong-correct;by_profile=np.array([delta[profile==r].mean()for r in range(24)])
            comparisons[name]['wrong_minus_same/'+condition]={'mean':float(delta.mean()),
                'profile_bootstrap95':np.quantile(by_profile[resample].mean(1),[.025,.975]).tolist(),
                'fraction_queries_wrong_is_worse':float(np.mean(delta>0)),
                'relative_mean_error_change':float(wrong.mean()/correct.mean()-1)}
        own=score(errors[name]['same/matched']).mean(1)
        for condition in ['cross_speed','cross_angle','cross_both']:
            cross=score(errors[name]['same/'+condition]).mean(1);delta=cross-own
            by_profile=np.array([delta[profile==r].mean()for r in range(24)])
            comparisons[name]['cross_minus_own/'+condition]={'mean':float(delta.mean()),
                'profile_bootstrap95':np.quantile(by_profile[resample].mean(1),[.025,.975]).tolist(),
                'relative_mean_error_change':float(cross.mean()/own.mean()-1)}
    split_summary={}
    for split,mask in [('training_types',np.isin(model,models[:3])),('unseen_types',np.isin(model,models[3:]))]+[(m,model==m)for m in models]:
        split_summary[split]={name:{v:stats(e[mask])for v,e in modes.items()}for name,modes in errors.items()}
    result={'checkpoints':checkpoints,'target_windows':len(rows),'independent_action_profiles':24,'support_history_max_difference_from_PCA_data':support_max_diff,
        'support_channel_max_difference_from_PCA_data':support_channel_max_diff,
        'summary':summary,'teacher_forced_one_step_summary':teacher_summary,'paired_comparisons':comparisons,'subgroups':split_summary,
        'donor_contexts_per_target':{k:int(v.shape[1])for k,v in donors.items()},
        'methodology':{'branch':'Auxiliary E+D only; no RSSM, actor or world-model intervention.',
            'time_alignment':'Support tokens end at t=5. Start from true state x5. Target x6..x10 uses executed canonical actions a5..a9. No future targets enter context.',
            'rollout':'Fixed donor context throughout five steps, matching the auxiliary training convention. New model predicts all four channels and uses its speed/angle predictions after step one; old model predicts two.',
            'teacher_forcing':'Separate diagnostic: every one-step prediction receives the true CURRENT state, including speed/angle for the new model; not imagination and not future endpoint leakage.',
            'averaging':'Evaluate each donor separately and average absolute errors per target; never average donor contexts before nonlinear prediction.',
            'normalization':'Common score = |acceleration error|/10 + |yaw-rate error|. All reporting excludes the new-only speed/angle targets.',
            'pairing':'Same target state, true trajectory and future actions for every donor comparison; other chassis use the same selected donor operating condition and throttle seed.',
            'bootstrap':'Resample 24 action-profile blocks, not individual donor hypotheses. Descriptive intervals conditional on fixed trained models and probe design.'},
        'limitations':['Only five fixed native parameter sets and one wide flat road; no continuous mass/steering identification claim.',
            'Common physical wheel angles require chassis-dependent normalized steering commands; steering gain can be inferred from observable angle/action pairs.',
            'Context swaps change history-derived motion information as well as chassis information; matched-condition contrasts reduce but do not eliminate this issue.',
            'Large-angle held-steering probes can differ from the on-policy training history distribution.',
            'Old best is 150k, new best 480k; architecture inputs and auxiliary targets both differ.',
            'Positive wrong-context degradation establishes local predictive dependence, not complete parameter disentanglement or RSSM/control benefit.']}
    (out/'prediction_results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'prediction_errors.npz',**{f'{n}__{v.replace("/","_")}':e for n,modes in errors.items()for v,e in modes.items()},
        old_own_predictions=own_predictions['old'],new_own_predictions=own_predictions['new'])
    with(out/'prediction_summary.csv').open('w')as f:
        fields=['version','variant','acceleration_mae_m_s2','yaw_rate_mae_rad_s','normalized_l1_mean_h1_h5','normalized_l1_h1','normalized_l1_h5']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for n,modes in summary.items():
            for v,s in modes.items():w.writerow({'version':n,'variant':v,**{k:s[k]for k in fields[2:]}})
    with(out/'per_target_errors.csv').open('w')as f:
        fields=['rollout','model','profile','initial_speed','angle','version','variant','normalized_l1','acceleration_mae','yaw_rate_mae']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for n,modes in errors.items():
            for v,e in modes.items():
                for i,r in enumerate(rows):w.writerow({'rollout':r['rollout'],'model':model[i],'profile':profile[i],'initial_speed':speed[i],'angle':angle[i],
                    'version':n,'variant':v,'normalized_l1':float(score(e[i]).mean()),'acceleration_mae':float(e[i,:,0].mean()),'yaw_rate_mae':float(e[i,:,1].mean())})
    # Figure contract: distinguish transfer failure from chassis specificity,
    # and show physical errors alongside the common normalized score.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    conditions=['matched','cross_speed','cross_angle','cross_both'];ticks=['Own condition','Other speed','Other angle','Other speed + angle']
    colors={'old':'#9D91B5','new':'#6A3D9A'}
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.5),layout='constrained')
    for name,label in [('old','Previous context'),('new','New context')]:
        for ax,key in [(axs[0,0],'normalized_l1_mean_h1_h5'),(axs[1,0],'acceleration_mae_m_s2'),(axs[1,1],'yaw_rate_mae_rad_s')]:
            ax.plot(range(4),[summary[name]['same/'+c][key]for c in conditions],marker='o',color=colors[name],label=label)
        vals=[comparisons[name]['wrong_minus_same/'+c]['mean']for c in conditions]
        cis=[comparisons[name]['wrong_minus_same/'+c]['profile_bootstrap95']for c in conditions]
        offset=-.035 if name=='old'else .035
        axs[0,1].errorbar(np.arange(4)+offset,vals,yerr=np.array([[v-ci[0]for v,ci in zip(vals,cis)],[ci[1]-v for v,ci in zip(vals,cis)]]),color=colors[name],marker='o',capsize=3,label=label)
    axs[0,0].set(title='Same chassis: transfer to another condition',ylabel='Mean normalized prediction error')
    axs[0,1].axhline(0,color='#666666',ls='--',lw=.8);axs[0,1].set(title='Wrong chassis minus same chassis',ylabel='Error increase (positive = wrong is worse)')
    axs[1,0].set(title='Same-chassis acceleration prediction',ylabel='MAE (m/s²)')
    axs[1,1].set(title='Same-chassis yaw-rate prediction',ylabel='MAE (rad/s)')
    for ax,letter in zip(axs.flat,'abcd'):
        ax.set_xticks(range(4),ticks,rotation=15,ha='right');ax.grid(alpha=.15);ax.set_axisbelow(True)
        ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Frozen auxiliary dynamics: context-transfer intervention\n1,680 targets; five-step open-loop prediction; paired donor comparisons',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'context_transfer_intervention.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.8),layout='constrained')
    styles=[('same/matched','Own context','#6A3D9A'),('same/cross_both','Same chassis / other condition','#4C78A8'),
            ('other/matched','Wrong chassis / matched condition','#E28B43'),('zero','Zero context','#999999'),('persistence','Persistence','#333333')]
    for ax,name,title in zip(axs,['old','new'],['Previous context','New context']):
        for variant,label,color in styles:ax.plot(np.arange(1,6),summary[name][variant]['normalized_l1_by_horizon'],label=label,color=color,marker='o',ms=3,ls='--'if variant=='persistence'else'-')
        ax.set(title=title,xlabel='Prediction horizon (control steps)',ylabel='Normalized prediction error',xticks=range(1,6));ax.grid(alpha=.15)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=7,frameon=False)
    fig.suptitle('Auxiliary prediction vs. persistence baseline',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'prediction_horizon_controls.{ext}',dpi=300)
    plt.close(fig)
    lines=['# Context 跨工况与跨底盘预测干预','','## 方法','',
        '本次检验冻结的辅助编码器E和预测器D，不是RSSM或策略干预。新旧模型均strict加载各自best。',
        '1680条物理轨迹，每车型336条；24条配对油门序列，初速5/10m/s和实际前轮角七档。每条先取五步历史，再采五步未来真值。',
        '每次替换只改变context：目标真实起点和未来动作完全相同。固定donor context预测五步，匹配辅助训练方式。',
        '同车型/不同车型×同工况/换速度/换角度/速度角度都换。换角度平均其余六档，换车型平均其余四类；逐个预测后平均误差，不先平均context。',
        '两模型仅比较共同的加速度、横摆角速度，归一化误差=|加速度误差|/10+|角速度误差|。另列物理单位MAE。',
        '','## 五步平均预测误差','','| 版本 | Context来源 | 加速度MAE m/s² | 角速度MAE rad/s | 归一化L1 | 第一步L1 | 第五步L1 |','|---|---|---:|---:|---:|---:|---:|']
    for name,modes in summary.items():
        for variant,s in modes.items():lines.append(f"| {name} | {variant} | {s['acceleration_mae_m_s2']:.4f} | {s['yaw_rate_mae_rad_s']:.4f} | {s['normalized_l1_mean_h1_h5']:.4f} | {s['normalized_l1_h1']:.4f} | {s['normalized_l1_h5']:.4f} |")
    lines+=['','## 换成其他底盘是否变差','','| 版本 | donor工况 | 错底盘减同底盘误差 | 动作序列bootstrap95% | 平均误差变化 | 目标中错底盘更差的比例 |','|---|---|---:|---|---:|---:|']
    for name,pairs in comparisons.items():
        for condition in conditions:
            s=pairs['wrong_minus_same/'+condition];lo,hi=s['profile_bootstrap95']
            lines.append(f"| {name} | {condition} | {s['mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {s['relative_mean_error_change']:+.1%} | {s['fraction_queries_wrong_is_worse']:.1%} |")
    lines+=['','## Teacher-forcing 单步诊断','',
        '每一步改用真实当前状态，仍使用固定donor context；用于区分当前状态条件和累计预测漂移，不是可部署的想象展开。',
        '| 版本 | Context来源 | 单步平均归一化L1 |','|---|---|---:|']
    for name,modes in teacher_summary.items():
        for variant,s in modes.items():lines.append(f"| {name} | {variant} | {s['normalized_l1_mean_h1_h5']:.4f} |")
    lines+=['','## 限制','']+[f'- {x}'for x in result['limitations']]+['',
        '所有区间按24条油门序列成组重采样，不能视为多训练seed证据。完整逐车型与训练/未见车型分解见prediction_results.json。',
        '匹配context本身是否优于保持当前值不变也必须一起看；如果预测器在此探测分布上不准确，替换实验不能直接证明参数辨识失败。','']
    lines+=['## 结果解读','',
        '新版在原工况的五步平均共同目标误差为0.0707，旧版为0.2207；新版绝对预测误差更低，但两者训练阶段、输入和预测目标均不同，不能视为纯编码输入消融。',
        '新版同底盘换初速后误差0.1505（+113.0%），换前轮角后0.0947（+34.0%），两者都换0.1644（+132.6%）。因此目前不支持同底盘context可稳定跨工况复用。',
        'donor工况匹配时，换底盘使新版误差从0.0707升至0.1125（+59.1%），84.8%的目标在对其他四种底盘求平均后变差，表明存在与底盘身份相关的预测信息。',
        '但donor车速和转角都不匹配时，同/异底盘仅为0.1644/0.1671（+1.7%），且仅54.0%的目标变差。训练三车型子集甚至为0.1642/0.1623，不能宣称对所有底盘均有稳定的正确context优势。',
        '保持当前加速度和角速度不变的五步误差为0.0565，低于新版原工况0.0707；第一步也为0.0270对0.0496。因探测油门平滑且转向固定，未来变化较小，当前预测器尚未超过此简单基准。',
        '每一步重置为真实当前状态后，新版原工况/换速度/两者都换误差仍为0.0528/0.1158/0.1235，因此问题并非仅由五步自回归漂移造成。',
        '结合前述PCA，较合理的解释是新版context同时承载底盘相关线索与当前运动工况；尚不能称为与工况分离的车辆参数编码。此干预没有测试RSSM或驾驶性能。',
        '新旧编码器在本次完全相同的历史上比较。独立重新采集与上次PCA历史的动作及前轮角逐值一致，物理数值最大差为0.00440m/s²、0.000694rad/s和0.000609m/s，已单独记录，未把两次采样当作逐值完全一致。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'summary':summary,'paired':comparisons,'teacher':teacher_summary},indent=2))


if __name__=='__main__':main()
