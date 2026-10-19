"""Same-trajectory old/new context PCA, factor associations and grouped vehicle probes."""
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def standardize(x,train):
    sd=x[train].std(0)
    # Do not amplify floating-point differences in nominally constant sensors.
    return (x-x[train].mean(0))/np.where(sd<1e-6,1.,sd)


def pca(x,train):
    x=standardize(x,train);_,s,v=np.linalg.svd(x[train],full_matrices=False)
    return x@v[:2].T,(s[:2]**2/(s*s).sum()).tolist()


def predict(x,y,train,test,kind='linear'):
    x=standardize(x,train)
    if kind=='linear':
        a=np.column_stack([np.ones(len(x)),x]);pen=np.eye(a.shape[1]);pen[0,0]=0
        weights=np.linalg.solve(a[train].T@a[train]+pen,a[train].T@np.eye(5)[y[train]])
        return (a[test]@weights).argmax(1)
    distance=(x[test]**2).sum(1)[:,None]+(x[train]**2).sum(1)[None,:]-2*x[test]@x[train].T
    nearest=np.argpartition(distance,4,axis=1)[:,:5]
    return np.eye(5)[y[train][nearest]].sum(1).argmax(1)


def design(labels):
    return np.column_stack([labels==u for u in np.unique(labels)[1:]])


def sse(y,features):
    a=np.column_stack([np.ones(len(y)),features]);res=y-a@np.linalg.lstsq(a,y,rcond=None)[0]
    return float((res*res).sum())


def eta(y,labels):
    return 1-sse(y,design(labels))/sse(y,np.empty((len(y),0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();out=args.directory;root=Path(__file__).resolve().parents[1]
    old=root/'logdir/context_chassis_500k_20261015_103141';new=root/'logdir/context_kinematics_500k_20261015_180340'
    rows=list(csv.DictReader((out/'windows.csv').open()));d=np.load(out/'representations.npz');H=d['history'].astype(np.float32)
    assert H.shape==(1680,5,6)
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('frozen_old_context',old/'source/context.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    E=module.HistoryEncoder().eval();payload=torch.load(old/'r2dreamer_context/best.pt',map_location='cpu',weights_only=False)
    prefix='context_encoder.';E.load_state_dict({k[len(prefix):]:v for k,v in payload['agent_state_dict'].items()if k.startswith(prefix)},strict=True)
    old_step=payload['step'];del payload
    old_h=H[...,[0,1,4,5]]
    with torch.inference_mode():old_c=E(torch.from_numpy(old_h),torch.ones(len(H),5,dtype=torch.bool)).numpy()
    C=d['learned_context'].astype(float)
    models=['default','m','l','s','xl'];labels=np.array([r['model']for r in rows]);y=np.array([models.index(m)for m in labels])
    group=np.array([int(r['repeat'])for r in rows]);speed0=np.rint([float(r['initial_speed_m_s'])for r in rows]).astype(int)
    angle=np.rint([float(r['front_left_angle_degrees'])for r in rows]).astype(int)
    actual_speed=np.array([float(r['final_speed_m_s'])for r in rows])
    assert all(sum(labels==m)==336 for m in models)
    assert set(angle)=={-30,-15,-5,0,5,15,30}
    known=np.column_stack([speed0/30,angle/60,H[...,4:].reshape(len(H),-1)])
    spaces={'old_context':old_c.astype(float),'new_context':C,
            'raw_old20':(old_h/np.array([10,1,1,1])).reshape(len(H),-1),
            'raw_new30':(H/np.array([10,1,30,np.pi/3,1,1])).reshape(len(H),-1),
            'protocol_only':known,'random_new_encoder':d['random_context'].astype(float)}
    groups=np.random.default_rng(20261016).permutation(np.unique(group));folds=np.array_split(groups,6)
    plot_train=~np.isin(group,folds[0]);fixed=(speed0==10)&(angle==15)
    factors={};coordinates={};ratios={};metrics={};predictions={}
    M=design(labels);I=design(speed0);A=design(angle);G=design(group)
    condition=design(np.array([f'{v}/{a}'for v,a in zip(speed0,angle)]));N=np.column_stack([condition,G])
    zspeed=(actual_speed-actual_speed.mean())/actual_speed.std();V=np.column_stack([zspeed,zspeed*zspeed])
    for name,x in spaces.items():
        coordinates[name],ratios[name]=pca(x,plot_train)
        xstd=standardize(x,plot_train)
        factors[name]={'model_eta2_full':eta(xstd,labels),'initial_speed_eta2_full':eta(xstd,speed0),
            'actual_wheel_angle_eta2_full':eta(xstd,angle),
            'actual_speed_quadratic_r2_full':1-sse(xstd,V)/sse(xstd,np.empty((len(H),0))),
            'model_partial_r2_given_speed_angle_profile':1-sse(xstd,np.column_stack([N,M]))/sse(xstd,N),
            'initial_speed_partial_r2_given_model_angle_profile':1-sse(xstd,np.column_stack([M,A,G,I]))/sse(xstd,np.column_stack([M,A,G])),
            'angle_partial_r2_given_model_speed_profile':1-sse(xstd,np.column_stack([M,I,G,A]))/sse(xstd,np.column_stack([M,I,G])),
            'PC_correlations':{f'PC{j+1}':{'speed_pearson':float(np.corrcoef(coordinates[name][:,j],actual_speed)[0,1]),
                                        'wheel_angle_pearson':float(np.corrcoef(coordinates[name][:,j],angle)[0,1]),
                                        'model_eta2':eta(coordinates[name][:,j],labels)}for j in range(2)}}
        metrics[name]={};predictions[name]={}
        for test_type in ['mixed','fixed_10mps_15deg','5_to_10','10_to_5']:
            for kind in ['linear','knn5','PCA2_linear']:
                pred=np.full(len(y),-1,dtype=int)
                for held in folds:
                    train=~np.isin(group,held);test=~train
                    if test_type=='fixed_10mps_15deg':train&=fixed;test&=fixed
                    elif test_type=='5_to_10':train&=speed0==5;test&=speed0==10
                    elif test_type=='10_to_5':train&=speed0==10;test&=speed0==5
                    feature=pca(x,train)[0]if kind=='PCA2_linear'else x
                    pred[test]=predict(feature,y,train,test,'linear'if kind=='PCA2_linear'else kind)
                valid=pred>=0;confusion=np.zeros((5,5),int);np.add.at(confusion,(y[valid],pred[valid]),1)
                metrics[name][test_type+'/'+kind]={'accuracy':float(np.mean(pred[valid]==y[valid])),
                    'test_samples':int(valid.sum()),'confusion':confusion.tolist()}
                predictions[name][test_type+'/'+kind]=pred
    # Paired differences by held-out action-profile group; descriptive bootstrap,
    # not independent training seeds and not an exact test of algorithm superiority.
    delta_ci={};rng=np.random.default_rng(20261016);unique=np.unique(group)
    for test_type in ['mixed','fixed_10mps_15deg','5_to_10','10_to_5']:
        key=test_type+'/linear';po=predictions['old_context'][key];pn=predictions['new_context'][key];valid=(po>=0)&(pn>=0)
        by_group=np.array([np.mean((pn[valid&(group==g)]==y[valid&(group==g)]).astype(float)-(po[valid&(group==g)]==y[valid&(group==g)]).astype(float))for g in unique])
        boot=by_group[rng.integers(0,len(unique),(10000,len(unique)))].mean(1)
        delta_ci[test_type]={'new_minus_old':float(by_group.mean()),'action_profile_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    null=[];rng=np.random.default_rng(1723);train=plot_train;test=~train
    for _ in range(100):
        yp=y.copy()
        for g in unique:
            for v in [5,10]:
                for a in np.unique(angle):
                    idx=np.flatnonzero((group==g)&(speed0==v)&(angle==a));yp[idx]=rng.permutation(yp[idx])
        null.append(float(np.mean(predict(C,yp,train,test)==yp[test])))
    result={'samples':len(H),'samples_per_model':336,'action_profiles':24,'old_checkpoint_step':old_step,
        'new_checkpoint_step':json.loads((out/'sampling_metadata.json').read_text())['checkpoint_step'],
        'old_checkpoint_sha256':hashlib.sha256((old/'r2dreamer_context/best.pt').read_bytes()).hexdigest(),
        'pca_variance_ratios':ratios,'factors':factors,'vehicle_probes':metrics,'new_minus_old_linear':delta_ci,
        'label_shuffle_control':{'runs':100,'mean_accuracy':float(np.mean(null)),'max_accuracy':float(max(null))},
        'methodology':{'PCA':'Each representation standardized and fitted on first-fold training profiles, all five types, then test profiles projected. One basis per representation, shared across factor-color panels.',
            'classification':'Six-fold blocked CV by 24 throttle-profile seeds. All five types are supervised probe labels, including S/XL; not zero-shot classification. Ridge one-hot alpha=1; kNN k=5; no tuning.',
            'cross_speed':'Train profiles at one initial speed; evaluate held-out profiles at the other speed. Profiles never cross train/test.',
            'eta2':'Weighted between-group centroid sum of squares / total sum of squares, using standardized representations. Descriptive variance association, not percentage of semantic information.',
            'conditional':'Partial R2 removes listed main effects; denominators differ and values are not additive or causal contributions.',
            'bootstrap':'Resample 24 held-out action-profile groups; descriptive interval conditional on these fitted CV probes, not training-seed uncertainty.',
            'numeric_precision':'Standard-deviation floor 1e-6 avoids amplifying near-constant floating-point sensor differences.'},
        'limitations':['One wide straight road, natural five-step responses; only starting speed is controlled. No velocity clamping.',
            'Five fixed native parameter sets; classification cannot isolate mass, steering range, force or geometry identification.',
            'Given actual wheel angle and normalized steering, max steering can be directly inferred for nonzero actions. Protocol-only control tests this source of type information.',
            'Encoder architecture/input and auxiliary targets changed together; this is a representation comparison, not a pure input ablation.',
            'No statement that higher probe accuracy improves driving or establishes complete condition invariance.']}
    (out/'association_results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'compared_representations.npz',old_context=old_c,new_context=C,**{f'{name}_PCA2':z for name,z in coordinates.items()})
    with(out/'association_summary.csv').open('w')as f:
        keys=[k for k in factors['new_context']if k!='PC_correlations'];w=csv.DictWriter(f,fieldnames=['representation']+keys);w.writeheader()
        for name,values in factors.items():w.writerow({'representation':name,**{k:values[k]for k in keys}})
    with(out/'probe_summary.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['representation','test','accuracy','test_samples'])
        for name,tests in metrics.items():
            for test,values in tests.items():w.writerow([name,test,values['accuracy'],values['test_samples']])
    with(out/'pca_source_data.csv').open('w')as f:
        fields=list(rows[0])+[f'{name}_pc{j+1}'for name in ['old_context','new_context','raw_new30']for j in range(2)]+['plot_split']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):w.writerow({**r,**{f'{name}_pc{j+1}':coordinates[name][i,j]for name in ['old_context','new_context','raw_new30']for j in range(2)},'plot_split':'train'if plot_train[i]else'test'})
    # Figure contract: compare model and condition associations on identical
    # physical trajectories; same coordinates across color panels, no synthetic clusters.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2']
    fig,axs=plt.subplots(2,3,figsize=(11,6.5),layout='constrained')
    for row,name in enumerate(['old_context','new_context']):
        z=coordinates[name];r=ratios[name]
        for j,ax in enumerate(axs[row]):
            if j==0:
                order=np.random.default_rng(13).permutation(len(y))
                for k,(m,col)in enumerate(zip(models,colors)):
                    idx=order[y[order]==k];ax.scatter(*z[idx].T,color=col,s=10,alpha=.35,label=m.upper(),edgecolors='none',rasterized=True)
            else:
                values=actual_speed if j==1 else angle
                sc=ax.scatter(*z.T,c=values,cmap='viridis'if j==1 else'RdBu_r',s=10,alpha=.4,edgecolors='none',rasterized=True)
                fig.colorbar(sc,ax=ax,shrink=.78,label='Speed (m/s)'if j==1 else'Wheel angle (deg)')
            ax.set(xlabel=f'PC1 ({r[0]:.1%})',ylabel=f'PC2 ({r[1]:.1%})',title=('Previous best (150k)'if row==0 else'New best (480k)')+' — '+['vehicle','speed','wheel angle'][j])
            ax.text(-.13,1.07,chr(97+row*3+j),transform=ax.transAxes,fontweight='bold',fontsize=11)
    legend=fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=8,markerscale=1.5)
    for handle in legend.legend_handles:handle.set_alpha(1)
    fig.suptitle('Old vs. new context on the same 1,680 physical windows\nEach row uses identical coordinates across colors; 336 windows per native vehicle type',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'context_pca_factors.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.6),layout='constrained')
    for ax,name,title in zip(axs,['old_context','new_context','raw_new30'],['Previous context (8D)','New context (8D)','Raw new history (30D)']):
        z,r=pca(spaces[name],plot_train&fixed)
        for k,(m,col)in enumerate(zip(models,colors)):
            q=fixed&(y==k);ax.scatter(*z[q].T,s=19,color=col,alpha=.55,label=m.upper())
        ax.set(title=title,xlabel=f'PC1 ({r[0]:.1%})',ylabel=f'PC2 ({r[1]:.1%})')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Fixed starting speed 10 m/s and measured wheel angle 15°\n24 distinct throttle profiles per type; PCA refitted within this preselected condition',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'fixed_condition_vehicle_pca.{ext}',dpi=300)
    plt.close(fig)
    lines=['# Context 车型、速度及前轮角关联对比','','## 采样与比较','',
        '同一批1680个物理窗口分别输入旧版best（150006步）和新版best（480006步），每车型336个窗口。五车型原生参数，不随机scale。',
        '初速5/10m/s，实际前轮角 -30/-15/-5/0/5/15/30度，24条配对油门序列。每次重置静置30步后设置初速，采集自然演化的五步历史。',
        '旧版读取[a_long,r,油门,转向]四维token，新版读取[a_long,r,v,实际轮角,油门,转向]六维token；物理响应逐条相同。',
        '','## 方差关联（完整标准化8维context）','','| 表示 | 车型η² | 起始速度η² | 实际轮角η² | 实际末速二次回归R² | 控制工况/油门序列后车型partial R² |','|---|---:|---:|---:|---:|---:|']
    for name in ['old_context','new_context']:
        r=factors[name];lines.append(f"| {name} | {r['model_eta2_full']:.2%} | {r['initial_speed_eta2_full']:.2%} | {r['actual_wheel_angle_eta2_full']:.2%} | {r['actual_speed_quadratic_r2_full']:.2%} | {r['model_partial_r2_given_speed_angle_profile']:.2%} |")
    lines+=['','η²是该样本与该标准化度量中的组均值方差关联，不是语义信息百分比。Partial R²的分母不同，不能相加或与η²直接当作同一贡献比例。',
        '','## 五车型识别（留出整条油门序列，机会水平20%）','','| 表示 | 混合工况线性 | 混合工况kNN5 | 仅前2PC线性 | 固定10m/s/15°线性 | 5→10m/s线性 | 10→5m/s线性 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in spaces:
        r=metrics[name];keys=['mixed/linear','mixed/knn5','mixed/PCA2_linear','fixed_10mps_15deg/linear','5_to_10/linear','10_to_5/linear']
        lines.append('| '+name+' | '+' | '.join(f"{r[k]['accuracy']:.1%}"for k in keys)+' |')
    lines+=['','所有识别探针都用五种车型标签训练，包括S/XL；这里检验表征是否可解码，不能称为零样本车型识别。',
        'protocol_only仅使用起始车速、实际轮角和动作历史，不使用加速度/角速度/后续车速响应。实际轮角÷归一化转向可显露最大转角，因此此对照尤其重要。',
        '','## 新版减旧版线性识别率（动作序列级描述性bootstrap）','']
    for test,r in delta_ci.items():
        lo,hi=r['action_profile_bootstrap95'];lines.append(f"- {test}: {100*r['new_minus_old']:+.1f} pp，区间[{100*lo:+.1f}, {100*hi:+.1f}] pp。")
    lines+=['','## 限制','']+[f'- {x}'for x in result['limitations']]+['',
        '本次只分析冻结编码器，没有训练或修改任何模型。不同PCA的绝对坐标不能直接比较；同一行按不同因素着色时坐标一致。固定工况图在预先选定的10m/s与15°条件内重新拟合，不是从结果中挑选分离最好的条件。','']
    lines+=['## 本次解释','',
        '新版车型组均值关联和留出动作序列的线性识别率均提高，但没有形成跨车速稳定的纯底盘表征。新版PC1与实际末速Pearson r=0.9459；PC2的车型η²=71.81%，表现为不同方向分别携带速度与车型信息。',
        '旧版并非没有车型信息：非线性kNN5探针已达97.7%，新版99.8%。线性识别改善更多说明信息更容易被线性读取，不能直接等同于新增加了同等比例的车型信息。',
        '固定10m/s与15°时，原始新历史的线性识别已达100%，未训练随机编码器也有98.3%。所以固定工况下的漂亮分簇本身不是训练学会参数辨识的充分证据。',
        '本次使用全新且更宽的对称转角采样；旧版指标也在同批新数据上重新计算，不能把它与此前小转角/单向转角采样的相关系数直接拼接比较。',
        '只比较各自best（旧150k、新480k），不是同训练阶段控制实验。结合此前驾驶测试，新版的车型可解码性改善尚未转化为驾驶收益。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'factors':{k:v for k,v in factors.items()if k in ['old_context','new_context']},
        'probes':{n:{k:v['accuracy']for k,v in tests.items()}for n,tests in metrics.items()},'delta':delta_ci,'null':result['label_shuffle_control']},indent=2))


if __name__=='__main__':main()
