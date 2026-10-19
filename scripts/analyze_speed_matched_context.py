"""Matched-speed PCA and held-out-action vehicle probes, with raw/random controls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def standardize(x, train):
    sd=x[train].std(0)
    return (x-x[train].mean(0))/np.where(sd<1e-8,1.,sd)


def classify(x, y, train, test, seed):
    x=standardize(x,train)
    a=np.column_stack([np.ones(len(x)),x])
    penalty=np.eye(a.shape[1]);penalty[0,0]=0
    weights=np.linalg.solve(a[train].T@a[train]+penalty,a[train].T@np.eye(5)[y[train]])
    pred=(a[test]@weights).argmax(1)
    return float(np.mean(pred==y[test])),pred


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();out=args.directory
    rows=list(csv.DictReader((out/'windows.csv').open()))
    data=np.load(out/'representations.npz')
    models=['default','m','l','s','xl'];y=np.array([models.index(r['model']) for r in rows])
    groups=np.array([int(r['group']) for r in rows]);unique=np.unique(groups)
    representations={
        'Learned context (8D)':data['learned_context'].astype(float),
        'Raw history (20D)':(data['history']/np.array([10,1,1,1])).reshape(len(rows),-1),
        'Untrained encoder (8D)':data['random_context'].astype(float),
        'Mean speed only (1D)':np.array([float(r['mean_speed_m_s']) for r in rows])[:,None],
    }
    assert len(rows)==len(unique)*5 and all((groups==g).sum()==5 for g in unique)
    results={k:[] for k in representations}
    results.update({k+' / PCA2':[] for k in list(representations)[:3]})
    first_z={};first_ratios={};first_predictions={};first_train=None;first_test=None
    for split in range(20):
        rng=np.random.default_rng(20261015+split)
        order=rng.permutation(unique);train=np.isin(groups,order[:int(.8*len(unique))]);test=~train
        for name,x in representations.items():
            acc,pred=classify(x,y,train,test,split);results[name].append(acc)
            if split==0:first_predictions[name]=pred
            if 'speed only' in name:continue
            sx=standardize(x,train);_,s,vt=np.linalg.svd(sx[train],full_matrices=False)
            z=sx@vt[:2].T
            acc,_=classify(z,y,train,test,split);results[name+' / PCA2'].append(acc)
            if split==0:
                first_z[name]=z;first_ratios[name]=(s[:2]**2/(s*s).sum()).tolist()
        if split==0:first_train,first_test=train.copy(),test.copy()
    summary={k:{'mean_accuracy':float(np.mean(v)),'min_accuracy':float(np.min(v)),
                'max_accuracy':float(np.max(v)),'first_split_accuracy':v[0],'split_accuracies':v} for k,v in results.items()}
    # Keep five labels balanced within every action group under the null.
    null=[];rng=np.random.default_rng(91832)
    for _ in range(200):
        perm_y=y.copy()
        for g in unique:
            idx=np.flatnonzero(groups==g);perm_y[idx]=rng.permutation(y[idx])
        null.append(classify(representations['Learned context (8D)'],perm_y,first_train,first_test,0)[0])
    pred=first_predictions['Learned context (8D)'];confusion=np.zeros((5,5),int)
    np.add.at(confusion,(y[first_test],pred),1)
    meta=json.loads((out/'sampling_metadata.json').read_text())
    report={'sampling':meta,'classification':summary,
        'classifier':'Feature-standardized ridge least-squares one-hot classifier, alpha=1 fixed; no hyperparameter tuning.',
        'split':'20 deterministic 80/20 splits by shared action-sequence group; no group crosses train/test; same splits for all representations. Splits reuse observations, ranges are not confidence intervals.',
        'pca_fit':'First split training action groups, all five types; held-out groups projected using that basis. Vehicle labels do not fit PCA.',
        'pca_variance_ratios':first_ratios,'first_split_confusion':confusion.tolist(),
        'permutation_control':{'runs':200,'mean_accuracy':float(np.mean(null)),
             'max_accuracy':float(np.max(null)),'observed_first_split_accuracy':summary['Learned context (8D)']['first_split_accuracy'],
             'shuffle_design':'Independently shuffle vehicle labels within each action group; repeat training and test. Descriptive null control for this diagnostic dataset.'},
        'limitations':['Approximately matched natural speed trajectories, not exact speed clamping.',
            'Single straight road, fixed initial heading and one initial speed; five-step probes start after a velocity initialization with history reset.',
            'S/XL were unseen by the encoder during RL training, but the diagnostic vehicle classifier is trained on all five labels.',
            'PCA separation and vehicle decoding do not prove parameter disentanglement or driving improvement.',
            'Residual speed differences are partly consequences of chassis-specific acceleration responses.']}
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'classification.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['representation','mean_accuracy','min_accuracy','max_accuracy','first_split_accuracy'])
        for name,r in summary.items():w.writerow([name,*[r[k] for k in ['mean_accuracy','min_accuracy','max_accuracy','first_split_accuracy']]])
    with (out/'pca_source_data.csv').open('w') as f:
        fields=list(rows[0])+['split']+[f'{name}_{pc}' for name in ['context','raw','random'] for pc in ['pc1','pc2']]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):
            w.writerow({**r,'split':'train' if first_train[i] else 'test',
                **{f'{short}_{pc}':first_z[name][i,j] for short,name in zip(['context','raw','random'],list(representations)[:3]) for j,pc in enumerate(['pc1','pc2'])}})
    # Figure contract: test vehicle separability at matched speed, with representation controls.
    # Three matched PCA panels and a held-out-action decoding panel; no invented intervals.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(9,7.3),layout='constrained')
    colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2']
    panel_names=['Raw history (20D)','Learned context (8D)','Untrained encoder (8D)']
    for ax,name in zip(axs.flat,panel_names):
        z=first_z[name];ratios=first_ratios[name]
        for k,(m,col) in enumerate(zip(models,colors)):
            q=y==k
            ax.scatter(*z[q&first_train].T,s=13,alpha=.3,color=col,edgecolors='none')
            ax.scatter(*z[q&first_test].T,s=24,alpha=.85,color=col,marker='x',linewidths=.8,label=m.upper())
        ax.set(xlabel=f'PC1 ({ratios[0]:.1%})',ylabel=f'PC2 ({ratios[1]:.1%})',title=name)
        ax.grid(alpha=.12);ax.set_axisbelow(True)
    axs[0,1].legend(ncol=3,fontsize=7)
    names=list(representations)[:3]
    ax=axs[1,1];pos=np.arange(3)
    full=[summary[n]['mean_accuracy'] for n in names]
    pca=[summary[n+' / PCA2']['mean_accuracy'] for n in names]
    ax.bar(pos-.17,full,.32,color='#7663A5',label='Full representation')
    ax.bar(pos+.17,pca,.32,color='#AABAD0',label='First two PCs')
    for k,(a,b) in enumerate(zip(full,pca)):
        for xx,val in [(k-.17,a),(k+.17,b)]:ax.text(xx,val+.015,f'{val:.0%}',ha='center',fontsize=8)
    ax.axhline(.2,color='#777777',ls='--',lw=.8)
    ax.set(xticks=pos,xticklabels=['Learned','Raw','Untrained'],ylim=(0,1.12),ylabel='Held-out action-group accuracy',title='Five-class linear vehicle decoding')
    ax.set_yticks([0,.2,.4,.6,.8,1.],['0%','20%','40%','60%','80%','100%'])
    ax.legend(fontsize=7,loc='upper right');ax.text(.02,.25,'Chance = 20%',transform=ax.transAxes,fontsize=7,color='#666666')
    for ax,letter in zip(axs.flat,'abcd'):ax.text(-.12,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    fig.suptitle('Vehicle information at matched speed\n100 shared action sequences × 5 models; initial speed 10 m/s',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'vehicle_classification_pca.{ext}',dpi=300)
    plt.close(fig)
    print(json.dumps({'classification':summary,'pca':first_ratios,'null':report['permutation_control'],'confusion':confusion.tolist()},indent=2))


if __name__=='__main__':main()
