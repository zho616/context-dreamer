"""Fixed-condition vehicle PCA: independent random 8D contexts vs 1M best encoder outputs."""
import argparse,csv,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_old_context_pca import fit_pca


def association(x,labels,fit):
    mean=x[fit].mean(0);sd=x[fit].std(0);sd=np.where(sd<1e-6,1.,sd)
    z=(x-mean)/sd;grand=z.mean(0);total=((z-grand)**2).sum()
    between=sum(np.sum(labels==v)*np.sum((z[labels==v].mean(0)-grand)**2) for v in np.unique(labels))
    return float(between/total)


def grouped_ridge_accuracy(x,y,groups,folds):
    predictions=np.full(len(y),-1)
    for fold in folds:
        test=np.isin(groups,fold);train=~test
        mu=x[train].mean(0);sd=x[train].std(0);sd=np.where(sd<1e-6,1.,sd)
        a=np.column_stack(((x[train]-mu)/sd,np.ones(train.sum())))
        b=np.eye(5)[y[train]];penalty=np.eye(a.shape[1]);penalty[-1,-1]=0
        weights=np.linalg.solve(a.T@a+penalty,a.T@b)
        predictions[test]=np.argmax(np.column_stack(((x[test]-mu)/sd,np.ones(test.sum())))@weights,axis=1)
    assert (predictions>=0).all()
    return float(np.mean(predictions==y)),predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    out=args.output.resolve()
    if out.exists():p.error('Use a fresh output directory')
    out.mkdir(parents=True);root=Path(__file__).resolve().parents[1]
    run=root/'logdir/retrain_1m_20261017_0115/training';data_dir=root/'logdir/context_kinematics_500k_20261015_180340/analysis/vehicle_context_pca'
    rows=list(csv.DictReader((data_dir/'pca_source_data.csv').open()))
    select=np.array([round(float(r['initial_speed_m_s']))==10 and round(float(r['front_left_angle_degrees']))==15 for r in rows])
    selected_rows=[r for r,q in zip(rows,select) if q]
    history=np.load(data_dir/'representations.npz')['history'][select][...,[0,1,4,5]].copy()
    types=['default','m','l','s','xl'];labels=np.array([r['model'] for r in selected_rows]);y=np.array([types.index(v) for v in labels]);groups=np.array([int(r['repeat']) for r in selected_rows])
    fit=np.array([r['plot_split']=='train' for r in selected_rows]);assert history.shape==(120,5,4) and fit.sum()==100
    contract={'claim':'Quantify whether the trained encoder has stronger vehicle-associated organization than random draws under the identical fixed-condition probes.',
      'archetype':'quantitative grid','backend':'python','panels':['Preselected random initialization PCA','Trained PCA','Full-8D vehicle association with 20-random-seed reference','Grouped linear probe accuracy'],
      'exports':['png','pdf','svg','csv','npz'],
      'risks':['Random contexts are iid standard Gaussian vectors, independent of history and type.','PCA bases independent; compactness not directly comparable.','Random-noise draw spread is not training-seed uncertainty.','Noise is a no-information baseline; this comparison alone does not isolate improvement caused by encoder training.']}
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('fixed_comparison_frozen_context',run/'source/context.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    seeds=list(range(20261015,20261035));random_outputs=[];mask=torch.ones(120,5,dtype=torch.bool)
    with torch.inference_mode():
        for seed in seeds:
            # Independent of input history, vehicle label, and other samples.
            random_outputs.append(np.random.default_rng(seed).standard_normal((120,8)).astype(np.float32))
        checkpoint=run/'r2dreamer_context/best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        encoder=module.HistoryEncoder().eval();prefix='context_encoder.'
        encoder.load_state_dict({k[len(prefix):]:v for k,v in payload['agent_state_dict'].items() if k.startswith(prefix)},strict=True)
        best=encoder(torch.from_numpy(history),mask).numpy();step=payload['step'];del payload
    random_outputs=np.array(random_outputs);assert random_outputs.shape==(20,120,8)
    previous=np.load(root/'logdir/retrained_vehicle_context_pca_20261020/pca_arrays.npz')['context'][select]
    np.testing.assert_allclose(best,previous,atol=1e-6,rtol=1e-6)
    folds=np.array_split(np.random.default_rng(20261016).permutation(np.unique(groups)),6)
    random_eta=np.array([association(x,labels,fit) for x in random_outputs]);best_eta=association(best,labels,fit)
    random_accuracy=np.array([grouped_ridge_accuracy(x,y,groups,folds)[0] for x in random_outputs]);best_accuracy,best_pred=grouped_ridge_accuracy(best,y,groups,folds)
    coords={};ratios={}
    for name,x in [('random',random_outputs[0]),('trained',best)]:coords[name],ratios[name],_,_,_=fit_pca(x.astype(float),fit)
    colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2'];pc=np.array(colors)[y]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,grid=plt.subplots(2,2,figsize=(10.3,8.3))
    axs=grid.ravel()
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.83,wspace=.31,hspace=.52)
    order=np.random.default_rng(13).permutation(len(y))
    for ax,name,title in zip(axs[:2],['random','trained'],[f'Random 8D noise | seed {seeds[0]}',f'Trained | best {step:,} steps']):
        z=coords[name];r=ratios[name]
        ax.scatter(*z[order].T,c=pc[order],s=27,alpha=.65,edgecolors='none')
        ax.set(title=title,xlabel=f'PC1 ({r[0]:.1%})',ylabel=f'PC2 ({r[1]:.1%})');ax.grid(alpha=.12);ax.set_axisbelow(True)
    for ax,random_values,trained_value,ylabel,title in [
        (axs[2],random_eta,best_eta,'Vehicle association η² (%)','Full 8D group association'),
        (axs[3],random_accuracy,best_accuracy,'Held-out classification accuracy (%)','Grouped linear ridge probe')]:
        ax.scatter(np.zeros(len(seeds))+np.linspace(-.12,.12,len(seeds)),random_values*100,color='#9AA9B7',s=19,alpha=.75)
        ax.errorbar([0],[random_values.mean()*100],yerr=[random_values.std(ddof=1)*100],fmt='D',color='#4C78A8',capsize=4,ms=6)
        ax.scatter([1],[trained_value*100],marker='D',s=48,color='#6A3D9A')
        ax.set(xlim=(-.45,1.45),ylim=(0,115),xticks=[0,1],xticklabels=['Random noise\n20 draws','Trained\n1 checkpoint'],ylabel=ylabel,title=title)
        ax.set_yticks([0,20,40,60,80,100])
        ax.text(0,random_values.max()*100+4,f'{random_values.mean():.1%}',ha='center',fontsize=9)
        ax.text(1,trained_value*100+4,f'{trained_value:.1%}',ha='center',fontsize=9)
    for a,letter in zip(axs,'abcd'):a.text(-.14,1.08,letter,transform=a.transAxes,fontweight='bold',fontsize=12)
    handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=m.upper(),markersize=6) for m,c in zip(types,colors)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=5)
    fig.suptitle('Fixed condition: independent random vs. trained context\nInitial speed 10 m/s; actual front-wheel angle 15°; 24 profiles per vehicle type',fontsize=12,y=.96)
    fig.text(.5,.012,'Identical histories; independent PCA bases. Error bars: random mean ± SD. Noise baseline: c ~ N(0, I8), independent of history and vehicle.',ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'fixed_random_vs_trained.{ext}',dpi=300)
    plt.close(fig)
    np.savez_compressed(out/'representations.npz',history=history,random_contexts=random_outputs,trained_context=best,random_pca=coords['random'],trained_pca=coords['trained'],random_eta=random_eta,random_ridge_accuracy=random_accuracy)
    def describe(a):return {'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),'min':float(a.min()),'max':float(a.max())}
    result={'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'random_seeds':seeds,'random_distribution':'iid N(0,I8), one independent vector per window, no encoder on random branch',
      'eta2_random':describe(random_eta),'eta2_trained':best_eta,'eta2_change_vs_random_mean_pp':float(100*(best_eta-random_eta.mean())),
      'ridge_accuracy_random':describe(random_accuracy),'ridge_accuracy_trained':best_accuracy,
      'ridge_accuracy_change_vs_random_mean_pp':float(100*(best_accuracy-random_accuracy.mean())),
      'pca_variance_ratios':{k:v.tolist() for k,v in ratios.items()},'n_windows':120,'fit_profiles':20,'display_profiles':24,
      'method':'η² = between-vehicle group-mean sum of squares / total, in 8D standardized on the same 20 fit profiles for each encoder. Ridge alpha1, six profile-grouped folds, standardized within each training fold; all five labels supervised only in posthoc diagnostic.',
      'limitations':contract['risks'],'torch_version':torch.__version__}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'random_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['random_seed','eta2','ridge_accuracy']);w.writeheader();w.writerows([{'random_seed':s,'eta2':e,'ridge_accuracy':a} for s,e,a in zip(seeds,random_eta,random_accuracy)])
    with (out/'source_data.csv').open('w') as f:
        fields=['vehicle','profile','pca_fit']+[f'{name}_{kind}{j}' for name in ['random','trained'] for kind,count in [('context',8),('pc',2)] for j in range(count)]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(120):
            row={'vehicle':labels[i],'profile':int(groups[i]),'pca_fit':bool(fit[i])}
            for name,x in [('random',random_outputs[0]),('trained',best)]:
                row.update({f'{name}_context{j}':float(x[i,j]) for j in range(8)});row.update({f'{name}_pc{j}':float(coords[name][i,j]) for j in range(2)})
            w.writerow(row)
    shutil.copy2(Path(__file__),out/'script_snapshot.py')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
