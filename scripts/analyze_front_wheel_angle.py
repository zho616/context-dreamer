"""Analyze physical and context changes under paired full-range wheel-angle probes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();out=args.directory
    rows=list(csv.DictReader((out/'windows.csv').open()));d=np.load(out/'representations.npz')
    models=['default','m','l','s','xl'];label=np.array([r['model']for r in rows]);fit=np.isin(label,models[:3])
    vals={k:np.array([float(r[k])for r in rows])for k in rows[0]if k!='model'}
    # Float32 Bullet initialization produces tiny representational speed differences.
    initial=np.rint(vals['initial_speed_m_s']).astype(int);frac=vals['angle_fraction'];angle=vals['front_left_angle_degrees']
    C=d['learned_context'].astype(float);sd=C[fit].std(0);X=(C-C[fit].mean(0))/np.maximum(sd,1e-8)
    _,s,vt=np.linalg.svd(X[fit],full_matrices=False);Z=X@vt[:2].T;ratios=s[:2]**2/(s*s).sum()
    def sse(y,f):
        a=np.column_stack([np.ones(len(y)),f]);res=y-a@np.linalg.lstsq(a,y,rcond=None)[0];return float((res*res).sum())
    def design(x):return np.column_stack([x==a for a in sorted(set(x))[1:]])
    def corr(a,b):return float(np.corrcoef(a,b)[0,1])
    look={(r['model'],int(round(float(r['initial_speed_m_s']))),int(r['repeat']),round(float(r['angle_fraction']),1)):i for i,r in enumerate(rows)}
    paired=[]
    for i,r in enumerate(rows):
        zero=look[(r['model'],initial[i],int(r['repeat']),0.)]
        paired.append(dict(model=r['model'],initial_speed=int(initial[i]),repeat=int(r['repeat']),fraction=frac[i],
            actual_angle_degrees=angle[i],context_distance_from_zero=float(np.linalg.norm(X[i]-X[zero])),
            delta_first_acceleration=vals['first_acceleration'][i]-vals['first_acceleration'][zero],
            delta_first_yaw_rate=vals['first_yaw_rate'][i]-vals['first_yaw_rate'][zero],
            delta_final_speed=vals['final_speed_m_s'][i]-vals['final_speed_m_s'][zero]))
    with(out/'paired_changes.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    summaries=[]
    for model in models:
        for speed in [5,10]:
            q=(label==model)&(initial==speed)
            nuisance=np.column_stack([design(label[q]),design(vals['repeat'][q])]) if len(set(label[q]))>1 else design(vals['repeat'][q])
            for fval in [0.,.2,.4,.6,.8,1.]:
                take=q&np.isclose(frac,fval);idx=np.flatnonzero(take)
                summaries.append(dict(model=model,initial_speed=speed,fraction=fval,n=int(take.sum()),angle_degrees=float(angle[take].mean()),
                    **{k:float(vals[k][take].mean())for k in ['first_acceleration','first_yaw_rate','final_acceleration','final_yaw_rate','final_speed_m_s']},
                    context_distance_from_zero=float(np.mean([paired[i]['context_distance_from_zero']for i in idx]))))
    with(out/'condition_means.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    associations={}
    for speed in [5,10]:
        q=initial==speed;N=np.column_stack([design(label[q]),design(vals['repeat'][q])]);A=design(frac[q])
        joint=np.column_stack([N,A])
        associations[str(speed)]={
            'pearson_pc1_actual_angle':corr(Z[q,0],angle[q]),'pearson_pc2_actual_angle':corr(Z[q,1],angle[q]),
            'fraction_partial_r2_full_context_given_model_and_throttle_profile':1-sse(X[q],joint)/sse(X[q],N),
            'fraction_partial_r2_pc1_given_model_and_throttle_profile':1-sse(Z[q,0],joint)/sse(Z[q,0],N),
            'fraction_partial_r2_pc2_given_model_and_throttle_profile':1-sse(Z[q,1],joint)/sse(Z[q,1],N),
            'final_speed_range_m_s':[float(vals['final_speed_m_s'][q].min()),float(vals['final_speed_m_s'][q].max())]}
    report={'samples':len(rows),'samples_per_model':int(sum(label==models[0])),'pca_variance_ratios':ratios.tolist(),
        'pca_fit':'Feature standardization and PCA fitted only on DEFAULT/M/L; S/XL projected. Same basis for both initial-speed panels.',
        'associations':associations,'condition_means':summaries,
        'limitations':['Only starting speed is fixed; subsequent speed differences are physical consequences of steering.',
            'Angle correlations measure the combined change of steering, motion response and resulting speed; not a pure direct dependence of the encoder on angle.',
            'Six held angles, two initial speeds, twelve paired positive-throttle profiles, one flat wide road.',
            'PCA and regression are descriptive, not held-out vehicle classification or evidence of disentanglement.',
            'Only positive steering tested; yaw rate uses the opposite sign (right positive).']}
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    with(out/'pca_source_data.csv').open('w')as f:
        fields=list(rows[0])+['pc1','pc2'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):w.writerow({**r,'pc1':Z[i,0],'pc2':Z[i,1]})

    # Figure contract: total response to a controlled steering sweep; show
    # physical responses separately from context geometry; no fixed-speed claim.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2']
    def curves(ax,key):
        for model,col in zip(models,colors):
            for speed,style in [(5,'-'),(10,'--')]:
                group=[r for r in summaries if r['model']==model and r['initial_speed']==speed]
                ax.plot([r['angle_degrees']for r in group],[r[key]for r in group],color=col,ls=style,marker='o',ms=3,lw=1.3)
        ax.set_xlabel('Measured front-wheel angle (deg)');ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig,axs=plt.subplots(2,2,figsize=(8.6,6.4),layout='constrained')
    keys=['first_acceleration','first_yaw_rate','final_acceleration','final_yaw_rate']
    titles=['First-step longitudinal acceleration','First-step yaw rate','End-of-window longitudinal acceleration','End-of-window yaw rate']
    for ax,key,title,letter in zip(axs.flat,keys,titles,'abcd'):
        curves(ax,key);ax.set_title(title);ax.set_ylabel('Acceleration (m/s²)'if'acceleration'in key else'Yaw rate (rad/s; right positive)')
        ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    handles=[Line2D([],[],color=col,label=model.upper())for model,col in zip(models,colors)]
    handles += [Line2D([],[],color='#555555',ls=style,label=f'Initial {speed} m/s')for speed,style in [(5,'-'),(10,'--')]]
    fig.legend(handles=handles,ncol=7,loc='outside lower center',fontsize=7,frameon=False)
    fig.suptitle('Physical response to front-wheel angle\nPaired throttle profiles; fixed starting speed, natural motion afterward',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'front_wheel_physical_response.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(8.6,6.5),layout='constrained')
    for ax,speed in zip(axs[0],[5,10]):
        q=initial==speed;sc=ax.scatter(*Z[q].T,c=angle[q],cmap='viridis',s=16,alpha=.65)
        fig.colorbar(sc,ax=ax,label='Measured wheel angle (deg)',shrink=.8)
        ax.set(title=f'Initial speed {speed} m/s',xlabel=f'PC1 ({ratios[0]:.1%})',ylabel=f'PC2 ({ratios[1]:.1%})')
    curves(axs[1,0],'context_distance_from_zero');axs[1,0].set(title='Context change from paired zero steering',ylabel='Distance in standardized 8D context')
    curves(axs[1,1],'final_speed_m_s');axs[1,1].set(title='Resulting speed after five control steps',ylabel='Speed (m/s)')
    for ax,letter in zip(axs.flat,'abcd'):ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.legend(handles=handles,ncol=7,loc='outside lower center',fontsize=7,frameon=False)
    fig.suptitle('Context changes under the full front-wheel-angle sweep\nSame PCA basis; context changes include the resulting motion and speed changes',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'front_wheel_context_response.{ext}',dpi=300)
    plt.close(fig)
    print(json.dumps({'pca':ratios.tolist(),'associations':associations,'endpoints':[r for r in summaries if r['fraction']in[0.,1.]]},indent=2))


if __name__=='__main__':main()
