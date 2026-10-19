"""One-factor physical mass / wheel-friction sweeps of a frozen state2 encoder."""
import argparse,csv,hashlib,itertools,json,shutil,sys,time
from pathlib import Path
import numpy as np


class StandardScaler:
    def fit(self,x):
        x=np.asarray(x,dtype=np.float64)
        self.mean_=x.mean(0);self.scale_=x.std(0)
        self.scale_=np.where(self.scale_>1e-10,self.scale_,1.)
        return self
    def transform(self,x):return (x-self.mean_)/self.scale_

class PCA:
    def __init__(self,n_components=2):self.n_components=n_components
    def fit(self,x):
        self.mean_=x.mean(0);centered=x-self.mean_
        _,s,v=np.linalg.svd(centered,full_matrices=False)
        signs=np.sign(v[np.arange(len(v)),np.abs(v).argmax(1)]);signs[signs==0]=1
        self.components_=(v*signs[:,None])[:self.n_components]
        self.explained_variance_ratio_=(s*s/(s*s).sum())[:self.n_components]
        np.testing.assert_allclose(self.components_@self.components_.T,np.eye(self.n_components),atol=1e-10)
        return self
    def transform(self,x):return (x-self.mean_)@self.components_.T


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--profiles',type=int,default=24)
    parser.add_argument('--speeds',type=float,nargs='+',default=[10.])
    parser.add_argument('--angles',type=float,nargs='+',default=[15.])
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():parser.error('Use a fresh output directory')
    if not args.speeds or min(args.speeds)<=0 or max(abs(a) for a in args.angles)>45:parser.error('Positive speeds and front angles within ±45 degrees required')
    mixed=len(args.speeds)*len(args.angles)>1
    conditions=list(itertools.product(args.speeds,args.angles,range(args.profiles)))
    out.mkdir(parents=True);sys.path.insert(0,str(args.source_root.resolve()))
    import torch
    from omegaconf import OmegaConf
    from context import HistoryEncoder
    from envs.metadrive import MetaDrive
    from metadrive.component.vehicle.vehicle_type import vehicle_type
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    cfg=OmegaConf.load(args.checkpoint.parent/'.hydra/config.yaml');cfg.device='cpu';OmegaConf.resolve(cfg)
    cfg.env.traffic_density=0.;cfg.env.spawn.randomize=False
    cfg.env.eval_start_seed=20000;cfg.env.eval_num_scenarios=1;cfg.env.time_limit=200
    cfg.env.termination.out_of_road_done=False;cfg.env.termination.on_continuous_line_done=False
    cfg.env.chassis.eval_models=['m'];cfg.env.chassis.eval_mass_scales=[1.];cfg.env.chassis.eval_steering_scales=[1.]
    OmegaConf.save(cfg,out/'sampling_config.yaml')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);encoder=HistoryEncoder().eval()
    encoder.load_state_dict({k[len('context_encoder.'):]:v for k,v in payload['agent_state_dict'].items() if k.startswith('context_encoder.')},strict=True)
    step=payload['step'];del payload
    scales=np.array([.8,.9,1.,1.1,1.2]);nominal_mass=int(vehicle_type['m'].MASS)
    sp=vehicle_type['m'].PARAMETER_SPACE['wheel_friction'];nominal_friction=float((sp.low[0]+sp.high[0])/2)
    contract={'purpose':'Show whether frozen latent context changes with mass or wheel-friction scale, one factor at a time.',
      'panels':{'a':'mass sweep, fixed nominal wheel friction','b':'wheel-friction sweep, fixed nominal mass'},
      'archetype':'quantitative grid','backend':'python','exports':['png','pdf','svg','csv','npz'],
      'risks':['PCA is descriptive, not proof of parameter identification.','Equal starting speed does not guarantee equal speed throughout the history.',
               'Friction slip is the Bullet wheel parameter, not a measured road friction coefficient.','Friction was not randomized during training.'],
      'normalization':f'One common StandardScaler fitted to all {2*5*len(conditions)} contexts; each factor panel fits its own PCA using all five scales.'}
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    env=MetaDrive(cfg.env,evaluation=True);env._env.config['map_config']['config']='';env._env.config['map_config']['lane_width']=20.
    histories=[];paths=[];rows=[];angles_all=[];start=time.monotonic()
    try:
        for factor in ['mass','friction']:
            for condition_id,(speed,angle,profile) in enumerate(conditions):
                rng=np.random.default_rng(910000+profile)
                throttle=np.interp(np.arange(5),[0,2,4],rng.uniform(.15,.4,3)).astype(np.float32)
                for scale in scales:
                    spec=env._env.config['chassis_spec'];spec['models']=['m'];spec['mass_scales']=[float(scale) if factor=='mass' else 1.];spec['steering_scales']=[1.]
                    env.reset(seed=20000);vehicle=env._env.vehicle
                    requested_friction=nominal_friction*(float(scale) if factor=='friction' else 1.)
                    for wi in range(vehicle.system.getNumWheels()):vehicle.system.getWheel(wi).setFrictionSlip(requested_friction)
                    vehicle.config['wheel_friction']=requested_friction
                    np.testing.assert_allclose([vehicle.system.getWheel(wi).getFrictionSlip() for wi in range(vehicle.system.getNumWheels())],requested_friction,rtol=1e-6)
                    requested_mass=int(np.floor(nominal_mass*(float(scale) if factor=='mass' else 1.)+.5))
                    np.testing.assert_allclose(vehicle.body.getMass(),requested_mass,rtol=1e-6,atol=1e-4)
                    for _ in range(30):
                        _,_,done,_=env.step(np.zeros(2,np.float32))
                        if done:raise RuntimeError('Termination during settling')
                    vehicle.set_velocity([1.,0.],value=float(speed),in_local_frame=True)
                    steer=float(angle)/vehicle.max_steering
                    vehicle._set_action(np.array([steer,throttle[0]],np.float32))
                    env._context_history.observe(vehicle)
                    initial_speed=float(np.linalg.norm(vehicle.velocity));initial_angle=[vehicle.system.getSteeringValue(i) for i in [0,1]]
                    assert abs(initial_speed-speed)<1e-4;np.testing.assert_allclose(initial_angle,angle,atol=1e-4)
                    physical=[];angles=[]
                    for t in range(5):
                        obs,_,done,_=env.step(np.array([steer,throttle[t]],np.float32))
                        if done or obs['ctx_collision']:raise RuntimeError(f'Invalid probe: {factor}/{scale}/{profile}; no selective filtering')
                        physical.append([*obs['ctx_ego_state'],float(np.linalg.norm(vehicle.velocity))])
                        angles.append([vehicle.system.getSteeringValue(i) for i in [0,1]])
                    np.testing.assert_allclose(angles,angle,atol=1e-4)
                    np.testing.assert_allclose([vehicle.system.getWheel(wi).getFrictionSlip() for wi in range(vehicle.system.getNumWheels())],requested_friction,rtol=1e-6)
                    assert obs['ctx_mask'].all()
                    histories.append(obs['ctx_history'].copy());paths.append(physical);angles_all.append(angles)
                    rows.append({'factor':factor,'scale':float(scale),'profile':profile,'condition_id':condition_id,'action_seed':910000+profile,
                      'vehicle_model':'m','mass_kg':float(vehicle.body.getMass()),'wheel_friction':float(vehicle.system.getWheel(0).getFrictionSlip()),
                      'max_steering_deg':float(vehicle.max_steering),'initial_speed_m_s':initial_speed,'initial_front_angle_deg':float(initial_angle[0]),
                      'final_speed_m_s':physical[-1][-1],'mean_speed_m_s':float(np.mean(np.array(physical)[:,-1])),
                      'final_acceleration_m_s2':physical[-1][0],'final_yaw_rad_s':physical[-1][1]})
                if (condition_id+1)%24==0 or condition_id+1==len(conditions):print(factor,condition_id+1,'/',len(conditions),'conditions; windows',len(rows),flush=True)
    finally:env.close()
    histories=np.array(histories);paths=np.array(paths)
    with torch.inference_mode():contexts=encoder(torch.from_numpy(histories),torch.ones(len(rows),5,dtype=torch.bool)).numpy()
    assert np.isfinite(contexts).all()
    np.savez_compressed(out/'representations.npz',history=histories,context=contexts,physical_paths=paths,front_wheel_angles=np.array(angles_all))
    scaled=StandardScaler().fit(contexts);standardized=scaled.transform(contexts)
    projection=np.zeros((len(rows),2));models={};stats={}
    for factor in ['mass','friction']:
        indices=np.array([i for i,r in enumerate(rows) if r['factor']==factor]);pc=PCA(n_components=2).fit(standardized[indices]);projection[indices]=pc.transform(standardized[indices]);models[factor]=pc
        levels=np.array([r['scale'] for r in rows])[indices];v=contexts[indices];grand=v.mean(0)
        between=sum(np.sum(levels==s)*np.sum((v[levels==s].mean(0)-grand)**2) for s in scales)
        within=sum(np.sum((v[levels==s]-v[levels==s].mean(0))**2) for s in scales)
        # Paired parameter effect relative to the nominal context, controlling profile.
        paired_shift={};physical_shift={}
        for s in scales:
            ids=[i for i in indices if np.isclose(rows[i]['scale'],s)]
            ref=[i for i in indices if rows[i]['scale']==1.]
            assert [rows[i]['profile'] for i in ids]==[rows[i]['profile'] for i in ref]
            paired_shift[str(s)]=float(np.linalg.norm(standardized[ids]-standardized[ref],axis=1).mean())
            physical_shift[str(s)]=np.abs(paths[ids]-paths[ref]).mean(axis=(0,1)).tolist()
        stats[factor]={'explained_variance_ratio':pc.explained_variance_ratio_.tolist(),'raw_context_scale_eta_squared':float(between/(between+within)),
            'paired_latent_shift_common_standardized':paired_shift,'paired_physical_mean_absolute_shift_acc_yaw_speed':physical_shift,
            'pca_components':pc.components_.tolist(),'pca_mean':pc.mean_.tolist()}
    for i,row in enumerate(rows):
        row.update(PC1=float(projection[i,0]),PC2=float(projection[i,1]))
        row.update({f'context_{j}':float(contexts[i,j]) for j in range(8)})
    with (out/'source_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    colors=plt.get_cmap('viridis')(np.linspace(.12,.88,len(scales)))
    def panel(ax,factor,letter=None):
        inds=np.array([i for i,r in enumerate(rows) if r['factor']==factor]);rr=np.array([rows[i]['scale'] for i in inds]);pts=projection[inds]
        for cid in sorted(set(rows[i]['condition_id'] for i in inds)):
            ii=[i for i in inds if rows[i]['condition_id']==cid]
            ax.plot(projection[ii,0],projection[ii,1],color='.7',lw=.5,alpha=.04 if mixed else .25,zorder=0)
        for level,color in zip(scales,colors):
            p=pts[np.isclose(rr,level)]
            ax.scatter(p[:,0],p[:,1],s=12 if mixed else 27,color=color,alpha=.36 if mixed else .7,edgecolors='none',label=f'{level:.1f}×')
            center=p.mean(0);ax.scatter(*center,s=100,marker='X',color=color,edgecolors='white',linewidths=.6,zorder=4)
        ratio=models[factor].explained_variance_ratio_*100
        ax.set_xlabel(f'PC1 ({ratio[0]:.1f}%)');ax.set_ylabel(f'PC2 ({ratio[1]:.1f}%)')
        ax.set_title('Mass scaling' if factor=='mass' else 'Wheel-friction scaling',pad=11)
        ax.grid(alpha=.13);ax.set_axisbelow(True);ax.margins(.12)
        if letter:ax.text(-.16,1.07,letter,transform=ax.transAxes,fontsize=13,fontweight='bold')
        return ax
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.7,4.4));fig.subplots_adjust(left=.09,right=.98,top=.77,bottom=.24,wspace=.32)
    for ax,factor,letter in zip(axs,['mass','friction'],['a','b']):panel(ax,factor,letter)
    subtitle=(f'Mixed initial speeds {args.speeds} m/s; front-wheel angles {args.angles}°' if mixed else f'Initial speed {args.speeds[0]:g} m/s; actual front-wheel angle {args.angles[0]:g}°')
    fig.suptitle(f'Frozen context encoder | M vehicle | best {step:,} steps\n{subtitle}\n{len(conditions)} windows/scale; {args.profiles} paired profiles per speed-angle combination',fontsize=10,y=1.03)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=5,columnspacing=1.5,title='Parameter / nominal value')
    fig.text(.5,.018,'Points: history windows; X: scale centroid. Shared 8D standardization; PCA fitted separately per panel.',ha='center',fontsize=7.5)
    export(fig,'mass_friction_context_pca')
    for factor in ['mass','friction']:
        fig,ax=plt.subplots(figsize=(5.5,5.1));fig.subplots_adjust(left=.16,right=.96,top=.76,bottom=.24)
        panel(ax,factor)
        fig.suptitle(f'Latent context: {factor} scale\n'+('Mixed initial speeds and front-wheel angles' if mixed else subtitle),fontsize=11,y=.98)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=5,columnspacing=.7,handletextpad=.2,title=f'{len(conditions)} windows per scale')
        export(fig,f'{factor}_context_pca')
    if mixed:
        fig,axs=plt.subplots(2,2,figsize=(8.7,7.2));fig.subplots_adjust(left=.09,right=.94,top=.86,bottom=.10,wspace=.38,hspace=.5)
        for row,factor in enumerate(['mass','friction']):
            inds=np.array([i for i,r in enumerate(rows) if r['factor']==factor])
            for col,key in enumerate(['initial_speed_m_s','initial_front_angle_deg']):
                values=np.array([rows[i][key] for i in inds]);ax=axs[row,col]
                sc=ax.scatter(projection[inds,0],projection[inds,1],c=values,cmap='viridis' if col==0 else 'coolwarm',s=7,alpha=.6,rasterized=True)
                ratio=models[factor].explained_variance_ratio_*100
                ax.set(xlabel=f'PC1 ({ratio[0]:.1f}%)',ylabel=f'PC2 ({ratio[1]:.1f}%)',title=f'{factor.title()} sweep | '+('initial speed' if col==0 else 'wheel angle'))
                fig.colorbar(sc,ax=ax,fraction=.045,pad=.03,label='m/s' if col==0 else 'degrees')
        fig.suptitle('Same PCA coordinates, colored by operating condition\nNo PCA refitting between colorings',fontsize=12,y=.98)
        export(fig,'operating_condition_colors')
    meta={'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      'windows':len(rows),'profiles_per_speed_angle_combination':args.profiles,'windows_per_scale':len(conditions),'initial_speeds_m_s':args.speeds,'initial_wheel_angles_degrees':args.angles,'scales':scales.tolist(),'nominal_mass_kg':nominal_mass,'nominal_wheel_friction':nominal_friction,
      'window_length':5,'control_dt_seconds':env._context_history.dt,'device':'cpu','stats':stats,
      'context_standardizer_mean':scaled.mean_.tolist(),'context_standardizer_scale':scaled.scale_.tolist(),
      'speed_range_over_windows_m_s':[float(paths[:,:,-1].min()),float(paths[:,:,-1].max())],
      'protocol':'Independent resets at the same flat, wide traffic-free road. Mass/wheel friction checked against Bullet. 30 settling steps, velocity initialized once; front steering set before the 5-step window; identical throttle profiles; no selective rejection or speed clamping.',
      'limitations':contract['risks'],'elapsed_seconds':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');shutil.copy2(Path(__file__),out/'script_snapshot.py')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
