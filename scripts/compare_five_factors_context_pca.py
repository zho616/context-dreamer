"""Independent random context vs best encoder: five one-factor physical probes."""
import argparse,csv,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from plot_old_context_pca import fit_pca
from compare_fixed_vehicle_random_context import association


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if out.exists():p.error('Use a fresh directory')
    out.mkdir(parents=True);source=root/'logdir/retrain_1m_20261017_0115/training/source';sys.path.insert(0,str(source))
    from omegaconf import OmegaConf
    from context import HistoryEncoder
    from envs.metadrive import MetaDrive
    from metadrive.component.vehicle.vehicle_type import vehicle_type
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    checkpoint=source.parent/'r2dreamer_context/best.pt';cfg=OmegaConf.load(checkpoint.parent/'.hydra/config.yaml');cfg.device='cpu';OmegaConf.resolve(cfg)
    cfg.env.traffic_density=0.;cfg.env.spawn.randomize=False;cfg.env.eval_start_seed=20000;cfg.env.eval_num_scenarios=1
    cfg.env.time_limit=200;cfg.env.termination.out_of_road_done=False;cfg.env.termination.on_continuous_line_done=False
    cfg.env.chassis.eval_models=['m'];cfg.env.chassis.eval_mass_scales=[1.];cfg.env.chassis.eval_steering_scales=[1.]
    OmegaConf.save(cfg,out/'sampling_config.yaml')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);encoder=HistoryEncoder().eval()
    encoder.load_state_dict({k[len('context_encoder.'):]:v for k,v in payload['agent_state_dict'].items() if k.startswith('context_encoder.')},strict=True)
    step=int(payload['step']);del payload
    cls=vehicle_type['m'];nominal_mass=int(cls.MASS);nominal_base=cls.FRONT_WHEELBASE+cls.REAR_WHEELBASE
    friction_space=cls.PARAMETER_SPACE['wheel_friction'];nominal_friction=float((friction_space.low[0]+friction_space.high[0])/2)
    scales=np.array([.8,.9,1.,1.1,1.2]);profiles=24
    factors={'speed':np.array([5.,7.5,10.,12.5,15.]),'wheel_angle':np.array([-30.,-15.,0.,15.,30.]),
             'friction':nominal_friction*scales,'mass':nominal_mass*scales,'wheelbase':nominal_base*scales}
    titles={'speed':'Initial speed','wheel_angle':'Actual front-wheel angle','friction':'Wheel friction','mass':'Mass','wheelbase':'Wheelbase'}
    units={'speed':'m/s','wheel_angle':'degrees','friction':'Bullet friction slip','mass':'kg','wheelbase':'m'}
    contract={'claim':'Show factor-associated organization of a trained context compared to independent no-information 8D noise, under separate one-factor probes.',
        'archetype':'quantitative grid','backend':'python','figures':'Five random-vs-best pairs and an overview; labels indicate physical factor levels, not vehicle type.',
        'risks':['OAT sensitivity does not establish cross-condition invariance.','Noise reference cannot isolate representation gain caused by training.',
                 'Wheelbase is a custom wheel-connection geometry intervention, holding the collision body and inertia unchanged.',
                 'Friction slip is not a measured road coefficient.','Angle varies the steering action explicitly present in the context input.'],
        'exports':['png','pdf','svg','csv','npz']}
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    env=MetaDrive(cfg.env,evaluation=True);env._env.config['map_config']['config']='';env._env.config['map_config']['lane_width']=20.
    rows=[];histories=[];paths=[];wheel_positions=[];started=time.monotonic()
    try:
        for factor,levels in factors.items():
            for profile in range(profiles):
                rng=np.random.default_rng(910000+profile);throttle=np.interp(np.arange(5),[0,2,4],rng.uniform(.15,.4,3)).astype(np.float32)
                for level_index,level in enumerate(levels):
                    spec=env._env.config['chassis_spec'];spec['models']=['m'];spec['steering_scales']=[1.];spec['mass_scales']=[float(level/nominal_mass) if factor=='mass' else 1.]
                    env.reset(seed=20000);car=env._env.vehicle
                    friction=float(level) if factor=='friction' else nominal_friction
                    wbscale=float(level/nominal_base) if factor=='wheelbase' else 1.
                    # Reset physical wheel geometry for every probe, including pooled vehicles.
                    for wi in range(car.system.getNumWheels()):
                        wheel=car.system.getWheel(wi);pos=wheel.getChassisConnectionPointCs()
                        pos.setY(float(cls.FRONT_WHEELBASE if wi<2 else -cls.REAR_WHEELBASE)*wbscale)
                        wheel.setChassisConnectionPointCs(pos);wheel.setFrictionSlip(friction)
                    car.config['wheel_friction']=friction;car.system.resetSuspension()
                    requested_mass=float(level) if factor=='mass' else nominal_mass
                    np.testing.assert_allclose(car.body.getMass(),requested_mass,rtol=1e-6,atol=1e-4)
                    for _ in range(30):
                        _,_,done,_=env.step(np.zeros(2,np.float32))
                        if done:raise RuntimeError('Settling terminated')
                    speed=float(level) if factor=='speed' else 10.;angle=float(level) if factor=='wheel_angle' else 15.
                    car.set_velocity([1.,0.],value=speed,in_local_frame=True)
                    steer=angle/car.max_steering;car._set_action(np.array([steer,throttle[0]],np.float32));env._context_history.observe(car)
                    np.testing.assert_allclose(np.linalg.norm(car.velocity),speed,rtol=1e-6,atol=1e-4)
                    initial_positions=np.array([list(car.system.getWheel(wi).getChassisConnectionPointCs()) for wi in range(4)])
                    front=initial_positions[:2,1].mean();rear=initial_positions[2:,1].mean();actual_wb=float(front-rear)
                    np.testing.assert_allclose(actual_wb,nominal_base*wbscale,rtol=1e-6)
                    trajectory=[]
                    for k in range(5):
                        obs,_,done,_=env.step(np.array([steer,throttle[k]],np.float32))
                        if done or obs['ctx_collision']:raise RuntimeError(f'Invalid probe {factor}/{level}/{profile}, no selective rejection')
                        trajectory.append([*obs['ctx_ego_state'],float(np.linalg.norm(car.velocity))])
                        np.testing.assert_allclose([car.system.getSteeringValue(wi) for wi in [0,1]],angle,atol=1e-4)
                    np.testing.assert_allclose([car.system.getWheel(wi).getFrictionSlip() for wi in range(4)],friction,rtol=1e-6)
                    final_positions=np.array([list(car.system.getWheel(wi).getChassisConnectionPointCs()) for wi in range(4)])
                    np.testing.assert_array_equal(initial_positions,final_positions)
                    assert obs['ctx_mask'].all()
                    histories.append(obs['ctx_history'].copy());paths.append(trajectory);wheel_positions.append(initial_positions)
                    rows.append({'factor':factor,'level_index':level_index,'factor_value':float(level),'profile':profile,'action_seed':910000+profile,
                        'initial_speed_m_s':speed,'actual_front_angle_deg':angle,'actual_mass_kg':float(car.body.getMass()),'actual_friction_slip':float(car.system.getWheel(0).getFrictionSlip()),
                        'actual_wheelbase_m':actual_wb,'final_speed_m_s':trajectory[-1][2]})
            print('COLLECTED',factor,len(rows),flush=True)
    finally:env.close()
    history=np.array(histories);physical=np.array(paths);assert history.shape==(600,5,4)
    with torch.inference_mode():best=encoder(torch.from_numpy(history),torch.ones(len(history),5,dtype=torch.bool)).numpy()
    seeds=list(range(20261015,20261035));noise=np.array([np.random.default_rng(s).standard_normal((len(rows),8)).astype(np.float32) for s in seeds])
    groups=np.array([r['profile'] for r in rows]);folds=np.array_split(np.random.default_rng(20261016).permutation(np.unique(groups)),6);fit=~np.isin(groups,folds[0])
    coords={};ratios={};stats={};projections=np.zeros((2,len(rows),2))
    for factor,levels in factors.items():
        ids=np.array([i for i,r in enumerate(rows) if r['factor']==factor]);labels=np.array([rows[i]['level_index'] for i in ids])
        for stage,x in [('random',noise[0,ids]),('best',best[ids])]:
            z,ratio,_,_,_=fit_pca(x.astype(float),fit[ids]);coords[factor,stage]=z;ratios[factor,stage]=ratio
            projections[0 if stage=='random' else 1,ids]=z
        random_eta=np.array([association(x[ids],labels,fit[ids]) for x in noise]);trained_eta=association(best[ids],labels,fit[ids])
        stats[factor]={'levels':levels.tolist(),'best_eta2':trained_eta,'random_eta2_mean':float(random_eta.mean()),'random_eta2_sd':float(random_eta.std(ddof=1)),
           'random_eta2_by_seed':random_eta.tolist(),'pca_variance_ratios':{stage:ratios[factor,stage].tolist() for stage in ['random','best']}}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    def panel(ax,factor,stage,size=25):
        ids=np.array([i for i,r in enumerate(rows) if r['factor']==factor]);values=np.array([rows[i]['factor_value'] for i in ids]);order=np.random.default_rng(13).permutation(len(ids))
        cm='coolwarm' if factor=='wheel_angle' else 'viridis';z=coords[factor,stage];ratio=ratios[factor,stage]
        sc=ax.scatter(*z[order].T,c=values[order],cmap=cm,s=size,alpha=.65,edgecolors='none',vmin=factors[factor].min(),vmax=factors[factor].max())
        ax.set_xlabel(f'PC1 ({ratio[0]:.1%})');ax.set_ylabel(f'PC2 ({ratio[1]:.1%})');ax.grid(alpha=.12);ax.set_axisbelow(True)
        return sc
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    for factor in factors:
        fig,axs=plt.subplots(1,2,figsize=(9.6,4.6));fig.subplots_adjust(left=.08,right=.89,bottom=.21,top=.77,wspace=.34)
        for ax,stage,title in zip(axs,['random','best'],['Independent random 8D context','Best encoder output']):
            sc=panel(ax,factor,stage);ax.set_title(title,fontsize=10)
        bar=fig.add_axes([.92,.23,.018,.50]);fig.colorbar(sc,cax=bar,ticks=factors[factor],label=units[factor])
        fig.suptitle(f'{titles[factor]} | M vehicle | 24 paired profiles per level\nOnly this factor varies; best at {step:,} steps',fontsize=11,y=.98)
        se=stats[factor];fig.text(.5,.075,f'Full-8D factor association η²: random {se["random_eta2_mean"]:.1%} ± {se["random_eta2_sd"]:.1%} SD; best {se["best_eta2"]:.1%}',ha='center',fontsize=9)
        fig.text(.5,.018,'Independent PCA bases; noise is N(0, I8). One-factor association is not parameter-only identification.',ha='center',fontsize=8)
        export(fig,f'{factor}_random_vs_best_pca')
    fig,axs=plt.subplots(2,5,figsize=(17,6.7));fig.subplots_adjust(left=.055,right=.99,bottom=.22,top=.85,wspace=.43,hspace=.40)
    for col,factor in enumerate(factors):
        for row,stage in enumerate(['random','best']):
            sc=panel(axs[row,col],factor,stage,size=13)
            if row==0:axs[row,col].set_title(titles[factor],fontsize=10)
        box=axs[1,col].get_position();cax=fig.add_axes([box.x0,.10,box.width,.013]);cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=factors[factor]);cb.ax.tick_params(labelsize=7);cb.set_label(units[factor],fontsize=8)
    fig.text(.012,.68,'Random 8D',rotation=90,va='center',fontsize=11);fig.text(.012,.35,'Best encoder',rotation=90,va='center',fontsize=11)
    fig.suptitle('Five separate one-factor probes: independent random context vs best encoder\nM vehicle; five levels × 24 profiles per factor; all context vectors are 8D',fontsize=13,y=.98)
    fig.text(.5,.015,'Each factor/stage has its own PCA basis. Parameter sensitivity under controlled conditions does not imply invariance across operating conditions.',ha='center',fontsize=9)
    export(fig,'five_factors_random_vs_best_pca')
    np.savez_compressed(out/'representations.npz',history=history,physical_paths=physical,wheel_connection_points=np.array(wheel_positions),best_context=best,random_contexts=noise,pca_coordinates=projections)
    for i,row in enumerate(rows):
        row.update({f'best_context_{j}':float(best[i,j]) for j in range(8)})
        for stage,k in [('random',0),('best',1)]:row.update({f'{stage}_PC{j+1}':float(projections[k,i,j]) for j in range(2)})
    with (out/'source_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata={'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'factors':stats,'windows':600,'profiles_per_level':24,
      'device':'cpu','nominal_mass_kg':nominal_mass,'nominal_friction_slip':nominal_friction,'nominal_wheelbase_m':nominal_base,'default_initial_speed_m_s':10.,'default_front_angle_deg':15.,
      'random_seeds':seeds,'random_distribution':'independent N(0,I8) per window; does not use an encoder or history',
      'wheelbase_intervention':'Scale signed front/rear chassis wheel connection y coordinates about the body origin by the same factor; preserve ratio, lateral/z, body dimensions, mass and body inertia. Actual connection distance read back before and after every rollout.',
      'pca_protocol':'Within each factor and stage, standardization and PCA fit only 20 profile groups; all 24 displayed. No factor labels enter PCA fitting. η² uses full standardized8D across all profiles.',
      'sampling_protocol':'Same wide flat road, zero traffic, 30 settling steps, same24 throttle profiles. Each factor varied alone. Initial velocity set once; wheel steering constant within each window, five real physics decision steps. No rejected windows.',
      'limitations':contract['risks'],'elapsed_seconds':time.monotonic()-started}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');shutil.copy2(Path(__file__),out/'script_snapshot.py')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
