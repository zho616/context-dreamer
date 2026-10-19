"""Frozen world-model interventions on identical held-out MetaDrive trajectories.

No training changes. Observation-free forecasts never consume future contexts.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np


def save(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['all','evaluate'],default='all')
    p.add_argument('--maps',type=int,default=10)
    p.add_argument('--draws',type=int,default=3)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args(); root=args.training_root.resolve(); out=args.output.resolve()
    if args.phase=='all' and out.exists(): p.error('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'source'))
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from dreamer import Dreamer
    from envs.metadrive import MetaDrive
    from context import refresh_context,canonical_action
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('high')
    device=args.device; started=time.monotonic()
    status={'phase':'starting','started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'pid':__import__('os').getpid()}
    def progress(phase,**kwargs):
        status.update(phase=phase,elapsed_seconds=time.monotonic()-started,**kwargs);save(out/'status.json',status)
    def tensors(obs):
        return TensorDict({k:torch.as_tensor(v,device=device).unsqueeze(0) for k,v in obs.items()},batch_size=[1])
    cfg=OmegaConf.load(root/'r2dreamer_context/.hydra/config.yaml');cfg.device=device;OmegaConf.resolve(cfg)
    cfg.model.compile=False;cfg.device=device
    # Nominal fixed parameter sets make same-chassis donors genuinely identical.
    cfg.env.chassis.eval_mass_scales=[1.];cfg.env.chassis.eval_steering_scales=[1.]
    cfg.env.eval_start_seed=30000;cfg.env.eval_num_scenarios=args.maps
    env=MetaDrive(cfg.env,evaluation=True)
    models={};provenance={}
    for label,folder in [('baseline','r2dreamer'),('context','r2dreamer_context')]:
        c=OmegaConf.load(root/folder/'.hydra/config.yaml');c.device=device;OmegaConf.resolve(c);c.model.compile=False
        path=root/folder/'best.pt';payload=torch.load(path,map_location=device,weights_only=False)
        model=Dreamer(c.model,env.observation_space,env.action_space).to(device)
        model.load_state_dict(payload['agent_state_dict'],strict=True);model.clone_and_freeze();model.eval()
        models[label]=model
        provenance[label]={'checkpoint':str(path),'step':payload['step'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        del payload
    save(out/'provenance.json',{'models':provenance,'source_root':str(root/'source'),'maps':args.maps,
        'protocol':'Shared real trajectories from both frozen policies; five nominal chassis; fixed weights; target-time physical contexts forbidden in forecasts.',
        'donor_calipers':{'speed_m_s':.75,'front_angle_degrees':3.,'native_steer':.15,'throttle':.15},
        'posterior_seed':314159,'forecast_draws':args.draws,'maximum_horizon':10})
    episodes=[]
    try:
        if args.phase=='all':
            (out/'episodes').mkdir()
            for map_id in range(args.maps):
                seed=30000+map_id
                for chassis in ['default','m','l','s','xl']:
                    for policy in ['baseline','context']:
                        progress('collecting',completed_episodes=len(episodes),total_episodes=args.maps*10)
                        env._env.config['chassis_spec']['models']=[chassis]
                        torch.manual_seed(seed+12345);np.random.seed(seed+12345)
                        obs=env.reset(seed=seed);state=models[policy].get_initial_state(1)
                        arrays={k:[] for k in obs};arrays['executed_action']=[];arrays['speed']=[];arrays['wheel_degrees']=[];arrays['reward']=[]
                        def append(observation,reward=0.):
                            for k,v in observation.items():arrays[k].append(np.asarray(v).copy())
                            arrays['reward'].append(float(reward))
                            car=env._env.vehicle
                            arrays['speed'].append(float(np.linalg.norm(car.velocity)))
                            arrays['wheel_degrees'].append(float(np.mean([car.system.getSteeringValue(i) for i in (0,1)])))
                        append(obs)
                        with torch.inference_mode():
                            for step in range(int(cfg.env.time_limit)):
                                action,state=models[policy].act(tensors(obs),state,eval=True)
                                act=action[0].cpu().numpy();arrays['executed_action'].append(act)
                                obs,reward,done,info=env.step(act);append(obs,reward)
                                if done:break
                        assert done
                        arrays['executed_action'].append(np.zeros(2,np.float32))
                        name=f'{len(episodes):04d}.npz';np.savez_compressed(out/'episodes'/name,**{k:np.array(v) for k,v in arrays.items()})
                        episodes.append({'file':name,'map_id':map_id,'seed':seed,'model':chassis,'policy':policy,
                                         'length':step+1,'chassis':env.chassis_parameters,'success':bool(info.get('arrive_dest',False))})
                        save(out/'episodes.json',episodes)
                        print('collected',len(episodes),chassis,policy,'steps',step+1,flush=True)
        else:episodes=json.loads((out/'episodes.json').read_text())
    finally:env.close()
    data=[dict(np.load(out/'episodes'/e['file'])) for e in episodes]
    for d in data:
        if 'reward' not in d:
            d['reward']=sum(d['log_reward_'+key] for key in ['progress','speed','lateral','terminal']).astype(np.float32)
    origins=[]
    for ei,(meta,d) in enumerate(zip(episodes,data)):
        end=meta['length']
        ts=sorted(set(range(20,end,10))|{end-1,end-5,end-10})
        for t in ts:
            if t<20 or not d['ctx_mask'][t].all() or d['is_last'][t] or d['ctx_collision'][t]:continue
            origins.append({'episode':ei,'t':t,'map_id':meta['map_id'],'model':meta['model'],'policy':meta['policy'],
                            'speed':float(d['speed'][t]),'angle':float(d['wheel_degrees'][t]),
                            'steer':float(d['executed_action'][t-1,0]),'throttle':float(d['executed_action'][t-1,1])})
    n=len(origins);assert n>0
    op=np.array([[o[k] for k in ['speed','angle','steer','throttle']] for o in origins])
    typ=np.array([o['model'] for o in origins]);ep=np.array([o['episode'] for o in origins]);pol=np.array([o['policy'] for o in origins])
    cal=np.array([.75,3.,.15,.15]);same=np.full(n,-1);other=np.full(n,-1)
    for i,o in enumerate(origins):
        distances=np.abs(op-op[i])/cal;allowed=(distances<=1).all(1)&(ep!=ep[i])&(pol==pol[i])
        # Different maps for both donor controls; avoid adjacent overlapping windows.
        allowed &= np.array([x['map_id']!=o['map_id'] for x in origins])
        for match,indices in [(typ==typ[i],same),(typ!=typ[i],other)]:
            candidates=np.flatnonzero(allowed&match)
            if len(candidates):indices[i]=candidates[np.argmin((distances[candidates]**2).sum(1))]
    matched=(same>=0)&(other>=0)
    assert matched.sum()>20,'Too few matched donors; report failure instead of silently relaxing controls'
    save(out/'origins.json',origins)
    np.savez_compressed(out/'matching.npz',same=same,other=other,matched=matched,conditions=op,calipers=cal)
    for i in np.flatnonzero(matched):
        assert typ[same[i]]==typ[i] and typ[other[i]]!=typ[i]
        assert (np.abs(op[same[i]]-op[i])<=cal+1e-7).all() and (np.abs(op[other[i]]-op[i])<=cal+1e-7).all()
        assert episodes[ep[i]]['chassis']==episodes[ep[same[i]]]['chassis']
    progress('filtering',origins=n,matched_origins=int(matched.sum()))
    histories=np.stack([data[o['episode']]['ctx_history'][o['t']] for o in origins])
    masks=np.stack([data[o['episode']]['ctx_mask'][o['t']] for o in origins])
    x0=np.stack([data[o['episode']]['ctx_ego_state'][o['t']] for o in origins])
    H=10;acts=np.zeros((n,H,2),np.float32);truth=np.zeros((n,H,2),np.float32)
    rewards=np.zeros((n,H,1),np.float32);cont=np.zeros((n,H,1),np.float32);valid=np.zeros((n,H),bool);physical=np.zeros_like(valid)
    for i,o in enumerate(origins):
        d=data[o['episode']];t=o['t'];length=min(H,episodes[o['episode']]['length']-t)
        acts[i,:length]=d['executed_action'][t:t+length];truth[i,:length]=d['ctx_ego_state'][t+1:t+length+1]
        rewards[i,:length,0]=d['reward'][t+1:t+length+1];cont[i,:length,0]=1-d['is_terminal'][t+1:t+length+1]
        valid[i,:length]=True
        physical[i,:length]=~(d['is_last'][t+1:t+length+1]|d['ctx_collision'][t+1:t+length+1])
        np.testing.assert_allclose(d['ctx_prev_action'][t+1:t+length+1],np.stack((acts[i,:length,1],-acts[i,:length,0]),-1),atol=1e-6)
    caches={};posterior_metrics={}
    with torch.inference_mode():
        for name,model in models.items():
            z_all=torch.empty(n,model.rssm._stoch,model.rssm._discrete,device=device);h_all=torch.empty(n,model.rssm._deter,device=device)
            torch.manual_seed(314159)
            for start in range(0,len(episodes),10):
                chunk=list(range(start,min(start+10,len(episodes))));maxlen=max(episodes[i]['length']+1 for i in chunk)
                keys=['state','is_first','ctx_history','ctx_mask']
                td={}
                for k in keys:
                    arr=np.zeros((len(chunk),maxlen,*data[chunk[0]][k].shape[1:]),dtype=data[chunk[0]][k].dtype)
                    for b,ei in enumerate(chunk):arr[b,:len(data[ei][k])]=data[ei][k]
                    td[k]=torch.as_tensor(arr,device=device)
                emb=model.encoder(TensorDict(td,batch_size=[len(chunk),maxlen]))
                cc=model.context_encoder(td['ctx_history'],td['ctx_mask']) if model.context_enabled else None
                prev=torch.zeros(len(chunk),maxlen,2,device=device)
                for b,ei in enumerate(chunk):prev[b,1:len(data[ei]['executed_action'])]=torch.as_tensor(data[ei]['executed_action'][:-1],device=device)
                z,h=model.rssm.initial(len(chunk))
                lookup={}
                for i,o in enumerate(origins):
                    if o['episode'] in chunk:lookup.setdefault(o['t'],[]).append((i,chunk.index(o['episode'])))
                for t in range(maxlen):
                    extra=(cc[:,t],) if cc is not None else ()
                    z,h,_=model.rssm.obs_step(z,h,prev[:,t],emb[:,t],td['is_first'][:,t],*extra)
                    for i,b in lookup.get(t,[]):z_all[i]=z[b];h_all[i]=h[b]
                print('filtered',name,start+len(chunk),'/',len(episodes),flush=True)
            caches[name]=(z_all,h_all)
        model=models['context'];contexts=model.context_encoder(torch.as_tensor(histories,device=device),torch.as_tensor(masks,device=device))
        torch.manual_seed(271828);shuffle=torch.randperm(n,device=device)
        variants=['baseline','dynamic','zero','fixed_own','fixed_same','fixed_other','fixed_shuffle','fixed_self_check']
        errors={v:{k:np.zeros((args.draws,n,H),np.float32) for k in ['reward_mae','reward_nll','continue_brier','continue_nll']} for v in variants}
        for v in variants:
            m=models['baseline' if v=='baseline' else 'context'];progress('forecasting',variant=v,origins=n,matched_origins=int(matched.sum()))
            for start in range(0,n,128):
                sl=slice(start,min(n,start+128));count=sl.stop-sl.start
                az=torch.as_tensor(acts[sl],device=device);rt=torch.as_tensor(rewards[sl],device=device);ct=torch.as_tensor(cont[sl],device=device)
                for draw in range(args.draws):
                    torch.manual_seed(900000+start+draw*100000)
                    z,h=[x[sl].clone() for x in caches['baseline' if v=='baseline' else 'context']]
                    state=torch.as_tensor(x0[sl],device=device);hist=torch.as_tensor(histories[sl],device=device);mask=torch.as_tensor(masks[sl],device=device)
                    c=contexts[sl].clone();fixed=c
                    if v in ['fixed_same','fixed_other']:
                        ids=(same if v=='fixed_same' else other)[sl];ids=np.where(ids>=0,ids,np.arange(start,sl.stop))
                        fixed=contexts[torch.as_tensor(ids,device=device)]
                    elif v=='fixed_shuffle':fixed=contexts[shuffle[sl]]
                    for k in range(H):
                        if v=='baseline':z,h=m.rssm.img_step(z,h,az[:,k])
                        else:
                            if v in ['dynamic','zero']:
                                state,c,hist,mask=refresh_context(m.context_encoder,m.context_dynamics,state,az[:,k],c,hist,mask,torch.ones(count,dtype=torch.bool,device=device))
                                injected=c if v=='dynamic' else torch.zeros_like(c)
                            else:injected=fixed
                            z,h=m.rssm.img_step(z,h,az[:,k],injected)
                        feat=m.rssm.get_feat(z,h);rw=m.reward(feat);cn=m.cont(feat)
                        for key,value in [('reward_mae',(rw.mode()-rt[:,k]).abs().reshape(-1)),('reward_nll',-rw.log_prob(rt[:,k])),
                                          ('continue_brier',(cn.mean.reshape(-1)-ct[:,k,0]).square()),('continue_nll',-cn.log_prob(ct[:,k]))]:
                            errors[v][key][draw,sl,k]=value.reshape(-1).cpu().numpy()
            print('forecast completed',v,flush=True)
        for key in errors['fixed_own']:
            np.testing.assert_array_equal(errors['fixed_own'][key],errors['fixed_self_check'][key])
        # Auxiliary E+D predicts physical response; this is not an RSSM output head.
        aux={}
        for v in ['own','zero','same','other','shuffle','persistence']:
            c=contexts.clone()
            if v=='zero':c=torch.zeros_like(c)
            if v in ['same','other']:
                ids=(same if v=='same' else other);ids=np.where(ids>=0,ids,np.arange(n));c=contexts[torch.as_tensor(ids,device=device)]
            if v=='shuffle':c=contexts[shuffle]
            x=torch.as_tensor(x0,device=device);tt=torch.as_tensor(truth,device=device);aa=torch.as_tensor(acts,device=device)
            e=[]
            for k in range(H):
                if v!='persistence':x=x+m.context_dynamics(c,x,canonical_action(aa[:,k]))
                e.append((x-tt[:,k]).abs().cpu().numpy())
            aux[v]=np.stack(e,1)
    np.savez_compressed(out/'errors.npz',**{f'{v}__{k}':a for v,d in errors.items() for k,a in d.items()},
                        **{f'aux__{v}':a for v,a in aux.items()},valid=valid,physical_valid=physical,truth=truth,actions=acts)
    # Inference is finished; retain arrays and write map-cluster paired summaries.
    progress('summarizing')
    maps=np.array([o['map_id'] for o in origins]);rng=np.random.default_rng(20261019)
    def mean(a,sel):return float(a[sel].mean()) if np.any(sel) else None
    summary={'world_model':{},'auxiliary':{},'paired':{},'origins':n,'matched_origins':int(matched.sum()),'independent_maps':len(set(maps))}
    for label,select in [('all',np.ones(n,bool)),('matched',matched),('train_types',np.isin(typ,['default','m','l'])),('unseen_types',np.isin(typ,['s','xl']))]:
        summary['world_model'][label]={v:{key:{str(k+1):mean(a.mean(0)[:,k],select&valid[:,k]&(matched if v in ['fixed_same','fixed_other'] else True)) for k in [0,4,9]} for key,a in d.items()} for v,d in errors.items() if v!='fixed_self_check'}
        summary['auxiliary'][label]={v:{'normalized_l1':{str(k+1):mean((a[:,:,0]/10+a[:,:,1])[:,k],select&physical[:,k]&(matched if v in ['same','other'] else True)) for k in [0,4,9]},
            'acceleration_mae':{str(k+1):mean(a[:,k,0],select&physical[:,k]&(matched if v in ['same','other'] else True)) for k in [0,4,9]},
            'yaw_mae':{str(k+1):mean(a[:,k,1],select&physical[:,k]&(matched if v in ['same','other'] else True)) for k in [0,4,9]}} for v,a in aux.items()}
    pairs=[('dynamic','baseline'),('zero','dynamic'),('fixed_other','fixed_own'),('fixed_other','fixed_same'),('fixed_same','fixed_own'),('fixed_shuffle','fixed_own'),('fixed_own','dynamic')]
    for a,b in pairs:
        label=f'{a}_minus_{b}';summary['paired'][label]={}
        select=matched if ('same' in a or 'other' in a or 'same' in b) else np.ones(n,bool)
        for key in ['reward_mae','reward_nll','continue_brier','continue_nll']:
            delta=errors[a][key].mean(0)-errors[b][key].mean(0)
            summary['paired'][label][key]={}
            for k in [0,4,9]:
                sel=select&valid[:,k];blocks=np.array([delta[sel&(maps==mi),k].mean() for mi in sorted(set(maps[sel]))])
                bs=blocks[rng.integers(0,len(blocks),(10000,len(blocks)))].mean(1)
                summary['paired'][label][key][str(k+1)]={'window_weighted_difference':mean(delta[:,k],sel),'map_weighted_difference':float(blocks.mean()),'map_bootstrap95':np.quantile(bs,[.025,.975]).tolist()}
    summary['matching']={'coverage':float(matched.mean()),'by_model':{m:int(np.sum(matched&(typ==m))) for m in sorted(set(typ))},
       'same_mean_abs_difference':np.abs(op[same[matched]]-op[matched]).mean(0).tolist(),
       'other_mean_abs_difference':np.abs(op[other[matched]]-op[matched]).mean(0).tolist()}
    save(out/'results.json',summary)
    lines=['# 冻结世界模型与context干预验证','','同一批真实轨迹、相同未来动作；两种行为策略×五种nominal车型。地图数量见元数据，无训练更新。',
      '预测前从reset按实际观测过滤至当前时刻；随后不再输入未来观测或真实未来context。动态context严格调用训练原refresh_context。',
      '所有context分支共享正确历史得到的起点RSSM状态，仅干预未来注入，不能排除过去context已存入h的作用。固定context对照用于隔离注入与预测器更新。',
      f"共{n}个预测起点，{matched.sum()}个满足同/异车型donor匹配；覆盖{matched.mean():.1%}。",
      '同车型donor参数完全相同；同/异车型donor均来自同一行为策略、其他地图/其他episode，按速度±0.75m/s、实际前轮角±3度、当前已执行转向/油门各±0.15筛选后最近邻。',
      '所有分支未来动作使用目标轨迹动作。3次未来RSSM采样共用随机seed；固定self-swap逐值一致检查通过。',
      '', '## 世界模型：全部起点', '', '| 分支 | Reward MAE h1/h5/h10 | Reward NLL h1/h5/h10 | Continue Brier h1/h5/h10 |', '|---|---|---|---|']
    for v,d in summary['world_model']['all'].items():
        lines.append('| '+v+' | '+' | '.join('/'.join(f'{d[key][str(k)]:.5f}' for k in [1,5,10]) for key in ['reward_mae','reward_nll','continue_brier'])+' |')
    lines+=['','## 匹配子集：固定context对照','','| 分支 | Reward MAE h1/h5/h10 | Reward NLL h1/h5/h10 |','|---|---|---|']
    for v in ['fixed_own','fixed_same','fixed_other','fixed_shuffle']:
        d=summary['world_model']['matched'][v];lines.append('| '+v+' | '+' | '.join('/'.join(f'{d[key][str(k)]:.5f}' for k in [1,5,10]) for key in ['reward_mae','reward_nll'])+' |')
    lines+=['','## 辅助E+D：匹配子集物理响应误差','','| context | 归一化L1 h1/h5/h10 | 加速度MAE h1/h5/h10 | 横摆角速度MAE h1/h5/h10 |','|---|---|---|---|']
    for v,d in summary['auxiliary']['matched'].items():
        lines.append('| '+v+' | '+' | '.join('/'.join(f'{d[key][str(k)]:.5f}' for k in [1,5,10]) for key in ['normalized_l1','acceleration_mae','yaw_mae'])+' |')
    lines+=['','## 解释范围','',
      '- RSSM没有加速度/横摆角速度重建头：世界模型直接比较共同可观测的reward与continuation预测；物理响应由独立E+D辅助分支验证，不能混为RSSM预测。',
      '- 单步及多步均为open-loop prior，不用未来真实观测。未来动作已知，这评估模型预测而非闭环驾驶性能。',
      '- 同车型donor比own差也可能来自未匹配历史/位置/交通工况；错车型减同车型更有解释力，但近邻并非严格随机化车型干预。',
      '- donor只匹配当前工况和最近动作，不保证五步完整历史逐值一致；匹配距离及覆盖率均报告，不外推至被排除工况。',
      '- 预测起点不跨reset；终止后目标均mask，物理辅助误差进一步剔除终止/碰撞目标。10步超出辅助训练5步，仅作外推诊断。',
      '- 固定nominal车辆参数测试用于隔离车型身份，非随机质量/转角全分布的驾驶性能测试。',
      '- 区间按新地图成组bootstrap，限于当前单训练seed、固定模型和本采样协议；不能作为多seed稳定性证据。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    progress('completed',origins=n,matched_origins=int(matched.sum()))
    print('COMPLETED',out,flush=True)

if __name__=='__main__':main()
