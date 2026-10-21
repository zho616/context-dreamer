"""Frozen response-predictor interventions on a shared natural trajectory corpus.

Keep recipient state/action fixed. Exchange only the encoded context, across real
independent episodes, with exact model/mass/max-steering matching for same chassis.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--donors',type=int,default=20)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=root/'logdir/world_model_context_validation_20261019_cpu'
    episodes=json.loads((cache/'episodes.json').read_text())
    data=[dict(np.load(cache/'episodes'/e['file'])) for e in episodes]
    original=json.loads((cache/'origins.json').read_text())
    origins=[];histories=[];initial=[]
    for o in original:
        d=data[o['episode']];t=o['t'];sl=slice(t-4,t+1)
        if not d['ctx_mask'][t].all() or any(d[k][sl].any() for k in ['is_first','is_last','is_terminal','ctx_collision']):continue
        np.testing.assert_allclose(d['ctx_history'][t,:,:2],d['ctx_ego_state'][sl],atol=1e-6)
        np.testing.assert_allclose(d['ctx_history'][t,:,2:],d['ctx_prev_action'][sl],atol=1e-6)
        state=np.column_stack((d['ctx_ego_state'][sl],d['speed'][sl],-np.deg2rad(d['wheel_degrees'][sl]))).astype(np.float32)
        histories.append(np.column_stack((state,d['ctx_prev_action'][sl])));initial.append(state[-1]);origins.append(o)
    n=len(origins);histories=np.array(histories,np.float32);initial=np.array(initial,np.float32)
    profiles=np.array([(episodes[o['episode']]['chassis']['model_id'],episodes[o['episode']]['chassis']['mass'],episodes[o['episode']]['chassis']['max_steering']) for o in origins])
    eid=np.array([o['episode'] for o in origins]);maps=np.array([o['map_id'] for o in origins]);policy=np.array([o['policy'] for o in origins])
    unique=np.unique(maps);rng=np.random.default_rng(20261021)
    donors={k:np.full((n,args.donors),-1,int) for k in ['same','other','same_matched']}
    operating=np.column_stack((initial[:,2:4],histories[:,-1,4:]))
    calipers=np.array([.75,np.deg2rad(3),.15,.15])
    for i in range(n):
        allowed=(eid!=eid[i])&(maps!=maps[i])&(policy==policy[i])
        same=(profiles==profiles[i]).all(-1)
        close=(np.abs(operating-operating[i])<=calipers).all(-1)
        for variant,filter_ in [('same',same),('other',~same),('same_matched',same&close)]:
            candidates=np.flatnonzero(allowed&filter_)
            if not len(candidates):continue
            # Equal episode probability, then equal endpoint probability within episode.
            eps=np.unique(eid[candidates])
            for j in range(args.donors):
                donor_ep=rng.choice(eps);donors[variant][i,j]=rng.choice(candidates[eid[candidates]==donor_ep])
    assert (donors['same']>=0).all() and (donors['other']>=0).all()
    for variant in donors:
        rows,cols=np.where(donors[variant]>=0);idx=donors[variant][rows,cols]
        assert np.all(eid[idx]!=eid[rows]) and np.all(maps[idx]!=maps[rows]) and np.all(policy[idx]==policy[rows])
        assert np.all((profiles[idx]==profiles[rows]).all(-1)==(variant!='other'))
        if variant=='same_matched':assert np.all(np.abs(operating[idx]-operating[rows])<=calipers+1e-7)
    H=10;actions=np.zeros((n,H,2),np.float32);truth=np.zeros((n,H,4),np.float32);valid=np.zeros((n,H),bool)
    for i,o in enumerate(origins):
        d=data[o['episode']];t=o['t'];length=min(H,episodes[o['episode']]['length']-t)
        actions[i,:length]=d['executed_action'][t:t+length]
        truth[i,:length]=np.column_stack((d['ctx_ego_state'][t+1:t+length+1],d['speed'][t+1:t+length+1],-np.deg2rad(d['wheel_degrees'][t+1:t+length+1])))
        events=d['is_last'][t+1:t+length+1]|d['is_terminal'][t+1:t+length+1]|d['ctx_collision'][t+1:t+length+1]
        valid[i,:length]=~np.maximum.accumulate(events)
        np.testing.assert_allclose(d['ctx_prev_action'][t+1:t+length+1],np.stack((actions[i,:length,1],-actions[i,:length,0]),-1),atol=1e-6)
    runs={
      '03_no_swap_best':('context_kinematics_500k_20261015_180340','r2dreamer_context','best.pt'),
      '09_deep_swap1_best':('context_swap_deep1_500k_20261020/training','r2dreamer_dali_deep','best.pt'),
      '09_deep_swap1_final':('context_swap_deep1_500k_20261020/training','r2dreamer_dali_deep','latest.pt'),
      '10_joint_swap1_best':('context_swap_dali_joint_500k_20261020_run1/training','r2dreamer_dali_deep_joint','best.pt'),
      '11_shallow_swap1_best':('context_swap_dali_shallow_500k_20261021/training','r2dreamer_dali_shallow','best.pt')}
    scale=np.array([10.,1.,30.,np.pi/3],np.float32)
    provenance={};summary={};raw={};device=args.device
    tensor=lambda x:torch.as_tensor(x,device=device)
    boot=np.random.default_rng(20261021).integers(0,len(unique),(10000,len(unique)))
    def interval(values,mask):
        sums=np.array([values[(maps==m)&mask].sum(dtype=np.float64) for m in unique]);counts=np.array([((maps==m)&mask).sum() for m in unique])
        denom=counts[boot].sum(1);sample=sums[boot].sum(1)/np.maximum(denom,1)
        return {'mean':float(values[mask].mean()),'map_bootstrap95':np.quantile(sample[denom>0],[.025,.975]).tolist()}
    with torch.inference_mode():
      for label,(folder,method,filename) in runs.items():
        training=root/'logdir'/folder;source=training/'source/context.py';checkpoint=training/method/filename
        spec=importlib.util.spec_from_file_location('exchange_'+label,source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        encoder=module.HistoryEncoder(4).to(device).eval();predictor=module.ResidualDynamics(4).to(device).eval()
        payload=torch.load(checkpoint,map_location=device,weights_only=False);weights=payload['agent_state_dict']
        for model,prefix in [(encoder,'context_encoder.'),(predictor,'context_dynamics.')]:
          model.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
        provenance[label]={'step':int(payload['step']),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'context_source':str(source),'context_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        del payload,weights
        c=encoder(tensor(histories),torch.ones(n,5,dtype=torch.bool,device=device))
        own_predictions=[];x=tensor(initial)
        for k in range(H):
          x=module.predict_next_state(predictor,c,x,module.canonical_action(tensor(actions[:,k])));own_predictions.append(x.cpu().numpy())
        own=np.stack(own_predictions,1);variants={'own':own}
        for variant in ['same','same_matched','other','zero']:
          predictions=[]
          for start in range(0,n,128):
            stop=min(n,start+128);count=stop-start
            if variant=='zero':cc=torch.zeros(count,args.donors,8,device=device)
            else:cc=c[tensor(np.maximum(donors[variant][start:stop],0))]
            cc=cc.reshape(-1,8);x=tensor(initial[start:stop])[:,None,:].expand(-1,args.donors,-1).reshape(-1,4)
            aa=tensor(actions[start:stop])[:,None,:,:].expand(-1,args.donors,-1,-1).reshape(-1,H,2)
            pred=[]
            for k in range(H):
              x=module.predict_next_state(predictor,cc,x,module.canonical_action(aa[:,k]));pred.append(x.reshape(count,args.donors,4).cpu().numpy())
            predictions.append(np.stack(pred,2))
          variants[variant]=np.concatenate(predictions,0)
        summary[label]={}
        own_error=np.abs(own-truth)
        for variant,pred in variants.items():
          error=own_error if variant=='own' else np.abs(pred-truth[:,None]).mean(1)
          change=np.zeros_like(error) if variant=='own' else np.abs(pred-own[:,None]).mean(1)
          raw[label+'__'+variant+'__error']=error;raw[label+'__'+variant+'__change']=change
          summary[label][variant]={}
          for h in [1,5,10]:
            k=h-1;mask=valid[:,k].copy()
            if variant in donors:mask&=(donors[variant]>=0).all(1)
            norm=(error[:,k]/scale).sum(-1);base=(own_error[:,k]/scale).sum(-1)
            normchange=(change[:,k]/scale).sum(-1)
            summary[label][variant][str(h)]={'count':int(mask.sum()),'normalized_l1':interval(norm,mask),'normalized_error_increase':interval(norm-base,mask),'normalized_prediction_change':interval(normchange,mask),
              'physical_mae':error[mask,k].mean(0).tolist(),'physical_prediction_change':change[mask,k].mean(0).tolist(),
              'relative_error_increase_percent':float(100*(norm[mask].mean()/base[mask].mean()-1))}
        print('completed',label,flush=True)
    np.savez_compressed(out/'arrays.npz',**raw,valid=valid,profiles=profiles,maps=maps,episode_ids=eid,history=histories,actions=actions,truth=truth,**{'donors_'+k:v for k,v in donors.items()})
    (out/'origins.json').write_text(json.dumps(origins,indent=2)+'\n')
    protocol={'corpus':str(cache),'origins':n,'episodes':len(episodes),'maps':len(unique),'donors_per_anchor':args.donors,
      'matching':'Exact model ID, actual mass and actual max steering; different episodes AND maps; same reference collection policy. Uniform donor episode then endpoint.',
      'same_matched_calipers':{'speed_mps':.75,'wheel_rad':float(np.deg2rad(3)),'previous_throttle':.15,'previous_right_steer':.15},
      'forecast':'Fixed initial context throughout each response forecast; recipient current state and all recorded future actions unchanged. No future observations. Exclude collision/end and all subsequent targets.',
      'scope':'Auxiliary physical-response predictor, not RSSM reward/continuation heads. 5 nominal chassis only; not all 75 randomized profiles. Single training seed; 03 vs 09 also changes injection.',
      'scale':scale.tolist(),'precision':'float32 highest matmul precision','device':device}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'provenance':provenance,'summary':summary},indent=2)+'\n')
    lines=['# 同底盘context交换预测测试','','## 方法','',json.dumps(protocol,ensure_ascii=False,indent=2),'',
      '## 一步预测：误差与交换变化','','归一化L1 = |加速度误差|/10 + |角速度误差| + |速度误差|/30 + |前轮角误差|/(π/3)。表内先按每个受体的20个供体平均，再按受体平均；并非只比较均值预测，避免抵消。own与same/other/zero使用相同受体。', '',
      '|方法|own误差|同底盘交换误差|相对误差增幅|预测本身变化L1|异底盘误差|零context误差|','|---|---:|---:|---:|---:|---:|---:|']
    for label,s in summary.items():
      own=s['own']['1'];same=s['same']['1'];other=s['other']['1'];zero=s['zero']['1']
      lines.append(f'|{label}|{own["normalized_l1"]["mean"]:.5f}|{same["normalized_l1"]["mean"]:.5f}|{same["relative_error_increase_percent"]:+.1f}%|{same["normalized_prediction_change"]["mean"]:.5f}|{other["normalized_l1"]["mean"]:.5f}|{zero["normalized_l1"]["mean"]:.5f}|')
    lines+=['','## 同底盘交换：物理单位的一步变化','','变化为相对使用自身context的预测绝对差；括号中的误差为交换后相对真实下一状态的MAE。','','|方法|加速度变化 m/s²（误差）|角速度变化 rad/s（误差）|车速变化 m/s（误差）|前轮角变化 rad（误差）|','|---|---:|---:|---:|---:|']
    for label,s in summary.items():
      a=s['same']['1'];lines.append('|'+label+'|'+'|'.join(f'{v:.6f} ({e:.6f})' for v,e in zip(a['physical_prediction_change'],a['physical_mae']))+'|')
    lines+=['','## 多步与工况匹配控制','','same_matched限制供体速度、轮角和前序动作接近，只用于诊断；该子集的own必须使用同一子集，下面增幅已按相同受体计算。','','|方法|交换方式|步长|受体数|交换误差|相对own增幅|误差增加95%区间|','|---|---|---:|---:|---:|---:|---|']
    for label,s in summary.items():
      for variant in ['same','same_matched','other','zero']:
        for h in [1,5,10]:
          v=s[variant][str(h)];lo,hi=v['normalized_error_increase']['map_bootstrap95']
          lines.append(f'|{label}|{variant}|{h}|{v["count"]}|{v["normalized_l1"]["mean"]:.5f}|{v["relative_error_increase_percent"]:+.1f}%|[{lo:+.5f}, {hi:+.5f}]|')
    lines+=['','## 解释边界','','- 相同底盘下误差增幅小，并且异底盘/零context明显变差，才支持context具有可互换的车辆响应信息；所有交换都无影响可能意味着忽略context。',
      '- 不能由03→09的差异单独证明一致性损失有效，因为Deep条件注入也改变了。08纯交换实验只有约47K，不能与500K作完整预算对照。',
      '- 区间以10张地图为聚类单位配对bootstrap10000次，包含固定抽样供体的平均效果，不覆盖训练种子或重新采样供体的不确定性。没有预先指定等效界值，因此不宣称严格统计等效。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(out/'REPORT.md',flush=True)


if __name__=='__main__':main()
