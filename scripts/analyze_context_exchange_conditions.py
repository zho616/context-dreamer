"""Stratify same-chassis context exchange by donor/recipient operating condition."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    root=Path(__file__).resolve().parents[1];out=root/'logdir/context_exchange_prediction_20261021'
    previous=json.loads((out/'results.json').read_text());a=dict(np.load(out/'arrays.npz'))
    histories=a['history'];donors=a['donors_same'];initial=histories[:,-1,:4];n,J=donors.shape;H=5;device='cuda:0'
    speed=np.abs(histories[donors,-1,2]-histories[:,-1,2,None]);angle=np.rad2deg(np.abs(histories[donors,-1,3]-histories[:,-1,3,None]))
    groups={'all':np.ones((n,J),bool),'close_speed_angle':(speed<1)&(angle<3),
      'speed_lt1':speed<1,'speed_1to3':(speed>=1)&(speed<3),'speed_ge3':speed>=3,
      'angle_lt3':angle<3,'angle_3to10':(angle>=3)&(angle<10),'angle_ge10':angle>=10,
      'both_far':(speed>=3)&(angle>=10),'speed_ge5':speed>=5,'angle_ge30':angle>=30}
    tensor=lambda x:torch.as_tensor(x,device=device)
    scale=np.array([10.,1.,30.,np.pi/3],np.float32);maps=a['maps'];unique=np.unique(maps)
    boot=np.random.default_rng(20261021).integers(0,len(unique),(10000,len(unique)))
    def interval(value,mask):
      sums=np.array([value[(maps==m)&mask].sum(dtype=np.float64) for m in unique]);counts=np.array([((maps==m)&mask).sum() for m in unique]);den=counts[boot].sum(1)
      sample=sums[boot].sum(1)/np.maximum(den,1)
      return np.quantile(sample[den>0],[.025,.975]).tolist()
    results={};save_arrays={}
    with torch.inference_mode():
      for label in ['03_no_swap_best','09_deep_swap1_best','09_deep_swap1_final','11_shallow_swap1_best']:
        provenance=previous['provenance'][label];source=Path(provenance['context_source'])
        spec=importlib.util.spec_from_file_location('conditions_'+label,source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        encoder=module.HistoryEncoder(4).to(device).eval();predictor=module.ResidualDynamics(4).to(device).eval()
        payload=torch.load(provenance['checkpoint'],map_location=device,weights_only=False);weights=payload['agent_state_dict']
        for model,prefix in [(encoder,'context_encoder.'),(predictor,'context_dynamics.')]:model.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
        del payload,weights
        c=encoder(tensor(histories),torch.ones(n,5,dtype=torch.bool,device=device));predictions=[];own_predictions=[]
        for start in range(0,n,128):
          stop=min(start+128,n);count=stop-start;x=tensor(initial[start:stop])[:,None].expand(-1,J+1,-1).reshape(-1,4)
          cc=torch.cat((c[start:stop,None],c[tensor(donors[start:stop])]),1).reshape(-1,8)
          aa=tensor(a['actions'][start:stop,:H])[:,None].expand(-1,J+1,-1,-1).reshape(-1,H,2);pred=[]
          for k in range(H):
            x=module.predict_next_state(predictor,cc,x,module.canonical_action(aa[:,k]));pred.append(x.reshape(count,J+1,4).cpu().numpy())
          pred=np.stack(pred,2);own_predictions.append(pred[:,0]);predictions.append(pred[:,1:])
        own=np.concatenate(own_predictions);pred=np.concatenate(predictions)
        error=np.abs(pred-a['truth'][:,None,:H]);own_error=np.abs(own-a['truth'][:,:H]);change=np.abs(pred-own[:,None])
        np.testing.assert_allclose(own_error,a[label+'__own__error'][:,:H],atol=1e-5,rtol=1e-5)
        save_arrays[label+'__pair_error']=error;save_arrays[label+'__pair_change']=change
        results[label]={}
        for group,selection in groups.items():
          results[label][group]={}
          counts=selection.sum(1);has=counts>0
          for h in [1,5]:
            k=h-1;mask=has&a['valid'][:,k]
            # Average donors within each anchor, then anchors, so groups do not
            # overweight recipient states with more qualifying donor contexts.
            weighted=lambda values:(values*selection[...,None]).sum(1)/np.maximum(counts[:,None],1)
            err=weighted(error[:,:,k]);chg=weighted(change[:,:,k]);base=own_error[:,k]
            normalized=(err/scale).sum(-1);own_norm=(base/scale).sum(-1);delta=normalized-own_norm
            results[label][group][str(h)]={'anchors':int(mask.sum()),'pairs':int(selection[mask].sum()),
              'own_normalized_l1':float(own_norm[mask].mean()),'swap_normalized_l1':float(normalized[mask].mean()),
              'relative_increase_percent':float(100*(normalized[mask].mean()/own_norm[mask].mean()-1)),
              'error_increase95':interval(delta,mask),'prediction_change_l1':float((chg[mask]/scale).sum(-1).mean()),
              'own_physical_mae':base[mask].mean(0).tolist(),'swap_physical_mae':err[mask].mean(0).tolist(),
              'physical_prediction_change':chg[mask].mean(0).tolist(),
              'mean_donor_speed_difference_mps':float((speed[mask]*selection[mask]).sum()/selection[mask].sum()),
              'mean_donor_angle_difference_deg':float((angle[mask]*selection[mask]).sum()/selection[mask].sum())}
        print('completed',label,flush=True)
    np.savez_compressed(out/'condition_arrays.npz',**save_arrays,speed_difference_mps=speed,angle_difference_deg=angle)
    (out/'condition_results.json').write_text(json.dumps({'summary':results,'provenance':previous['provenance'],
      'protocol':'Same exact donor pairs and fixed state/action as main exchange audit; thresholds descriptive, not tuned. Donor contexts from independent maps/episodes; recipient context fixed throughout 5-step auxiliary forecast. Equal anchor weighting within condition group. Prefix natural, no imposed speed/steering.',
      'units':['m/s^2','rad/s','m/s','rad'],'interval':'10-map paired cluster bootstrap, conditional on fixed donors. No training-seed uncertainty.'},indent=2)+'\n')
    lines=['# 同底盘context跨速度、转角预测测试','','保持受体当前状态和动作不变，仅交换其他回合、其他地图的同实际车型/质量/最大转角context；每个起点20个随机供体。按供体末端速度和实际前轮角与受体的差分组。同组内每个受体等权，各组受体不同，不直接把组间绝对误差变化归因于工况差。预测对象是辅助车辆响应模型，非RSSM主模型。','','## 一步误差（归一化L1，越小越好）','','|方法|供体工况|受体数|配对数|自身context误差|交换后误差|增幅|误差增加95%区间|','|---|---|---:|---:|---:|---:|---:|---|']
    for label,s in results.items():
      for group in ['close_speed_angle','speed_ge3','angle_ge10','both_far','speed_ge5','angle_ge30']:
        v=s[group]['1'];lo,hi=v['error_increase95']
        lines.append(f'|{label}|{group}|{v["anchors"]}|{v["pairs"]}|{v["own_normalized_l1"]:.5f}|{v["swap_normalized_l1"]:.5f}|{v["relative_increase_percent"]:+.1f}%|[{lo:+.5f}, {hi:+.5f}]|')
    lines+=['','## 速度与转角同时差较大的物理误差','','both_far：速度差≥3m/s（10.8km/h），实际前轮角差≥10度。','','|方法|步长|加速度MAE|角速度MAE|车速MAE|前轮角MAE|','|---|---:|---:|---:|---:|---:|']
    for label,s in results.items():
      for h in [1,5]:lines.append(f'|{label}|{h}|'+'|'.join(f'{v:.6f}' for v in s['both_far'][str(h)]['swap_physical_mae'])+'|')
    lines+=['','## 限制','','供体速度和转角取各自历史窗口末端；交换时受体当前速度/前轮角仍按真实值输入预测器。该实验检验context能否跨工况使用，并不要求响应与当前速度/转角无关。只覆盖5种名义底盘、10张地图、一个训练seed。没有预设准确性/等效阈值，不能称为严格统计等效，也不能以03/09差异证明损失单独有效。']
    (out/'CONDITION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out/'CONDITION_REPORT.md')


if __name__=='__main__':main()
