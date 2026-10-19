"""Add map-cluster intervals and simple prediction controls to frozen diagnostics."""
import json,sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]);d=np.load(out/'errors.npz');matching=np.load(out/'matching.npz')
origins=json.loads((out/'origins.json').read_text());episodes=json.loads((out/'episodes.json').read_text())
arrays=[dict(np.load(out/'episodes'/e['file'])) for e in episodes]
for a in arrays:
    if 'reward' not in a:a['reward']=sum(a['log_reward_'+k] for k in ['progress','speed','lateral','terminal']).astype(np.float32)
maps=np.array([o['map_id'] for o in origins]);typ=np.array([o['model'] for o in origins]);matched=matching['matched']
rng=np.random.default_rng(20261019);result={'auxiliary_paired':{},'simple_controls':{},'matching_distribution':{}}

def compare(delta,mask):
    blocks=np.array([delta[mask&(maps==m)].mean() for m in sorted(set(maps[mask]))])
    samples=blocks[rng.integers(0,len(blocks),(10000,len(blocks)))].mean(1)
    return {'windows':int(mask.sum()),'maps':len(blocks),'window_weighted_difference':float(delta[mask].mean()),
            'map_weighted_difference':float(blocks.mean()),'map_bootstrap95':np.quantile(samples,[.025,.975]).tolist()}

for a,b in [('zero','own'),('other','own'),('other','same'),('same','own'),('own','persistence')]:
    ea=d['aux__'+a];eb=d['aux__'+b]
    delta=((ea-eb)/np.array([10.,1.])).sum(-1)
    select=matched if ('other' in a or 'same' in a or 'same' in b) else np.ones(len(origins),bool)
    result['auxiliary_paired'][a+'_minus_'+b]={str(k+1):compare(delta[:,k],select&d['physical_valid'][:,k]) for k in [0,4,9]}
for split,select in [('all',np.ones(len(origins),bool)),('matched',matched)]:
    persistence=np.zeros_like(d['valid'],dtype=np.float32)
    terminal=np.zeros_like(d['valid'],dtype=np.float32)
    for i,o in enumerate(origins):
        source=arrays[o['episode']];t=o['t'];length=min(10,episodes[o['episode']]['length']-t)
        persistence[i,:length]=np.abs(source['reward'][t]-source['reward'][t+1:t+length+1])
        terminal[i,:length]=source['is_terminal'][t+1:t+length+1]
    result['simple_controls'][split]={'reward_persistence_mae':{str(k+1):float(persistence[select&d['valid'][:,k],k].mean()) for k in [0,4,9]},
        'always_continue_brier':{str(k+1):float(terminal[select&d['valid'][:,k],k].mean()) for k in [0,4,9]}}
    op=matching['conditions'][select]
    result['matching_distribution'][split]={'speed_m_s_quantiles':np.quantile(op[:,0],[0,.25,.5,.75,1]).tolist(),
        'absolute_wheel_angle_deg_quantiles':np.quantile(np.abs(op[:,1]),[0,.25,.5,.75,1]).tolist(),
        'abs_angle_below_3_degrees_fraction':float((np.abs(op[:,1])<3).mean())}
# Behavior-policy strata use exactly the same target windows in every model.
result['world_model_by_behavior']={}
for policy in ['baseline','context']:
    select=np.array([o['policy']==policy for o in origins]);result['world_model_by_behavior'][policy]={}
    for variant in ['baseline','dynamic','zero','fixed_own']:
        result['world_model_by_behavior'][policy][variant]={key:{str(k+1):float(d[variant+'__'+key].mean(0)[select&d['valid'][:,k],k].mean()) for k in [0,4,9]} for key in ['reward_mae','reward_nll','continue_brier']}
assert all(np.isfinite(d[key]).all() for key in d.files)
(out/'additional_checks.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['\n## 辅助分支配对差（归一化L1）','', '| 比较：前者减后者 | h1均值 [地图bootstrap95%] | h5均值 [地图bootstrap95%] | h10均值 [地图bootstrap95%] |','|---|---|---|---|']
for pair,values in result['auxiliary_paired'].items():
    cells=[]
    for k in [1,5,10]:
        x=values[str(k)];lo,hi=x['map_bootstrap95'];cells.append(f"{x['map_weighted_difference']:+.5f} [{lo:+.5f},{hi:+.5f}]")
    lines.append('| '+pair+' | '+' | '.join(cells)+' |')
lines+=['','## 简单基准与匹配覆盖','',json.dumps(result['simple_controls'],indent=2),'',json.dumps(result['matching_distribution'],indent=2),
        '', '全部浮点误差数组finite检查通过；同车型/错车型donor数量、匹配距离见results.json。简单continue=1基准提示终止预测任务的类别不平衡。',
        '物理误差为加速度绝对误差/10 + 横摆角速度绝对误差；两项的物理单位MAE另列在主表。','']
with (out/'REPORT.md').open('a') as f:f.write('\n'.join(lines))
print(json.dumps(result,indent=2))
