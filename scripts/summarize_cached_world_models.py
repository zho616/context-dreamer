"""Summarize the fixed-corpus comparison and separately audit training losses."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logdir/world_model_injection_comparison_20261021'
CORPUS = ROOT/'logdir/world_model_context_validation_20261019_cpu'
RUNS = {
    'baseline': 'chassis_baselines_500k_20261015_020330/r2dreamer',
    '03_recurrent': 'context_kinematics_500k_20261015_180340/r2dreamer_context',
    '09_deep': 'context_swap_deep1_500k_20261020/training/r2dreamer_dali_deep',
    '10_joint': 'context_swap_dali_joint_500k_20261020_run1/training/r2dreamer_dali_deep_joint',
    '11_shallow': 'context_swap_dali_shallow_500k_20261021/training/r2dreamer_dali_shallow',
}


def main():
    origins = json.loads((CORPUS/'origins.json').read_text())
    maps = np.array([o['map_id'] for o in origins]); unique = np.unique(maps)
    arrays = {name: dict(np.load(OUT/name/'errors.npz')) for name in RUNS}
    summaries = {name: json.loads((OUT/name/'summary.json').read_text()) for name in RUNS}
    provenance = {name: json.loads((OUT/name/'provenance.json').read_text()) for name in RUNS}
    assert len({p['origins_sha256'] for p in provenance.values()}) == 1
    valid = arrays['baseline']['valid']
    for a in arrays.values(): np.testing.assert_array_equal(a['valid'], valid)
    rng = np.random.default_rng(20261021)
    draws = rng.integers(0, len(unique), (10000, len(unique)))
    pairs = {}
    for new, ref in [('10_joint','baseline'), ('11_shallow','baseline'), ('10_joint','03_recurrent'), ('11_shallow','03_recurrent'), ('10_joint','09_deep'), ('11_shallow','09_deep')]:
        pairs[new+'_minus_'+ref] = metrics = {}
        for metric in ['reward_mae', 'reward_nll', 'continue_brier', 'continue_nll']:
            delta = arrays[new][metric].mean(0)-arrays[ref][metric].mean(0)
            metrics[metric] = {}
            for k in [0,4,9]:
                sums = np.array([delta[(maps==mi)&valid[:,k],k].sum(dtype=np.float64) for mi in unique])
                counts = np.array([((maps==mi)&valid[:,k]).sum() for mi in unique])
                samples = sums[draws].sum(1)/counts[draws].sum(1)
                metrics[metric][str(k+1)] = {'difference': float(delta[valid[:,k],k].mean()), 'map_cluster_bootstrap95': np.quantile(samples,[.025,.975]).tolist()}
    training = {}
    for name, rel in RUNS.items():
        rows = [json.loads(x) for x in (ROOT/'logdir'/rel/'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if 'train/loss/dyn' in r][-20:]
        training[name] = {'first_step': rows[0]['step'], 'last_step': rows[-1]['step'], 'n': len(rows),
            'mean': {k: float(np.mean([r[k] for r in rows if k in r])) for k in rows[-1] if k.startswith('train/loss/') or k.startswith('train/context/one_step_mae')}}
    metadata = json.loads((CORPUS/'episodes.json').read_text())
    terminals = [np.load(CORPUS/'episodes'/e['file'])['is_terminal'] for e in metadata]
    result = {'summary': summaries, 'provenance': provenance, 'training_last20': training, 'paired': pairs,
        'counts': {str(k+1): {'valid': int(valid[:,k].sum()), 'terminal': int(sum(1 for i,o in enumerate(origins) if valid[i,k] and terminals[o['episode']][o['t']+k+1]))} for k in [0,4,9]}}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# DALI注入实验：共同轨迹世界模型评估', '',
        '冻结每组驾驶验证选出的best检查点，严格加载各自训练源码。100条共同真实轨迹、10张独立地图（30000–30009）、5种名义底盘、2153个预测起点，每个起点3次随机隐变量采样。真实轨迹来自两个既有冻结参考策略，各模型使用完全相同的观测前缀和未来记录动作。预测1/5/10步；仅初始前缀使用真实观测，未来context由自身响应预测器滚动生成；不使用未来真实状态。Shallow在先验滚动中没有直接context注入。', '',
        '这是动作条件下的世界模型预测评估，不是策略驾驶评估。持续概率Brier为均方概率误差；所有下表预测误差越低越好。奖励NLL使用模型训练所用的离散symlog分布，因此不能与MAE直接比较数值。', '',
        '## 共同轨迹预测', '', '|方法|best step|奖励MAE 1步|5步|10步|持续Brier 1步|5步|10步|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        a=s['all']; values=[a[m][str(k)] for m in ['reward_mae','continue_brier'] for k in [1,5,10]]
        lines.append(f'|{name}|{provenance[name]["step"]}|'+'|'.join(f'{v:.5f}' for v in values)+'|')
    lines += ['', '|方法|奖励NLL 1步|5步|10步|持续NLL 1步|5步|10步|', '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        a=s['all']; values=[a[m][str(k)] for m in ['reward_nll','continue_nll'] for k in [1,5,10]]
        lines.append(f'|{name}|'+'|'.join(f'{v:.5f}' for v in values)+'|')
    lines += ['', '## 相对差值与地图聚类区间', '', '10张地图配对聚类bootstrap 10000次；区间只反映该小规模地图样本，不覆盖训练种子不确定性。正数表示新方法误差较高。', '', '|对比|指标|1步差值 [95%区间]|5步|10步|', '|---|---|---|---|---|']
    for pair,metrics in pairs.items():
        for key in ['reward_mae','continue_brier']:
            cells=[]
            for k in [1,5,10]:
                v=metrics[key][str(k)]; lo,hi=v['map_cluster_bootstrap95'];cells.append(f'{v["difference"]:+.5f} [{lo:+.5f}, {hi:+.5f}]')
            lines.append('|'+pair+'|'+key+'|'+'|'.join(cells)+'|')
    lines += ['', '## 末段训练损失（独立于上面的best检查点评估）', '',
        '每组最后20条训练日志，约400K–495.5K步的均值，均为未乘权重原值。各组自身策略采样分布不同。dyn与rep数值相同但梯度路径不同；KL有free-nats下限，不能按越低代表能力越强来排序。Barlow为表征一致性目标，不是观测重建误差。', '',
        '|方法|dyn/rep KL|Barlow|reward NLL|continue NLL|context aux|swap|', '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in training.items():
        m=v['mean']; keys=['dyn','barlow','rew','con','context_aux','context_swap']
        lines.append('|'+name+'|'+'|'.join(f'{m["train/loss/"+k]:.6f}' if 'train/loss/'+k in m else '—' for k in keys)+'|')
    lines += ['', '## 独立的车辆响应辅助预测器（不是RSSM状态输出）', '',
        '固定起点context，沿相同记录动作滚动辅助残差模型；排除目标时刻碰撞或回合结束。单位：纵向加速度m/s²、横摆角速度rad/s、车速m/s、前轮角rad。', '',
        '|方法|步长|加速度MAE|角速度MAE|车速MAE|前轮角MAE|', '|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        if 'auxiliary_mae' not in s['all']: continue
        for k,values in s['all']['auxiliary_mae'].items(): lines.append(f'|{name}|{k}|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    lines += ['', '## 解释与边界', '',
        '- 10 Joint在共同轨迹上的奖励和持续概率预测均明显退化，其训练reward/continue损失更低不能证明泛化更好。此前驾驶测试车速偏低，存在自身训练分布与共同轨迹分布不同的可能；本评估不能单独确定因果。',
        '- 11 Shallow的奖励MAE没有普遍改善；持续概率NLL较无context基线下降，但并未优于03/09。整体不能宣称世界模型全面提升。',
        '- 响应辅助预测器与RSSM奖励/持续预测必须分开看。11的速度与前轮角预测较09好，但相对03并非全指标提升；10在这个共同自然轨迹样本上更差。',
        '- 本次使用各组best（不同训练步数），不是全部500K final；这里只比较实际部署检查点。一个训练种子，10张地图，底盘名义参数；不能外推到全部随机车辆参数或断言某个损失项的因果收益。没有独立观测重建头，因此未报告像素/状态重建MSE。',
        '- 所有终止目标定义为1-is_terminal（time-limit截断仍可持续）；真实回合之后的目标不计分。预测不中途按预测持续概率截断。各步终止样本量见comparison.json，类别不均衡会影响Brier解释。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(OUT/'REPORT.md')
    for pair in ['10_joint_minus_baseline','11_shallow_minus_baseline']:
        print(pair, pairs[pair])


if __name__ == '__main__': main()
