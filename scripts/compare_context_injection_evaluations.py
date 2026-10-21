"""Audit completed paired 100-episode evaluations for context injection runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    old = json.loads((root / 'logdir/context_kinematics_500k_20261015_180340/comparison_20261016_011017/evaluation_comparison.json').read_text())
    runs = {
        '09_deep_swap1': ('context_swap_deep1_500k_20261020', 'r2dreamer_dali_deep'),
        '10_joint_swap1': ('context_swap_dali_joint_500k_20261020_run1', 'r2dreamer_dali_deep_joint'),
        '11_shallow_swap1': ('context_swap_dali_shallow_500k_20261021', 'r2dreamer_dali_shallow'),
    }
    result = {'summaries': {}, 'pairwise': {}, 'provenance': {},
        'methodology': {'episodes_per_split': 100, 'splits': ['new_maps', 's', 'xl'],
            'checkpoint': 'Each method best chosen by 20-episode training validation; independent test never selects checkpoint.',
            'pairing': 'Exact seed, model, physical mass and max steering equality across all methods.',
            'intervals': 'Paired bootstrap, 10000 resamples of 100 scenario seeds per split; descriptive, no multiple-comparison adjustment.',
            'scope': 'One training seed; map intervals do not estimate training-seed uncertainty. S/XL are unseen types with in-training mass/steering scale grid.'}}
    checkpoint_digests = {}
    for split in ('new_maps', 's', 'xl'):
        directories = {'baseline': Path(old[split]['data_paths']['baseline'])}
        for label, (folder, method) in runs.items():
            state = json.loads((root / 'logdir' / folder / 'training/status.json').read_text())
            assert state['phase'] == 'completed'
            assert state['methods'][method]['evaluations'][split]['state'] == 'completed'
            directories[label] = root / 'logdir' / folder / 'training' / method / 'evaluation' / split
        rows, result['summaries'][split], result['provenance'][split] = {}, {}, {}
        for label, folder in directories.items():
            metadata = json.loads((folder / 'provenance.json').read_text())
            summary_file = json.loads((folder / 'summary.json').read_text())
            assert summary_file['status'] == 'completed'
            data = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
            assert len(data) == 100 and len({r['seed'] for r in data}) == 100
            data.sort(key=lambda row: row['seed'])
            assert sum(r['success'] for r in data) == summary_file['summary']['success_count']
            ckpt = metadata['checkpoint']
            if ckpt not in checkpoint_digests:
                checkpoint_digests[ckpt] = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
            assert checkpoint_digests[ckpt] == metadata['checkpoint_sha256']
            result['summaries'][split][label] = summary_file['summary']
            result['provenance'][split][label] = metadata
            rows[label] = data
        identity = lambda r: (r['seed'], r['chassis']['model'], r['chassis']['mass'], r['chassis']['max_steering'])
        base_ids = [identity(r) for r in rows['baseline']]
        assert all([identity(r) for r in data] == base_ids for data in rows.values())
        result['pairwise'][split] = {}
        for new, ref in [('10_joint_swap1', 'baseline'), ('11_shallow_swap1', 'baseline'),
                         ('11_shallow_swap1', '10_joint_swap1'), ('10_joint_swap1', '09_deep_swap1'),
                         ('11_shallow_swap1', '09_deep_swap1')]:
            rng = np.random.default_rng(20261021)
            indices = rng.integers(0, 100, (10000, 100))
            comparisons = {}
            for metric in ('success', 'crash', 'out_of_road', 'timeout', 'reward', 'route_completion', 'mean_speed_km_h'):
                delta = np.array([float(a[metric]) - float(b[metric]) for a, b in zip(rows[new], rows[ref])])
                comparisons[metric] = {'difference': float(delta.mean()),
                    'paired_bootstrap95': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
            result['pairwise'][split][new + '_minus_' + ref] = comparisons
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    labels = {'baseline': '无context基线', '09_deep_swap1': '09 Deep适配+swap1',
        '10_joint_swap1': '10 Joint Deep+swap1', '11_shallow_swap1': '11 Shallow+swap1'}
    lines = ['# Joint Deep与Shallow独立测试汇总', '',
        '核验已有完整独立测试，未重复运行已完成的回合。每组每类100回合，共同地图种子、实际车型、质量和最大转角逐局配对一致；检查点SHA256与评估记录一致。',
        '', '## 成功率', '', '| 方法 | best步数 | 新地图 | S | XL |', '|---|---:|---:|---:|---:|']
    for label in labels:
        step = result['provenance']['new_maps'][label]['checkpoint_step']
        values = [result['summaries'][s][label]['success'] for s in ('new_maps', 's', 'xl')]
        lines.append(f'|{labels[label]}|{step}|' + '|'.join(f'{v:.0%}' for v in values) + '|')
    lines += ['', '## 各项指标', '', '| 条件 | 方法 | 成功率 | 碰撞率 | 出界率 | 超时率 | 奖励 | 路线完成 | 速度km/h |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, summaries in result['summaries'].items():
        for label, v in summaries.items():
            lines.append(f'|{split}|{labels[label]}|{v["success"]:.0%}|{v["crash"]:.0%}|{v["out_of_road"]:.0%}|{v["timeout"]:.0%}|{v["reward"]:.2f}|{v["route_completion"]:.1%}|{v["mean_speed_km_h"]:.2f}|')
    lines += ['', '碰撞/出界等为环境报告标志，个别终止原因可重叠，不要求比例严格相加为100%。', '',
              '## 相对基线的配对成功率差', '', '| 方法 | 新地图 | S | XL |', '|---|---|---|---|']
    for label in ('10_joint_swap1', '11_shallow_swap1'):
        cells = []
        for split in ('new_maps', 's', 'xl'):
            v = result['pairwise'][split][label + '_minus_baseline']['success']
            lo, hi = v['paired_bootstrap95']
            cells.append(f'{v["difference"]*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}]')
        lines.append(f'|{labels[label]}|' + '|'.join(cells) + '|')
    lines += ['', '## 解读', '',
        'Shallow在新地图/S的成功率点估计较高（83%/91%），但XL只有31%，出界率49%，未表现为全面泛化提升。',
        'Joint Deep较慢，新地图/S/XL平均速度8.26/6.74/6.34km/h；S有21%超时。不能仅凭成功率或低碰撞概括安全优势。',
        '09旧Deep适配组仍有最高XL成功率54%；Joint并未优于该适配组。不同Deep入口同时改变输入投影/参数化，不能只归因于拼接顺序。',
        '各方法单训练seed，按20回合验证选择best；配对区间仅反映固定训练模型在这100个测试场景上的不确定性。所有新组交换损失权重均1，不能作为swap0/1损失消融。',
        '', '完整来源、全部配对指标与区间见comparison.json。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'success': {s: {k: v['success'] for k, v in d.items()} for s, d in result['summaries'].items()},
        'paired_success': {s: {k: v['success'] for k, v in d.items()} for s, d in result['pairwise'].items()}}, indent=2))


if __name__ == '__main__': main()
