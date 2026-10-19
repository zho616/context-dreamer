"""Replot the state2 best encoder with audited cached physical probe windows."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


def fit_pca(x, fit):
    mean = x[fit].mean(0)
    sd = x[fit].std(0)
    sd = np.where(sd < 1e-6, 1., sd)
    standardized = (x - mean) / sd
    _, singular, axes = np.linalg.svd(standardized[fit], full_matrices=False)
    return standardized @ axes[:2].T, singular[:2] ** 2 / (singular ** 2).sum(), mean, sd, axes[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'logdir/context_kinematics_500k_20261015_180340/analysis/vehicle_context_pca'
    run = root / 'logdir/context_chassis_500k_20261015_103141'
    out = args.output.resolve(); out.mkdir(exist_ok=True, parents=True)
    checkpoint = run / 'r2dreamer_context/best.pt'
    previous = json.loads((data_dir / 'association_results.json').read_text())
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert fingerprint == previous['old_checkpoint_sha256']
    rows = list(csv.DictReader((data_dir / 'pca_source_data.csv').open()))
    history = np.load(data_dir / 'representations.npz')['history'][..., [0, 1, 4, 5]]
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('frozen_state2_encoder', run / 'source/context.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    encoder = module.HistoryEncoder().eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['step'] == 150006
    prefix = 'context_encoder.'
    encoder.load_state_dict({k[len(prefix):]: v for k, v in payload['agent_state_dict'].items() if k.startswith(prefix)}, strict=True)
    del payload
    with torch.inference_mode():
        context = encoder(torch.from_numpy(history.copy()), torch.ones(len(history), 5, dtype=torch.bool)).numpy()
    cached = np.load(data_dir / 'compared_representations.npz')['old_context']
    np.testing.assert_allclose(context, cached, atol=1e-6, rtol=1e-6)
    models = ['default', 'm', 'l', 's', 'xl']
    labels = np.array([r['model'] for r in rows]); y = np.array([models.index(v) for v in labels])
    train = np.array([r['plot_split'] == 'train' for r in rows])
    speed0 = np.rint([float(r['initial_speed_m_s']) for r in rows]).astype(int)
    angle = np.rint([float(r['front_left_angle_degrees']) for r in rows]).astype(int)
    speed = np.array([float(r['final_speed_m_s']) for r in rows])
    fixed = (speed0 == 10) & (angle == 15)
    z, ratio, mean, scale, axes = fit_pca(context.astype(float), train)
    z_fixed, r_fixed, m_fixed, s_fixed, a_fixed = fit_pca(context.astype(float), train & fixed)
    np.testing.assert_allclose(z, np.load(data_dir / 'compared_representations.npz')['old_context_PCA2'], atol=1e-5)
    assert context.shape == (1680, 8) and fixed.sum() == 120
    assert all((fixed & (labels == model)).sum() == 24 for model in models)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    point_colors = np.array(colors)[y]
    def scatter_types(ax, coords, select, size):
        order = np.random.default_rng(13).permutation(np.flatnonzero(select))
        ax.scatter(*coords[order].T, c=point_colors[order], s=size, alpha=.55, edgecolors='none', rasterized=True)
        ax.grid(alpha=.12); ax.set_axisbelow(True)
    def axis_labels(ax, variance):
        ax.set(xlabel=f'PC1 ({variance[0]:.2%} variance)', ylabel=f'PC2 ({variance[1]:.2%} variance)')
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), layout='constrained')
    scatter_types(axs[0], z, np.ones(len(y), bool), 14)
    scatter_types(axs[1], z_fixed, fixed, 35)
    axs[0].set_title('Mixed operating conditions\n1,680 windows; 336 per vehicle type', fontsize=10)
    axs[1].set_title('Initial speed 10 m/s; front-wheel angle 15°\n120 windows; 24 per vehicle type', fontsize=10)
    for ax, variance, letter in zip(axs, [ratio, r_fixed], 'ab'):
        axis_labels(ax, variance); ax.text(-.1, 1.08, letter, transform=ax.transAxes, weight='bold', fontsize=12)
    handles = [Line2D([], [], marker='o', linestyle='none', color=col, label=model.upper(), markersize=6) for model, col in zip(models, colors)]
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle('State2 Context-R2Dreamer: best checkpoint at 150,006 steps\nPCA fitted separately in each panel; points colored only by vehicle type', fontsize=11)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'old_context_vehicle_pca.{ext}', dpi=300)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.1), layout='constrained')
    scatter_types(axs[0], z, np.ones(len(y), bool), 12)
    axs[0].set_title('Vehicle type')
    order = np.random.default_rng(13).permutation(len(y))
    for ax, values, title, cmap, label in [(axs[1], speed, 'Measured endpoint speed', 'viridis', 'Speed (m/s)'),
                                         (axs[2], angle, 'Actual front-wheel angle', 'RdBu_r', 'Angle (deg; left positive)')]:
        sc = ax.scatter(*z[order].T, c=values[order], s=12, alpha=.55, cmap=cmap, edgecolors='none', rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=.75, label=label)
        ax.set_title(title); ax.grid(alpha=.12); ax.set_axisbelow(True)
    for ax, letter in zip(axs, 'abc'):
        axis_labels(ax, ratio); ax.text(-.1, 1.06, letter, transform=ax.transAxes, weight='bold', fontsize=12)
        ax.set_xlim(axs[0].get_xlim()); ax.set_ylim(axs[0].get_ylim())
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle('One PCA projection, three alternative color labels\nState2 context; identical 1,680 points and axes in all panels', fontsize=11)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'old_context_pca_factors.{ext}', dpi=300)
    plt.close(fig)
    with (out / 'pca_source_data.csv').open('w') as f:
        fields = ['rollout', 'vehicle', 'profile', 'initial_speed_m_s', 'endpoint_speed_m_s', 'actual_left_wheel_angle_deg', 'pca_fit',
                  'pc1', 'pc2', 'fixed_condition', 'fixed_pc1', 'fixed_pc2'] + [f'context_{j}' for j in range(8)]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(dict(rollout=row['rollout'], vehicle=labels[i], profile=row['repeat'], initial_speed_m_s=speed0[i],
                endpoint_speed_m_s=speed[i], actual_left_wheel_angle_deg=angle[i], pca_fit=bool(train[i]), pc1=z[i, 0], pc2=z[i, 1],
                fixed_condition=bool(fixed[i]), fixed_pc1=z_fixed[i, 0] if fixed[i] else '', fixed_pc2=z_fixed[i, 1] if fixed[i] else '',
                **{f'context_{j}': context[i, j] for j in range(8)}))
    np.savez_compressed(out / 'pca_arrays.npz', context=context, mixed_coordinates=z, fixed_coordinates=z_fixed[fixed],
                        mixed_mean=mean, mixed_scale=scale, mixed_axes=axes, fixed_mean=m_fixed, fixed_scale=s_fixed, fixed_axes=a_fixed)
    metadata = {'checkpoint_step': 150006, 'checkpoint_sha256': fingerprint, 'source_windows': str(data_dir),
        'source_history_sha256': hashlib.sha256((data_dir / 'representations.npz').read_bytes()).hexdigest(),
        'samples': 1680, 'fixed_samples': 120, 'mixed_variance_ratios': ratio.tolist(), 'fixed_variance_ratios': r_fixed.tolist(),
        'fit_profile_count': len(set(r['repeat'] for r in rows if r['plot_split'] == 'train')),
        'probes_from_existing_audit': previous['vehicle_probes']['old_context'], 'factor_associations': previous['factors']['old_context'],
        'method': 'Standardized context8; fit first-fold training profiles only; all profiles displayed; separate PCA refit in predefined fixed condition.',
        'limitations': ['Controlled probes, not histories from the 600 driving evaluations.',
            'Only initial speed fixed; subsequent speeds naturally diverge.', 'Held fixed native parameters; not mass/steering-scale generalization.',
            'Actual wheel angle uses left-positive degrees for coloring; encoder uses right-positive yaw and canonical actions.',
            'Type separability is not parameter identification, causal dynamics information or RSSM benefit.',
            'PCA includes all five types in the unsupervised fit; the supervised probe labels also include S/XL.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'README.md').write_text(f'''# 旧版 state2 context PCA

冻结150006步best，严格加载旧版E；重新编码1680个已有物理探测窗口，与缓存context逐值核对通过。
主图左：混合两档初速、七档实际转角、24个配对油门profile、五车型，每车型336点。
主图右：预先指定10m/s初速、15°实际前轮角，每车型24个profile；在该条件内独立拟合PCA。
每个点为一个5步历史窗口的8维context；颜色只表示车型。固定的是初速，后续速度自然分化。
所有PCA仅用20条训练profile拟合标准化和投影，4条留出profile投影；图中均展示。
混合图PC1/PC2解释标准化context方差 {ratio[0]:.2%}/{ratio[1]:.2%}；固定图 {r_fixed[0]:.2%}/{r_fixed[1]:.2%}。
百分比不是车型信息比例；两主图坐标轴由不同PCA拟合，不能直接比较绝对位置。

辅助图是同一混合PCA分别按车型、实际末速、实际前轮角着色，三幅坐标完全一致。
这些是受控物理探测数据，并非刚完成的600回合驾驶轨迹；没有重训模型。
既有六折profile分组探针：混合工况线性37.1%、kNN5为97.7%；固定工况线性92.5%。
这些探针使用完整8维context（不是只用画出的2维），且所有五类都有监督标签，不是未见车型零样本分类。
车型信息可能非线性存在；工况关联不能视为车型信息不存在。不能由分簇证明参数辨识或世界模型收益。
原始历史也有车型线索；不同车型实现共同物理轮角使用不同归一化动作，可能暴露转向增益。
均为固定模型/单训练seed分析，无伪造误差条或多seed区间。逐点数据、PCA变换和checkpoint校验见同目录文件。
''')
    print(json.dumps({'output': str(out), 'mixed_variance': ratio.tolist(), 'fixed_variance': r_fixed.tolist()}))


if __name__ == '__main__':
    main()
