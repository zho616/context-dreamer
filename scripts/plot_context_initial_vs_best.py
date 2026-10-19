"""Compare matched random-initialized and best encoders on saved physical windows."""
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
import numpy as np
import torch

from analyze_vehicle_context_associations import pca


def main():
    # Figure contract: show whether training changes vehicle-associated structure,
    # without treating PCA separation as proof of dynamics identification.
    # Quantitative 2x2 grid; 8D in every panel; PNG/PDF/editable SVG plus source data.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional separate export directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    old = root / 'logdir/context_chassis_500k_20261015_103141'
    new = root / 'logdir/context_kinematics_500k_20261015_180340'
    source = new / 'analysis/vehicle_context_pca'
    out = args.output.resolve() if args.output else source / 'initial_vs_best'
    out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    seed = 20261015
    data = np.load(source / 'representations.npz')
    history = data['history'].astype(np.float32)
    rows = list(csv.DictReader((source / 'windows.csv').open()))
    groups = np.array([int(row['repeat']) for row in rows])
    folds = np.array_split(np.random.default_rng(20261016).permutation(np.unique(groups)), 6)
    fit = ~np.isin(groups, folds[0])
    fixed = np.array([round(float(r['initial_speed_m_s'])) == 10 and
                      round(float(r['front_left_angle_degrees'])) == 15 for r in rows])
    labels = np.array([r['model'] for r in rows])
    mask = torch.ones(len(history), 5, dtype=torch.bool)
    spaces, provenance, initial_weights = {}, {}, {}
    for name, run, h in [('old', old, history[..., [0, 1, 4, 5]]), ('new', new, history)]:
        spec = importlib.util.spec_from_file_location(f'pca_{name}_context', run / 'source/context.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.manual_seed(seed)
        encoder = module.HistoryEncoder().eval()
        initial_weights[name] = {k: v.clone() for k, v in encoder.state_dict().items()}
        with torch.inference_mode():
            spaces[f'{name}_random'] = encoder(torch.from_numpy(h), mask).numpy()
        checkpoint = run / 'r2dreamer_context/best.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        prefix = 'context_encoder.'
        encoder.load_state_dict({k[len(prefix):]: v for k, v in payload['agent_state_dict'].items()
                                 if k.startswith(prefix)}, strict=True)
        with torch.inference_mode():
            spaces[f'{name}_best'] = encoder(torch.from_numpy(h), mask).numpy()
        provenance[name] = {'checkpoint': str(checkpoint), 'step': int(payload['step']),
                            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        del payload
    torch.save(initial_weights, out / 'random_encoder_weights.pt')
    np.testing.assert_allclose(spaces['new_random'], data['random_context'], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(spaces['new_best'], data['learned_context'], atol=1e-6, rtol=1e-6)
    saved_old = np.load(source / 'compared_representations.npz')['old_context']
    np.testing.assert_allclose(spaces['old_best'], saved_old, atol=1e-6, rtol=1e-6)
    assert all(v.shape == (1680, 8) and np.isfinite(v).all() for v in spaces.values())
    assert fixed.sum() == 120 and all(sum(fixed & (labels == m)) == 24
                                         for m in ['default', 'm', 'l', 's', 'xl'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    models = ['default', 'm', 'l', 's', 'xl']
    coords, ratios = {}, {}
    for condition, selected in [('fixed', fixed), ('mixed', np.ones(len(history), bool))]:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8.4))
        fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.85, wspace=.27, hspace=.40)
        for row, version in enumerate(['old', 'new']):
            for col, stage in enumerate(['random', 'best']):
                ax = axs[row, col]
                name = f'{version}_{stage}'
                z, ratio = pca(spaces[name], fit & selected)
                key = f'{condition}_{name}'
                coords[key], ratios[key] = z, ratio
                for model, color in zip(models, colors):
                    q = selected & (labels == model)
                    ax.scatter(*z[q].T, s=24 if condition == 'fixed' else 12,
                               color=color, alpha=.65 if condition == 'fixed' else .40,
                               edgecolors='none', label=model.upper())
                heading = 'Old state2' if version == 'old' else 'New state4'
                detail = f'Random init (seed {seed})' if stage == 'random' else f'Best ({provenance[version]["step"]:,} steps)'
                ax.set(title=f'{heading} | {detail}', xlabel=f'PC1 ({ratio[0]:.1%})',
                       ylabel=f'PC2 ({ratio[1]:.1%})')
                ax.text(-.13, 1.06, chr(97 + row * 2 + col), transform=ax.transAxes,
                        fontweight='bold', fontsize=13)
        heading = ('Starting speed 10 m/s; measured wheel angle 15°; 24 profiles/type'
                   if condition == 'fixed' else 'Two starting speeds × seven wheel angles × 24 profiles/type')
        fig.suptitle('Random-initialized vs. trained context encoders (all outputs 8D)\n' + heading,
                     fontsize=12, y=.97)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(.5, .055), ncol=5, frameon=False, markerscale=1.3)
        fig.text(.5, .025, 'Same physical windows; PCA fitted independently per panel. Random init is not the original training step 0.',
                 ha='center', fontsize=8)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{condition}_initial_vs_best.{ext}', dpi=250)
        plt.close(fig)
    np.savez_compressed(out / 'representations_and_pca.npz', **spaces, **coords)
    with (out / 'pca_source_data.csv').open('w') as f:
        fields = list(rows[0]) + ['pca_fit_group', 'fixed_condition'] + [f'{k}_pc{i+1}' for k in coords for i in range(2)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({**r, 'pca_fit_group': bool(fit[i]), 'fixed_condition': bool(fixed[i]),
                             **{f'{k}_pc{j+1}': float(z[i, j]) for k, z in coords.items() for j in range(2)}})
    metadata = {'random_seed': seed, 'checkpoints': provenance, 'explained_variance': ratios,
                'samples': len(history), 'fixed_samples': int(fixed.sum()),
                'protocol': 'Original saved windows and original grouped PCA fit split; standardize dimensions on fit groups only. Each panel has independent PCA axes.',
                'limitation': 'One random initialization per architecture, not the actual step-0 checkpoint. PCA alone does not establish training benefit, parameter identification, or driving improvement.',
                'checks': 'All 8D outputs finite; both best outputs and new random output reproduce original saved representations within 1e-6.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'QA.md').write_text('# 图形与数据检查\n\n'
        '- 四个面板均为8维context，同批物理轨迹，旧版取state2/action2，新版取state4/action2。\n'
        '- 每个架构各自随机初始化；seed20261015，不声称是原训练第0步。\n'
        '- 原best和新版随机输出与已有数据复算一致；1680个窗口，固定工况120个窗口，每车型24个。\n'
        '- 与原图相同的按profile分组拟合子集；每栏独立标准化和PCA，跨栏坐标方向不对应。\n'
        '- 未添加抖动、未选取最优随机seed；单次随机初始化不表示初始化稳定性。\n'
        '- 提供PNG、可编辑SVG、PDF、原始8维输出、PCA坐标CSV、随机权重及checkpoint哈希。\n')
    print(out)


if __name__ == '__main__':
    main()
