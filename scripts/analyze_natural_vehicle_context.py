"""Vehicle probes on unconstrained natural driving, split by entire map."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_swap_context_associations import load_context


def scale(x, train):
    sd = x[train].std(0)
    return (x - x[train].mean(0)) / np.where(sd < 1e-6, 1, sd)


def classify(x, y, train, test, episode, kind):
    z = scale(x, train)
    if kind in ('ridge', 'softmax'):
        a = np.column_stack((np.ones(len(z)), z))
        _, ids, counts = np.unique(episode[train], return_inverse=True, return_counts=True)
        weight = 1 / counts[ids]
        weight /= weight.mean()
        penalty = np.eye(a.shape[1]); penalty[0, 0] = 0
        if kind == 'softmax':
            xx, yy = a[train], np.eye(5)[y[train]]
            def objective(flat):
                w = flat.reshape(xx.shape[1], 5)
                logits = xx @ w
                logprob = logits - logsumexp(logits, axis=1, keepdims=True)
                loss = -(weight[:, None] * yy * logprob).sum() + .5 * (w[1:] ** 2).sum()
                grad = xx.T @ (weight[:, None] * (np.exp(logprob) - yy)) + penalty @ w
                return loss, grad.ravel()
            result = minimize(objective, np.zeros(a.shape[1] * 5), jac=True, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-10})
            if not result.success:
                raise RuntimeError('Softmax probe did not converge: ' + result.message)
            return (a[test] @ result.x.reshape(a.shape[1], 5)).argmax(1)
        w = np.linalg.solve(a[train].T @ (weight[:, None] * a[train]) + penalty,
                            a[train].T @ (weight[:, None] * np.eye(5)[y[train]]))
        return (a[test] @ w).argmax(1)
    distance = (z[test] ** 2).sum(1)[:, None] + (z[train] ** 2).sum(1)[None] - 2 * z[test] @ z[train].T
    near = np.argpartition(distance, 4, axis=1)[:, :5]
    return np.eye(5)[y[train][near]].sum(1).argmax(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=args.overwrite)
    cache = root / 'logdir/world_model_context_validation_20261019_cpu'
    episodes = json.loads((cache / 'episodes.json').read_text())
    provenance = json.loads((cache / 'provenance.json').read_text())
    models = ['default', 'm', 'l', 's', 'xl']
    histories, rows = [], []
    for eid, meta in enumerate(episodes):
        d = np.load(cache / 'episodes' / meta['file'])
        candidates = []
        for t in range(5, len(d['speed']), 5):
            sl = slice(t - 4, t + 1)
            if not d['ctx_mask'][t].all() or any(d[key][sl].any() for key in ('is_first', 'is_last', 'is_terminal', 'ctx_collision')):
                continue
            candidates.append(t)
        # Up to 40 non-overlapping histories per episode, selected without labels/kinematics.
        if len(candidates) > 40:
            candidates = np.array(candidates)[np.linspace(0, len(candidates) - 1, 40).astype(int)].tolist()
        for t in candidates:
            sl = slice(t - 4, t + 1)
            np.testing.assert_allclose(d['ctx_history'][t, :, :2], d['ctx_ego_state'][sl], atol=1e-6)
            np.testing.assert_allclose(d['ctx_history'][t, :, 2:], d['ctx_prev_action'][sl], atol=1e-6)
            h = np.column_stack((d['ctx_ego_state'][sl], d['speed'][sl],
                                  -np.deg2rad(d['wheel_degrees'][sl]), d['ctx_prev_action'][sl])).astype(np.float32)
            assert h.shape == (5, 6) and np.isfinite(h).all()
            histories.append(h)
            rows.append({'episode': eid, 'map_id': meta['map_id'], 'map_seed': meta['seed'],
                'model': meta['model'], 'policy': meta['policy'], 'step': t,
                'speed_mps': float(d['speed'][t]), 'wheel_degrees_left': float(d['wheel_degrees'][t])})
    history = np.array(histories)
    maps = np.array([r['map_id'] for r in rows]); ep = np.array([r['episode'] for r in rows])
    policy = np.array([r['policy'] for r in rows]); y = np.array([models.index(r['model']) for r in rows])
    torch.set_num_threads(1)
    old = root / 'logdir/context_kinematics_500k_20261015_180340'
    new = root / 'logdir/context_swap_deep1_500k_20261020/training'
    checkpoints = {'original_best': (old / 'source', old / 'r2dreamer_context/best.pt'),
        'swap_best': (new / 'source', new / 'r2dreamer_dali_deep/best.pt'),
        'swap_final': (new / 'source', new / 'r2dreamer_dali_deep/latest.pt')}
    contexts, ckpts = {}, {}
    for name, (source, checkpoint) in checkpoints.items():
        _, contexts[name], ckpts[name] = load_context(source, checkpoint, history, name)
    spaces = dict(contexts)
    spaces['raw_history'] = (history / np.array([10, 1, 30, np.pi / 3, 1, 1])).reshape(len(history), -1)
    spaces['kinematics_only'] = (history[..., 2:4] / np.array([30, np.pi / 3])).reshape(len(history), -1)
    spaces['kinematics_actions'] = (history[..., 2:] / np.array([30, np.pi / 3, 1, 1])).reshape(len(history), -1)
    results, preds = {}, {}
    for collector in ('baseline', 'context'):
        selected = policy == collector
        results[collector], preds[collector] = {}, {}
        assert len(np.unique(ep[selected])) == 50
        for name, x in spaces.items():
            results[collector][name], preds[collector][name] = {}, {}
            for kind in ('ridge', 'softmax', 'knn5'):
                pred = np.full(len(y), -1)
                for held in np.unique(maps):
                    train, test = selected & (maps != held), selected & (maps == held)
                    pred[test] = classify(x, y, train, test, ep, kind)
                per_episode = {int(e): float((pred[ep == e] == y[ep == e]).mean()) for e in np.unique(ep[selected])}
                by_map = [np.mean([per_episode[int(e)] for e in np.unique(ep[selected & (maps == m)])])
                          for m in np.unique(maps)]
                confusion = np.zeros((5, 5))
                for e in np.unique(ep[selected]):
                    mask = ep == e
                    counts = np.bincount(pred[mask], minlength=5) / mask.sum()
                    confusion[y[mask][0]] += counts / 10
                results[collector][name][kind] = {'episode_balanced_accuracy': float(np.mean(by_map)),
                    'window_accuracy': float((pred[selected] == y[selected]).mean()),
                    'map_accuracies': np.array(by_map).tolist(), 'episode_accuracies': per_episode,
                    'confusion_row_normalized': confusion.tolist()}
                preds[collector][name][kind] = pred
    deltas = {}
    for collector in ('baseline', 'context'):
        deltas[collector] = {}
        for name in ('swap_best', 'swap_final'):
            for kind in ('ridge', 'softmax', 'knn5'):
                delta = np.array(results[collector][name][kind]['map_accuracies']) - np.array(results[collector]['original_best'][kind]['map_accuracies'])
                rng = np.random.default_rng(20261021)
                samples = delta[rng.integers(0, 10, (10000, 10))].mean(1)
                deltas[collector][name + '/' + kind] = {'mean': float(delta.mean()),
                    'map_bootstrap95': np.quantile(samples, [.025, .975]).tolist()}
    (out / 'figure_contract.json').write_text(json.dumps({'question': 'Can vehicle type be decoded during unconstrained natural driving?',
        'archetype': 'quantitative grid', 'backend': 'python', 'exports': ['png', 'pdf', 'svg', 'csv', 'npz'],
        'evidence': ['Same natural histories encoded by both models', 'Leave-one-map-out supervised probes', 'Type/speed/angle recoloring of identical PCA coordinates'],
        'risks': ['Natural kinematics differ by type and are not controlled', 'Fixed native chassis only',
                  'Map-group inference has 10 groups, not independent windows', 'Supervised S/XL labels are not zero-shot classification']}, indent=2))
    report = {'samples': len(y), 'baseline_policy_windows': int((policy == 'baseline').sum()),
        'context_policy_windows': int((policy == 'context').sum()), 'episode_count': 100, 'map_count': 10,
        'sampling': 'Natural rollouts, no set_velocity/set_steering, no speed/angle filtering; stride5, at most40 windows/episode.',
        'reconstruction': 'state4 histories reconstructed from cached endpoint state2 + measured speed + measured wheel settings + executed actions; exact old-token alignment asserted.',
        'collection_provenance': provenance, 'encoder_checkpoints': ckpts, 'results': results, 'paired_deltas': deltas,
        'class_counts_baseline': {m: int(((y == k) & (policy == 'baseline')).sum()) for k, m in enumerate(models)},
        'condition_ranges_baseline': {field: [float(history[policy == 'baseline', -1, ch].min()), float(history[policy == 'baseline', -1, ch].max())]
                                      for field, ch in [('speed_mps', 2), ('right_wheel_rad', 3)]},
        'limitations': ['Deep and swap loss changed together; no loss-only attribution.',
            'Leave-one-map-out, all windows from each episode kept together; train-only scaling.',
            'Main accuracy gives each episode equal weight; all 5 classes have 10 episodes per collector.',
            'Natural model separation can reflect speed, angle, maneuver, and steering-gain cues rather than pure dynamics.',
            'One training seed, 10 maps, fixed native profiles; probes supervise all five types including S/XL.']}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(out / 'representations.npz', history=history, **contexts)
    with (out / 'windows.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / 'probe_summary.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['collector', 'representation', 'probe', 'episode_balanced_accuracy'])
        for collector, rep in results.items():
            for name, kinds in rep.items():
                for kind, values in kinds.items(): w.writerow([collector, name, kind, values['episode_balanced_accuracy']])
    selected = policy == 'baseline'; fit = selected & (maps != 0)
    plot_ids = np.flatnonzero(selected)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.1), layout='constrained')
    order = np.random.default_rng(13).permutation(plot_ids)
    coordinates = {}
    for row, name in enumerate(('original_best', 'swap_best')):
        zz = scale(contexts[name], fit)
        _, sv, axes = np.linalg.svd(zz[fit], full_matrices=False)
        coord = zz @ axes[:2].T; ratio = sv[:2] ** 2 / (sv ** 2).sum()
        coordinates[name] = coord
        for col in range(3):
            ax = axs[row, col]
            if col == 0:
                for k, model in enumerate(models):
                    ix = order[y[order] == k]
                    ax.scatter(*coord[ix].T, s=9, alpha=.4, color=colors[k], edgecolors='none', label=model.upper())
                if row == 0: ax.legend(ncol=3, fontsize=7)
            else:
                values = history[:, -1, 2] if col == 1 else -np.rad2deg(history[:, -1, 3])
                v = values[selected]
                sc = ax.scatter(*coord[order].T, c=values[order], s=9, alpha=.45, edgecolors='none',
                    cmap='viridis' if col == 1 else 'coolwarm', vmin=v.min(), vmax=v.max())
                fig.colorbar(sc, ax=ax, shrink=.7, label='m/s' if col == 1 else 'deg (left positive)')
            if row == 0: ax.set_title(['Vehicle type', 'Natural measured speed', 'Natural front-wheel angle'][col])
            ax.set_xlabel(f'PC1 ({ratio[0]:.1%})')
            ax.set_ylabel(('Original state4\n' if row == 0 else 'Deep + swap1\n') + f'PC2 ({ratio[1]:.1%})')
            ax.grid(alpha=.12)
    fig.suptitle(f'Natural driving: same reference policy, 50 episodes across 10 maps\n{selected.sum():,} non-overlapping windows; no speed or steering intervention', fontsize=11)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(out / f'natural_context_pca.{ext}', dpi=250)
    plt.close(fig)
    np.savez_compressed(out / 'pca_coordinates.npz', **coordinates)
    print(json.dumps({'windows': report['class_counts_baseline'], 'range': report['condition_ranges_baseline'],
        'probes': {c: {n: {k: v['episode_balanced_accuracy'] for k, v in ks.items()} for n, ks in rs.items()}
                   for c, rs in results.items()}, 'deltas': deltas}, indent=2))


if __name__ == '__main__': main()
