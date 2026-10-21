"""Paired frozen-encoder audit of operating-condition and vehicle associations."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_vehicle_context_associations import standardize, design, sse, eta, predict, pca


def load_context(source, checkpoint, history, name):
    spec = importlib.util.spec_from_file_location('audit_' + name, source / 'context.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    encoder = module.HistoryEncoder(state_dim=4).eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    prefix = 'context_encoder.'
    encoder.load_state_dict({k[len(prefix):]: v for k, v in payload['agent_state_dict'].items()
                             if k.startswith(prefix)}, strict=True)
    step = int(payload['step'])
    del payload
    mask = torch.ones(len(history), 5, dtype=torch.bool)
    with torch.inference_mode():
        context = encoder(torch.from_numpy(history.copy()), mask).numpy()
    return encoder, context.astype(float), {'step': step, 'checkpoint': str(checkpoint),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'source': str(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Regenerate this analysis directory')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.overwrite)
    root = Path(__file__).resolve().parents[1]
    cache = root / 'logdir/context_kinematics_500k_20261015_180340/analysis/vehicle_context_pca'
    rows = list(csv.DictReader((cache / 'windows.csv').open()))
    history = np.load(cache / 'representations.npz')['history'].astype(np.float32)
    assert history.shape == (1680, 5, 6)
    models = ['default', 'm', 'l', 's', 'xl']
    labels = np.array([r['model'] for r in rows])
    y = np.array([models.index(v) for v in labels])
    group = np.array([int(r['repeat']) for r in rows])
    speed = np.rint([float(r['initial_speed_m_s']) for r in rows]).astype(int)
    angle = np.rint([float(r['front_left_angle_degrees']) for r in rows]).astype(int)
    endpoint_speed = history[:, -1, 2].astype(float)
    assert set(speed) == {5, 10} and len(np.unique(group)) == 24
    assert all((labels == m).sum() == 336 for m in models)
    folds = np.array_split(np.random.default_rng(20261016).permutation(np.unique(group)), 6)
    fit = ~np.isin(group, folds[0])
    fixed = (speed == 10) & (angle == 15)
    assert fixed.sum() == 120
    (out / 'figure_contract.json').write_text(json.dumps({
        'question': 'Does the swap+Deep encoder reduce operating-condition associations and retain vehicle information?',
        'evidence': ['Full-8D partial R2 controlling other factors/action profile',
                     'Vehicle probes on held-out action profiles and cross-speed tests',
                     'Identical PCA basis within each row, recolored by factors'],
        'archetype': 'quantitative grid', 'backend': 'python', 'exports': ['png', 'pdf', 'svg', 'csv', 'npz'],
        'risks': ['Deep injection and swap loss changed together; no swap-only causal attribution.',
                  'Cached native vehicle profiles; not randomized mass/steering grid.',
                  'Actual angle and executed steer can reveal max steering/type.',
                  'S/XL appear as supervised probe labels; not zero-shot classification.',
                  'Standardized association is not semantic information percentage.',
                  'Natural short responses; endpoint speed is not clamped.']}, indent=2) + '\n')
    old = root / 'logdir/context_kinematics_500k_20261015_180340'
    new = root / 'logdir/context_swap_deep1_500k_20261020/training'
    configs = {
        'state4_original_best': (old / 'source', old / 'r2dreamer_context/best.pt'),
        'deep_swap1_best': (new / 'source', new / 'r2dreamer_dali_deep/best.pt'),
        'deep_swap1_final': (new / 'source', new / 'r2dreamer_dali_deep/latest.pt'),
    }
    torch.set_num_threads(1)
    spaces, provenance, encoders = {}, {}, {}
    for name, (source, checkpoint) in configs.items():
        encoders[name], spaces[name], provenance[name] = load_context(source, checkpoint, history, name)
    spaces['raw_history'] = (history / np.array([10, 1, 30, np.pi / 3, 1, 1])).reshape(len(history), -1)
    spaces['protocol_only'] = np.column_stack((speed / 30, angle / 60, history[..., 4:].reshape(len(history), -1)))
    M, I, A, G = design(labels), design(speed), design(angle), design(group)
    condition = design(np.array([f'{v}/{a}' for v, a in zip(speed, angle)]))
    zspeed = (endpoint_speed - endpoint_speed.mean()) / endpoint_speed.std()
    V = np.column_stack((zspeed, zspeed ** 2, zspeed ** 3))
    factors, metrics, predictions, coordinates, ratios, sensitivity = {}, {}, {}, {}, {}, {}
    for name, x in spaces.items():
        z = standardize(x, fit)
        partial = lambda base, added: 1 - sse(z, np.column_stack((base, added))) / max(sse(z, base), 1e-12)
        factors[name] = {
            'vehicle_eta2': eta(z, labels), 'initial_speed_eta2': eta(z, speed), 'wheel_angle_eta2': eta(z, angle),
            'vehicle_partial_r2': partial(np.column_stack((condition, G)), M),
            'initial_speed_partial_r2': partial(np.column_stack((M, A, G)), I),
            'wheel_angle_partial_r2': partial(np.column_stack((M, I, G)), A),
            'endpoint_speed_partial_r2': partial(np.column_stack((M, A, G)), V),
        }
        coordinates[name], ratios[name] = pca(x, fit)
        metrics[name], predictions[name] = {}, {}
        for test in ('mixed', 'fixed_10mps_15deg', 'speed5_to10', 'speed10_to5'):
            for kind in ('linear', 'knn5'):
                pred = np.full(len(y), -1, dtype=int)
                for held in folds:
                    train = ~np.isin(group, held)
                    select = ~train
                    if test == 'fixed_10mps_15deg':
                        train &= fixed
                        select &= fixed
                    elif test == 'speed5_to10':
                        train &= speed == 5
                        select &= speed == 10
                    elif test == 'speed10_to5':
                        train &= speed == 10
                        select &= speed == 5
                    pred[select] = predict(x, y, train, select, 'linear' if kind == 'linear' else 'knn')
                valid = pred >= 0
                confusion = np.zeros((5, 5), dtype=int)
                np.add.at(confusion, (y[valid], pred[valid]), 1)
                metrics[name][test + '/' + kind] = {'accuracy': float((pred[valid] == y[valid]).mean()),
                    'samples': int(valid.sum()), 'confusion': confusion.tolist()}
                predictions[name][test + '/' + kind] = pred
        # Held-out continuous probes quantify information about current measured state.
        for field, target in [('endpoint_speed', endpoint_speed), ('wheel_angle', angle.astype(float))]:
            pred = np.zeros(len(y))
            for held in folds:
                train = ~np.isin(group, held)
                test = ~train
                zz = standardize(x, train)
                xx = np.column_stack((np.ones(len(x)), zz))
                penalty = np.eye(xx.shape[1]); penalty[0, 0] = 0
                w = np.linalg.solve(xx[train].T @ xx[train] + penalty, xx[train].T @ target[train])
                pred[test] = xx[test] @ w
            metrics[name][field + '/linear_regression'] = {
                'r2': float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()),
                'mae': float(np.abs(target - pred).mean())}
        if name in encoders:
            encoder = encoders[name]
            sd = x[fit].std(0).clip(min=1e-6)
            sensitivity[name] = {}
            # Supplementary input-channel intervention only; not a physical rollout.
            for field, channel, step in [('speed_channel', 2, .1), ('angle_channel', 3, np.deg2rad(1)),
                                         ('current_speed_channel', 2, .1), ('current_angle_channel', 3, np.deg2rad(1))]:
                plus, minus = history.copy(), history.copy()
                if field.startswith('current_'):
                    plus[:, -1, channel] += step
                    minus[:, -1, channel] -= step
                else:
                    plus[..., channel] += step
                    minus[..., channel] -= step
                with torch.inference_mode():
                    mask = torch.ones(len(x), 5, dtype=torch.bool)
                    cp = encoder(torch.from_numpy(plus), mask).numpy()
                    cm = encoder(torch.from_numpy(minus), mask).numpy()
                normalized_derivative = (cp - cm) / (2 * step) / sd
                sensitivity[name][field] = {'median_rms_derivative': float(np.median(np.sqrt((normalized_derivative ** 2).mean(1)))),
                    'mean_rms_derivative': float(np.sqrt((normalized_derivative ** 2).mean(1)).mean()),
                    'unit': 'per m/s' if channel == 2 else 'per rad',
                    'warning': ('Endpoint only' if field.startswith('current_') else 'All five history tokens')
                               + ': one state channel perturbed; actions/other channels held. Off-manifold diagnostic.'}
    deltas = {}
    for stage in ('deep_swap1_best', 'deep_swap1_final'):
        deltas[stage] = {}
        for test in ('mixed', 'fixed_10mps_15deg', 'speed5_to10', 'speed10_to5'):
            key = test + '/linear'
            oldpred, newpred = predictions['state4_original_best'][key], predictions[stage][key]
            valid = oldpred >= 0
            values = np.array([((newpred[valid & (group == g)] == y[valid & (group == g)]).astype(float)
                               - (oldpred[valid & (group == g)] == y[valid & (group == g)]).astype(float)).mean()
                               for g in np.unique(group)])
            rng = np.random.default_rng(20261021)
            draws = values[rng.integers(0, 24, (10000, 24))].mean(1)
            deltas[stage][test] = {'accuracy_delta': float(values.mean()),
                'profile_bootstrap95': np.quantile(draws, [.025, .975]).tolist()}
    result = {'samples': len(y), 'action_profile_groups': 24, 'vehicle_types': models,
        'cache': str(cache), 'history_sha256': hashlib.sha256(history.tobytes()).hexdigest(),
        'checkpoints': provenance, 'factors': factors, 'probes': metrics,
        'paired_probe_deltas': deltas, 'channel_sensitivity': sensitivity,
        'methodology': {'factors': 'Full standardized representation; descriptive linear partial R2 controlling other factors/profile. Not additive/causal.',
            'cv': 'Six folds, split by 24 action profiles, training-only scaling. Linear ridge alpha=1, kNN5.',
            'bootstrap': '24 held-out profile group accuracies resampled 10000 times; no training-seed uncertainty.',
            'cache': 'Identical physical histories freshly encoded by all three checkpoints; no policy confounding.',
            'attribution': 'Deep injection and exchange loss both changed; improvement cannot be assigned to loss alone.',
            'type_confound': 'Given angle and normalized steer, max-steering/type cues can be recovered. Compare protocol_only.',
            'scope': 'Five native profiles on one straight road; only initial speed/actual wheel settings controlled, no velocity clamping.'}}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'representations.npz', history=history, **spaces,
                        **{name + '_pca2': z for name, z in coordinates.items()})
    with (out / 'association_summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['representation'] + list(next(iter(factors.values()))))
        w.writeheader()
        for name, values in factors.items(): w.writerow({'representation': name, **values})
    with (out / 'probe_summary.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['representation', 'test', 'accuracy'])
        for name, tests in metrics.items():
            for test, values in tests.items():
                if 'accuracy' in values: w.writerow([name, test, values['accuracy']])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.1), layout='constrained')
    order = np.random.default_rng(13).permutation(len(y))
    for row, name in enumerate(('state4_original_best', 'deep_swap1_best')):
        z = coordinates[name]
        for col, title in enumerate(('Vehicle type', 'Measured endpoint speed', 'Actual front-wheel angle')):
            ax = axs[row, col]
            if col == 0:
                for k, label in enumerate(models):
                    ids = order[y[order] == k]
                    ax.scatter(*z[ids].T, s=10, alpha=.5, color=colors[k], edgecolors='none', label=label.upper())
                if row == 0: ax.legend(ncol=3, fontsize=7, loc='best')
            else:
                values = endpoint_speed if col == 1 else angle
                sc = ax.scatter(*z[order].T, c=values[order], s=10, alpha=.5, edgecolors='none',
                    cmap='viridis' if col == 1 else 'coolwarm', vmin=values.min(), vmax=values.max())
                fig.colorbar(sc, ax=ax, shrink=.7, label='m/s' if col == 1 else 'deg (left positive)')
            ax.set_title(title if row == 0 else '')
            ax.set_xlabel(f'PC1 ({ratios[name][0]:.1%})')
            ax.set_ylabel(('Original state4\n' if row == 0 else 'Deep + swap1\n') + f'PC2 ({ratios[name][1]:.1%})')
            ax.grid(alpha=.12)
    fig.suptitle('Identical 1,680 physical windows; five vehicle types, 24 held-out action profiles\nEach row uses its own PCA basis; columns recolor the same coordinates', fontsize=11)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(out / f'context_factor_pca.{ext}', dpi=250)
    plt.close(fig)
    print(json.dumps({'factors': factors, 'probes': metrics, 'deltas': deltas, 'sensitivity': sensitivity}, indent=2))


if __name__ == '__main__':
    main()
