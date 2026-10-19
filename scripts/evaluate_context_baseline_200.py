"""Sequential frozen-checkpoint, paired 200-episode comparison on three pools."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def compare(out, status, reference=None):
    import numpy as np
    results = {}
    lines = ['# R2Dreamer 与 Context：每种200回合', '',
             '冻结各自验证集选出的best；seed0训练模型。每个测试池两种方法使用相同地图种子和实际底盘参数。',
             '新版context为state4，旧版为state2，均使用recurrent_concat；不是AdaRMS或直接Actor/Critic条件化。',
             '200回合为每组总量，包含此前100回合的种子范围，不与旧测试相加为300。', '',
             '| 测试池 | 方法 | 回合数 | 成功率 | 碰撞率 | 出界率 | 超时率 | 奖励 | 路线完成率 | 速度km/h |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rng = np.random.default_rng(20261016)
    for pool in ('new_maps', 's', 'xl'):
        paths = {name: out / (pool + '_' + name) for name in ('r2dreamer', 'R2_recurrent_context')}
        required = list(paths.values())
        if reference is not None:
            paths = {name: reference / (pool + '_' + name) for name in ('r2dreamer', 'R2_recurrent_context')}
            paths['old_state2_context'] = out / (pool + '_old_state2_context')
            required = [paths['old_state2_context']]
        if not all(status['runs'].get(path.name, {}).get('state') == 'completed' for path in required):
            continue
        summaries = {name: json.loads((p / 'summary.json').read_text()) for name, p in paths.items()}
        traces = {name: [json.loads(x) for x in (p / 'episodes.jsonl').read_text().splitlines()] for name, p in paths.items()}
        target = 'old_state2_context' if reference is not None else 'R2_recurrent_context'
        a, b = traces['r2dreamer'], traces[target]
        assert len(a) == len(b) == 200
        assert all(x['seed'] == y['seed'] and x['chassis'] == y['chassis'] for x, y in zip(a, b)), pool
        for trace in traces.values():
            assert len(trace) == 200
            assert all(x['seed'] == y['seed'] and x['chassis'] == y['chassis'] for x, y in zip(trace, b)), pool
        assert all(s['status'] == 'completed' and s['inference_seed_offset'] == 12345 for s in summaries.values())
        resamples = rng.integers(0, 200, (10000, 200))
        differences = {}
        for key in ('success', 'crash', 'out_of_road', 'timeout', 'reward', 'route_completion', 'mean_speed_km_h'):
            delta = np.array([float(y[key]) - float(x[key]) for x, y in zip(a, b)])
            differences[key] = {'context_minus_baseline': float(delta.mean()),
                                'paired_bootstrap95': np.quantile(delta[resamples].mean(1), [.025, .975]).tolist()}
        results[pool] = {'summaries': {n: s['summary'] for n, s in summaries.items()}, 'differences': differences}
        if reference is not None:
            delta = np.array([float(y['success']) - float(x['success']) for x, y in zip(traces['R2_recurrent_context'], b)])
            results[pool]['old_minus_new_success'] = {'difference': float(delta.mean()),
                'paired_bootstrap95': np.quantile(delta[resamples].mean(1), [.025, .975]).tolist()}
        for name, summary in summaries.items():
            s = summary['summary']
            lines.append(f"| {pool} | {name} | {s['episodes']} | {s['success']:.1%} | {s['crash']:.1%} | {s['out_of_road']:.1%} | {s['timeout']:.1%} | {s['reward']:.2f} | {s['route_completion']:.1%} | {s['mean_speed_km_h']:.2f} |")
    lines += ['', '## 配对成功率差', '', '| 测试池 | 比较 | 成功率差 | 地图配对bootstrap95% |', '|---|---|---:|---|']
    for pool, r in results.items():
        d = r['differences']['success']; lo, hi = d['paired_bootstrap95']
        label = '旧版state2减基线' if reference is not None else '新版state4减基线'
        lines.append(f"| {pool} | {label} | {100*d['context_minus_baseline']:+.1f}pp | [{100*lo:+.1f}, {100*hi:+.1f}]pp |")
        if reference is not None:
            d = r['old_minus_new_success']; lo, hi = d['paired_bootstrap95']
            lines.append(f"| {pool} | 旧版state2减新版state4 | {100*d['difference']:+.1f}pp | [{100*lo:+.1f}, {100*hi:+.1f}]pp |")
    lines += ['', '区间按200个配对地图种子重采样10000次，条件于单个训练seed，不代表多训练seed稳定性。',
              '车型S/XL未用于训练，但质量和转角scale仍取训练网格；不同测试池的部分地图种子可重叠，不将它们当作600个独立地图样本。',
              '报告仅包含已完整完成并核对配对参数的测试池；全部完成与否以status.json为准。', '']
    save(out / 'comparison.json', results)
    (out / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--old-context-reference', type=Path,
                        help='Evaluate only legacy state2 context and compare to an existing completed baseline/state4 200-episode run')
    args = parser.parse_args()
    out = args.output.resolve(); root = args.root.resolve()
    if (out / 'status.json').exists():
        parser.error('Use a fresh output directory')
    out.mkdir(exist_ok=True, parents=True)
    runs = {'r2dreamer': root / 'logdir/chassis_baselines_500k_20261015_020330',
            'R2_recurrent_context': root / 'logdir/context_kinematics_500k_20261015_180340'}
    reference = args.old_context_reference.resolve() if args.old_context_reference else None
    if reference is not None:
        assert json.loads((reference / 'status.json').read_text())['phase'] == 'completed'
        runs = {'old_state2_context': root / 'logdir/context_chassis_500k_20261015_103141'}
    specs = []
    for pool, seed, models in [('new_maps', 10020, []), ('s', 10320, ['s']), ('xl', 10420, ['xl'])]:
        for name, folder in runs.items():
            checkpoint = folder / ('r2dreamer' if name == 'r2dreamer' else 'r2dreamer_context') / 'best.pt'
            assert checkpoint.is_file() and (folder / 'source/dreamer.py').is_file()
            label = pool + '_' + name
            command = [sys.executable, '-u', str(out / 'evaluate_metadrive.py'), '--checkpoint', str(checkpoint),
                       '--source-root', str(folder / 'source'), '--output', str(out / label),
                       '--episodes', '200', '--seed-start', str(seed)]
            if models:
                command += ['--eval-models', *models]
            specs.append((label, command))
    status = {'phase': 'starting', 'runner_pid': os.getpid(), 'started_at': now(),
              'episodes_per_run': 200, 'total_requested': 200 * len(specs),
              'reference_directory': str(reference) if reference is not None else None,
              'evaluation_script_sha256': hashlib.sha256((out / 'evaluate_metadrive.py').read_bytes()).hexdigest(),
              'runs': {label: {'state': 'queued', 'command': cmd} for label, cmd in specs}}
    save(out / 'status.json', status)
    current = None
    def stop(signum, frame):
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGINT)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try:
        for label, command in specs:
            status['phase'] = label
            record = status['runs'][label]
            record.update(state='running', started_at=now())
            print(now(), 'Starting', label, flush=True)
            with (out / (label + '.log')).open('w') as log:
                current = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record['pid'] = current.pid; save(out / 'status.json', status)
                code = current.wait()
            if code:
                raise RuntimeError(f'{label} exited {code}')
            summary = json.loads((out / label / 'summary.json').read_text())
            assert summary['status'] == 'completed' and summary['completed_episodes'] == 200
            record.update(state='completed', finished_at=now(), summary=summary['summary'])
            compare(out, status, reference); save(out / 'status.json', status)
            print(now(), 'Completed', label, summary['summary'], flush=True)
        status.update(phase='completed', finished_at=now())
        save(out / 'status.json', status)
    except BaseException as exc:
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGINT)
        status.update(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=traceback.format_exc(), finished_at=now())
        if 'record' in locals():
            record['state'] = status['phase']
        save(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
