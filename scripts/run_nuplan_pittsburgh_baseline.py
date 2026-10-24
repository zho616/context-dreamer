"""Supervise official Pittsburgh download, conversion, checks and baseline training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_status(path, state):
    state['updated_at'] = now()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


def process_matches(pid, substring):
    try:
        return substring in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/home/server/datasets/nuplan'))
    parser.add_argument('--conversion-python', type=Path, default=Path('/home/server/.venvs/nuplan-convert/bin/python'))
    parser.add_argument('--download-pid', type=int)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--steps', type=int, default=500000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    repo = Path(__file__).resolve().parents[1]
    output, root = args.output.resolve(), args.root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    source.mkdir()
    for file in repo.glob('*.py'):
        shutil.copy2(file, source/file.name)
    for name in ('configs', 'envs', 'optim', 'scripts'):
        shutil.copytree(repo/name, source/name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(repo/'pyproject.toml', source/'pyproject.toml')
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    (output/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    dataset, run = root/'scenarionet_pittsburgh_8000', output/'baseline'
    status_path = output/'status.json'
    state = dict(phase='starting', runner_pid=os.getpid(), created_at=now(), source=str(source),
                 dataset=str(dataset), method='r2dreamer_baseline', steps=args.steps,
                 input='LiDAR + native trajectory navigation (271D)',
                 reward='unchanged paper reward from existing closed-loop training', stages={})
    current = None

    def stop(signum, frame):
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HYDRA_FULL_ERROR='1')

    def refresh_download_progress():
        try:
            state['download_progress'] = json.loads((root/'download_progress.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def execute(phase, command, cpu=False):
        nonlocal current
        command = list(map(str, command))
        state['phase'] = phase
        state['stages'][phase] = dict(state='running', command=command, started_at=now())
        write_status(status_path, state)
        print(f'{now()} {phase}', flush=True)
        env = environment.copy()
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        with (output/(phase+'.log')).open('w') as stream:
            current = subprocess.Popen(command, cwd=source, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['stages'][phase]['pid'] = current.pid
            write_status(status_path, state)
            while current.poll() is None:
                if phase == 'download':
                    refresh_download_progress()
                    write_status(status_path, state)
                time.sleep(5)
            result = current.returncode
        state['stages'][phase].update(returncode=result, state='completed' if result==0 else 'failed', finished_at=now())
        write_status(status_path, state)
        if result:
            raise RuntimeError(f'{phase} exited {result}; see {output/(phase+".log")}')

    try:
        write_status(status_path, state)
        marker = root/'nuplan-v1.1_train_pittsburgh.zip.extracted.json'
        if args.download_pid and not marker.exists():
            state.update(phase='waiting_for_download', download_pid=args.download_pid)
            write_status(status_path, state)
            while not marker.exists() and process_matches(args.download_pid, 'prepare_nuplan_mini.py'):
                refresh_download_progress()
                write_status(status_path, state)
                time.sleep(15)
        if not marker.exists():
            execute('download', [sys.executable, '-u', source/'scripts/prepare_nuplan_mini.py',
                '--root', root, '--workers', args.workers, '--archives', 'nuplan-v1.1_train_pittsburgh.zip'], cpu=True)
        manifest = json.loads((root/'download_manifest.json').read_text())
        if not manifest['files']['nuplan-v1.1_train_pittsburgh.zip'].get('etag_verified'):
            raise RuntimeError('Official Pittsburgh archive has not passed integrity verification')
        execute('convert', [args.conversion_python, '-u', source/'scripts/prepare_nuplan_pittsburgh.py',
                           '--root', root, '--output', dataset], cpu=True)
        sys.path.insert(0, str(source/'scripts'))
        from prepare_nuplan_pittsburgh import audit_dataset
        audit = audit_dataset(dataset, dict(train=8000, val=100, test=100))
        (output/'split_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        execute('physics_audit', [sys.executable, source/'scripts/audit_nuplan_physics.py',
                '--data', dataset, '--output', output/'physics_audit.json'], cpu=True)
        execute('baseline_smoke', [sys.executable, source/'scripts/smoke_nuplan_training.py',
                '--data', dataset, '--output', output/'smoke', '--gpu-short', '--methods', 'baseline'])
        command = [sys.executable, '-u', source/'train.py', '--config-name', 'metadrive_nuplan_pittsburgh',
                   f'env.steps={args.steps}', f'logdir={run}']
        state['formal_training_command'] = list(map(str, command))
        execute('training', command)
        for inference_seed in (0, 1, 2):
            execute(f'test_seed{inference_seed}', [sys.executable, source/'scripts/evaluate_metadrive.py',
                '--checkpoint', run/'best.pt', '--source-root', source,
                '--output', output/f'test_seed{inference_seed}', '--episodes', '100', '--seed-start', '0',
                '--inference-seed-offset', str(12345+inference_seed*100000),
                '--scenario-dir', dataset/'test'])
        state['phase'] = 'completed'
        write_status(status_path, state)
    except BaseException as error:
        state.update(phase='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        write_status(status_path, state)
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
