"""Continue one prepared recovery, then run the original independent test protocol."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    run = Path(sys.argv[1]).resolve()
    meta = json.loads((run / 'recovery.json').read_text())
    experiment = run.parent.parent
    source = Path(meta['source'])
    env = dict(os.environ, PYTHONPATH='/home/server/projects/metadrive', PYTHONUNBUFFERED='1')
    status = dict(state='starting', runner_pid=os.getpid(), started_at=now(), recovery=meta)

    def update():
        write(run / 'status.json', status)
        original = json.loads((experiment / 'status.json').read_text())
        original.setdefault('original_runner_pid', original['runner_pid'])
        original['runner_pid'] = os.getpid()
        original['phase'] = 'r2dreamer_recurrent_actor_critic/recovery/' + status['state']
        record = original['methods']['r2dreamer_recurrent_actor_critic']
        record.setdefault('interrupted_pid', record.get('pid'))
        record.update(state=status['state'], recovery_logdir=str(run),
                      recovery_origin_step=meta['origin_step'], recovery_status=str(run / 'status.json'))
        if 'pid' in status:
            record['pid'] = status['pid']
        write(experiment / 'status.json', original)

    def execute(command, log):
        status['command'] = command
        with (run / log).open('w') as output:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=output,
                                       stderr=subprocess.STDOUT)
            status['pid'] = process.pid
            update()
            code = process.wait()
        if code:
            raise RuntimeError(f'Child exited with code {code}: {log}')

    try:
        status['state'] = 'training'
        execute([sys.executable, '-u', str(source / 'train.py'), '--config-name', 'recovery',
                 '+resume_checkpoint=' + meta['resume_checkpoint'], 'logdir=' + str(run)], 'launcher.log')
        assert (run / 'latest.pt').exists() and (run / 'best.pt').exists()
        for name, seed, model in [('new_maps', 10020, None), ('s', 10320, 's'), ('xl', 10420, 'xl')]:
            status['state'] = 'evaluation_' + name
            command = [sys.executable, '-u', str(experiment / 'evaluate_metadrive.py'),
                '--checkpoint', str(run / 'best.pt'), '--source-root', str(source),
                '--output', str(run / 'evaluation' / name), '--episodes', '100', '--seed-start', str(seed)]
            if model:
                command += ['--eval-models', model]
            execute(command, 'evaluation_' + name + '.log')
        status.update(state='completed', finished_at=now())
        update()
    except BaseException:
        status.update(state='failed', error=traceback.format_exc(), finished_at=now())
        update()
        raise


if __name__ == '__main__':
    main()
