"""Evaluate a frozen baseline/state2 pair on the established three 200-episode pools."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def report(out, status):
    import numpy as np
    results = {}
    lines = ['# 1M训练：R2Dreamer 与旧版 state2 Context', '',
             '各自验证集选出的best；每个测试池200回合，地图、底盘参数和推理随机种子逐回合配对。',
             '训练车型DEFAULT/M/L；S和XL为未见车型，质量及转角scale沿用训练网格。', '',
             '| 测试池 | 方法 | 回合数 | 成功率 | 碰撞率 | 出界率 | 超时率 | 奖励 | 路线完成率 | 速度km/h |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rng = np.random.default_rng(20261018)
    for pool in ('new_maps', 's', 'xl'):
        paths = {m: out / f'{pool}_{m}' for m in ('r2dreamer', 'old_state2_context')}
        for method, path in paths.items():
            if status['runs'][path.name]['state'] != 'completed':
                continue
            s = json.loads((path/'summary.json').read_text())['summary']
            lines.append(f"| {pool} | {method} | {s['episodes']} | {s['success']:.1%} | {s['crash']:.1%} | {s['out_of_road']:.1%} | {s['timeout']:.1%} | {s['reward']:.2f} | {s['route_completion']:.1%} | {s['mean_speed_km_h']:.2f} |")
        if not all(status['runs'][p.name]['state'] == 'completed' for p in paths.values()):
            continue
        traces = {m: [json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()] for m,p in paths.items()}
        a,b = traces['r2dreamer'], traces['old_state2_context']
        assert len(a) == len(b) == 200
        assert all(x['seed'] == y['seed'] and x['chassis'] == y['chassis'] for x,y in zip(a,b)), pool
        indices = rng.integers(0,200,(10000,200))
        differences = {}
        for key in ('success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h'):
            delta = np.array([float(y[key])-float(x[key]) for x,y in zip(a,b)])
            differences[key] = {'context_minus_baseline':float(delta.mean()),
                                'paired_bootstrap95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
        results[pool] = {'summaries':{m:json.loads((p/'summary.json').read_text())['summary'] for m,p in paths.items()},
                         'differences':differences}
    lines += ['', '| 测试池 | Context成功率提升 | 配对bootstrap95%区间 |', '|---|---:|---|']
    for pool,r in results.items():
        d=r['differences']['success']; lo,hi=d['paired_bootstrap95']
        lines.append(f"| {pool} | {100*d['context_minus_baseline']:+.1f}pp | [{100*lo:+.1f}, {100*hi:+.1f}]pp |")
    lines += ['', '区间基于每池200个配对地图重采样10000次，只代表当前单训练seed的测试不确定性。',
              '部分测试池地图种子重叠，不将三个测试池合并成600个独立地图。仅完成的测试纳入表格；总状态见status.json。','']
    (out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=2,choices=(1,2,3))
    args=parser.parse_args()
    root=args.training_root.resolve(); out=args.output.resolve()
    if out.exists(): parser.error('Use a fresh output directory')
    assert json.loads((root/'status.json').read_text())['phase']=='completed'
    out.mkdir(parents=True)
    evaluator=Path(__file__).resolve().parent/'evaluate_metadrive.py'
    shutil.copy2(evaluator,out/evaluator.name)
    shutil.copy2(Path(__file__),out/'runner_snapshot.py')
    runs={}
    for pool,seed,models in [('new_maps',10020,[]),('s',10320,['s']),('xl',10420,['xl'])]:
        for method,folder in [('r2dreamer','r2dreamer'),('old_state2_context','r2dreamer_context')]:
            checkpoint=root/folder/'best.pt'
            assert checkpoint.is_file()
            label=f'{pool}_{method}'
            command=[sys.executable,'-u',str(out/evaluator.name),'--checkpoint',str(checkpoint),
                     '--source-root',str(root/'source'),'--output',str(out/label),
                     '--episodes','200','--seed-start',str(seed)]
            if models: command+=['--eval-models',*models]
            runs[label]={'state':'queued','command':command}
    status={'phase':'running','runner_pid':os.getpid(),'started_at':now(),'total_requested':1200,
            'workers':args.workers,'training_root':str(root),'runs':runs,
            'evaluation_sha256':hashlib.sha256((out/evaluator.name).read_bytes()).hexdigest()}
    lock=threading.Lock(); children={}; stopped=threading.Event()
    def save():
        tmp=out/'status.json.tmp'; tmp.write_text(json.dumps(status,indent=2)+'\n'); tmp.replace(out/'status.json')
    save()
    def stop(signum,frame):
        stopped.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    def evaluate(label):
        with lock:
            if stopped.is_set(): return
            record=runs[label]
            log=(out/(label+'.log')).open('w')
            child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,
                                   cwd=root/'source',start_new_session=True)
            children[label]=child
            record.update(state='running',pid=child.pid,started_at=now()); save()
        code=child.wait(); log.close()
        with lock:
            record.update(returncode=code,finished_at=now(),state='completed' if code==0 else 'failed')
            if code==0:
                summary=json.loads((out/label/'summary.json').read_text())
                assert summary['status']=='completed' and summary['completed_episodes']==200
                record['summary']=summary['summary']
            report(out,status); save()
            print(now(),label,record['state'],record.get('summary',{}),flush=True)
        if code: raise RuntimeError(f'{label} exited {code}')
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(evaluate,label) for label in runs]
            for future in as_completed(futures): future.result()
        status.update(phase='completed',finished_at=now()); save()
    except BaseException:
        status.update(phase='interrupted' if stopped.is_set() else 'failed',finished_at=now()); save()
        raise

if __name__=='__main__':
    main()
