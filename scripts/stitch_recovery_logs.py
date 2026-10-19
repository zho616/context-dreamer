"""Serve one live log lineage: pre-checkpoint events followed by recovered training."""
import argparse
import json
import os
from pathlib import Path
import time

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recovery', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    recovery, output = args.recovery.resolve(), args.output.resolve()
    origin = recovery.parent
    boundary = json.loads((recovery / 'recovery.json').read_text())['origin_step']
    output.mkdir(parents=True, exist_ok=False)
    writer = EventFileWriter(str(output))
    loaders = {}
    counts = {'original_events': 0, 'recovery_events': 0, 'metrics_rows': 0}
    # Original files stay untouched, including the discarded 140K-302K branch.
    metrics = (output / 'metrics.jsonl').open('w')
    with (origin / 'metrics.jsonl').open() as source:
        for line in source:
            if json.loads(line)['step'] <= boundary:
                metrics.write(line)
                counts['metrics_rows'] += 1
    live_metrics = (recovery / 'metrics.jsonl').open()
    manifest = {'pid': os.getpid(), 'boundary_step': boundary, 'original': str(origin),
        'recovery': str(recovery), 'policy': 'original <= checkpoint step; recovery > checkpoint step',
        'original_files_preserved': True, 'smoke_excluded': True}
    try:
        while True:
            for directory, prefix in [(origin, 'original'), (recovery, 'recovery')]:
                # Nonrecursive glob explicitly excludes smoke and nested runs.
                for path in sorted(directory.glob('events.out.tfevents.*')):
                    loader = loaders.setdefault(path, EventFileLoader(str(path)))
                    for event in loader.Load():
                        if not event.HasField('summary'):
                            continue
                        keep = event.step <= boundary if prefix == 'original' else event.step > boundary
                        if keep:
                            writer.add_event(event)
                            counts[prefix + '_events'] += 1
            while True:
                position = live_metrics.tell()
                line = live_metrics.readline()
                if not line or not line.endswith('\n'):
                    live_metrics.seek(position)
                    break
                row = json.loads(line)
                if row['step'] > boundary:
                    metrics.write(line)
                    counts['metrics_rows'] += 1
                    manifest['latest_step'] = row['step']
            writer.flush()
            metrics.flush()
            manifest.update(counts, updated_at=time.time())
            temp = output / 'stitch_status.json.tmp'
            temp.write_text(json.dumps(manifest, indent=2) + '\n')
            temp.replace(output / 'stitch_status.json')
            status = json.loads((recovery / 'status.json').read_text())
            if status['state'] in ('completed', 'failed', 'stopped_by_user'):
                break
            time.sleep(5)
    finally:
        writer.close()
        metrics.close()
        live_metrics.close()


if __name__ == '__main__':
    main()
