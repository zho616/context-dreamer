"""Build a resumable Pittsburgh-only 8000/100/100 ScenarioNet dataset.

Train/test counts follow VDD. An additional validation split, session grouping,
seed and extraction rules are explicit local conventions, not author token lists.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np

MAP = 'us-pa-pittsburgh-hazelwood'


def session(path):
    return Path(path).stem.rsplit('_', 2)[0]


def partition_logs(logs, seed):
    groups = sorted({session(path) for path in logs})
    if len(groups) < 3:
        raise ValueError('At least three independent recording sessions are required')
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    ntrain = min(len(groups) - 2, max(1, int(len(groups) * .7)))
    nval = min(len(groups) - ntrain - 1, max(1, int(len(groups) * .15)))
    assignments = dict(train=groups[:ntrain], val=groups[ntrain:ntrain+nval],
                       test=groups[ntrain+nval:])
    # Interleave sessions so a bounded subset covers many recording sessions.
    result = {}
    for split, names in assignments.items():
        queues = [sorted(path for path in logs if session(path) == name) for name in names]
        result[split] = [paths[index] for index in range(max(map(len, queues)))
                         for paths in queues if index < len(paths)]
    return assignments, result


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def atomic_pickle(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def audit_dataset(output, expected):
    from metadrive.scenario.utils import read_dataset_summary
    rows, identities = {}, {}
    for split, count in expected.items():
        summary, names, _ = read_dataset_summary(str(output / split))
        if len(names) != count:
            raise ValueError(f'{split}: expected {count}, found {len(names)}')
        if any(item['map'] != MAP or item['dataset'] != 'nuplan' for item in summary.values()):
            raise ValueError(f'{split}: dataset or map mismatch')
        tokens = {item['scenario_token'] for item in summary.values()}
        logs = {item['log_name'] for item in summary.values()}
        if len(tokens) != count:
            raise ValueError('Duplicate scenario tokens')
        identities[split] = dict(tokens=tokens, logs=logs, sessions={session(log) for log in logs})
        rows[split] = dict(scenes=count, logs=len(logs), sessions=len(identities[split]['sessions']))
    for left in identities:
        for right in identities:
            if left >= right:
                continue
            for key in ('tokens', 'logs', 'sessions'):
                if identities[left][key] & identities[right][key]:
                    raise ValueError(f'Leakage between {left}/{right}: {key}')
    return dict(passed=True, map=MAP, splits=rows, token_log_session_disjoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--db-dir', type=Path)
    parser.add_argument('--train-scenes', type=int, default=8000)
    parser.add_argument('--val-scenes', type=int, default=100)
    parser.add_argument('--test-scenes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    expected = dict(train=args.train_scenes, val=args.val_scenes, test=args.test_scenes)
    if min(expected.values()) < 1:
        parser.error('All split sizes must be positive')
    root, output = args.root.resolve(), args.output.resolve()
    maps = root / 'dataset/nuplan-maps-v1.0'
    dbdir = args.db_dir or root / 'dataset/data/cache/train_pittsburgh'
    logs = sorted(dbdir.rglob('*.db'))
    if not logs:
        parser.error(f'No Pittsburgh logs: {dbdir}')
    for log in logs:
        with sqlite3.connect(log.resolve().as_uri()+'?mode=ro', uri=True) as connection:
            if {row[0] for row in connection.execute('SELECT location FROM log')} != {MAP}:
                parser.error(f'Wrong city: {log}')
    assignments, split_logs = partition_logs(logs, args.seed)
    contract = dict(map=MAP, expected=expected, seed=args.seed,
                    recording_session_assignments=assignments,
                    logs={key: [str(p.resolve()) for p in paths] for key, paths in split_logs.items()},
                    extraction=dict(duration_s=20, sample_hz=10, start_spacing_s=10,
                                    remove_invalid_goals=True, minimum_displacement_m=10.,
                                    all_scenario_types=True),
                    conventions='Local split/extraction; author tokens and filters are not published')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'split_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract'] != contract:
            raise ValueError('Existing output has a different data contract; choose a fresh directory')
    else:
        manifest = dict(contract=contract, phase='converting', converted={}, failures=[])
        atomic_json(manifest_path, manifest)
    os.environ.update(NUPLAN_DATA_ROOT=str(root/'dataset'), NUPLAN_MAPS_ROOT=str(maps),
                      NUPLAN_EXP_ROOT=str(root/'exp'), CUDA_VISIBLE_DEVICES='')
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    from scenarionet.converter.nuplan.utils import convert_nuplan_scenario
    from metadrive.scenario import ScenarioDescription as SD
    for split, paths in split_logs.items():
        folder = output / split
        folder.mkdir(exist_ok=True)
        summary_path = folder / SD.DATASET.SUMMARY_FILE
        summary = pickle.loads(summary_path.read_bytes()) if summary_path.exists() else {}
        progress = manifest['converted'].setdefault(split, dict(scene_count=0, completed_logs=[]))
        for log in paths:
            if len(summary) >= expected[split]:
                break
            if str(log) in progress['completed_logs']:
                continue
            builder = NuPlanScenarioBuilder(data_root=str(log.parent), map_root=str(maps),
                sensor_root=str(root/'unused_sensors'), db_files=[str(log)],
                map_version='nuplan-maps-v1.0', include_cameras=False, max_workers=1,
                verbose=False, scenario_mapping=ScenarioMapping({}, .5))
            filters = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None,
                map_names=[MAP], num_scenarios_per_type=None, limit_total_scenarios=None,
                timestamp_threshold_s=10., ego_displacement_minimum_m=10.,
                expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
            candidates = builder.get_scenarios(filters, Sequential())
            # Stable per-log order, independent of conversion outcomes and model performance.
            salt = int.from_bytes(hashlib.sha256(log.stem.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, salt]))
            for index in rng.permutation(len(candidates)):
                scenario = candidates[index]
                name = SD.get_export_file_name('nuplan', 'v1.1', scenario.scenario_name)
                if name in summary:
                    continue
                try:
                    data = convert_nuplan_scenario(scenario, 'v1.1')
                    if data[SD.LENGTH] < 2:
                        raise ValueError('Recording contains fewer than two control frames')
                    SD.update_summaries(data)
                    data = data.to_dict()
                    SD.sanity_check(data, check_self_type=True)
                    atomic_pickle(folder/name, data)
                    summary[name] = copy.deepcopy(data[SD.METADATA])
                except Exception as error:
                    manifest['failures'].append(dict(split=split, log=log.stem,
                        token=scenario.scenario_name, error=repr(error)))
                if len(summary) >= expected[split]:
                    break
            atomic_pickle(summary_path, summary)
            atomic_pickle(folder/SD.DATASET.MAPPING_FILE, {name: '' for name in summary})
            progress['completed_logs'].append(str(log))
            progress['scene_count'] = len(summary)
            atomic_json(manifest_path, manifest)
            print(f'{split}: {len(summary)}/{expected[split]} scenes; {log.stem}', flush=True)
        if len(summary) != expected[split]:
            raise RuntimeError(f'{split}: only {len(summary)} eligible scenarios; need {expected[split]}')
    audit = audit_dataset(output, expected)
    atomic_json(output/'split_audit.json', audit)
    manifest['phase'] = 'ready'
    atomic_json(manifest_path, manifest)
    print('PITTSBURGH_READY '+str(output), flush=True)


if __name__ == '__main__':
    main()
