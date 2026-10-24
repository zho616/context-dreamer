"""Convert a bounded, original-log-disjoint nuPlan Mini subset at10Hz."""
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
import sys
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--db-dir',type=Path);p.add_argument('--train-scenes',type=int,default=32);p.add_argument('--eval-scenes',type=int,default=8);p.add_argument('--logs-per-split',type=int,default=64)
 p.add_argument('--map-name',help='Restrict the experiment to one nuPlan map, e.g. us-pa-pittsburgh-hazelwood')
 a=p.parse_args();root=a.root.resolve();out=a.output.resolve();assert not out.exists(),out
 maps=root/'dataset/nuplan-maps-v1.0';assert(maps/'nuplan-maps-v1.0.json').exists()
 dbs=sorted((a.db_dir if a.db_dir else root/'dataset').rglob('*.db'))
 if a.db_dir and (a.db_dir/'manifest.json').exists():
  priority=json.loads((a.db_dir/'manifest.json').read_text())
  if 'all_log_names' in priority:dbs=sorted(a.db_dir/name for name in priority['all_log_names'])
 if a.map_name:
  selected=[]
  for path in dbs:
   with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)as connection:
    locations={row[0]for row in connection.execute('SELECT location FROM log')}
   if locations=={a.map_name}:selected.append(path)
  dbs=selected
 session=lambda path:path.stem.rsplit('_',2)[0]
 session_names=sorted({session(path)for path in dbs});rng=np.random.default_rng(20261022);order=rng.permutation(len(session_names));n=len(session_names)
 if n<3:p.error(f'At least three recording sessions are needed for independent train/val/test; found {n}')
 if min(a.train_scenes,a.eval_scenes,a.logs_per_split)<1:p.error('Scene and log limits must be positive')
 ntrain=min(n-2,max(1,int(n*.7)));nval=min(n-ntrain-1,max(1,int(n*.15)))
 grouped={'train':[session_names[i]for i in order[:ntrain]],'val':[session_names[i]for i in order[ntrain:ntrain+nval]],'test':[session_names[i]for i in order[ntrain+nval:]]}
 splits={name:[path for sid in sessions for path in dbs if session(path)==sid]for name,sessions in grouped.items()}
 sets=[{session(path)for path in ps}for ps in splits.values()];assert all(not x&y for i,x in enumerate(sets)for y in sets[i+1:])
 out.mkdir(parents=True);manifest={'source':'Official nuPlan Mini v1.1, log DBs and maps only','partition':'70/15/15 recording sessions (timestamp+vehicle), fixed seed20261022; fragments from the same recording never cross splits; scan assigned logs deterministically until the bounded integration quota is reached','recording_session_assignments':grouped,'all_log_assignments':{k:[str(p)for p in ps]for k,ps in splits.items()},'converted':{}}
 if a.map_name:
  manifest.update(map_name=a.map_name,partition='Map-specific recording-session split, fixed seed20261022; approximate70/15/15 with at least one session per split; independent of the all-city integration subset')
 (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
 os.environ['NUPLAN_DATA_ROOT']=str(root/'dataset');os.environ['NUPLAN_MAPS_ROOT']=str(maps)
 os.environ['NUPLAN_EXP_ROOT']=str(root/'exp');os.environ['CUDA_VISIBLE_DEVICES']=''
 from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
 from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
 from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
 from nuplan.planning.utils.multithreading.worker_sequential import Sequential
 from scenarionet.converter.nuplan.utils import convert_nuplan_scenario
 from scenarionet.converter.utils import write_to_directory_single_worker
 from metadrive.scenario.utils import read_dataset_summary
 for split,logs in splits.items():
  available=[p for p in logs if p.exists()][:a.logs_per_split];assert available,f'No downloaded logs in {split}';limit=a.train_scenes if split=='train'else a.eval_scenes
  scenarios=[];selected=[]
  for log in available:
   selected.append(log)
   builder=NuPlanScenarioBuilder(data_root=str(log.parent),map_root=str(maps),sensor_root=str(root/'unused_sensors'),db_files=[str(log)],map_version='nuplan-maps-v1.0',include_cameras=False,max_workers=1,verbose=False,scenario_mapping=ScenarioMapping({},.5))
   filters=ScenarioFilter(scenario_types=None,scenario_tokens=None,log_names=None,map_names=[a.map_name]if a.map_name else None,num_scenarios_per_type=None,limit_total_scenarios=None,timestamp_threshold_s=20.,ego_displacement_minimum_m=10.,expand_scenarios=False,remove_invalid_goals=True,shuffle=False)
   valid=builder.get_scenarios(filters,Sequential());scenarios.extend(valid[:limit-len(scenarios)])
   print(split,'log',log.stem,'valid',len(valid),'selected',len(scenarios),flush=True)
   if len(scenarios)>=limit:break
  assert scenarios,f'No valid scenes for {split}'
  print(split,'convert',len(scenarios),'from',len(selected),'logs',flush=True)
  write_to_directory_single_worker(convert_func=convert_nuplan_scenario,scenarios=scenarios,output_path=str(out/split),dataset_version='v1.1',dataset_name='nuplan',overwrite=False)
  summary,names,mapping=read_dataset_summary(str(out/split));assert len(names)==len(scenarios)
  manifest['converted'][split]={'scene_count':len(names),'selected_logs':[str(p)for p in selected],'scenario_tokens':[s.scenario_name for s in scenarios],'used_logs':sorted({s.log_name for s in scenarios})}
  for other in manifest['converted']:
   if other!=split:
    assert not {s.rsplit('_',2)[0]for s in manifest['converted'][split]['used_logs']}&{s.rsplit('_',2)[0]for s in manifest['converted'][other]['used_logs']}
    assert not set(manifest['converted'][split]['scenario_tokens'])&set(manifest['converted'][other]['scenario_tokens'])
  (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('CONVERSION_READY',out,flush=True)
if __name__=='__main__':main()
