#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/home/ysh/codecase/SCOPE-repro/SCOPE
python3 - <<'PY'
from pathlib import Path
from typing import TypedDict
import contextlib,hashlib,io,json,os,random,subprocess
from src.goatbench_utils import prepare_goatbench_navigation_goals

class TaskMetadata(TypedDict):
    task_type: str
    answer: str
    scene_id: str
    episode_id: int
    subtask_index: int
    split: int
    source_filename: str

def mapping(value: object) -> dict[str, object]:
    if not isinstance(value,dict) or any(not isinstance(key,str) for key in value):
        raise ValueError('Expected object with string keys')
    return {key:item for key,item in value.items()}

def sequence(value: object) -> list[object]:
    if not isinstance(value,list):raise ValueError('Expected JSON array')
    return list(value)

def string(value: object) -> str:
    if not isinstance(value,str):raise ValueError('Expected string')
    return value

def persist(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=content:raise ValueError(f'Frozen artifact differs: {path}')
    else:
        path.write_text(content)

def save_json(path: Path, value: object) -> None:
    persist(path,json.dumps(value,indent=2,ensure_ascii=False)+'\n')

base=Path('/home/ysh/Datasets/SCOPE')
source=Path('/home/ysh/codecase/SCOPE-repro/SCOPE')
manifest_dir=base/'manifests'
test_dir=base/'datasets/goat_bench/val_unseen/content'
exp_name='REPRO_20261005_SCOPE_Gemini38Flash_full2669'
output_dir=base/'outputs'/exp_name
upstream_cfg=source/'cfg/eval_goatbench.yaml'
raw_order=os.listdir(test_dir)
assert len(raw_order)==36 and all(name.endswith('.json') for name in raw_order)
shuffled_order=list(raw_order)
random.Random(3407).shuffle(shuffled_order)
scene_root=base/'scene_datasets/hm3d'
scene_order=os.listdir(scene_root/'train')+os.listdir(scene_root/'val')
scene_ids:dict[str,str]={}
input_files:dict[str,object]={}
for filename in raw_order:
    scene_name=Path(filename).stem
    matches=[scene_id for scene_id in scene_order if scene_name in scene_id]
    assert len(matches)==1
    scene_ids[filename]=matches[0]
    path=test_dir/filename
    input_files[filename]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for suffix in ('.basis.glb','.basis.navmesh','.semantic.glb','.semantic.txt'):
        resource=scene_root/'val'/matches[0]/(scene_name+suffix)
        assert resource.is_file() and resource.stat().st_size>0

all_ids:list[str]=[]
metadata:dict[str,TaskMetadata]={}
ids_by_split:dict[str,list[str]]={str(split):[] for split in range(1,11)}
worker_records:list[dict[str,object]]=[]
for worker_index,(gpu,start,end) in enumerate(((0,0.0,0.5),(2,0.5,1.0))):
    names=shuffled_order[int(start*36):int(end*36)]
    assert len(names)==18
    worker_records.append({'worker_id':f'gpu{gpu}','physical_gpu':gpu,'cuda_visible_devices':str(gpu),'start_ratio':start,'end_ratio':end,'scene_files':names,'scene_ids':[scene_ids[name] for name in names],'expected_ids_by_split':{str(split):[] for split in range(1,11)},'subtask_counts_by_split':{str(split):0 for split in range(1,11)},'logger_suffix_by_split':{str(split):f'{start}_{end}_{split}' for split in range(1,11)}})
for split in range(1,11):
    for worker in worker_records:
        names=worker['scene_files']
        assert isinstance(names,list)
        worker_ids:list[str]=[]
        for filename_raw in names:
            filename=string(filename_raw)
            raw=mapping(json.loads((test_dir/filename).read_text()))
            episodes=sequence(raw['episodes'])
            assert len(episodes)==10
            episode=mapping(episodes[split-1])
            goals=mapping(raw['goals'])
            with contextlib.redirect_stdout(io.StringIO()):
                types_raw,goals_raw=prepare_goatbench_navigation_goals(Path(filename).stem,episode,goals)
            types=sequence(types_raw)
            selected_goals=sequence(goals_raw)
            assert len(types)==len(sequence(episode['tasks']))==len(selected_goals)
            episode_id=episode['episode_id']
            assert isinstance(episode_id,int) and not isinstance(episode_id,bool)
            for subtask_index,(task_type_raw,goal_raw) in enumerate(zip(types,selected_goals)):
                task_type=string(task_type_raw)
                assert task_type in {'object','description','image'}
                goal=sequence(goal_raw)
                assert goal
                answer=string(mapping(goal[0])['object_category'])
                subtask_id=f'{scene_ids[filename]}_{episode_id}_{subtask_index}'
                assert subtask_id not in metadata
                metadata[subtask_id]={'task_type':task_type,'answer':answer,'scene_id':scene_ids[filename],'episode_id':episode_id,'subtask_index':subtask_index,'split':split,'source_filename':filename}
                all_ids.append(subtask_id)
                ids_by_split[str(split)].append(subtask_id)
                worker_ids.append(subtask_id)
        expected=mapping(worker['expected_ids_by_split'])
        counts=mapping(worker['subtask_counts_by_split'])
        expected[str(split)]=worker_ids
        counts[str(split)]=len(worker_ids)
        worker['expected_ids_by_split']=expected
        worker['subtask_counts_by_split']=counts
assert len(all_ids)==len(set(all_ids))==len(metadata)==2669
assert len(ids_by_split['1'])==278
assert set(metadata)==set(all_ids)
for split in range(1,11):
    left=sequence(mapping(worker_records[0]['expected_ids_by_split'])[str(split)])
    right=sequence(mapping(worker_records[1]['expected_ids_by_split'])[str(split)])
    assert not set(left)&set(right)
    assert set(left)|set(right)==set(ids_by_split[str(split)])
assert len(set(worker_records[0]['scene_files'])|set(worker_records[1]['scene_files']))==36

updates={'exp_name':exp_name,'output_parent_dir':str(base/'outputs'),'scene_dataset_config_path':str(scene_root/'hm3d_annotated_basis.scene_dataset_config.json'),'scene_data_path':str(scene_root),'test_data_dir':str(test_dir),'concept_graph_config_path':str(source/'cfg/concept_graph_default.yaml'),'yolo_model_name':str(base/'weights/yolov8x-world.pt'),'sam_model_name':str(base/'weights/sam_l.pt')}
old_lines=upstream_cfg.read_text().splitlines(keepends=True)
new_lines:list[str]=[]
changed:set[str]=set()
for line in old_lines:
    key=line.split(':',1)[0]
    if key in updates:
        new_lines.append(f'{key}: {json.dumps(updates[key])}\n')
        changed.add(key)
    else:new_lines.append(line)
assert changed==set(updates)
assert all(old==new for old,new in zip(old_lines,new_lines) if old.split(':',1)[0] not in updates)
configuration=''.join(new_lines)
for key in ('scene_dataset_config_path','concept_graph_config_path','yolo_model_name','sam_model_name'):assert Path(updates[key]).is_file()
config_path=manifest_dir/'eval_full2669.yaml'
persist(config_path,configuration)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
frozen={'experiment':exp_name,'output_dir':str(output_dir),'config_path':str(config_path),'seed':3407,'test_dir':str(test_dir),'raw_os_listdir_file_order':raw_order,'seeded_shuffled_file_order':shuffled_order,'scene_root_os_listdir_order':scene_order,'source_repository':'https://github.com/mrwangyou/SCOPE','source_commit':commit,'source_config_sha256':hashlib.sha256(upstream_cfg.read_bytes()).hexdigest(),'runtime_config_sha256':hashlib.sha256(configuration.encode()).hexdigest(),'upstream_goal_helper_sha256':hashlib.sha256((source/'src/goatbench_utils.py').read_bytes()).hexdigest(),'input_files':input_files,'modified_config_keys':sorted(updates),'workers':worker_records,'subtask_counts_by_split':{key:len(value) for key,value in ids_by_split.items()},'all_subtasks':len(all_ids),'canonical_firstsplit_subtasks':len(ids_by_split['1']),'backend':'Gemini38Flash; model substitution, does not claim exact GPT-4o paper reproduction','preparation_only':True}
save_json(manifest_dir/'full2669_frozen_manifest.json',frozen)
save_json(manifest_dir/'canonical278_ids.json',ids_by_split['1'])
save_json(manifest_dir/'full2669_ids_by_split.json',ids_by_split)
save_json(output_dir/'expected_subtask_ids.json',all_ids)
save_json(output_dir/'subtask_metadata.json',metadata)
verification={'status':'passed','scenes':36,'episodes':360,'all_subtasks':2669,'canonical_subtasks':278,'worker_firstsplit_counts':[mapping(worker['subtask_counts_by_split'])['1'] for worker in worker_records],'worker_all_subtask_counts':[sum(mapping(worker['subtask_counts_by_split']).values()) for worker in worker_records],'duplicate_ids':0,'task_filter_differences':0,'config_changed_keys':sorted(updates),'upstream_remaining_parameters':'unchanged','evaluation_started':False}
save_json(manifest_dir/'full2669_preparation_verification.json',verification)
print(json.dumps(verification))
PY
