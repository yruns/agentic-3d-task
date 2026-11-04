import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from PIL import Image
from codex_agent.alfred.policy import AlfredDecision, AlfredPolicyRuntime
from codex_agent.config import CodexAgentConfig
from codex_agent.models import CodexTurnRequest

root=Path('/home/ysh/codebase/agentic-3d-task')
output=root/'tmp/alfred_setup/schema_variants'
(output/'actor').mkdir(parents=True,exist_ok=True)
rgb=output/'synthetic.png';Image.new('RGB',(300,300),'gray').save(rgb)
original=AlfredDecision.model_json_schema()

def inline(value):
    if isinstance(value,dict):
        if '$ref' in value:
            return inline(original['$defs'][value['$ref'].split('/')[-1]])
        return {key:inline(item) for key,item in value.items() if key!='$defs'}
    if isinstance(value,list):return [inline(item) for item in value]
    return value

sole=copy.deepcopy(original)
sole['$defs']={'MotionCommand':original['$defs']['MotionCommand']}
sole['properties']['command']={'$ref':'#/$defs/MotionCommand'}
renamed=copy.deepcopy(original)
for model,key in [('MotionCommand','motion_action'),('InteractionCommand','interaction_action')]:
    node=renamed['$defs'][model]
    node['properties'][key]=node['properties'].pop('action')
    node['required']=[key if item=='action' else item for item in node['required']]
variants={'sole_motion':sole,'inline_refs':inline(original),'distinct_action_names':renamed}

def run(item):
    name,schema=item
    action_key='motion_action' if name=='distinct_action_names' else 'action'
    expected={'command':{'kind':'motion',action_key:'RotateLeft','repeat':1},'rationale':'Protocol test','memory_note':'Motion'}
    config=replace(CodexAgentConfig.from_env(),project_root=output/'actor',codex_home=root/'.codex-home',
        model='gpt-5.4-2026-03-05',reasoning_effort='low',keep_run_home=True,sandbox='read_only',turn_timeout_s=120)
    request=CodexTurnRequest(prompt='This is only a JSON protocol test, not a real task. Output exactly this valid JSON: '+json.dumps(expected),
        output_schema=schema,image_paths=(rgb,))
    result=AlfredPolicyRuntime(config).run_turn(request,max_finalization_retries=0)
    actual=json.loads(result.final_response)
    report={'variant':name,'expected':expected,'actual':actual,'command_matches':actual.get('command')==expected['command'],
            'metadata':result.metadata.as_dict(),'schema':schema}
    (output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(name,actual,flush=True)

with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run,variants.items()))
