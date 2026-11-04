import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from PIL import Image
from codex_agent.alfred.policy import AlfredDecision, AlfredPolicyRuntime
from codex_agent.config import CodexAgentConfig
from codex_agent.models import CodexTurnRequest

root=Path('/home/ysh/codebase/agentic-3d-task')
output=root/'tmp/alfred_setup/all_command_probe'
(output/'actor').mkdir(parents=True,exist_ok=True)
rgb=output/'synthetic.png';Image.new('RGB',(300,300),'gray').save(rgb)
schema=AlfredDecision.model_json_schema()
commands={
    'motion':{'kind':'motion','motion_action':'RotateLeft','repeat':1},
    'interact':{'kind':'interact','interaction_action':'PickupObject','target':{'type':'object','object_id':0}},
    'navigate_object':{'kind':'navigate','target':{'type':'object','object_id':0}},
    'navigate_cell':{'kind':'navigate','target':{'type':'cell','cell':[2,-1]}},
    'retrieve_object':{'kind':'retrieve','target':{'type':'object','object_id':0}},
    'retrieve_cell':{'kind':'retrieve','target':{'type':'cell','cell':[2,-1]}},
    'stop':{'kind':'stop'},
}
def run(item):
    name,command=item
    expected={'command':command,'rationale':'Protocol test','memory_note':name}
    AlfredDecision.model_validate(expected)
    config=replace(CodexAgentConfig.from_env(),project_root=output/'actor',codex_home=root/'.codex-home',
        model='gpt-5.4-2026-03-05',reasoning_effort='low',keep_run_home=True,sandbox='read_only',turn_timeout_s=120)
    request=CodexTurnRequest(prompt='This is only a JSON protocol test; no simulator or real action is involved. Output exactly this valid JSON: '+json.dumps(expected),
        output_schema=schema,image_paths=(rgb,))
    result=AlfredPolicyRuntime(config).run_turn(request,max_finalization_retries=0)
    actual=json.loads(result.final_response)
    parsed=AlfredDecision.model_validate(actual)
    report={'command_name':name,'expected':expected,'actual':actual,'command_matches':actual.get('command')==command,
            'metadata':result.metadata.as_dict(),'parsed_action':parsed.action}
    (output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(name,report['command_matches'],actual,flush=True)
    return report['command_matches']

(output/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(run,commands.items()))
assert all(results), 'provider changed one or more requested valid commands'
