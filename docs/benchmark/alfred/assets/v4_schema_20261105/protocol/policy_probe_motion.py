from dataclasses import replace
import json
from pathlib import Path
from PIL import Image
from codex_agent.config import CodexAgentConfig
from codex_agent.cli.runtime_preflight import preflight_codex_runtime
from codex_agent.alfred.policy import AlfredPolicyRuntime, AlfredPolicyTask
from codex_agent.alfred.dataset import AlfredInstruction

root = Path('/home/ysh/codebase/agentic-3d-task')
probe = root / 'tmp/alfred_setup/policy_probe_motion'
(probe / 'actor').mkdir(parents=True, exist_ok=True)
image = probe / 'synthetic.png'
Image.new('RGB', (300, 300), color=(100, 120, 140)).save(image)
config = replace(CodexAgentConfig.from_env(), project_root=probe / 'actor',
    codex_home=root / '.codex-home', model='gpt-5.4-2026-03-05',
    reasoning_effort='low', max_tool_calls=1, turn_timeout_s=120,
    keep_run_home=True, sandbox='read_only')
preflight_codex_runtime(config)
runtime = AlfredPolicyRuntime(config)
task = AlfredPolicyTask(AlfredInstruction(
    'This is a structured output protocol test, not a navigation task. Output exactly a motion command: command.kind is motion, command.action is RotateLeft, command.repeat is 1. rationale is Protocol test and memory_note is Motion. Do not use Stop, Navigate or Retrieve. There is no simulator being controlled.', ()),
    {'synthetic': True}, (image,))
result = runtime.execute(task)
report = {'decision': result.outcome.model_dump(mode='json'), 'metadata': result.turn.metadata.as_dict(),
    'kind': 'unscored synthetic SDK structured-output and tool-isolation check',
    'source_status': 'uncommitted development code; not a benchmark evaluation'}
(probe / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
