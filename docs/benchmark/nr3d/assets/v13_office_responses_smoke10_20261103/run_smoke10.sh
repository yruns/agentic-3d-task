#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/ysh/codebase/agentic-3d-task
BASE="$ROOT/tmp/nr3d_office_smoke_20261103"
RUN_ID=v13_office_responses_smoke10_20261103
OUT="$ROOT/tmp/nr3d_eval_$RUN_ID"
exec > >(tee "$BASE/preflight/run-console.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$BASE/status/run.exitcode"; printf "NR3D_RUN_EXIT=%s\n" "$rc"' EXIT
cd "$ROOT"
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
source scripts/datasets/processed_v1.sh
set -u
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
 printf 'Tracked working tree must be clean before evaluation.\n' >&2; exit 1
fi
export CODEX_AGENT_KEEP_RUN_HOME=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID="$RUN_ID"
export PYTHONUNBUFFERED=1
python -I - <<'PY'
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/ysh/codebase/agentic-3d-task')
base=root/'tmp/nr3d_office_smoke_20261103'
fold=Path('/home/ysh/Datasets/nr3d/processed_v1/folds/strat10_smoke_sample_ids.json')
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
d={'run_id':'v13_office_responses_smoke10_20261103','branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'head_commit_at_launch':sha,'runtime_code_commit':sha,'worktree_drift':False,'started_at_unix':time.time(),'sample_ids':str(fold),'sample_ids_sha256':hashlib.sha256(fold.read_bytes()).hexdigest(),'fold_n':len(json.loads(fold.read_text())),'tools_enabled':False,'workers':2,'model':'gpt-5.4-2026-03-05','reasoning_effort':'medium','reasoning_summary':'auto','upstream_env':'office','responses_path':'/api/modelhub/online/unified/v1/responses','sdk_version':'0.163.0','cli_version':'0.163.0','transport_setup':'user ~/.zshrc inherited by adapter; no manual proxy override','comparison_scope':'Frozen diagnostic smoke10, not decision-grade strat600','compact_endpoint_probe_http':404}
(base/'preflight/launch.json').write_text(json.dumps(d,indent=2)+'\n')
print('LAUNCH_CODE_SHA:',sha,flush=True)
PY
printf 'NR3D_RUN_START: %s\n' "$RUN_ID"
python -m codex_agent.cli.run_nr3d \
 --sample-ids "$NR3D_DATA_ROOT/../folds/strat10_smoke_sample_ids.json" \
 --data-root "$NR3D_DATA_ROOT" --pack-name "$NR3D_PACK_NAME" \
 --output-dir "$OUT" --workers 2 --sample-retries 0 \
 --reasoning-effort medium --reasoning-summary auto \
 --turn-timeout 180 --max-tool-calls 12 --max-repeated-tool-calls 3
printf 'NR3D_RUN_COMPLETE: %s\n' "$OUT"
