#!/usr/bin/env bash
# Verify the first complete episode, then run split 1 and finally splits 2..10.
set -euo pipefail
scope_assets=/home/ysh/Datasets/SCOPE
scope_full=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_full2669
scope_public=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0
scope_gate_python=/home/ysh/miniconda3/envs/conceptgraph/bin/python
read -r scope_project_commit < "$scope_assets/manifests/full_project_commit.txt"
trap 'scope_exit=$?; printf "%s\n" "$scope_exit" > "$scope_assets/logs/full_run.exit"' EXIT

wait_for_exit() {
  local scope_status=$1 scope_session=$2
  while [[ ! -f "$scope_status" ]]; do
    if ! tmux has-session -t "$scope_session" 2>/dev/null; then
      [[ -f "$scope_status" ]] || { printf 'Missing terminal status: %s\n' "$scope_session" >&2; return 2; }
      break
    fi
    sleep 10
  done
  [[ "$(cat "$scope_status")" == 0 ]] || { printf 'Failed session: %s\n' "$scope_session" >&2; return 1; }
}

ingest_and_check() {
  local scope_dir=$1 scope_run_id=$2 scope_expected=$3 scope_commit=$4
  "$scope_gate_python" "$scope_assets/scripts/ingest_goat_bench_run.py" \
    --output-dir "$scope_dir" --run-id "$scope_run_id" \
    --branch feat/realtime-model --commit "$scope_commit" \
    --runtime-commit 46f1e1f78077219c30e35440540e4ef22995194c \
    --model gemini-3.8-flash --expected-subtasks "$scope_expected" \
    --db "$scope_assets/outputs/runs.sqlite" \
    --notes 'Verified distance metrics; model replacement; two disjoint scene shards' \
    > "$scope_dir/ingested_summary.json"
  PYTHONPATH="$scope_assets/scripts" "$scope_gate_python" - "$scope_dir/ingested_summary.json" <<'PY'
from pathlib import Path
import sys
from ingest_goat_bench_run import RunSummary
summary = RunSummary.model_validate_json(Path(sys.argv[1]).read_text())
if not (summary.complete and summary.expected_ids_verified and summary.modality_complete):
    raise SystemExit('Result coverage or modality validation failed')
print(summary.model_dump_json())
PY
}

wait_for_exit "$scope_assets/logs/public_episode.exit" scope-public-ep0-20261005
scope_public_commit=$(
  "$scope_gate_python" - "$scope_public/run_manifest.json" <<'PY'
from pathlib import Path
import json
import sys
print(json.loads(Path(sys.argv[1]).read_text())['head_commit_at_launch'])
PY
)
ingest_and_check "$scope_public" REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0 5 "$scope_public_commit"

for scope_split in {1..10}; do
  for scope_worker in 0 1; do
    scope_session=scope-s${scope_split}-w${scope_worker}-20261005
    scope_status=$scope_assets/logs/full_split${scope_split}_worker${scope_worker}.exit
    [[ ! -f "$scope_status" ]] || { printf 'Refusing stale status: %s\n' "$scope_status" >&2; exit 2; }
    tmux new-session -d -s "$scope_session" \
      "bash $scope_assets/scripts/run_worker.sh $scope_split $scope_worker > $scope_assets/logs/full_split${scope_split}_worker${scope_worker}.log 2>&1"
  done
  for scope_worker in 0 1; do
    wait_for_exit "$scope_assets/logs/full_split${scope_split}_worker${scope_worker}.exit" "scope-s${scope_split}-w${scope_worker}-20261005"
  done
  if [[ "$scope_split" == 1 ]]; then
    scope_canonical=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_canonical278
    mkdir -p "$scope_canonical"
    cp "$scope_full"/*distance*.pkl "$scope_canonical/"
    cp "$scope_full/subtask_metadata.json" "$scope_canonical/subtask_metadata.json"
    cp "$scope_assets/manifests/canonical278_ids.json" "$scope_canonical/expected_subtask_ids.json"
    ingest_and_check "$scope_canonical" REPRO_20261005_SCOPE_Gemini38Flash_canonical278 278 "$scope_project_commit"
  fi
  printf 'Finished split %s\n' "$scope_split"
done
ingest_and_check "$scope_full" REPRO_20261005_SCOPE_Gemini38Flash_full2669 2669 "$scope_project_commit"
