#!/usr/bin/env bash
# Verify the first complete episode, then run split 1 and finally splits 2..10.
set -euo pipefail
scope_assets=/home/ysh/Datasets/SCOPE
scope_full=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_full2669
scope_public=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0
scope_gate_python=/home/ysh/miniconda3/envs/conceptgraph/bin/python
scope_plan=$scope_assets/manifests/storage_plan_parallel_v2.json
scope_lineage_source_map=$scope_assets/manifests/frozen_result_lineage.json
scope_original_runtime=46f1e1f78077219c30e35440540e4ef22995194c
read -r scope_controller_launch_commit < "$scope_assets/manifests/lineage_controller_commit.txt"
exec 9>"$scope_assets/logs/full_controller.lock"
flock -n 9 || { printf 'Another SCOPE controller owns the run\n' >&2; exit 2; }
trap 'scope_exit=$?; printf "%s\n" "$scope_exit" > "$scope_assets/logs/full_run.exit"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

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

# A finished primary shard is immutable. Publish through a temporary file so
# observers never read a partially copied pickle; refuse conflicting history.
publish_shard() {
  local scope_source=$1 scope_target=$2
  [[ -s "$scope_source" ]] || { printf 'Missing primary shard: %s\n' "$scope_source" >&2; return 2; }
  if [[ "$scope_source" == "$scope_target" ]]; then
    return
  fi
  if [[ -f "$scope_target" ]]; then
    cmp -s "$scope_source" "$scope_target" || { printf 'Conflicting primary shard: %s\n' "$scope_target" >&2; return 2; }
  else
    cp "$scope_source" "$scope_target.pending.$$"
    mv "$scope_target.pending.$$" "$scope_target"
  fi
}

collect_worker() {
  local scope_split=$1 scope_worker=$2
  local scope_source scope_suffix scope_gate scope_expected scope_metric
  read -r scope_source scope_suffix scope_gate scope_expected < <(
    "$scope_gate_python" - "$scope_plan" "$scope_split" "$scope_worker" <<'PY'
from pathlib import Path
import json
import sys
plan = json.loads(Path(sys.argv[1]).read_text())
stages = [stage for stage in plan['stages'] if stage['split'] == int(sys.argv[2]) and stage['worker'] == int(sys.argv[3])]
if len(stages) != 1:
    raise SystemExit('Expected exactly one frozen worker stage')
stage = stages[0]
print(stage['source_output_dir'], stage['logger_suffix'], stage['gate_dir'], stage['expected_subtasks'])
PY
  )
  for scope_metric in success_by_distance spl_by_distance; do
    publish_shard "$scope_source/${scope_metric}_${scope_suffix}.pkl" "$scope_gate/${scope_metric}_${scope_suffix}.pkl"
  done
  ingest_and_check "$scope_gate" "REPRO_20261005_SCOPE_Gemini38Flash_split${scope_split}_worker${scope_worker}" "$scope_expected" "$scope_controller_launch_commit"
  for scope_metric in success_by_distance spl_by_distance; do
    publish_shard "$scope_gate/${scope_metric}_${scope_suffix}.pkl" "$scope_full/${scope_metric}_${scope_suffix}.pkl"
  done
}

ingest_and_check() {
  local scope_dir=$1 scope_run_id=$2 scope_expected=$3 scope_commit=$4
  local scope_runtime=${5:-} scope_notes
  if [[ -z "$scope_runtime" ]]; then
    if ! scope_runtime=$(
      "$scope_gate_python" "$scope_assets/scripts/runtime_lineage.py" \
        --output-dir "$scope_dir" --source-map "$scope_lineage_source_map"
    ); then
      printf 'Runtime lineage validation failed: %s\n' "$scope_run_id" >&2
      return 1
    fi
    [[ -n "$scope_runtime" ]] || { printf 'Empty runtime lineage label: %s\n' "$scope_run_id" >&2; return 2; }
    scope_notes='Verified distance metrics; model replacement; runtime_lineage.json records per-ID sources; commit archives controller source'
  else
    scope_notes='Verified public episode; model replacement; original public launch head and runtime retained'
  fi
  "$scope_gate_python" "$scope_assets/scripts/ingest_goat_bench_run.py" \
    --output-dir "$scope_dir" --run-id "$scope_run_id" \
    --branch feat/realtime-model --commit "$scope_commit" \
    --runtime-commit "$scope_runtime" \
    --model gemini-3.8-flash --expected-subtasks "$scope_expected" \
    --db "$scope_assets/outputs/runs.sqlite" \
    --notes "$scope_notes" \
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

launch_split() {
  local scope_split=$1 scope_worker scope_session scope_status
  for scope_worker in 0 1; do
    scope_session=scope-s${scope_split}-w${scope_worker}-20261005
    scope_status=$scope_assets/logs/full_split${scope_split}_worker${scope_worker}.exit
    if [[ -f "$scope_status" ]]; then
      printf 'Reusing terminal worker: %s\n' "$scope_session"
    elif tmux has-session -t "$scope_session" 2>/dev/null; then
      printf 'Waiting for existing worker: %s\n' "$scope_session"
    else
      tmux new-session -d -s "$scope_session" \
        "bash $scope_assets/scripts/run_worker.sh $scope_split $scope_worker > $scope_assets/logs/full_split${scope_split}_worker${scope_worker}.log 2>&1"
      printf 'Started worker: %s\n' "$scope_session"
    fi
  done
}

complete_split() {
  local scope_split=$1 scope_worker scope_canonical scope_suffix scope_metric
  for scope_worker in 0 1; do
    wait_for_exit "$scope_assets/logs/full_split${scope_split}_worker${scope_worker}.exit" "scope-s${scope_split}-w${scope_worker}-20261005"
    collect_worker "$scope_split" "$scope_worker"
  done
  if [[ "$scope_split" == 1 ]]; then
    scope_canonical=$scope_assets/outputs/REPRO_20261005_SCOPE_Gemini38Flash_canonical278
    mkdir -p "$scope_canonical"
    for scope_suffix in 0.0_0.5_1 0.5_1.0_1; do
      for scope_metric in success_by_distance spl_by_distance; do
        publish_shard "$scope_full/${scope_metric}_${scope_suffix}.pkl" "$scope_canonical/${scope_metric}_${scope_suffix}.pkl"
      done
    done
    cp "$scope_full/subtask_metadata.json" "$scope_canonical/subtask_metadata.json"
    cp "$scope_assets/manifests/canonical278_ids.json" "$scope_canonical/expected_subtask_ids.json"
    ingest_and_check "$scope_canonical" REPRO_20261005_SCOPE_Gemini38Flash_canonical278 278 "$scope_controller_launch_commit"
  fi
  printf 'Finished split %s\n' "$scope_split"
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
ingest_and_check "$scope_public" REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0 5 "$scope_public_commit" "$scope_original_runtime"

# Retain the two active first-episode workers. Subsequent batches run three
# independent episodes per scene, with one process per physical GPU and output.
launch_split 1
complete_split 1
for scope_batch_start in 2 5 8; do
  for scope_split in "$scope_batch_start" "$((scope_batch_start + 1))" "$((scope_batch_start + 2))"; do
    launch_split "$scope_split"
  done
  for scope_split in "$scope_batch_start" "$((scope_batch_start + 1))" "$((scope_batch_start + 2))"; do
    complete_split "$scope_split"
  done
done
ingest_and_check "$scope_full" REPRO_20261005_SCOPE_Gemini38Flash_full2669 2669 "$scope_controller_launch_commit"
