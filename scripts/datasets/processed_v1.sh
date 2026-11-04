#!/usr/bin/env bash
# Source this file before passing the exported paths to the benchmark CLIs.
# It only configures paths; it never launches an evaluation or modifies data.

_agentic3d_processed_v1_paths() {
  local root="${AGENTIC3D_DATASETS_ROOT:-/home/ysh/Datasets}"
  local nr3d="$root/nr3d/processed_v1"
  local openeqa="$root/open-eqa/processed_v1"
  local required

  for required in \
    "$nr3d/scannet" \
    "$openeqa/scannet"; do
    if [[ ! -d "$required" ]]; then
      printf 'Processed dataset directory is missing: %s\n' "$required" >&2
      return 1
    fi
  done
  for required in \
    "$nr3d/folds/full_test_sample_ids.json" \
    "$nr3d/folds/strat600_sample_ids.json" \
    "$openeqa/open-eqa-v0.json" \
    "$openeqa/folds/v5_pilot_fold_300.json"; do
    if [[ ! -f "$required" ]]; then
      printf 'Processed dataset input file is missing: %s\n' "$required" >&2
      return 1
    fi
  done

  export NR3D_DATA_ROOT="$nr3d/scannet"
  export NR3D_FULL_SAMPLE_IDS="$nr3d/folds/full_test_sample_ids.json"
  export NR3D_PILOT_SAMPLE_IDS="$nr3d/folds/strat600_sample_ids.json"
  export NR3D_PACK_NAME="pack_nr3d_v9_catalog_first"
  export OPENEQA_DATA_ROOT="$openeqa/scannet"
  export OPENEQA_QUESTIONS="$openeqa/open-eqa-v0.json"
  export OPENEQA_PILOT_QUESTION_IDS="$openeqa/folds/v5_pilot_fold_300.json"
}

if _agentic3d_processed_v1_paths; then
  unset -f _agentic3d_processed_v1_paths
else
  unset -f _agentic3d_processed_v1_paths
  return 1 2>/dev/null || exit 1
fi
