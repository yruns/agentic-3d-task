#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/SceneFuncVal-CG}"

export NO_COLOR=1
export TERM=dumb
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${DATASET_ROOT}/models/cache/hf_modules}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-${DATASET_ROOT}/models/Molmo-7B-D-0924}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/molmo_sam3d_full_smallest_20260627}"
LOG_PATH="${LOG_PATH:-${DATASET_ROOT}/logs/molmo_sam3d_full_smallest_421254_000050.log}"
SAM_SELECTION="${SAM_SELECTION:-smallest}"
ALLOW_FAILURE="${ALLOW_FAILURE:-0}"
PROMPT="${PROMPT:-Point to the small dark round drawer knob handle on the lower wooden cabinet drawer near the right side of the image. Return exactly one XML point tag with x and y percentage attributes and no other text.}"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")"

EXTRA_ARGS=()
if [[ "${ALLOW_FAILURE}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-failure)
fi

"${PYTHON_BIN}" docs/benchmark/scenefunc_molmo_sam3d/assets/molmo_sam3d_smoke.py \
  --scene-id 421254 \
  --frame-id 000050 \
  --desc-id af0b7790-028c-4eed-945b-d90386d4f16b \
  --molmo-model-id "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --sam-selection "${SAM_SELECTION}" \
  --prompt "${PROMPT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"
