#!/usr/bin/env bash
set -euo pipefail

export NO_COLOR=1
export TERM=dumb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/SceneFuncVal-CG}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/sam_transformers_debug_20260629}"
SAM_MODEL_PATH="${SAM_MODEL_PATH:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610}"
HF_HOME="${HF_HOME:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export REPO_ROOT
export DATASET_ROOT
export RUN_ROOT
export SAM_MODEL_PATH
export PYTHON_BIN
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from codex_agent.scenefunc3d.servers.sam2_mask_server import TransformersSam2Runner
from codex_agent.scenefunc3d.servers.schemas import SamMaskRequest, SamPointPrompt


def main() -> None:
    dataset_root = Path(os.environ["DATASET_ROOT"])
    run_root = Path(os.environ["RUN_ROOT"])
    sam_model_path = Path(os.environ["SAM_MODEL_PATH"])
    scene_root = dataset_root / "421254"
    image_path = scene_root / "raw" / "000050-rgb.jpg"
    staging_dir = run_root / "sam" / "000050" / "candidates"
    runner = TransformersSam2Runner(
        model_name="SAM2.1-Hiera-L",
        model_reference=str(sam_model_path),
        staging_root=run_root,
        device="cuda:0",
    )
    request = SamMaskRequest(
        request_id="000050_sam_debug",
        image_path=image_path,
        points=(
            SamPointPrompt(
                x_px=517.1142857142856,
                y_px=925.7704918032787,
                label="small dark round knob handle",
                source="retry5_molmo_point",
            ),
        ),
        staging_dir=staging_dir,
    )
    try:
        candidates = runner.masks(request)
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostic boundary.
        traceback.print_exc()
        (run_root / "sam_debug_error.json").write_text(
            json.dumps(
                {
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    (run_root / "sam_debug_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
PY
