#!/usr/bin/env bash
set -euo pipefail

export NO_COLOR=1
export TERM=dumb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/SceneFuncVal-CG}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/scenefunc3d/sidecar_tool_smoke_20260629}"
MOLMO_PROCESSOR_SNAPSHOT="${MOLMO_PROCESSOR_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_CODE_SNAPSHOT="${MOLMO_CODE_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/hub/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_MODEL_PATH="${MOLMO_MODEL_PATH:-${RUN_ROOT}/molmopoint_merged_model}"
SAM_MODEL_PATH="${SAM_MODEL_PATH:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610}"
HF_HOME="${HF_HOME:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SIDE_CAR_DEPS="${INSTALL_SIDE_CAR_DEPS:-0}"
export REPO_ROOT
export DATASET_ROOT
export RUN_ROOT
export MOLMO_PROCESSOR_SNAPSHOT
export MOLMO_CODE_SNAPSHOT
export MOLMO_MODEL_PATH
export SAM_MODEL_PATH
export PYTHON_BIN
export INSTALL_SIDE_CAR_DEPS

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

if [[ "${INSTALL_SIDE_CAR_DEPS}" == "1" ]]; then
  mkdir -p "${RUN_ROOT}/python_deps"
  "${PYTHON_BIN}" -m pip install --target "${RUN_ROOT}/python_deps" --upgrade \
    "transformers==4.57.1" \
    "accelerate>=1.8.0" \
    "huggingface_hub>=0.36.0,<1.0" \
    "hf_transfer>=0.1.9" \
    "hf_xet>=1.1.0" \
    "safetensors>=0.5.3" \
    "pydantic==2.11.7" \
    "tomli>=2.0; python_version<'3.11'" \
    "pillow>=10.0.0,<12.0"
  export PYTHONPATH="${RUN_ROOT}/python_deps:${REPO_ROOT}/src"
else
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import importlib
from importlib import metadata

required_modules = ("torch", "transformers", "pydantic", "PIL", "safetensors")
missing_modules = [
    module_name
    for module_name in required_modules
    if importlib.util.find_spec(module_name) is None
]
if missing_modules:
    missing = ", ".join(missing_modules)
    raise RuntimeError(
        "missing sidecar dependency modules; rerun with "
        f"INSTALL_SIDE_CAR_DEPS=1: {missing}"
    )

print(
    "sidecar dependency check: "
    f"torch={metadata.version('torch')} "
    f"transformers={metadata.version('transformers')} "
    f"pydantic={metadata.version('pydantic')} "
    f"pillow={metadata.version('pillow')} "
    f"safetensors={metadata.version('safetensors')}",
    flush=True,
)
PY
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


repo_root = Path(os.environ["REPO_ROOT"])
dataset_root = Path(os.environ["DATASET_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
molmo_model_path = Path(os.environ["MOLMO_MODEL_PATH"])
molmo_processor_snapshot = Path(os.environ["MOLMO_PROCESSOR_SNAPSHOT"])
molmo_code_snapshot = Path(os.environ["MOLMO_CODE_SNAPSHOT"])
sam_model_path = Path(os.environ["SAM_MODEL_PATH"])
python_bin = os.environ["PYTHON_BIN"]
scene_root = dataset_root / "421254"
image_path = scene_root / "raw" / "000050-rgb.jpg"
depth_path = scene_root / "raw" / "000050-depth.png"
intrinsics_path = scene_root / "raw" / "000050-intrinsic.txt"
pose_path = scene_root / "raw" / "000050.txt"
logs_dir = run_root / "logs"
out_dir = run_root / "tool_outputs"
config_path = run_root / "scenefunc3d_backends.toml"
default_merged_molmo_path = run_root / "molmopoint_merged_model"


def prepare_molmo_model_dir() -> None:
    if molmo_model_path != default_merged_molmo_path:
        return
    if not molmo_processor_snapshot.is_dir():
        raise RuntimeError(
            f"missing Molmo processor snapshot: {molmo_processor_snapshot}"
        )
    if not molmo_code_snapshot.is_dir():
        raise RuntimeError(f"missing Molmo code snapshot: {molmo_code_snapshot}")
    molmo_model_path.mkdir(parents=True, exist_ok=True)
    for snapshot in (molmo_code_snapshot, molmo_processor_snapshot):
        for source_path in snapshot.iterdir():
            if not source_path.is_file():
                continue
            target_path = molmo_model_path / source_path.name
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            target_path.symlink_to(source_path.resolve())


def write_config() -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            (
                'molmo_url = "http://127.0.0.1:8711"',
                'sam_url = "http://127.0.0.1:8712"',
                "request_timeout_seconds = 300.0",
                f'allowed_image_roots = ["{dataset_root.parent}", "{run_root}"]',
                f'allowed_output_roots = ["{run_root}"]',
                "",
            )
        ),
        encoding="utf-8",
    )


def start_process(name: str, args: list[str]) -> subprocess.Popen[bytes]:
    log_path = logs_dir / f"{name}.log"
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        args,
        cwd=repo_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    print(f"started {name}: pid={process.pid} log={log_path}", flush=True)
    return process


def wait_health(name: str, url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 900.0
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited before health: code={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            print(f"{name} health: {payload}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - diagnostic smoke harness.
            last_error = f"{exc.__class__.__name__}: {exc}"
            time.sleep(5.0)
    raise TimeoutError(f"{name} health timed out: last_error={last_error}")


def run_tool(tool_name: str, args: dict[str, object], output_file: Path) -> dict[str, object]:
    command = [
        python_bin,
        "-m",
        "codex_agent.scenefunc3d.tools",
        tool_name,
        "--scene-root",
        str(scene_root),
        "--backend-config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--args",
        json.dumps(args),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    output_file.write_text(completed.stdout, encoding="utf-8")
    payload = json.loads(completed.stdout)
    print(f"{tool_name} payload written: {output_file}", flush=True)
    if "error" in payload:
        raise RuntimeError(f"{tool_name} failed: {payload['error']}")
    return payload


def choose_sam_candidate(sam_payload: dict[str, object]) -> dict[str, object]:
    raw_candidates = sam_payload["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RuntimeError("SAM payload has no candidates")
    candidates = [candidate for candidate in raw_candidates if isinstance(candidate, dict)]
    if not candidates:
        raise RuntimeError("SAM payload candidates are malformed")
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate["coverage_percent"]),
            -float(candidate["score"]),
            str(candidate["candidate_id"]),
        ),
    )


def main() -> None:
    prepare_molmo_model_dir()
    write_config()
    molmo = start_process(
        "molmo_server",
        [
            python_bin,
            "-u",
            "-m",
            "codex_agent.scenefunc3d.servers.molmo_point_server",
            "--port",
            "8711",
            "--model-name",
            "MolmoPoint-8B",
            "--model-path",
            str(molmo_model_path),
            "--device",
            "cuda:0",
        ],
    )
    sam = start_process(
        "sam_server",
        [
            python_bin,
            "-u",
            "-m",
            "codex_agent.scenefunc3d.servers.sam2_mask_server",
            "--port",
            "8712",
            "--model-name",
            "SAM2.1-Hiera-L",
            "--backend",
            "transformers",
            "--model-path",
            str(sam_model_path),
            "--staging-root",
            str(run_root),
            "--device",
            "cuda:0",
        ],
    )
    try:
        wait_health("molmo", "http://127.0.0.1:8711/health", molmo)
        wait_health("sam", "http://127.0.0.1:8712/health", sam)
        molmo_payload = run_tool(
            "molmo_point",
            {
                "frame_id": "000050",
                "image_path": str(image_path),
                "prompt": (
                    "Point to the small dark round knob handle on the bottom "
                    "drawer of the cabinet to the left of the TV."
                ),
                "image_width": 1440,
                "image_height": 1920,
            },
            run_root / "molmo_point.json",
        )
        sam_payload = run_tool(
            "sam_mask",
            {
                "frame_id": "000050",
                "image_path": str(image_path),
                "points": molmo_payload["points"],
            },
            run_root / "sam_mask.json",
        )
        candidate = choose_sam_candidate(sam_payload)
        lift_payload = run_tool(
            "lift_mask_to_3d",
            {
                "frame_id": "000050",
                "candidate_id": candidate["candidate_id"],
                "mask_path": candidate["mask_npz_path"],
                "depth_path": str(depth_path),
                "intrinsics_path": str(intrinsics_path),
                "pose_path": str(pose_path),
            },
            run_root / "lift_mask_to_3d.json",
        )
        inspect_payload = run_tool(
            "inspect_mask_artifact",
            {
                "mask_npz_path": lift_payload["mask_npz_path"],
                "mask_ply_path": lift_payload["mask_ply_path"],
                "overlay_paths": [lift_payload["overlay_path"]],
            },
            run_root / "inspect_mask_artifact.json",
        )
        fragment_id = f"000050_{candidate['candidate_id']}"
        suggest_payload = run_tool(
            "suggest_additional_views",
            {
                "seed_fragment_id": fragment_id,
                "accepted_frame_id": "000050",
                "seed_mask_npz_path": lift_payload["mask_npz_path"],
                "seed_mask_ply_path": lift_payload["mask_ply_path"],
                "seed_lift_overlay_path": lift_payload["overlay_path"],
                "min_seed_point_count": 1,
                "k": 3,
            },
            run_root / "suggest_additional_views.json",
        )
        fuse_payload = run_tool(
            "fuse_accepted_masks",
            {
                "fragments": [
                    {
                        "fragment_id": fragment_id,
                        "frame_id": "000050",
                        "mask_npz_path": lift_payload["mask_npz_path"],
                        "mask_ply_path": lift_payload["mask_ply_path"],
                        "approval_actions": [
                            "select_evidence",
                            "propose_molmo_point",
                            "approve_molmo_point",
                            "propose_sam_candidates",
                            "approve_sam_candidate",
                            "create_first_lift",
                            "approve_first_lift",
                        ],
                        "review_artifacts": {
                            "molmo_raw_text_path": molmo_payload["raw_text_path"],
                            "molmo_overlay_path": molmo_payload["overlay_path"],
                            "sam_contact_sheet_path": sam_payload["contact_sheet_path"],
                            "sam_candidate_overlay_path": candidate["overlay_path"],
                            "lift_overlay_path": lift_payload["overlay_path"],
                        },
                    }
                ],
                "multi_view_decision": {
                    "seed_fragment_id": fragment_id,
                    "action": suggest_payload["expansion_recommendation"],
                    "reason": suggest_payload["expansion_reason"],
                    "suggested_frame_ids": [
                        view["frame_id"] for view in suggest_payload["views"]
                    ],
                },
            },
            run_root / "fuse_accepted_masks.json",
        )
        summary = {
            "status": "ok",
            "run_root": str(run_root),
            "molmo_points": molmo_payload["points"],
            "sam_candidate_count": len(sam_payload["candidates"]),
            "selected_candidate": candidate,
            "lift": lift_payload,
            "inspect": inspect_payload,
            "suggest": suggest_payload,
            "fuse": fuse_payload,
        }
        (run_root / "sidecar_tool_smoke_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        for process in (sam, molmo):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30.0)


if __name__ == "__main__":
    main()
PY
