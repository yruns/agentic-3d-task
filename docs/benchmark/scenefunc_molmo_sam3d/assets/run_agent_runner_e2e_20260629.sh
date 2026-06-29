#!/usr/bin/env bash
set -euo pipefail

export NO_COLOR=1
export TERM=dumb

REPO_ROOT="${REPO_ROOT:-/mlx_devbox/users/yueshuhao/playground/repos/agentic-3d-task/.worktrees/scenefunc3d-agent-tools}"
DATASET_ROOT="${DATASET_ROOT:-/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/agent_runner_e2e_20260629}"
SAMPLE_ID="${SAMPLE_ID:-421254::af0b7790-028c-4eed-945b-d90386d4f16b}"
SIDECAR_PYTHON_BIN="${SIDECAR_PYTHON_BIN:-/usr/bin/python}"
RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-/mlx_devbox/users/yueshuhao/miniforge3/envs/conceptgraph/bin/python}"
MOLMO_PROCESSOR_SNAPSHOT="${MOLMO_PROCESSOR_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_CODE_SNAPSHOT="${MOLMO_CODE_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/hub/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_MODEL_PATH="${MOLMO_MODEL_PATH:-${RUN_ROOT}/molmopoint_merged_model}"
SAM_MODEL_PATH="${SAM_MODEL_PATH:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610}"
HF_HOME="${HF_HOME:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home}"
CODEX_HOME="${CODEX_HOME:-${REPO_ROOT}/.codex-home}"
ADAPTER_HEALTH_URL="${ADAPTER_HEALTH_URL:-http://127.0.0.1:8787/health}"

export REPO_ROOT
export DATASET_ROOT
export RUN_ROOT
export SAMPLE_ID
export SIDECAR_PYTHON_BIN
export RUNNER_PYTHON_BIN
export MOLMO_PROCESSOR_SNAPSHOT
export MOLMO_CODE_SNAPSHOT
export MOLMO_MODEL_PATH
export SAM_MODEL_PATH
export HF_HOME
export CODEX_HOME
export ADAPTER_HEALTH_URL
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CODEX_AGENT_PROJECT_ROOT="${REPO_ROOT}"
export CODEX_AGENT_TURN_TIMEOUT_S="${CODEX_AGENT_TURN_TIMEOUT_S:-1800}"
export CODEX_AGENT_MAX_TOOL_CALLS="${CODEX_AGENT_MAX_TOOL_CALLS:-48}"
export CODEX_AGENT_MAX_REPEATED_TOOL_CALLS="${CODEX_AGENT_MAX_REPEATED_TOOL_CALLS:-3}"
export CODEX_AGENT_KEEP_RUN_HOME="${CODEX_AGENT_KEEP_RUN_HOME:-1}"

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

"${RUNNER_PYTHON_BIN}" - <<'PY'
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
sample_id = os.environ["SAMPLE_ID"]
sidecar_python_bin = os.environ["SIDECAR_PYTHON_BIN"]
runner_python_bin = os.environ["RUNNER_PYTHON_BIN"]
molmo_processor_snapshot = Path(os.environ["MOLMO_PROCESSOR_SNAPSHOT"])
molmo_code_snapshot = Path(os.environ["MOLMO_CODE_SNAPSHOT"])
molmo_model_path = Path(os.environ["MOLMO_MODEL_PATH"])
sam_model_path = Path(os.environ["SAM_MODEL_PATH"])
codex_home = Path(os.environ["CODEX_HOME"])
adapter_health_url = os.environ["ADAPTER_HEALTH_URL"]
logs_dir = run_root / "logs"
output_dir = run_root / "agent_outputs"
config_path = run_root / "scenefunc3d_backends.toml"
default_merged_molmo_path = run_root / "molmopoint_merged_model"


def ensure_codex_home() -> None:
    if (codex_home / "config.toml").is_file():
        return
    subprocess.run(
        [
            str(repo_root / "scripts" / "codex_agent" / "bootstrap_codex_home.sh"),
            "--project-root",
            str(repo_root),
            "--codex-home",
            str(codex_home),
        ],
        cwd=repo_root,
        check=True,
    )


def check_adapter_ready() -> None:
    try:
        with urllib.request.urlopen(adapter_health_url, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Codex ModelHub adapter is not reachable. Start it before this "
            f"E2E script: health_url={adapter_health_url}"
        ) from exc
    config = payload.get("config")
    if not isinstance(config, dict) or not bool(config.get("has_upstream_ak")):
        raise RuntimeError(
            "Codex ModelHub adapter is reachable but has no private upstream "
            "credential configured. Set its gitignored .env or upstream TOML."
        )
    print(
        "adapter ready: "
        f"status={payload.get('status')} "
        f"upstream_base_url={config.get('upstream_base_url')} "
        f"toml_upstream_count={config.get('modelhub_toml_upstream_count')}",
        flush=True,
    )


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


def write_backend_config() -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            (
                'molmo_url = "http://127.0.0.1:8711"',
                'sam_url = "http://127.0.0.1:8712"',
                "request_timeout_seconds = 300.0",
                f'artifact_staging_root = "{run_root}"',
                f'allowed_image_roots = ["{dataset_root.parent}"]',
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
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            time.sleep(5.0)
    raise TimeoutError(f"{name} health timed out: last_error={last_error}")


def run_agent_runner() -> None:
    stdout_path = run_root / "runner_stdout.json"
    log_path = logs_dir / "runner.log"
    command = [
        runner_python_bin,
        "-m",
        "codex_agent.scenefunc3d.runner",
        "--dataset-root",
        str(dataset_root),
        "--sample-id",
        sample_id,
        "--backend-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--score",
    ]
    with log_path.open("wb") as log_handle:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=log_handle,
            check=True,
        )
    stdout_path.write_bytes(completed.stdout)
    print(completed.stdout.decode("utf-8"), flush=True)
    print(f"runner stdout: {stdout_path}", flush=True)
    print(f"runner log: {log_path}", flush=True)


def main() -> None:
    ensure_codex_home()
    check_adapter_ready()
    prepare_molmo_model_dir()
    write_backend_config()
    molmo = start_process(
        "molmo_server",
        [
            sidecar_python_bin,
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
            sidecar_python_bin,
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
        run_agent_runner()
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
    try:
        main()
    except RuntimeError as exc:
        print(f"agent runner E2E failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
PY
