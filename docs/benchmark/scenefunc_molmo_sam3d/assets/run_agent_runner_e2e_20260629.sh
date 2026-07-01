#!/usr/bin/env bash
set -euo pipefail

export NO_COLOR=1
export TERM=dumb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/SceneFuncVal-CG}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/agent_runner_e2e_20260629}"
SAMPLE_ID_RAW="${SAMPLE_ID-}"
SAMPLE_ID_WAS_SET="0"
if [[ "${SAMPLE_ID+x}" == "x" ]]; then
  SAMPLE_ID_WAS_SET="1"
fi
SAMPLE_ID="${SAMPLE_ID:-421254::af0b7790-028c-4eed-945b-d90386d4f16b}"
SAMPLE_IDS_PATH="${SAMPLE_IDS_PATH:-}"
ALL_SAMPLES="${ALL_SAMPLES:-0}"
SIDECAR_PYTHON_BIN="${SIDECAR_PYTHON_BIN:-python3}"
RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-python3}"
USE_CODEX_AUTH="${USE_CODEX_AUTH:-0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
HOST_UNAME="${HOST_UNAME:-$(uname -s)}"
SIDECAR_MODE="${SIDECAR_MODE:-remote}"
SCENEFUNC3D_SIDECAR_BASE_URL="${SCENEFUNC3D_SIDECAR_BASE_URL:-https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws54367p10640t1782831706e9geojby}"
SCENEFUNC3D_SIDECAR_HEADERS_PATH="${SCENEFUNC3D_SIDECAR_HEADERS_PATH:-${REPO_ROOT}/configs/scenefunc3d_sidecar_headers.toml}"
MOLMO_PROCESSOR_SNAPSHOT="${MOLMO_PROCESSOR_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_CODE_SNAPSHOT="${MOLMO_CODE_SNAPSHOT:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/hub/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba}"
MOLMO_MODEL_PATH="${MOLMO_MODEL_PATH:-${RUN_ROOT}/molmopoint_merged_model}"
SAM_MODEL_PATH="${SAM_MODEL_PATH:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home/transformers/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610}"
HF_HOME="${HF_HOME:-${DATASET_ROOT}/molmopoint_sam21_20260628/hf_home}"
if [[ "${USE_CODEX_AUTH}" == "1" ]]; then
  CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
  CODEX_AGENT_MODEL="${CODEX_AGENT_MODEL:-gpt-5.5}"
  CODEX_AGENT_MODEL_PROVIDER="openai"
  CODEX_AGENT_COPY_AUTH="1"
  CODEX_AGENT_ENABLE_PREFIX_CACHE="0"
  CODEX_AGENT_KEEP_RUN_HOME="0"
else
  CODEX_HOME="${CODEX_HOME:-${REPO_ROOT}/.codex-home}"
fi
DEFAULT_START_ADAPTER="0"
if [[ "${HOST_UNAME}" == "Linux" ]]; then
  DEFAULT_START_ADAPTER="1"
fi
START_ADAPTER="${START_ADAPTER:-${DEFAULT_START_ADAPTER}}"
ADAPTER_PYTHON_BIN="${ADAPTER_PYTHON_BIN:-python3}"
ADAPTER_HOST="${ADAPTER_HOST:-127.0.0.1}"
ADAPTER_PORT="${ADAPTER_PORT:-8787}"
ADAPTER_ENV_PATH="${ADAPTER_ENV_PATH:-${REPO_ROOT}/codex_modelhub_adapter/.env}"
ADAPTER_UPSTREAMS_TOML_PATH="${ADAPTER_UPSTREAMS_TOML_PATH:-${REPO_ROOT}/codex_modelhub_adapter/.modelhub_upstreams.toml}"
ADAPTER_HEALTH_URL="${ADAPTER_HEALTH_URL:-http://${ADAPTER_HOST}:${ADAPTER_PORT}/health}"

export REPO_ROOT
export DATASET_ROOT
export RUN_ROOT
export SAMPLE_ID_RAW
export SAMPLE_ID
export SAMPLE_ID_WAS_SET
export SAMPLE_IDS_PATH
export ALL_SAMPLES
export SIDECAR_PYTHON_BIN
export RUNNER_PYTHON_BIN
export USE_CODEX_AUTH
export PRECHECK_ONLY
export HOST_UNAME
export SIDECAR_MODE
export SCENEFUNC3D_SIDECAR_BASE_URL
export SCENEFUNC3D_SIDECAR_HEADERS_PATH
export MOLMO_PROCESSOR_SNAPSHOT
export MOLMO_CODE_SNAPSHOT
export MOLMO_MODEL_PATH
export SAM_MODEL_PATH
export HF_HOME
export CODEX_HOME
export START_ADAPTER
export ADAPTER_PYTHON_BIN
export ADAPTER_HOST
export ADAPTER_PORT
export ADAPTER_ENV_PATH
export ADAPTER_UPSTREAMS_TOML_PATH
export ADAPTER_HEALTH_URL
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CODEX_AGENT_PROJECT_ROOT="${REPO_ROOT}"
export CODEX_AGENT_MODEL="${CODEX_AGENT_MODEL:-gpt-5.4-2026-03-05}"
export CODEX_AGENT_MODEL_PROVIDER="${CODEX_AGENT_MODEL_PROVIDER:-modelhub_adapter}"
export CODEX_AGENT_COPY_AUTH="${CODEX_AGENT_COPY_AUTH:-0}"
export CODEX_AGENT_ENABLE_PREFIX_CACHE="${CODEX_AGENT_ENABLE_PREFIX_CACHE:-1}"
export CODEX_AGENT_TURN_TIMEOUT_S="${CODEX_AGENT_TURN_TIMEOUT_S:-900}"
export CODEX_AGENT_MAX_TOOL_CALLS="${CODEX_AGENT_MAX_TOOL_CALLS:-128}"
export CODEX_AGENT_MAX_REPEATED_TOOL_CALLS="${CODEX_AGENT_MAX_REPEATED_TOOL_CALLS:-6}"
export CODEX_AGENT_KEEP_RUN_HOME="${CODEX_AGENT_KEEP_RUN_HOME:-1}"

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_ROOT}"

"${RUNNER_PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.runner import check_sidecar_health
from codex_agent.scenefunc3d.sample import SceneFunc3dSampleId


repo_root = Path(os.environ["REPO_ROOT"])
dataset_root = Path(os.environ["DATASET_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
sample_id_raw = os.environ["SAMPLE_ID_RAW"]
sample_id = os.environ["SAMPLE_ID"]
sample_id_was_set = os.environ["SAMPLE_ID_WAS_SET"] == "1"
sample_ids_path = os.environ["SAMPLE_IDS_PATH"]
all_samples_value = os.environ["ALL_SAMPLES"]
sidecar_python_bin = os.environ["SIDECAR_PYTHON_BIN"]
runner_python_bin = os.environ["RUNNER_PYTHON_BIN"]
sidecar_mode = os.environ["SIDECAR_MODE"]
sidecar_base_url = os.environ["SCENEFUNC3D_SIDECAR_BASE_URL"].rstrip("/")
sidecar_headers_path = Path(os.environ["SCENEFUNC3D_SIDECAR_HEADERS_PATH"])
molmo_processor_snapshot = Path(os.environ["MOLMO_PROCESSOR_SNAPSHOT"])
molmo_code_snapshot = Path(os.environ["MOLMO_CODE_SNAPSHOT"])
molmo_model_path = Path(os.environ["MOLMO_MODEL_PATH"])
sam_model_path = Path(os.environ["SAM_MODEL_PATH"])
codex_home = Path(os.environ["CODEX_HOME"])
adapter_health_url = os.environ["ADAPTER_HEALTH_URL"]
start_adapter = os.environ["START_ADAPTER"] == "1"
use_codex_auth = os.environ["USE_CODEX_AUTH"] == "1"
precheck_only = os.environ["PRECHECK_ONLY"] == "1"
adapter_python_bin = os.environ["ADAPTER_PYTHON_BIN"]
adapter_host = os.environ["ADAPTER_HOST"]
adapter_port = os.environ["ADAPTER_PORT"]
adapter_env_path = Path(os.environ["ADAPTER_ENV_PATH"])
adapter_upstreams_toml_path = Path(os.environ["ADAPTER_UPSTREAMS_TOML_PATH"])
logs_dir = run_root / "logs"
output_dir = run_root / "agent_outputs"
config_path = run_root / "scenefunc3d_backends.toml"
default_merged_molmo_path = run_root / "molmopoint_merged_model"
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def load_adapter_env() -> dict[str, str]:
    env = os.environ.copy()
    adapter_dir = repo_root / "codex_modelhub_adapter"
    env["PYTHONPATH"] = f"{adapter_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if adapter_env_path.is_file():
        for raw_line in adapter_env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            env[key] = value
    if (
        adapter_upstreams_toml_path.is_file()
        and not env.get("AIDP_MODELHUB_UPSTREAMS_TOML")
    ):
        env["AIDP_MODELHUB_UPSTREAMS_TOML"] = str(adapter_upstreams_toml_path)
    return env


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, raw_value = line.split("=", 1)
    key = key.strip()
    if not _ENV_KEY_PATTERN.fullmatch(key):
        return None
    value = raw_value.strip()
    try:
        split_value = shlex.split(value, comments=False, posix=True)
    except ValueError:
        split_value = ()
    if len(split_value) == 1:
        value = split_value[0]
    return key, value


def start_adapter_if_requested() -> subprocess.Popen[bytes] | None:
    if not start_adapter:
        return None
    adapter_dir = repo_root / "codex_modelhub_adapter"
    log_path = logs_dir / "adapter_server.log"
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        [
            adapter_python_bin,
            "-m",
            "uvicorn",
            "adapter.app:app",
            "--host",
            adapter_host,
            "--port",
            adapter_port,
        ],
        cwd=adapter_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=load_adapter_env(),
    )
    print(f"started adapter: pid={process.pid} log={log_path}", flush=True)
    return process


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


def check_codex_auth_ready() -> None:
    auth_path = codex_home / "auth.json"
    if not auth_path.is_file():
        raise RuntimeError(
            "USE_CODEX_AUTH=1 requires an authenticated CODEX_HOME with "
            f"auth.json: {auth_path}"
        )
    print(
        "codex auth ready: "
        f"codex_home={codex_home} "
        f"model={os.environ.get('CODEX_AGENT_MODEL')} "
        f"provider={os.environ.get('CODEX_AGENT_MODEL_PROVIDER')} "
        f"copy_auth={os.environ.get('CODEX_AGENT_COPY_AUTH')} "
        f"prefix_cache={os.environ.get('CODEX_AGENT_ENABLE_PREFIX_CACHE')} "
        f"keep_run_home={os.environ.get('CODEX_AGENT_KEEP_RUN_HOME')}",
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
    if sidecar_mode == "remote":
        backend_lines = (
            f"sidecar_base_url = {json.dumps(sidecar_base_url)}",
            f"request_headers_path = {json.dumps(str(sidecar_headers_path))}",
            "request_timeout_seconds = 300.0",
            f"artifact_staging_root = {json.dumps(str(run_root))}",
            (
                "allowed_image_roots = "
                f"[{json.dumps(str(dataset_root.parent))}, {json.dumps(str(run_root))}]"
            ),
            f"allowed_output_roots = [{json.dumps(str(run_root))}]",
            "",
        )
    elif sidecar_mode == "local":
        backend_lines = (
            'molmo_url = "http://127.0.0.1:8711"',
            'sam_url = "http://127.0.0.1:8712"',
            "request_timeout_seconds = 300.0",
            f"artifact_staging_root = {json.dumps(str(run_root))}",
            (
                "allowed_image_roots = "
                f"[{json.dumps(str(dataset_root.parent))}, {json.dumps(str(run_root))}]"
            ),
            f"allowed_output_roots = [{json.dumps(str(run_root))}]",
            "",
        )
    else:
        raise RuntimeError("SIDECAR_MODE must be remote or local")
    config_path.write_text(
        "\n".join(backend_lines),
        encoding="utf-8",
    )


def check_remote_sidecars_ready() -> None:
    check_sidecar_health(config_path)
    print(f"remote sidecars ready: base_url={sidecar_base_url}", flush=True)


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


def sample_source_args() -> list[str]:
    normalized_sample_id = sample_id.strip()
    normalized_sample_ids_path = sample_ids_path.strip()
    normalized_all_samples = all_samples_value.strip()
    if normalized_all_samples not in ("0", "1"):
        raise RuntimeError("ALL_SAMPLES must be 0 or 1")
    all_samples = normalized_all_samples == "1"
    explicit_sample_id = sample_id_was_set and bool(sample_id_raw.strip())
    if explicit_sample_id and (all_samples or normalized_sample_ids_path):
        raise RuntimeError(
            "SAMPLE_ID cannot be combined with SAMPLE_IDS_PATH or ALL_SAMPLES=1"
        )
    if all_samples and normalized_sample_ids_path:
        raise RuntimeError("ALL_SAMPLES=1 cannot be combined with SAMPLE_IDS_PATH")
    if all_samples:
        print("runner sample source: all samples", flush=True)
        return ["--all-samples"]
    if normalized_sample_ids_path:
        _validate_sample_ids_file(Path(normalized_sample_ids_path))
        print(
            f"runner sample source: sample_ids_path={normalized_sample_ids_path}",
            flush=True,
        )
        return ["--sample-ids-path", normalized_sample_ids_path]
    if not normalized_sample_id:
        raise RuntimeError(
            "one runner sample source is required: set SAMPLE_ID, "
            "SAMPLE_IDS_PATH, or ALL_SAMPLES=1"
        )
    print(f"runner sample source: sample_id={normalized_sample_id}", flush=True)
    return ["--sample-id", normalized_sample_id]


def _validate_sample_ids_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"SAMPLE_IDS_PATH must point to a JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SAMPLE_IDS_PATH is not valid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"SAMPLE_IDS_PATH must contain a JSON array: {path}")
    if not payload:
        raise RuntimeError(f"SAMPLE_IDS_PATH must contain at least one sample id: {path}")
    seen_sample_ids: set[str] = set()
    for item_index, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(
                "SAMPLE_IDS_PATH entries must be non-empty strings: "
                f"path={path}; index={item_index}"
            )
        normalized_sample_id = item.strip()
        try:
            SceneFunc3dSampleId.parse(normalized_sample_id)
        except SceneFunc3dDataError as exc:
            raise RuntimeError(
                "SAMPLE_IDS_PATH entry is not a valid sample id: "
                f"path={path}; index={item_index}; value={normalized_sample_id!r}"
            ) from exc
        if normalized_sample_id in seen_sample_ids:
            raise RuntimeError(
                "SAMPLE_IDS_PATH must not contain duplicate sample ids: "
                f"path={path}; sample_id={normalized_sample_id!r}"
            )
        seen_sample_ids.add(normalized_sample_id)


def run_agent_runner(sample_args: list[str]) -> None:
    stdout_path = run_root / "runner_stdout.json"
    log_path = logs_dir / "runner.log"
    command = [
        runner_python_bin,
        "-m",
        "codex_agent.cli.run_scenefunc3d",
        "--dataset-root",
        str(dataset_root),
    ] + sample_args + [
        "--backend-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--score",
    ]
    with log_path.open("wb") as log_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=log_handle,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout is None:
                stdout_path.write_bytes(b"")
            else:
                stdout_path.write_bytes(exc.stdout)
                print(exc.stdout.decode("utf-8", errors="replace"), flush=True)
            print(f"runner stdout: {stdout_path}", flush=True)
            print(f"runner log: {log_path}", flush=True)
            raise
    stdout_path.write_bytes(completed.stdout)
    print(completed.stdout.decode("utf-8"), flush=True)
    print(f"runner stdout: {stdout_path}", flush=True)
    print(f"runner log: {log_path}", flush=True)


def main() -> None:
    runner_sample_args = sample_source_args()
    ensure_codex_home()
    write_backend_config()
    adapter: subprocess.Popen[bytes] | None = None
    molmo: subprocess.Popen[bytes] | None = None
    sam: subprocess.Popen[bytes] | None = None
    try:
        if use_codex_auth:
            check_codex_auth_ready()
        else:
            adapter = start_adapter_if_requested()
            if adapter is not None:
                wait_health("adapter", adapter_health_url, adapter)
            check_adapter_ready()
        if precheck_only:
            print("precheck complete", flush=True)
            return
        if sidecar_mode == "remote":
            check_remote_sidecars_ready()
            run_agent_runner(runner_sample_args)
            return
        prepare_molmo_model_dir()
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
        wait_health("molmo", "http://127.0.0.1:8711/health", molmo)
        wait_health("sam", "http://127.0.0.1:8712/health", sam)
        run_agent_runner(runner_sample_args)
    finally:
        for process in (sam, molmo, adapter):
            if process is None:
                continue
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
