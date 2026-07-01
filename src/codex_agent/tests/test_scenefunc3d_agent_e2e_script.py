"""Tests for the SceneFunc3D real agent-runner E2E script contract."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def test_agent_e2e_script_runs_runner_with_sidecars_and_scoring() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "codex_agent.scenefunc3d.servers.molmo_point_server" in script_text
    assert "codex_agent.scenefunc3d.servers.sam2_mask_server" in script_text
    assert "codex_agent.cli.run_scenefunc3d" in script_text
    assert '"--score"' in script_text
    assert "--backend-config" in script_text
    assert "check_adapter_ready()" in script_text
    assert "ensure_codex_home()" in script_text


def test_agent_e2e_script_allows_full_molmo_sam_fuse_tool_budget() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert 'CODEX_AGENT_TURN_TIMEOUT_S="${CODEX_AGENT_TURN_TIMEOUT_S:-900}"' in (
        script_text
    )
    assert 'CODEX_AGENT_MAX_TOOL_CALLS="${CODEX_AGENT_MAX_TOOL_CALLS:-128}"' in (
        script_text
    )
    assert (
        'CODEX_AGENT_MAX_REPEATED_TOOL_CALLS="${CODEX_AGENT_MAX_REPEATED_TOOL_CALLS:-6}"'
        in script_text
    )


def test_agent_e2e_script_preserves_runner_stdout_on_failure() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "except subprocess.CalledProcessError as exc:" in script_text
    assert "stdout_path.write_bytes(exc.stdout)" in script_text
    assert "runner stdout:" in script_text


def test_agent_e2e_script_supports_batch_runner_sample_sources() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert 'SAMPLE_IDS_PATH="${SAMPLE_IDS_PATH:-}"' in script_text
    assert 'ALL_SAMPLES="${ALL_SAMPLES:-0}"' in script_text
    assert 'sample_ids_path = os.environ["SAMPLE_IDS_PATH"]' in script_text
    assert 'all_samples_value = os.environ["ALL_SAMPLES"]' in script_text
    assert 'sample_id_was_set = os.environ["SAMPLE_ID_WAS_SET"] == "1"' in script_text
    assert "def sample_source_args() -> list[str]:" in script_text
    assert '"--sample-ids-path"' in script_text
    assert '"--all-samples"' in script_text


def test_agent_e2e_script_runner_command_uses_one_sample_source_helper() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "sample_source_args()" in script_text
    assert "command = [" in script_text
    assert "] + sample_args + [" in script_text
    assert "runner_sample_args = sample_source_args()" in script_text
    assert "run_agent_runner(runner_sample_args)" in script_text
    assert "ALL_SAMPLES=1 cannot be combined with SAMPLE_IDS_PATH" in script_text


def test_agent_e2e_script_can_start_project_local_adapter() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert 'HOST_UNAME="${HOST_UNAME:-$(uname -s)}"' in script_text
    assert 'if [[ "${HOST_UNAME}" == "Linux" ]]; then' in script_text
    assert 'START_ADAPTER="${START_ADAPTER:-${DEFAULT_START_ADAPTER}}"' in script_text
    assert "ADAPTER_UPSTREAMS_TOML_PATH" in script_text
    assert "AIDP_MODELHUB_UPSTREAMS_TOML" in script_text
    assert 'adapter_python_bin = os.environ["ADAPTER_PYTHON_BIN"]' in script_text
    assert "load_adapter_env()" in script_text
    assert "start_adapter_if_requested()" in script_text
    assert "codex_modelhub_adapter" in script_text
    assert "adapter.app:app" in script_text
    assert "adapter_server.log" in script_text


def test_agent_e2e_script_can_use_codex_auth_provider() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert 'USE_CODEX_AUTH="${USE_CODEX_AUTH:-0}"' in script_text
    assert "CODEX_AGENT_COPY_AUTH" in script_text
    assert "CODEX_AGENT_MODEL_PROVIDER" in script_text
    assert "CODEX_AGENT_ENABLE_PREFIX_CACHE" in script_text
    assert "check_codex_auth_ready()" in script_text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-0}"' in script_text


def test_agent_e2e_script_supports_remote_workspace_proxy_sidecars() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert 'SIDECAR_MODE="${SIDECAR_MODE:-remote}"' in script_text
    assert "SCENEFUNC3D_SIDECAR_BASE_URL" in script_text
    assert "workspace-proxy-candy-maliva-tce.tiktok-row.org" in script_text
    assert "SCENEFUNC3D_SIDECAR_HEADERS_PATH" in script_text
    assert "sidecar_base_url = {json.dumps(sidecar_base_url)}" in script_text
    assert "request_headers_path = {json.dumps(str(sidecar_headers_path))}" in (
        script_text
    )
    assert "check_remote_sidecars_ready()" in script_text
    assert 'if sidecar_mode == "remote":' in script_text


def test_agent_e2e_script_allows_agent_generated_frame_artifacts() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "allowed_image_roots = " in script_text
    assert "json.dumps(str(dataset_root.parent))" in script_text
    assert "json.dumps(str(run_root))" in script_text


def test_agent_e2e_script_defaults_outputs_to_repo_tmp() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert (
        'RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/scenefunc3d/agent_runner_e2e_20260629}"'
        in script_text
    )
    assert 'RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/agent_runner_e2e_20260629}"' not in (
        script_text
    )


def test_agent_e2e_script_defaults_are_repo_relative() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "/mlx_devbox/" not in script_text
    assert "/usr/bin/python" not in script_text
    assert "/home/tiger/.codex" not in script_text
    assert "BASH_SOURCE[0]" in script_text
    assert 'RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-python3}"' in script_text


def test_benchmark_asset_shell_scripts_use_portable_defaults() -> None:
    for script_path in _benchmark_asset_script_paths():
        script_text = script_path.read_text(encoding="utf-8")

        assert "/mlx_devbox/" not in script_text
        assert "/usr/bin/python" not in script_text
        assert "BASH_SOURCE[0]" in script_text
        assert "python3" in script_text


def test_benchmark_python_smoke_uses_portable_defaults() -> None:
    smoke_script = (
        Path.cwd()
        / "docs"
        / "benchmark"
        / "scenefunc_molmo_sam3d"
        / "assets"
        / "molmo_sam3d_smoke.py"
    )
    script_text = smoke_script.read_text(encoding="utf-8")

    assert "/mlx_devbox/" not in script_text
    assert "DATASET_ROOT" in script_text


def test_agent_e2e_script_preflight_stops_before_gpu_sidecars(
    tmp_path: Path,
) -> None:
    _skip_without_adapter_runtime()
    port = _unused_port()
    run_root = tmp_path / "run"
    env = _script_env(tmp_path)
    env.update(
        {
            "START_ADAPTER": "1",
            "ADAPTER_PORT": str(port),
            "ADAPTER_HEALTH_URL": f"http://127.0.0.1:{port}/health",
            "ADAPTER_ENV_PATH": str(tmp_path / "missing.env"),
            "ADAPTER_UPSTREAMS_TOML_PATH": str(tmp_path / "missing.toml"),
            "RUN_ROOT": str(run_root),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "started adapter:" in completed.stdout
    assert "has no private upstream credential configured" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout
    assert not _port_is_open(port)


def test_agent_e2e_script_linux_defaults_to_project_local_adapter(
    tmp_path: Path,
) -> None:
    _skip_without_adapter_runtime()
    port = _unused_port()
    run_root = tmp_path / "run"
    env = _script_env(tmp_path)
    env.update(
        {
            "HOST_UNAME": "Linux",
            "ADAPTER_PORT": str(port),
            "ADAPTER_HEALTH_URL": f"http://127.0.0.1:{port}/health",
            "ADAPTER_ENV_PATH": str(tmp_path / "missing.env"),
            "ADAPTER_UPSTREAMS_TOML_PATH": str(tmp_path / "missing.toml"),
            "RUN_ROOT": str(run_root),
        }
    )
    env.pop("START_ADAPTER", None)

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "started adapter:" in completed.stdout
    assert "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online" in completed.stdout
    assert "has no private upstream credential configured" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout
    assert not _port_is_open(port)


def test_agent_e2e_script_codex_auth_preflight_stops_before_sidecars(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "CODEX_HOME": str(codex_home),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "auth.json" in completed.stdout
    assert "started adapter:" not in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_codex_auth_precheck_only_succeeds(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "codex auth ready:" in completed.stdout
    assert "precheck complete" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_conflicting_batch_sources(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text('["421254::desc-a"]\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "ALL_SAMPLES": "1",
            "SAMPLE_IDS_PATH": str(sample_ids_path),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "ALL_SAMPLES=1 cannot be combined with SAMPLE_IDS_PATH" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_explicit_sample_id_with_all_samples(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "ALL_SAMPLES": "1",
            "SAMPLE_ID": "421254::desc-a",
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "SAMPLE_ID cannot be combined with SAMPLE_IDS_PATH or ALL_SAMPLES=1" in (
        completed.stdout
    )
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_missing_sample_ids_path(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "SAMPLE_IDS_PATH": str(tmp_path / "missing_sample_ids.json"),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "SAMPLE_IDS_PATH must point to a JSON file" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_invalid_all_samples_value(
    tmp_path: Path,
) -> None:
    env = _script_env(tmp_path)
    env.update(
        {
            "HOST_UNAME": "Linux",
            "ALL_SAMPLES": "true",
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "ALL_SAMPLES must be 0 or 1" in completed.stdout
    assert "started adapter:" not in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_codex_auth_precheck_accepts_sample_ids_path(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text('["421254::desc-a"]\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "SAMPLE_IDS_PATH": str(sample_ids_path),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert (
        f"runner sample source: sample_ids_path={sample_ids_path}" in completed.stdout
    )
    assert "precheck complete" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_accepts_empty_sample_id_with_sample_ids_path(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text('["421254::desc-a"]\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "SAMPLE_ID": "",
            "SAMPLE_IDS_PATH": str(sample_ids_path),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert (
        f"runner sample source: sample_ids_path={sample_ids_path}" in completed.stdout
    )
    assert "precheck complete" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_duplicate_sample_ids(
    tmp_path: Path,
) -> None:
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text(
        '["421254::desc-a", " 421254::desc-a "]\n',
        encoding="utf-8",
    )
    env = _script_env(tmp_path)
    env.update(
        {
            "SAMPLE_IDS_PATH": str(sample_ids_path),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "SAMPLE_IDS_PATH must not contain duplicate sample ids" in completed.stdout
    assert "started adapter:" not in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_precheck_rejects_malformed_sample_id(
    tmp_path: Path,
) -> None:
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text('["not-a-sample-id"]\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "SAMPLE_IDS_PATH": str(sample_ids_path),
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert "SAMPLE_IDS_PATH entry is not a valid sample id" in completed.stdout
    assert "started adapter:" not in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_codex_auth_precheck_accepts_all_samples(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "ALL_SAMPLES": "1",
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "runner sample source: all samples" in completed.stdout
    assert "precheck complete" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_codex_auth_overrides_inherited_modelhub_env(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}\n', encoding="utf-8")
    env = _script_env(tmp_path)
    env.update(
        {
            "USE_CODEX_AUTH": "1",
            "PRECHECK_ONLY": "1",
            "CODEX_HOME": str(codex_home),
            "CODEX_AGENT_MODEL_PROVIDER": "modelhub_adapter",
            "CODEX_AGENT_COPY_AUTH": "0",
            "CODEX_AGENT_ENABLE_PREFIX_CACHE": "1",
            "CODEX_AGENT_KEEP_RUN_HOME": "1",
            "RUN_ROOT": str(tmp_path / "run"),
        }
    )

    completed = subprocess.run(
        ["bash", str(_agent_e2e_script_path())],
        cwd=Path.cwd(),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "provider=openai" in completed.stdout
    assert "copy_auth=1" in completed.stdout
    assert "prefix_cache=0" in completed.stdout
    assert "keep_run_home=0" in completed.stdout
    assert "started molmo_server" not in completed.stdout
    assert "started sam_server" not in completed.stdout


def test_agent_e2e_script_does_not_manual_chain_mask_tools() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert '"molmo_point",' not in script_text
    assert '"sam_mask",' not in script_text
    assert '"lift_mask_to_3d",' not in script_text
    assert '"fuse_accepted_masks",' not in script_text


def _agent_e2e_script_path() -> Path:
    return (
        Path.cwd()
        / "docs"
        / "benchmark"
        / "scenefunc_molmo_sam3d"
        / "assets"
        / "run_agent_runner_e2e_20260629.sh"
    )


def _benchmark_asset_script_paths() -> tuple[Path, ...]:
    asset_dir = Path.cwd() / "docs" / "benchmark" / "scenefunc_molmo_sam3d" / "assets"
    return (
        asset_dir / "run_full_421254_000050.sh",
        asset_dir / "run_sam_transformers_debug_20260629.sh",
        asset_dir / "run_sidecar_tool_smoke_20260629.sh",
    )


def _script_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in _SCRIPT_ENV_KEYS:
        env.pop(name, None)
    repo_root = Path.cwd()
    env.update(
        {
            "REPO_ROOT": str(repo_root),
            "DATASET_ROOT": str(tmp_path / "dataset"),
            "RUNNER_PYTHON_BIN": sys.executable,
            "SIDECAR_PYTHON_BIN": sys.executable,
            "ADAPTER_PYTHON_BIN": sys.executable,
        }
    )
    return env


_SCRIPT_ENV_KEYS: tuple[str, ...] = (
    "ADAPTER_PYTHON_BIN",
    "ALL_SAMPLES",
    "CODEX_AGENT_COPY_AUTH",
    "CODEX_AGENT_ENABLE_PREFIX_CACHE",
    "CODEX_AGENT_KEEP_RUN_HOME",
    "CODEX_AGENT_MODEL_PROVIDER",
    "CODEX_HOME",
    "DATASET_ROOT",
    "HOST_UNAME",
    "PRECHECK_ONLY",
    "REPO_ROOT",
    "RUNNER_PYTHON_BIN",
    "SAMPLE_ID",
    "SAMPLE_IDS_PATH",
    "SIDECAR_PYTHON_BIN",
    "START_ADAPTER",
    "USE_CODEX_AUTH",
)


def _skip_without_adapter_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import fastapi, uvicorn"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"{sys.executable} cannot import fastapi and uvicorn")


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0
