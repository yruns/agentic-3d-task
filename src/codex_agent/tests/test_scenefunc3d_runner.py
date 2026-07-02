"""Tests for the SceneFunc3D runtime runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pytest

import codex_agent.scenefunc3d.runner as runner
from codex_agent.cli import runtime_preflight
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import (
    CodexResponseError,
    CodexTurnError,
    SceneFunc3dDataError,
)
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.scenefunc3d.runner import (
    SCENEFUNC3D_ALLOWED_TOOL_NAMES,
    SceneFunc3dMaskTask,
    SceneFunc3dRunnerConfig,
    build_arg_parser,
    check_sidecar_health,
    main,
    run_samples,
    run_single_sample,
)
from codex_agent.scenefunc3d.sample import SceneFunc3dSample, SceneFuncMotionHint
from codex_agent.scenefunc3d.servers.http_json import (
    JsonObject,
    make_json_handler,
)
from codex_agent.scenefunc3d.task import SEED_APPROVAL_ACTIONS
from codex_agent.tasks.base import CodexExecutor, CodexTask

ResultT = TypeVar("ResultT")

_SEED_APPROVAL_VALUES = tuple(action.value for action in SEED_APPROVAL_ACTIONS)


def _capture_runtime_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> list[CodexAgentConfig]:
    calls: list[CodexAgentConfig] = []

    def fake_preflight(config: CodexAgentConfig) -> None:
        calls.append(config)

    monkeypatch.setattr(runtime_preflight, "preflight_codex_runtime", fake_preflight)
    return calls


def test_build_arg_parser_accepts_single_case_runtime_options(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.dataset_root == tmp_path / "data"
    assert args.sample_id == "421254::desc-a"
    assert args.backend_config == tmp_path / "backends.toml"
    assert args.output_dir == tmp_path / "out"


def test_build_arg_parser_accepts_all_samples_batch_runtime_options(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--all-samples",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--score",
        ]
    )

    assert args.dataset_root == tmp_path / "data"
    assert args.all_samples is True
    assert args.sample_id is None
    assert args.backend_config == tmp_path / "backends.toml"
    assert args.output_dir == tmp_path / "out"
    assert args.score is True


def test_build_arg_parser_accepts_positive_workers(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-ids-path",
            str(tmp_path / "sample_ids.json"),
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--workers",
            "20",
        ]
    )

    assert args.workers == 20


def test_build_arg_parser_rejects_non_positive_workers(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset-root",
                str(tmp_path / "data"),
                "--sample-ids-path",
                str(tmp_path / "sample_ids.json"),
                "--backend-config",
                str(tmp_path / "backends.toml"),
                "--output-dir",
                str(tmp_path / "out"),
                "--workers",
                "0",
            ]
        )


def test_build_arg_parser_help_omits_skip_sidecar_health_check() -> None:
    parser = build_arg_parser()

    assert "--skip-sidecar-health-check" not in parser.format_help()


def test_main_prints_one_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_path = tmp_path / "out" / "result.json"

    def _fake_build_executor() -> CodexExecutor:
        return cast(CodexExecutor, object())

    def _fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: CodexExecutor,
        check_sidecars: bool,
    ) -> Path:
        assert check_sidecars is True
        _ = (config, sample_id, executor)
        return result_path

    monkeypatch.setattr(runner, "_build_executor", _fake_build_executor)
    monkeypatch.setattr(runner, "run_single_sample", _fake_run_single_sample)

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [json.dumps({"result_path": str(result_path)})]


def test_main_propagates_single_sample_failure_without_success_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_scene(tmp_path / "data")
    turn_error = CodexTurnError("cli turn failed")

    def _fake_build_executor() -> CodexExecutor:
        return _FailingCodexTurnExecutor(turn_error)

    def _skip_sidecar_health(_backend_config_path: Path) -> None:
        return None

    monkeypatch.setattr(runner, "_build_executor", _fake_build_executor)
    monkeypatch.setattr(runner, "check_sidecar_health", _skip_sidecar_health)

    with pytest.raises(CodexTurnError, match="cli turn failed") as exc_info:
        main(
            [
                "--dataset-root",
                str(tmp_path / "data"),
                "--sample-id",
                "421254::desc-a",
                "--backend-config",
                str(tmp_path / "backends.toml"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )

    assert exc_info.value is turn_error
    assert capsys.readouterr().out == ""
    assert (tmp_path / "out" / "421254" / "desc-a" / "failure.json").is_file()


def test_run_from_args_always_checks_sidecars_for_single_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "out" / "result.json"
    args = argparse.Namespace(
        dataset_root=tmp_path / "data",
        sample_id="421254::desc-a",
        sample_ids_path=None,
        all_samples=False,
        backend_config=tmp_path / "backends.toml",
        output_dir=tmp_path / "out",
        score=False,
        skip_sidecar_health_check=True,
    )

    def _fake_build_executor() -> CodexExecutor:
        return cast(CodexExecutor, object())

    def _fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: CodexExecutor,
        check_sidecars: bool,
    ) -> Path:
        assert config.dataset_root == tmp_path / "data"
        assert sample_id == "421254::desc-a"
        assert check_sidecars is True
        _ = executor
        return result_path

    monkeypatch.setattr(runner, "_build_executor", _fake_build_executor)
    monkeypatch.setattr(runner, "run_single_sample", _fake_run_single_sample)

    assert runner.run_from_args(args) == 0


def test_run_from_args_always_checks_sidecars_for_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text(json.dumps(["421254::desc-a"]), encoding="utf-8")
    args = argparse.Namespace(
        dataset_root=tmp_path / "data",
        sample_id=None,
        sample_ids_path=sample_ids_path,
        all_samples=False,
        backend_config=tmp_path / "backends.toml",
        output_dir=tmp_path / "out",
        score=True,
        workers=3,
        skip_sidecar_health_check=True,
    )

    def _fake_build_executor() -> CodexExecutor:
        return cast(CodexExecutor, object())

    def _fake_run_samples(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_ids: tuple[str, ...],
        executor: CodexExecutor,
        check_sidecars: bool = True,
        score: bool = True,
        continue_on_error: bool = True,
        workers: int = 1,
    ) -> runner.SceneFunc3dBatchRunSummary:
        assert config.output_dir == tmp_path / "out"
        assert sample_ids == ("421254::desc-a",)
        assert check_sidecars is True
        assert score is True
        assert workers == 3
        _ = (executor, continue_on_error)
        return runner.SceneFunc3dBatchRunSummary(
            scoring_enabled=True,
            sample_count=1,
            completed_count=1,
            failed_count=0,
            scored_count=1,
            mean_iou=0.25,
            mean_precision=0.5,
            mean_recall=0.75,
            mean_f1=0.6,
            results=(),
        )

    monkeypatch.setattr(runner, "_build_executor", _fake_build_executor)
    monkeypatch.setattr(runner, "run_samples", _fake_run_samples)

    assert runner.run_from_args(args) == 0


def test_run_samples_rejects_non_positive_workers(tmp_path: Path) -> None:
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(ValueError, match="workers must be positive"):
        run_samples(
            config,
            sample_ids=("421254::desc-a",),
            executor=cast(CodexExecutor, object()),
            check_sidecars=False,
            score=False,
            workers=0,
        )


def test_run_samples_uses_workers_for_multiple_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    active_count = 0
    max_active_count = 0
    lock = threading.Lock()
    first_sample_started = threading.Event()
    allow_first_sample_to_finish = threading.Event()

    def _fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: CodexExecutor,
        check_sidecars: bool,
    ) -> Path:
        nonlocal active_count, max_active_count
        assert check_sidecars is False
        _ = (config, executor)
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        if sample_id == "421254::desc-a":
            first_sample_started.set()
            assert allow_first_sample_to_finish.wait(timeout=2.0)
        else:
            assert first_sample_started.wait(timeout=2.0)
            allow_first_sample_to_finish.set()
        with lock:
            active_count -= 1
        return tmp_path / "out" / sample_id.replace("::", "_") / "result.json"

    monkeypatch.setattr(runner, "run_single_sample", _fake_run_single_sample)

    summary = run_samples(
        config,
        sample_ids=("421254::desc-a", "421254::desc-b"),
        executor=cast(CodexExecutor, object()),
        check_sidecars=False,
        score=False,
        workers=2,
    )

    assert max_active_count == 2
    assert summary.sample_count == 2
    assert summary.completed_count == 2
    assert [result.sample_id for result in summary.results] == [
        "421254::desc-a",
        "421254::desc-b",
    ]


def test_run_samples_fail_stop_does_not_start_later_samples_with_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    started_sample_ids: list[str] = []

    def _fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: CodexExecutor,
        check_sidecars: bool,
    ) -> Path:
        assert check_sidecars is False
        _ = (config, executor)
        started_sample_ids.append(sample_id)
        if sample_id == "421254::desc-a":
            raise CodexTurnError("first sample failed")
        return tmp_path / "out" / sample_id.replace("::", "_") / "result.json"

    monkeypatch.setattr(runner, "run_single_sample", _fake_run_single_sample)

    with pytest.raises(CodexTurnError, match="first sample failed"):
        run_samples(
            config,
            sample_ids=("421254::desc-a", "421254::desc-b"),
            executor=cast(CodexExecutor, object()),
            check_sidecars=False,
            score=False,
            continue_on_error=False,
            workers=2,
        )

    assert started_sample_ids == ["421254::desc-a"]


def test_mask_task_prompt_inlines_tools_without_attachments(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "421254"
    backend_config_path = tmp_path / "backends.toml"
    output_dir = tmp_path / "out"
    tool_context_path = output_dir / "tool_context.json"
    tool_context_path.parent.mkdir(parents=True)
    tool_context_path.write_text("{}", encoding="utf-8")
    task = _mask_task(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=backend_config_path,
        tool_context_path=tool_context_path,
    )

    request = task.build_turn_request()

    assert request.skills == ()
    assert request.image_paths == ()
    assert "Molmo point" in request.prompt
    assert "SAM candidates" in request.prompt
    assert f"--context {tool_context_path}" in request.prompt
    assert (
        f"python -m codex_agent.scenefunc3d.tools <tool> "
        f"--context {tool_context_path} --args '<json>'"
    ) in request.prompt
    tool_command = request.prompt.split("- invoke: ", maxsplit=1)[1].split(
        "\n", maxsplit=1
    )[0]
    assert "--scene-root" not in tool_command
    assert "--backend-config" not in tool_command
    assert "--out-dir" not in tool_command
    assert "set yield_time_ms=30000" in request.prompt
    assert "Process running with session ID" in request.prompt
    assert "wait on that same session until it exits and returns JSON" in (
        request.prompt
    )
    hard_limits = request.prompt.split("Hard limits:\n", maxsplit=1)[1].split(
        "\nFinal JSON schema:", maxsplit=1
    )[0]
    assert "Use ONLY these SceneFunc3D tools plus view_image" in hard_limits
    assert ", ".join(SCENEFUNC3D_ALLOWED_TOOL_NAMES) in hard_limits
    for tool_name in SCENEFUNC3D_ALLOWED_TOOL_NAMES:
        assert tool_name in hard_limits
    assert "Do NOT read, cat, sed, head, grep, rg, or open" in hard_limits
    assert "Never re-run a tool with identical arguments" in hard_limits


def test_mask_task_rejects_missing_tool_context_before_prompt(
    tmp_path: Path,
) -> None:
    task = _mask_task(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
        tool_context_path=tmp_path / "out" / "tool_context.json",
    )

    with pytest.raises(SceneFunc3dDataError, match="tool context is missing"):
        task.build_turn_request()


def test_check_sidecar_health_passes_for_healthy_fake_servers(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecar_health_sends_configured_headers(tmp_path: Path) -> None:
    molmo_server = _start_header_health_server(model_name="fake-molmo")
    sam_server = _start_header_health_server(model_name="fake-sam")
    headers_path = tmp_path / "headers.toml"
    headers_path.write_text(
        """
[headers]
X-Sidecar-Auth = "expected-secret"
""",
        encoding="utf-8",
    )
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
        request_headers_path=headers_path,
    )

    try:
        check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecar_health_rejects_unloaded_sidecar(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server_with_state(
        model_name="fake-sam",
        model_loaded=False,
    )
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        with pytest.raises(RuntimeError, match="model_loaded=false"):
            check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecars_script_uses_backend_config(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )
    script_path = Path("scripts/scenefunc3d/check_sidecars.sh").resolve()
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = "src"

    try:
        result = subprocess.run(
            [str(script_path), str(backend_config_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        molmo_server.close()
        sam_server.close()

    assert "sidecars ok" in result.stdout


def test_run_single_sample_writes_failure_json_when_codex_turn_fails(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    turn_error = CodexTurnError("executor turn failed")
    executor = _FailingCodexTurnExecutor(turn_error)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexTurnError, match="executor turn failed") as exc_info:
        run_single_sample(
            config,
            sample_id="421254::desc-a",
            executor=executor,
            check_sidecars=False,
        )

    assert exc_info.value is turn_error
    failure_path = sample_output_dir / "failure.json"
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload == {
        "task_name": "scenefunc3d_mask_generation",
        "sample_id": "421254::desc-a",
        "status": "failed",
        "failure_stage": "codex_turn",
        "error_type": "CodexTurnError",
        "error_message": "executor turn failed",
        "events_path": str(sample_output_dir / "events.jsonl"),
        "tool_context_path": str(sample_output_dir / "tool_context.json"),
        "turn": _empty_failure_turn_payload(),
    }


def test_run_single_sample_writes_pipeline_result_from_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")

    monkeypatch.setattr(
        runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz)
    )
    monkeypatch.setattr(
        runner,
        "run_anchor_multiview_pipeline",
        lambda **_: _fake_pipeline_result((0, 1)),
    )
    executor = _SeedFakeExecutor(
        _seed_confirmation_payload(),
        on_execute=lambda: _write_seed_sam_event(sample_output_dir),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    result_path = run_single_sample(
        config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["result_kind"] == "anchor_multiview_pipeline"
    assert payload["outcome"]["seed"]["seed_fragment_id"] == "000010_cand-a"
    assert payload["outcome"]["seed"]["seed_point_count"] == 2
    assert payload["outcome"]["selected_frame_ids"] == ["000010"]
    assert payload["outcome"]["confidence"] == pytest.approx(0.66)
    fused_npz = sample_output_dir / "fused" / "mask_data.npz"
    assert fused_npz.is_file()
    assert payload["outcome"]["mask_npz_path"] == str(fused_npz)


def test_run_single_sample_fails_closed_on_missing_seed_sam_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    monkeypatch.setattr(
        runner,
        "_lift_seed_fragment",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not lift")),
    )
    executor = _SeedFakeExecutor(_seed_confirmation_payload(), on_execute=lambda: None)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    with pytest.raises(CodexResponseError, match="no successful sam_mask"):
        run_single_sample(
            config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
        )
    assert (tmp_path / "out" / "421254" / "desc-a" / "failure.json").is_file()


def test_run_single_sample_fails_closed_on_empty_fused_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")
    monkeypatch.setattr(
        runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz)
    )
    monkeypatch.setattr(
        runner, "run_anchor_multiview_pipeline", lambda **_: _fake_pipeline_result(())
    )
    executor = _SeedFakeExecutor(
        _seed_confirmation_payload(),
        on_execute=lambda: _write_seed_sam_event(sample_output_dir),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    with pytest.raises(SceneFunc3dDataError, match="empty fused mask"):
        run_single_sample(
            config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
        )


def test_run_samples_writes_batch_evaluation_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_two_sample_scene(tmp_path / "data")
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")
    monkeypatch.setattr(
        runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz)
    )
    monkeypatch.setattr(
        runner,
        "run_anchor_multiview_pipeline",
        lambda **_: _fake_pipeline_result((10, 12)),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    executor = _SeedDynamicExecutor()

    summary = run_samples(
        config,
        sample_ids=("421254::desc-a", "421254::desc-b"),
        executor=executor,
        check_sidecars=False,
        score=True,
    )

    assert executor.sample_output_dirs == (
        tmp_path / "out" / "421254" / "desc-a",
        tmp_path / "out" / "421254" / "desc-b",
    )
    assert summary.sample_count == 2
    assert summary.completed_count == 2
    assert summary.failed_count == 0
    assert summary.mean_iou == pytest.approx((2.0 / 3.0 + 0.5) / 2.0)
    assert summary.mean_precision == pytest.approx((1.0 + 0.5) / 2.0)
    assert summary.mean_recall == pytest.approx((2.0 / 3.0 + 1.0) / 2.0)
    assert summary.mean_f1 == pytest.approx((0.8 + 2.0 / 3.0) / 2.0)

    summary_path = tmp_path / "out" / "evaluation_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "scenefunc3d_mask_generation"
    assert payload["sample_count"] == 2
    assert payload["completed_count"] == 2
    assert payload["failed_count"] == 0
    assert payload["mean_metrics"]["iou"] == pytest.approx(summary.mean_iou)
    assert [item["sample_id"] for item in payload["results"]] == [
        "421254::desc-a",
        "421254::desc-b",
    ]
    assert payload["results"][0]["result_path"] == str(
        tmp_path / "out" / "421254" / "desc-a" / "result.json"
    )
    assert payload["results"][0]["score"]["metrics"]["predicted_count"] == 2
    assert payload["results"][0]["score"]["metrics"]["gt_count"] == 3


def test_run_samples_preserves_result_path_when_scoring_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_unscorable_scene(tmp_path / "data")
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")
    monkeypatch.setattr(
        runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz)
    )
    monkeypatch.setattr(
        runner,
        "run_anchor_multiview_pipeline",
        lambda **_: _fake_pipeline_result((10, 12)),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    summary = run_samples(
        config,
        sample_ids=("421254::desc-a",),
        executor=_SeedDynamicExecutor(),
        check_sidecars=False,
        score=True,
    )

    result_path = tmp_path / "out" / "421254" / "desc-a" / "result.json"
    assert result_path.is_file()
    assert summary.sample_count == 1
    assert summary.completed_count == 0
    assert summary.failed_count == 1
    assert summary.scoring_enabled is True
    assert summary.scored_count == 0
    assert summary.mean_iou is None
    assert summary.results[0].status == "failed"
    assert summary.results[0].failure_stage == "score"
    assert summary.results[0].result_path == result_path
    assert summary.results[0].score is None
    assert "SceneFunc3dDataError" in summary.results[0].error

    payload = json.loads(
        (tmp_path / "out" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert payload["scoring_enabled"] is True
    assert payload["scored_count"] == 0
    assert payload["mean_metrics"] is None
    assert payload["results"][0]["result_path"] == str(result_path)
    assert payload["results"][0]["failure_stage"] == "score"


def test_run_samples_records_failure_path_when_run_writes_failure_json(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    summary = run_samples(
        config,
        sample_ids=("421254::desc-a",),
        executor=_SeedFakeExecutor(
            _seed_confirmation_payload(), on_execute=lambda: None
        ),
        check_sidecars=False,
        score=False,
    )

    failure_path = sample_output_dir / "failure.json"
    assert failure_path.is_file()
    assert summary.sample_count == 1
    assert summary.completed_count == 0
    assert summary.failed_count == 1
    assert summary.results[0].status == "failed"
    assert summary.results[0].failure_stage == "run"
    assert summary.results[0].result_path is None
    assert summary.results[0].failure_path == failure_path
    assert summary.results[0].score is None

    payload = json.loads(
        (tmp_path / "out" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert payload["results"][0]["result_path"] is None
    assert payload["results"][0]["failure_path"] == str(failure_path)


def test_run_samples_marks_unscored_batch_metrics_as_not_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_two_sample_scene(tmp_path / "data")
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")
    monkeypatch.setattr(
        runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz)
    )
    monkeypatch.setattr(
        runner,
        "run_anchor_multiview_pipeline",
        lambda **_: _fake_pipeline_result((10, 12)),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    summary = run_samples(
        config,
        sample_ids=("421254::desc-a", "421254::desc-b"),
        executor=_SeedDynamicExecutor(),
        check_sidecars=False,
        score=False,
    )

    assert summary.sample_count == 2
    assert summary.completed_count == 2
    assert summary.failed_count == 0
    assert summary.scoring_enabled is False
    assert summary.scored_count == 0
    assert summary.mean_iou is None
    assert summary.mean_precision is None
    assert summary.mean_recall is None
    assert summary.mean_f1 is None
    assert all(result.score is None for result in summary.results)

    payload = json.loads(
        (tmp_path / "out" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert payload["scoring_enabled"] is False
    assert payload["scored_count"] == 0
    assert payload["mean_metrics"] is None


def test_main_with_score_prints_result_path_and_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_scene(tmp_path / "data")

    def fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: object,
        check_sidecars: bool = True,
    ) -> Path:
        assert sample_id == "421254::desc-a"
        assert check_sidecars is True
        _ = executor
        sample_output_dir = config.output_dir / "421254" / "desc-a"
        result_path = sample_output_dir / "result.json"
        _write_pipeline_result_json(
            result_path, sample_id=sample_id, point_indices=(10, 12)
        )
        return result_path

    def fake_build_executor() -> CodexExecutor:
        return _UnusedExecutor()

    monkeypatch.setattr(runner, "run_single_sample", fake_run_single_sample)
    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--score",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result_path"] == str(
        tmp_path / "out" / "421254" / "desc-a" / "result.json"
    )
    score = payload["score"]
    assert score["sample_id"] == "421254::desc-a"
    assert score["metrics"]["iou"] == 0.0
    assert score["metrics"]["predicted_count"] == 2
    assert score["metrics"]["gt_count"] == 3


def test_main_all_samples_prints_batch_summary_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_two_sample_scene(tmp_path / "data")
    captured_sample_ids: list[str] = []

    def fake_run_samples(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_ids: tuple[str, ...],
        executor: CodexExecutor,
        check_sidecars: bool = True,
        score: bool = True,
        continue_on_error: bool = True,
        workers: int = 1,
    ) -> runner.SceneFunc3dBatchRunSummary:
        _ = executor
        _ = continue_on_error
        assert workers == 1
        assert config.dataset_root == tmp_path / "data"
        assert config.output_dir == tmp_path / "out"
        assert config.backend_config_path == tmp_path / "backends.toml"
        assert check_sidecars is True
        assert score is True
        captured_sample_ids.extend(sample_ids)
        return runner.SceneFunc3dBatchRunSummary(
            scoring_enabled=True,
            sample_count=2,
            completed_count=2,
            failed_count=0,
            scored_count=2,
            mean_iou=0.25,
            mean_precision=0.5,
            mean_recall=0.75,
            mean_f1=0.6,
            results=(),
        )

    def fake_build_executor() -> CodexExecutor:
        return _UnusedExecutor()

    monkeypatch.setattr(runner, "run_samples", fake_run_samples)
    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--all-samples",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--score",
        ]
    )

    assert exit_code == 0
    assert captured_sample_ids == ["421254::desc-a", "421254::desc-b"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary_path"] == str(tmp_path / "out" / "evaluation_summary.json")
    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["mean_metrics"]["iou"] == 0.25


def test_main_sample_ids_path_strips_file_entries_before_batch_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_two_sample_scene(tmp_path / "data")
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text(
        json.dumps([" 421254::desc-b ", "421254::desc-a"]),
        encoding="utf-8",
    )
    captured_sample_ids: list[str] = []

    def fake_run_samples(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_ids: tuple[str, ...],
        executor: CodexExecutor,
        check_sidecars: bool = True,
        score: bool = True,
        continue_on_error: bool = True,
        workers: int = 1,
    ) -> runner.SceneFunc3dBatchRunSummary:
        _ = config
        _ = executor
        _ = check_sidecars
        _ = score
        _ = continue_on_error
        assert workers == 1
        captured_sample_ids.extend(sample_ids)
        return runner.SceneFunc3dBatchRunSummary(
            scoring_enabled=True,
            sample_count=2,
            completed_count=2,
            failed_count=0,
            scored_count=2,
            mean_iou=0.25,
            mean_precision=0.5,
            mean_recall=0.75,
            mean_f1=0.6,
            results=(),
        )

    def fake_build_executor() -> CodexExecutor:
        return _UnusedExecutor()

    monkeypatch.setattr(runner, "run_samples", fake_run_samples)
    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-ids-path",
            str(sample_ids_path),
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--score",
        ]
    )

    assert exit_code == 0
    assert captured_sample_ids == ["421254::desc-b", "421254::desc-a"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary_path"] == str(tmp_path / "out" / "evaluation_summary.json")


def test_main_validates_sample_ids_path_before_building_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_executor() -> CodexExecutor:
        raise AssertionError("executor should not be built before input validation")

    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    with pytest.raises(SceneFunc3dDataError, match="sample ids file is missing"):
        main(
            [
                "--dataset-root",
                str(tmp_path / "data"),
                "--sample-ids-path",
                str(tmp_path / "missing_sample_ids.json"),
                "--backend-config",
                str(tmp_path / "backends.toml"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_main_rejects_empty_all_samples_before_building_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data").mkdir()

    def fake_build_executor() -> CodexExecutor:
        raise AssertionError("executor should not be built before input validation")

    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    with pytest.raises(SceneFunc3dDataError, match="sample_ids must not be empty"):
        main(
            [
                "--dataset-root",
                str(tmp_path / "data"),
                "--all-samples",
                "--backend-config",
                str(tmp_path / "backends.toml"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_build_executor_uses_tool_writable_runtime_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    monkeypatch.delenv("CODEX_AGENT_SANDBOX_NETWORK", raising=False)
    preflight_calls = _capture_runtime_preflight(monkeypatch)

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert preflight_calls == [executor.config]
    assert executor.config.sandbox == "workspace_write"
    assert executor.config.sandbox_network_access is True


def test_build_executor_preserves_full_access_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "full_access")
    monkeypatch.delenv("CODEX_AGENT_SANDBOX_NETWORK", raising=False)
    preflight_calls = _capture_runtime_preflight(monkeypatch)

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert preflight_calls == [executor.config]
    assert executor.config.sandbox == "full_access"
    assert executor.config.sandbox_network_access is False


def test_build_executor_upgrades_read_only_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "read_only")
    monkeypatch.setenv("CODEX_AGENT_SANDBOX_NETWORK", "false")
    preflight_calls = _capture_runtime_preflight(monkeypatch)

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert preflight_calls == [executor.config]
    assert executor.config.sandbox == "workspace_write"
    assert executor.config.sandbox_network_access is True


class _SeedFakeExecutor:
    """Fake executor returning a canned SeedConfirmation final response."""

    def __init__(
        self, response_payload: Mapping[str, object], *, on_execute: Callable[[], None]
    ) -> None:
        self._response_payload = dict(response_payload)
        self._on_execute = on_execute
        self.task_name = ""
        self.prompt = ""

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        request = task.build_turn_request()
        self.task_name = task.task_name
        self.prompt = request.prompt
        self._on_execute()
        outcome = task.parse_response(json.dumps(self._response_payload))
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(ResultT, outcome),
            turn=CodexTurnResult(
                final_response=json.dumps(self._response_payload),
                metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
            ),
        )


class _SeedDynamicExecutor:
    """Seed-aware executor that writes the SAM event and parses the seed outcome."""

    def __init__(self) -> None:
        self.sample_output_dirs: tuple[Path, ...] = ()

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        if not isinstance(task, SceneFunc3dMaskTask):
            raise AssertionError("expected SceneFunc3dMaskTask")
        sample_output_dir = task.output_dir
        _write_seed_sam_event(sample_output_dir)
        self.sample_output_dirs = (*self.sample_output_dirs, sample_output_dir)
        response = json.dumps(_seed_confirmation_payload())
        outcome = task.parse_response(response)
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(ResultT, outcome),
            turn=CodexTurnResult(
                final_response=response,
                metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
            ),
        )


class _FailingCodexTurnExecutor:
    def __init__(self, error: CodexTurnError) -> None:
        self.error = error

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        _ = task.build_turn_request()
        raise self.error


class _UnusedExecutor:
    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        _ = task
        raise AssertionError("test patched run_single_sample must not execute tasks")


def _empty_failure_turn_payload() -> dict[str, object]:
    return {
        "turn_id": None,
        "status": None,
        "duration_ms": None,
        "usage": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "reasoning_summary": None,
        "run_home": None,
        "attempts": [],
    }


def _seed_confirmation_payload(
    *, seed_frame_id: str = "000010", seed_candidate_id: str = "cand-a"
) -> dict[str, object]:
    return {
        "seed_frame_id": seed_frame_id,
        "seed_candidate_id": seed_candidate_id,
        "affordance_concept": "the drawer handle",
        "approval_actions": list(_SEED_APPROVAL_VALUES),
        "uncertainties": ["partial occlusion"],
    }


def _write_seed_sam_event(
    output_dir: Path, *, frame_id: str = "000010", candidate_id: str = "cand-a"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_type": "tool_completed",
                    "tool_name": "sam_mask",
                    "status": "success",
                    "args": {"frame_id": frame_id, "image_path": f"{frame_id}-rgb.png"},
                    "result": {
                        "frame_id": frame_id,
                        "candidates": [
                            {
                                "candidate_id": candidate_id,
                                "mask_npz_path": str(
                                    output_dir
                                    / "sam"
                                    / frame_id
                                    / f"{candidate_id}.npz"
                                ),
                                "overlay_path": f"{candidate_id}_overlay.jpg",
                            }
                        ],
                        "contact_sheet_path": "contact_sheet.jpg",
                    },
                    "error": "",
                }
            )
            + "\n"
        )


def _fake_lift_seed_fragment(fragment_npz: Path) -> Callable[..., object]:
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult

    def _fake(*, scene: object, seed_outcome: object, out_dir: Path) -> LiftMaskResult:
        _ = (scene, seed_outcome, out_dir)
        return LiftMaskResult(
            frame_id="000010",
            candidate_id="cand-a",
            lifted_point_count=2,
            mask_npz_path=fragment_npz,
            mask_ply_path=fragment_npz.with_name("lifted_points.ply"),
            overlay_path=fragment_npz.with_name("lift_overlay.txt"),
        )

    return _fake


def _fake_pipeline_result(point_indices: tuple[int, ...]) -> object:
    from codex_agent.scenefunc3d.backends.fusion import FusedMask, FusionParams
    from codex_agent.scenefunc3d.pipeline import (
        AnchorMultiViewResult,
        PerFrameOutcome,
    )

    return AnchorMultiViewResult(
        fused=FusedMask(
            point_indices=point_indices,
            confidence=0.66,
            instances=(),
            params=FusionParams(),
        ),
        selected_frames=(),
        per_frame=(
            PerFrameOutcome(
                frame_id="000010",
                point_indices=point_indices,
                centroid=(0.0, 0.0, 2.0),
                accepted=True,
                reject_reason="",
                molmo_fallback_used=False,
            ),
        ),
    )


def _write_seed_fragment_npz(path: Path) -> Path:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz

    path.parent.mkdir(parents=True, exist_ok=True)
    return write_lift_npz(
        path,
        np.array([[0.0, 0.0, 2.0], [0.01, 0.0, 2.0]], dtype=np.float64),
        point_indices=np.array([0, 1], dtype=np.int64),
    )


def _write_pipeline_result_json(
    result_path: Path, *, sample_id: str, point_indices: tuple[int, ...]
) -> None:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz

    fused_dir = result_path.parent / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)
    fused_npz = write_lift_npz(
        fused_dir / "mask_data.npz",
        np.zeros((len(point_indices), 3), dtype=np.float64),
        point_indices=np.array(point_indices, dtype=np.int64),
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": sample_id,
                "result_kind": "anchor_multiview_pipeline",
                "outcome": {
                    "result_kind": "anchor_multiview_pipeline",
                    "mask_npz_path": str(fused_npz),
                    "mask_ply_path": str(fused_dir / "lifted_points.ply"),
                    "confidence": 0.5,
                    "selected_frame_ids": ["000010"],
                    "seed": {
                        "seed_frame_id": "000010",
                        "seed_candidate_id": "cand-a",
                        "seed_fragment_id": "000010_cand-a",
                        "affordance_concept": "the drawer handle",
                        "seed_point_count": 2,
                    },
                    "per_frame": [],
                    "fusion_params": {
                        "agreement_tau": 0.5,
                        "radius_scale": 1.0,
                        "cluster_link_eps_m": 0.02,
                        "min_cluster_points": 10,
                    },
                    "uncertainties": [],
                },
                "turn": {
                    "turn_id": "fake-turn",
                    "status": "completed",
                    "duration_ms": None,
                    "reasoning_summary": None,
                    "run_home": None,
                },
            }
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _HealthServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _start_health_server(*, model_name: str) -> _HealthServer:
    return _start_health_server_with_state(model_name=model_name, model_loaded=True)


def _start_header_health_server(*, model_name: str) -> _HealthServer:
    class HeaderHealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            if self.headers.get("X-Sidecar-Auth") != "expected-secret":
                self.send_response(403)
                self.end_headers()
                return
            body = json.dumps(
                {
                    "status": "ok",
                    "model_name": model_name,
                    "model_loaded": True,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress noisy access logs in tests."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), HeaderHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return _HealthServer(
        base_url=f"http://127.0.0.1:{port}",
        server=server,
        thread=thread,
    )


def _start_health_server_with_state(
    *,
    model_name: str,
    model_loaded: bool,
) -> _HealthServer:
    def health_route(_payload: JsonObject) -> JsonObject:
        return {
            "status": "ok",
            "model_name": model_name,
            "model_loaded": model_loaded,
        }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_json_handler({"/health": health_route})
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return _HealthServer(
        base_url=f"http://127.0.0.1:{port}",
        server=server,
        thread=thread,
    )


def _sample() -> SceneFunc3dSample:
    return SceneFunc3dSample(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the lower drawer.",
        annotation_ids=("annot-a",),
        motion_hints=(
            SceneFuncMotionHint(
                motion_id="motion-a",
                annotation_id="annot-a",
                motion_type="trans",
                motion_dir=(1.0, 0.0, 0.0),
            ),
        ),
    )


def _mask_task(
    *,
    sample: SceneFunc3dSample,
    scene_root: Path,
    output_dir: Path,
    backend_config_path: Path,
    tool_context_path: Path | None = None,
    tool_cli_module: str = runner.DEFAULT_TOOL_CLI_MODULE,
) -> SceneFunc3dMaskTask:
    return SceneFunc3dMaskTask(
        sample=sample,
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=backend_config_path,
        tool_context_path=(
            output_dir / "tool_context.json"
            if tool_context_path is None
            else tool_context_path
        ),
        tool_cli_module=tool_cli_module,
    )


def _write_backend_config(
    path: Path,
    *,
    molmo_url: str,
    sam_url: str,
    root: Path,
    request_headers_path: Path | None = None,
) -> None:
    header_lines: list[str] = []
    if request_headers_path is not None:
        header_lines.append(
            f"request_headers_path = {json.dumps(str(request_headers_path))}"
        )
    path.write_text(
        "\n".join(
            (
                f"molmo_url = {json.dumps(molmo_url)}",
                f"sam_url = {json.dumps(sam_url)}",
                *header_lines,
                "request_timeout_seconds = 2.0",
                f"allowed_image_roots = [{json.dumps(str(root))}]",
                f"allowed_output_roots = [{json.dumps(str(root / 'out'))}]",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_scene(root: Path) -> None:
    _write_backend_config(
        root.parent / "backends.toml",
        molmo_url="http://127.0.0.1:8001",
        sam_url="http://127.0.0.1:8002",
        root=root.parent,
    )
    scene_dir = root / "421254"
    _write_scene_root(scene_dir)
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_two_sample_scene(root: Path) -> None:
    _write_backend_config(
        root.parent / "backends.toml",
        molmo_url="http://127.0.0.1:8001",
        sam_url="http://127.0.0.1:8002",
        root=root.parent,
    )
    scene_dir = root / "421254"
    _write_scene_root(scene_dir)
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    },
                    {
                        "desc_id": "desc-b",
                        "annot_id": ["annot-b"],
                        "description": "Press the lower button.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    },
                    {
                        "motion_id": "motion-b",
                        "annot_id": "annot-b",
                        "motion_type": "press",
                        "motion_dir": [0.0, 0.0, -1.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [10, 12, 99],
                    },
                    {
                        "annot_id": "annot-b",
                        "label": "press",
                        "indices": [10],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_unscorable_scene(root: Path) -> None:
    _write_two_sample_scene(root)
    scene_dir = root / "421254"
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                    },
                    {
                        "annot_id": "annot-b",
                        "label": "press",
                        "indices": [10],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_scene_root(
    scene_dir: Path, *, frame_ids: tuple[str, ...] = ("000010", "000020")
) -> Path:
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"fake-png")
    _write_raw_mesh(raw_dir / "mesh.ply", vertex_count=100)
    return scene_dir


def _write_raw_mesh(path: Path, *, vertex_count: int) -> None:
    points = np.zeros(
        vertex_count,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]),
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + points.tobytes())
