"""ALFRED orchestration tests with real memory/execution and fake external ports."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import TracebackType
from typing import TypeVar, cast

import pytest
from PIL import Image

from codex_agent.alfred import provenance
from codex_agent.alfred.dataset import create_pilot_manifest
from codex_agent.alfred.execution import ExecutionConfig
from codex_agent.alfred.perception import (
    PERCEPTION_PROFILE,
    PerceptionRuntimeStats,
    ScenePerceptionConfig,
)
from codex_agent.alfred.policy import AlfredEpisodeTurnOutcome, AlfredPolicyTask
from codex_agent.alfred.simulator import (
    SimulatorConfig,
    SimulatorObservation,
    SimulatorScore,
)
from codex_agent.alfred.tools.__main__ import call_episode_server
from codex_agent.cli import run_alfred
from codex_agent.config import CodexAgentConfig
from codex_agent.evaluation import alfred_runner
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.tasks.base import CodexTask
from codex_agent.tests.test_alfred_dataset import _write_dataset
from codex_agent.tests.test_alfred_execution import FakePerception
from codex_agent.tests.test_alfred_settings import local_scene_config

OutcomeT = TypeVar("OutcomeT")


class _FakePerception(FakePerception):
    instances = 0

    def __init__(self, config: ScenePerceptionConfig, *, log_path: Path) -> None:
        type(self).instances += 1
        super().__init__(include_sink_and_faucet=False)

    def __enter__(self) -> _FakePerception:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    @property
    def runtime_provenance(self) -> provenance.RuntimeProvenance:
        return provenance.RuntimeProvenance(
            executable="/fixture/worker/bin/python",
            prefix="/fixture/worker",
            package_root=str(Path(provenance.__file__).parent),
            package_source_sha256=provenance.alfred_source_sha256(),
            visible_devices="",
            packages={
                name: provenance.PackageProvenance(
                    version="worker-fixture-version",
                    distribution_record_sha256="f" * 64,
                )
                for name in provenance.PERCEPTION_PACKAGES
            },
        )

    def statistics(self) -> PerceptionRuntimeStats:
        return PerceptionRuntimeStats(
            load_seconds=0,
            inference_seconds=0,
            frames=len(self.frames),
            peak_cuda_memory_bytes=0,
            device="cpu",
            depth_seconds=0,
            detection_seconds=0,
            segmentation_seconds=0,
            feature_seconds=0,
            capped_detections=0,
            nms_suppressed=0,
            empty_masks=0,
            gpu_wait_seconds=0,
            residency="sequential_cpu_offload",
        )


class _FakeSimulator:
    instances = 0
    closed = 0
    score_calls = 0
    fail_instance: int | None = None

    def __init__(self, config: SimulatorConfig) -> None:
        saved = json.loads((config.output_dir.parents[2] / "run.json").read_text())
        assert saved["perception"]["provenance_status"] == "worker_reported"
        type(self).instances += 1
        self.index = type(self).instances
        self.output = config.output_dir
        self.steps = 0

    def __enter__(self) -> _FakeSimulator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        type(self).closed += 1

    def reset(self, trajectory_path: Path) -> SimulatorObservation:
        if self.index == type(self).fail_instance:
            raise RuntimeError("simulator disconnected")
        return self._observation(True)

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation:
        self.steps += 1
        return self._observation(self.steps != 2)

    def score(self) -> SimulatorScore:
        assert self.steps == 2
        type(self).score_calls += 1
        return SimulatorScore(
            success=self.index == 1,
            completed_goal_conditions=1,
            total_goal_conditions=1 if self.index == 1 else 3,
            snapshot_path=self.output / "terminal_evaluation.json",
        )

    def _observation(self, success: bool) -> SimulatorObservation:
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"{self.steps}.png"
        Image.new("RGB", (300, 300), "gray").save(path)
        return SimulatorObservation(
            frame_id=self.steps,
            rgb_path=path,
            yaw_degrees=0,
            horizon_degrees=30,
            last_action_success=success,
            last_action_error="" if success else "collision",
        )


class _FakeRuntime:
    def __init__(self, config: CodexAgentConfig) -> None:
        assert config.sandbox == "workspace_write"
        assert config.sandbox_network_access is True
        self.calls = 0

    def execute(self, task: CodexTask[OutcomeT]) -> CodexTaskResult[OutcomeT]:
        self.calls += 1
        assert isinstance(task, AlfredPolicyTask)
        episode_dir = task.episode_dir
        # Trigger 2 primitive steps via scan_surroundings (which stops on failed step 2)
        call_episode_server(episode_dir, "scan_surroundings", {"pitches": [30.0]})
        call_episode_server(
            episode_dir, "finish_episode", {"summary": "Finished test episode"}
        )
        outcome = AlfredEpisodeTurnOutcome(
            status="completed",
            summary="Completed test turn",
            completed_subgoals=["scan"],
        )
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(OutcomeT, outcome),
            turn=CodexTurnResult(
                final_response=outcome.model_dump_json(),
                metadata=CodexTurnMetadata(
                    usage={"total": {"input_tokens": 120, "output_tokens": 30}}
                ),
            ),
        )


@pytest.fixture
def run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> alfred_runner.AlfredRunConfig:
    source = tmp_path / "source"
    source.mkdir()
    dataset_root = tmp_path / "dataset"
    splits = _write_dataset(dataset_root)
    fold = tmp_path / "fold.json"
    fold.write_text(
        create_pilot_manifest(dataset_root, splits, per_split=7).model_dump_json()
    )
    perception = local_scene_config(tmp_path / "assets")
    monkeypatch.setattr(
        alfred_runner, "code_provenance", lambda path: {"head_sha": "abc"}
    )
    monkeypatch.setattr(
        alfred_runner, "external_source_provenance", lambda path: {"head_sha": "def"}
    )
    monkeypatch.setattr(
        alfred_runner,
        "model_source_provenance",
        lambda: {"groundingdino": {"version": "fixture", "source_sha256": "f" * 64}},
    )
    monkeypatch.setattr(alfred_runner, "preflight_codex_runtime", lambda config: None)
    monkeypatch.setattr(alfred_runner, "ScenePerception", _FakePerception)
    monkeypatch.setattr(alfred_runner, "ThorSimulator", _FakeSimulator)
    monkeypatch.setattr(alfred_runner, "AlfredPolicyRuntime", _FakeRuntime)
    _FakePerception.instances = 0
    _FakeSimulator.instances = _FakeSimulator.closed = 0
    _FakeSimulator.score_calls = 0
    _FakeSimulator.fail_instance = None
    output = tmp_path / "run"
    return alfred_runner.AlfredRunConfig(
        source_root=source,
        data_root=dataset_root,
        splits_path=splits,
        fold_path=fold,
        output_dir=output,
        run_id="test-run",
        database_path=tmp_path / "runs.sqlite",
        simulator=SimulatorConfig(Path(sys.executable), tmp_path, output),
        perception=perception,
        runtime=CodexAgentConfig(
            project_root=source, codex_home=source / ".codex-home"
        ),
        execution=ExecutionConfig(max_actions=2, max_decisions=4, initial_scan=False),
        command=("python", "-m", "codex_agent.cli.run_alfred"),
        shard_index=0,
        num_shards=7,
    )


def test_runner_counts_failed_actions_and_aggregates_official_splits(
    run_config: alfred_runner.AlfredRunConfig,
) -> None:
    summary = alfred_runner.run_validation(run_config)
    assert summary.selected == 2
    assert summary.completed == 2
    assert summary.metrics is not None
    assert summary.metrics.success_rate == 0.5
    assert summary.metrics.goal_condition_success_rate == 0.5
    assert summary.metrics.executed_actions == 4
    assert summary.by_split["valid_seen"] is not None
    assert summary.by_split["valid_seen"].success_rate == 1
    assert summary.by_split["valid_unseen"] is not None
    assert summary.by_split["valid_unseen"].success_rate == 0
    assert _FakePerception.instances == 1
    assert _FakeSimulator.closed == 2
    assert _FakeSimulator.score_calls == 2
    saved = json.loads((run_config.output_dir / "run.json").read_text())
    assert saved["perception"]["provenance_status"] == "worker_reported"
    assert saved["perception"]["runtime_provenance"]["prefix"] == "/fixture/worker"
    assert (
        saved["perception"]["libraries"]["torch"]["version"] == "worker-fixture-version"
    )
    with sqlite3.connect(run_config.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (2,)


def test_error_preserves_partial_results_and_is_reraised(
    run_config: alfred_runner.AlfredRunConfig,
) -> None:
    _FakeSimulator.fail_instance = 2
    with pytest.raises(RuntimeError, match="simulator disconnected"):
        alfred_runner.run_validation(run_config)
    summary = json.loads((run_config.output_dir / "summary.json").read_text())
    assert summary["selected"] == 2
    assert summary["completed"] == 1
    assert summary["errors"] == 1
    assert summary["metrics"]["num_evals"] == 1
    assert _FakeSimulator.score_calls == 1
    with sqlite3.connect(run_config.database_path) as connection:
        assert connection.execute(
            "SELECT status, score_json FROM attempts WHERE status = 'error'"
        ).fetchone() == ("error", None)


def test_initialization_failure_preserves_unknown_model_provenance(
    run_config: alfred_runner.AlfredRunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        raise RuntimeError("model worker did not initialize")

    monkeypatch.setattr(alfred_runner, "ScenePerception", unavailable)
    with pytest.raises(RuntimeError, match="did not initialize"):
        alfred_runner.run_validation(run_config)
    saved = json.loads((run_config.output_dir / "run.json").read_text())
    assert saved["status"] == "error"
    assert saved["perception"]["libraries"] is None
    assert saved["perception"]["runtime_provenance"] is None
    assert (
        saved["perception"]["provenance_status"] == "unavailable_initialization_failed"
    )
    assert _FakeSimulator.instances == 0


def test_existing_run_directory_is_not_overwritten(
    run_config: alfred_runner.AlfredRunConfig,
) -> None:
    run_config.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        alfred_runner.run_validation(run_config)
    assert _FakePerception.instances == 0


def test_run_metadata_fingerprints_actual_scene_assets_and_records_new_profile(
    run_config: alfred_runner.AlfredRunConfig,
) -> None:
    before = alfred_runner._run_metadata(run_config, (), {}, full_fold_size=50)
    assert before["perception_profile"] == PERCEPTION_PROFILE
    assert "film" not in cast(dict[str, object], before["external_sources"])
    run_config.perception.dino_checkpoint.write_bytes(b"changed-model-fixture")
    after = alfred_runner._run_metadata(run_config, (), {}, full_fold_size=50)
    assert before["perception"] != after["perception"]
    assert before["external_sources"] == after["external_sources"]


def test_cli_rejects_nonpositive_limit() -> None:
    with pytest.raises(SystemExit):
        run_alfred.build_arg_parser().parse_args(["--limit", "0"])


def test_external_source_digest_detects_untracked_python_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = tmp_path / "tracked.py"
    untracked = tmp_path / "untracked.py"
    tracked.write_text("tracked = True\n")
    untracked.write_text("version = 1\n")

    def git_output(command: list[str], **kwargs: object) -> bytes | str:
        if "rev-parse" in command:
            return "fixed-head\n"
        if "diff" in command:
            return b""
        return b"tracked.py\0untracked.py\0"

    monkeypatch.setattr(provenance.subprocess, "check_output", git_output)
    before = alfred_runner.external_source_provenance(tmp_path)
    untracked.write_text("version = 2\n")
    after = alfred_runner.external_source_provenance(tmp_path)
    assert before["head_sha"] == after["head_sha"]
    assert before["tracked_diff_sha256"] == after["tracked_diff_sha256"]
    assert before["python_source_sha256"] != after["python_source_sha256"]
    assert after["python_source_count"] == 2


def test_continue_on_error_retries_transient_failure_and_tracks_concurrency_state(
    run_config: alfred_runner.AlfredRunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    monkeypatch.setattr(alfred_runner.time, "sleep", lambda _: None)
    _FakeSimulator.fail_instance = 1
    state_file = run_config.output_dir.parent / "concurrency.json"
    cfg = replace(
        run_config,
        continue_on_error=True,
        concurrency_state_path=state_file,
    )
    summary = alfred_runner.run_validation(cfg)
    assert summary.selected == 2
    assert summary.completed == 2
    assert summary.errors == 0
    state = json.loads(state_file.read_text())
    assert len(state["recent_outcomes"]) == 2
    assert state["recent_outcomes"][0]["had_retry"] is True
