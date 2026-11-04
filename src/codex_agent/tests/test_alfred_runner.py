"""ALFRED orchestration tests with real memory/execution and fake external ports."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import TracebackType
from typing import TypeVar, cast

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.dataset import (
    VALIDATION_SPLITS,
    AlfredEpisode,
    AlfredInstruction,
)
from codex_agent.alfred.execution import ExecutionConfig
from codex_agent.alfred.perception import (
    FILM_CHECKPOINTS,
    FilmPerceptionConfig,
    PerceptionObservation,
    PerceptionRuntimeStats,
)
from codex_agent.alfred.policy import AlfredDecision
from codex_agent.alfred.simulator import (
    SimulatorConfig,
    SimulatorObservation,
    SimulatorScore,
)
from codex_agent.cli import run_alfred
from codex_agent.config import CodexAgentConfig
from codex_agent.evaluation import alfred_runner
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.tasks.base import CodexTask

OutcomeT = TypeVar("OutcomeT")


class _FakePerception:
    instances = 0

    def __init__(self, config: FilmPerceptionConfig, *, log_path: Path) -> None:
        type(self).instances += 1
        self.frames = 0

    def __enter__(self) -> _FakePerception:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def predict(
        self, rgb_path: Path, *, frame_id: str, horizon_degrees: float, output_dir: Path
    ) -> PerceptionObservation:
        self.frames += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / f"{frame_id}.npy"
        np.save(depth_path, np.full((300, 300), 2, dtype=np.float32))
        return PerceptionObservation(
            frame_id=frame_id,
            depth_path=depth_path,
            depth_model="downward",
            detections=(),
        )

    def statistics(self) -> PerceptionRuntimeStats:
        return PerceptionRuntimeStats(
            load_seconds=0,
            inference_seconds=0,
            frames=self.frames,
            peak_cuda_memory_bytes=0,
            device="cpu",
        )


class _FakeSimulator:
    instances = 0
    closed = 0
    fail_instance: int | None = None

    def __init__(self, config: SimulatorConfig) -> None:
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
        assert self.steps == 2  # Includes the failed second primitive.
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
        assert config.project_root.name == "actor"
        self.calls = 0

    def execute(self, task: CodexTask[OutcomeT]) -> CodexTaskResult[OutcomeT]:
        self.calls += 1
        action = self.calls % 2 == 1
        decision = AlfredDecision.model_validate(
            {
                "command": (
                    {"kind": "motion", "action": "MoveAhead", "repeat": 2}
                    if action
                    else {"kind": "stop"}
                ),
                "rationale": "test external policy",
                "memory_note": "",
            }
        )
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(OutcomeT, decision),
            turn=CodexTurnResult(
                final_response=decision.model_dump_json(),
                metadata=CodexTurnMetadata(input_tokens=12),
            ),
        )


@pytest.fixture
def run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> alfred_runner.AlfredRunConfig:
    source = tmp_path / "source"
    source.mkdir()
    fold, splits = tmp_path / "fold.json", tmp_path / "splits.json"
    fold.write_text("{}")
    splits.write_text("{}")
    weights = tmp_path / "weights"
    for name in FILM_CHECKPOINTS:
        path = weights / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    episodes = tuple(
        AlfredEpisode(
            split=split,
            task=f"type/trial{index}",
            repeat_idx=0,
            task_id=f"trial{index}",
            task_type="pick_and_place_simple",
            trajectory_path=tmp_path / f"trajectory{index}.json",
            trajectory_sha256="a" * 64,
            expert_actions=10,
            instruction=AlfredInstruction("Move an object.", ()),
        )
        for index, split in enumerate(VALIDATION_SPLITS)
    )
    monkeypatch.setattr(alfred_runner, "load_manifest", lambda *args: episodes)
    monkeypatch.setattr(
        alfred_runner, "code_provenance", lambda path: {"head_sha": "abc"}
    )
    monkeypatch.setattr(
        alfred_runner, "external_source_provenance", lambda path: {"head_sha": "def"}
    )
    monkeypatch.setattr(alfred_runner, "preflight_codex_runtime", lambda config: None)
    monkeypatch.setattr(alfred_runner, "FilmPerception", _FakePerception)
    monkeypatch.setattr(alfred_runner, "ThorSimulator", _FakeSimulator)
    monkeypatch.setattr(alfred_runner, "AlfredPolicyRuntime", _FakeRuntime)
    _FakePerception.instances = 0
    _FakeSimulator.instances = _FakeSimulator.closed = 0
    _FakeSimulator.fail_instance = None
    output = tmp_path / "run"
    return alfred_runner.AlfredRunConfig(
        source_root=source,
        data_root=tmp_path,
        splits_path=splits,
        fold_path=fold,
        output_dir=output,
        run_id="test-run",
        database_path=tmp_path / "runs.sqlite",
        simulator=SimulatorConfig(Path(sys.executable), tmp_path, output),
        perception=FilmPerceptionConfig(
            Path(sys.executable), tmp_path, weights, device="cpu"
        ),
        runtime=CodexAgentConfig(
            project_root=source, codex_home=source / ".codex-home"
        ),
        execution=ExecutionConfig(max_actions=5, max_decisions=4),
        command=("python", "-m", "codex_agent.cli.run_alfred"),
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
    with sqlite3.connect(run_config.database_path) as connection:
        assert connection.execute(
            "SELECT status, score_json FROM attempts WHERE status = 'error'"
        ).fetchone() == ("error", None)


def test_existing_run_directory_is_not_overwritten(
    run_config: alfred_runner.AlfredRunConfig,
) -> None:
    run_config.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        alfred_runner.run_validation(run_config)
    assert _FakePerception.instances == 0


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

    monkeypatch.setattr(alfred_runner.subprocess, "check_output", git_output)
    before = alfred_runner.external_source_provenance(tmp_path)
    untracked.write_text("version = 2\n")
    after = alfred_runner.external_source_provenance(tmp_path)
    assert before["head_sha"] == after["head_sha"]
    assert before["tracked_diff_sha256"] == after["tracked_diff_sha256"]
    assert before["python_source_sha256"] != after["python_source_sha256"]
    assert after["python_source_count"] == 2
