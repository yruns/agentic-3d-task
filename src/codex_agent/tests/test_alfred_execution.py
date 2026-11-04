"""Closed-loop execution tests with fake external services and real scene memory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.execution import (
    EpisodeExecutor,
    ExecutionConfig,
    ExecutionOutcome,
)
from codex_agent.alfred.perception import PerceptionDetection, PerceptionObservation
from codex_agent.alfred.policy import AlfredDecision, AlfredPolicyTask
from codex_agent.alfred.simulator import SimulatorObservation
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.tasks.base import CodexTask

ResultT = TypeVar("ResultT")


class FakeEnvironment:
    """Produce RGB/proprioception while making any hidden scoring call fail."""

    def __init__(
        self,
        root: Path,
        *,
        successes: Sequence[bool] = (),
        yaw_degrees: float = 0.0,
    ) -> None:
        self.root = root
        self.root.mkdir()
        self.successes = successes
        self.yaw_degrees = yaw_degrees
        self.frame_id = 0
        self.calls: list[tuple[str, Path | None]] = []
        self.score_calls = 0
        self.hidden_state = "PRIVATE_SCORING_LABELS"

    def observe(self, *, success: bool = True) -> SimulatorObservation:
        path = self.root / f"{self.frame_id}.png"
        Image.new("RGB", (300, 300), (200, 30, 20)).save(path)
        return SimulatorObservation(
            frame_id=self.frame_id,
            rgb_path=path,
            yaw_degrees=self.yaw_degrees,
            horizon_degrees=0,
            last_action_success=success,
            last_action_error="" if success else "Action blocked",
        )

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation:
        index = len(self.calls)
        success = self.successes[index] if index < len(self.successes) else True
        self.calls.append((action, mask_path))
        if success and action == "RotateRight":
            self.yaw_degrees = (self.yaw_degrees + 90) % 360
        if success and action == "RotateLeft":
            self.yaw_degrees = (self.yaw_degrees - 90) % 360
        self.frame_id += 1
        return self.observe(success=success)

    def score(self) -> None:
        self.score_calls += 1
        raise AssertionError("terminal scorer must not be queried by execution")


class FakePerception:
    """Supply current-frame arrays while recording exactly which frames were read."""

    def __init__(self, *, visible_frames: set[int] | None = None) -> None:
        self.visible_frames = visible_frames
        self.frames: list[int] = []

    def predict(
        self, rgb_path: Path, *, frame_id: str, horizon_degrees: float, output_dir: Path
    ) -> PerceptionObservation:
        assert rgb_path.is_file()
        index = int(frame_id)
        self.frames.append(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / f"depth_{index}.npy"
        np.save(depth_path, np.full((300, 300), 2.0, dtype=np.float32))
        detections: tuple[PerceptionDetection, ...] = ()
        if self.visible_frames is None or index in self.visible_frames:
            mask_path = output_dir / f"mask_{index}.npy"
            mask = np.zeros((300, 300), dtype=np.uint8)
            mask[120:180, 120:180] = 1
            np.save(mask_path, mask)
            detections = (
                PerceptionDetection(
                    category="Mug",
                    mask_path=mask_path,
                    bbox_xyxy=(120, 120, 180, 180),
                    confidence=0.9,
                ),
            )
        return PerceptionObservation(
            frame_id=frame_id,
            depth_path=depth_path,
            depth_model="level",
            detections=detections,
        )


class FakeRuntime:
    """Return planned decisions through the real task parser and capture actor inputs."""

    def __init__(self, decisions: Sequence[AlfredDecision]) -> None:
        self.decisions = iter(decisions)
        self.tasks: list[AlfredPolicyTask] = []
        self.usage_known = True

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        assert isinstance(task, AlfredPolicyTask)
        self.tasks.append(task)
        decision = next(self.decisions)
        response = decision.model_dump_json()
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=task.parse_response(response),
            turn=CodexTurnResult(
                final_response=response,
                metadata=CodexTurnMetadata(
                    input_tokens=5,
                    usage=(
                        {
                            "total": {"input_tokens": 10, "output_tokens": 7},
                            "last": {"input_tokens": 5, "output_tokens": 3},
                        }
                        if self.usage_known
                        else None
                    ),
                ),
            ),
        )


def _decision(**updates: object) -> AlfredDecision:
    return AlfredDecision.model_validate(
        {
            "kind": "stop",
            "action": None,
            "object_id": None,
            "cell": None,
            "repeat": 1,
            "rationale": "Synthetic service-boundary test",
            "memory_note": "Keep this note",
            **updates,
        }
    )


def _executor(
    tmp_path: Path,
    decisions: Sequence[AlfredDecision],
    *,
    config: ExecutionConfig | None = None,
    visible_frames: set[int] | None = None,
    successes: Sequence[bool] = (),
    yaw_degrees: float = 0.0,
) -> EpisodeExecutor:
    return EpisodeExecutor(
        environment=FakeEnvironment(
            tmp_path / "simulator", successes=successes, yaw_degrees=yaw_degrees
        ),
        perception=FakePerception(visible_frames=visible_frames),
        runtime=FakeRuntime(decisions),
        output_dir=tmp_path / "episode",
        config=config if config is not None else ExecutionConfig(),
    )


def _run(executor: EpisodeExecutor) -> ExecutionOutcome:
    assert isinstance(executor.environment, FakeEnvironment)
    return executor.run(
        AlfredInstruction("Put the mug on a table", ("PRIVATE_STEP_INSTRUCTION",)),
        executor.environment.observe(),
    )


def test_repeated_actions_integrate_every_intermediate_observation(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="act", action="MoveAhead", repeat=3), _decision()],
    )
    outcome = _run(executor)
    assert outcome.actions == 3
    assert outcome.stop_reason == "policy_stop"
    assert tuple(executor.memory.frames) == (0, 1, 2, 3)
    assert isinstance(executor.perception, FakePerception)
    assert executor.perception.frames == [0, 1, 2, 3]
    assert executor.odometry.agent_position == (0.0, 0.75, 0.0)
    assert (outcome.input_tokens, outcome.output_tokens) == (20, 14)
    assert isinstance(executor.runtime, FakeRuntime)
    assert executor.runtime.tasks[1].context["previous_note"] == "Keep this note"


def test_interaction_uses_reobserved_mask_after_movement(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision(kind="act", action="MoveAhead"),
            _decision(kind="act", action="PickupObject", object_id=0),
            _decision(),
        ],
    )
    outcome = _run(executor)
    assert outcome.actions == 2
    assert isinstance(executor.environment, FakeEnvironment)
    assert executor.environment.calls[0] == ("MoveAhead", None)
    action, mask_path = executor.environment.calls[1]
    assert action == "PickupObject"
    assert mask_path is not None and mask_path.name == "mask_1.npy"
    assert executor.memory.held_object_id == 0


def test_missing_provider_usage_remains_unknown_instead_of_zero(tmp_path: Path) -> None:
    executor = _executor(tmp_path, [_decision()])
    assert isinstance(executor.runtime, FakeRuntime)
    executor.runtime.usage_known = False
    outcome = _run(executor)
    assert outcome.input_tokens is None
    assert outcome.output_tokens is None


def test_historical_target_without_current_mask_cannot_be_interacted_with(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision(kind="act", action="MoveAhead"),
            _decision(kind="act", action="PickupObject", object_id=0),
            _decision(),
        ],
        visible_frames={0},
    )
    outcome = _run(executor)
    assert outcome.actions == 1
    assert outcome.failures == 0
    assert isinstance(executor.runtime, FakeRuntime)
    assert "no mask in the current observation" in str(
        executor.runtime.tasks[-1].context["feedback"]
    )


def test_stop_uses_no_simulator_action_and_actor_never_reads_terminal_truth(
    tmp_path: Path,
) -> None:
    executor = _executor(tmp_path, [_decision()])
    outcome = _run(executor)
    assert (outcome.actions, outcome.failures, outcome.decisions) == (0, 0, 1)
    assert isinstance(executor.environment, FakeEnvironment)
    assert executor.environment.calls == []
    assert executor.environment.score_calls == 0
    assert isinstance(executor.runtime, FakeRuntime)
    prompt = executor.runtime.tasks[0].build_turn_request().prompt
    assert "PRIVATE_SCORING_LABELS" not in prompt
    assert "PRIVATE_STEP_INSTRUCTION" not in prompt
    assert "completed_goal_conditions" not in prompt


def test_action_budget_interrupts_repeat_before_extra_step(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="act", action="MoveAhead", repeat=8)],
        config=ExecutionConfig(max_actions=2),
    )
    outcome = _run(executor)
    assert outcome.actions == 2
    assert outcome.stop_reason == "action_budget"
    assert len(executor.memory.frames) == 3


def test_failure_budget_stops_repetition_and_preserves_failed_observation(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="act", action="MoveAhead", repeat=8)],
        config=ExecutionConfig(max_failures=1),
        successes=[False],
    )
    outcome = _run(executor)
    assert (outcome.actions, outcome.failures) == (1, 1)
    assert outcome.stop_reason == "failure_budget"
    assert executor.odometry.agent_position == (0.0, 0.0, 0.0)
    assert executor.memory.cell_state((0, 1)) == "occupied"
    assert tuple(executor.memory.frames) == (0, 1)


def test_failed_westward_move_marks_attempted_cell_without_float_drift(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="act", action="MoveAhead")],
        config=ExecutionConfig(max_failures=1),
        successes=[False],
        yaw_degrees=270.0,
    )
    _run(executor)
    assert executor.memory.cell_state((-1, 0)) == "occupied"


def test_action_budget_reached_on_last_decision_reports_action_limit(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="act", action="MoveAhead", repeat=8)],
        config=ExecutionConfig(max_actions=1, max_decisions=1),
    )
    outcome = _run(executor)
    assert outcome.actions == 1
    assert outcome.stop_reason == "action_budget"


def test_retrieval_uses_only_seen_frames_and_does_not_move(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="retrieve", object_id=0), _decision()],
    )
    outcome = _run(executor)
    assert outcome.actions == 0
    assert isinstance(executor.runtime, FakeRuntime)
    task = executor.runtime.tasks[-1]
    assert task.context["retrieved_frame_ids"] == [0]
    assert len(task.image_paths) == 3
    assert task.image_paths[-1] == executor.memory.frames[0].rgb_path


def test_unknown_retrieval_target_is_feedback_and_does_not_make_up_evidence(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision(kind="retrieve", object_id=999), _decision()],
    )
    outcome = _run(executor)
    assert outcome.actions == 0
    assert isinstance(executor.runtime, FakeRuntime)
    task = executor.runtime.tasks[-1]
    assert task.context["retrieved_frame_ids"] == []
    assert "never been observed" in str(task.context["feedback"])
    assert len(task.image_paths) == 2


@pytest.mark.parametrize("name", ["max_actions", "max_failures", "max_decisions"])
def test_budget_rejects_boolean_and_zero(name: str) -> None:
    for value in (False, 0):
        with pytest.raises(ValueError, match=name):
            # Deliberately cross the typed boundary with an invalid dynamic value.
            ExecutionConfig(**{name: value})  # type: ignore[arg-type]
