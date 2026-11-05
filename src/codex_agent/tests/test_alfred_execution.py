"""Closed-loop execution tests with fake external services and real scene memory."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.execution import (
    EpisodeExecutor,
    ExecutionConfig,
    ExecutionOutcome,
)
from codex_agent.alfred.memory import MappingConfig, MemoryConfig
from codex_agent.alfred.perception import (
    PerceptionDetection,
    PerceptionDiagnostics,
    PerceptionObservation,
)
from codex_agent.alfred.policy import AlfredDecision, AlfredPolicyTask
from codex_agent.alfred.simulator import SimulatorObservation
from codex_agent.alfred.vocabulary import ALFRED_CATEGORIES, detection_batches
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
        self.horizon_degrees = 0.0
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
            horizon_degrees=self.horizon_degrees,
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
        if success and action == "LookDown":
            self.horizon_degrees = min(60.0, self.horizon_degrees + 30)
        if success and action == "LookUp":
            self.horizon_degrees = max(-30.0, self.horizon_degrees - 30)
        self.frame_id += 1
        return self.observe(success=success)

    def score(self) -> None:
        self.score_calls += 1
        raise AssertionError("terminal scorer must not be queried by execution")


class FakePerception:
    """Supply current-frame arrays while recording exactly which frames were read."""

    def __init__(
        self,
        *,
        visible_frames: set[int] | None = None,
        features_by_frame: Mapping[int, tuple[float, float, float]] | None = None,
        include_container: bool = False,
    ) -> None:
        self.visible_frames = visible_frames
        self.features_by_frame = features_by_frame or {}
        self.include_container = include_container
        self.frames: list[int] = []

    def predict(
        self,
        rgb_path: Path,
        *,
        frame_id: str,
        intrinsics: NDArray[np.float64],
        output_dir: Path,
    ) -> PerceptionObservation:
        assert rgb_path.is_file()
        assert intrinsics.shape == (3, 3)
        index = int(frame_id)
        self.frames.append(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / f"depth_{index}.npy"
        np.save(
            depth_path,
            np.full((300, 300), max(0.75, 2.0 - index * 0.25), dtype=np.float32),
        )
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
            if self.include_container:
                container_path = output_dir / f"container_{index}.npy"
                container_mask = np.zeros_like(mask)
                container_mask[90:210, 90:210] = 1
                np.save(container_path, container_mask)
                detections = (
                    PerceptionDetection(
                        category="Cabinet",
                        mask_path=container_path,
                        bbox_xyxy=(90, 90, 210, 210),
                        confidence=0.95,
                    ),
                    *detections,
                )
        features_path = output_dir / f"features_{index}.npz"
        feature = np.asarray(
            self.features_by_frame.get(index, (1.0, 0.0, 0.0)), dtype=np.float32
        )
        np.savez(
            features_path,
            frame_feature=feature,
            image_features=np.tile(feature, (len(detections), 1)),
            text_features=np.tile(
                np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(detections), 1)
            ),
        )
        return PerceptionObservation(
            frame_id=frame_id,
            depth_path=depth_path,
            depth_model="da3metric-large",
            image_width=300,
            image_height=300,
            features_path=features_path,
            detections=detections,
            diagnostics=PerceptionDiagnostics(
                vocabulary_categories=len(ALFRED_CATEGORIES),
                caption_batches=len(detection_batches(24)),
                candidate_boxes=len(detections),
                nms_suppressed=0,
                capped_detections=0,
                empty_masks=0,
            ),
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
        response = decision.to_wire().model_dump_json()
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


def _decision(command: dict[str, object] | None = None) -> AlfredDecision:
    return AlfredDecision.model_validate(
        {
            "command": command if command is not None else {"kind": "stop"},
            "rationale": "Synthetic service-boundary test",
            "memory_note": "Keep this note",
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
    memory_config: MemoryConfig | None = None,
    features_by_frame: Mapping[int, tuple[float, float, float]] | None = None,
) -> EpisodeExecutor:
    return EpisodeExecutor(
        environment=FakeEnvironment(
            tmp_path / "simulator", successes=successes, yaw_degrees=yaw_degrees
        ),
        perception=FakePerception(
            visible_frames=visible_frames, features_by_frame=features_by_frame
        ),
        runtime=FakeRuntime(decisions),
        output_dir=tmp_path / "episode",
        config=config if config is not None else ExecutionConfig(),
        memory_config=memory_config if memory_config is not None else MemoryConfig(),
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
        [
            _decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 3}),
            _decision(),
        ],
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
            _decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 1}),
            _decision(
                {
                    "kind": "interact",
                    "interaction_action": "PickupObject",
                    "target": {"type": "object", "object_id": 0},
                }
            ),
            _decision(),
        ],
    )
    outcome = _run(executor)
    assert outcome.actions == 2
    assert isinstance(executor.environment, FakeEnvironment)
    assert executor.environment.calls[0] == ("MoveAhead", None)
    action, mask_path = executor.environment.calls[1]
    assert action == "PickupObject"
    assert mask_path is not None and mask_path.name == "000002.npy"
    np.testing.assert_array_equal(
        np.load(mask_path), np.load(executor.output_dir / "perception" / "mask_1.npy")
    )
    record = json.loads(
        (executor.output_dir / "actions.jsonl").read_text().splitlines()[1]
    )
    assert record["interaction_mask"]["source_detection_index"] == 0
    assert record["interaction_mask"]["source"] == "current_mapping_mask"
    assert record["source_frame"] == 1
    assert executor.memory.held_object_id == 0


def test_interaction_dispatch_excludes_a_contained_objects_pixels(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision(
                {
                    "kind": "interact",
                    "interaction_action": "OpenObject",
                    "target": {"type": "object", "object_id": 0},
                }
            ),
            _decision(),
        ],
    )
    executor.perception = FakePerception(include_container=True)
    outcome = _run(executor)
    assert outcome.actions == 1
    assert isinstance(executor.environment, FakeEnvironment)
    action, mask_path = executor.environment.calls[0]
    assert action == "OpenObject" and mask_path is not None
    parent = np.load(executor.output_dir / "perception" / "container_0.npy").astype(
        bool
    )
    child = np.load(executor.output_dir / "perception" / "mask_0.npy").astype(bool)
    actual = np.load(mask_path)
    assert np.any(parent & child)
    assert not np.any(actual & child)
    np.testing.assert_array_equal(actual, parent & ~child)


def test_interaction_resolves_a_real_merge_alias_and_uses_the_current_mask(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 1}),
            _decision(
                {
                    "kind": "interact",
                    "interaction_action": "PickupObject",
                    "target": {"type": "object", "object_id": 1},
                }
            ),
            _decision(),
        ],
        memory_config=MemoryConfig(
            mapping=MappingConfig(
                association_threshold=1.9,
                merge_image_similarity=0.6,
                maintenance_interval=2,
            )
        ),
        features_by_frame={1: (0.7, float(np.sqrt(0.51)), 0.0)},
    )
    outcome = _run(executor)
    assert outcome.actions == 2
    assert executor.memory.aliases[1] == 0
    assert executor.memory.held_object_id == 0
    assert isinstance(executor.environment, FakeEnvironment)
    action, mask_path = executor.environment.calls[1]
    assert action == "PickupObject"
    assert mask_path is not None and mask_path.name == "000002.npy"
    np.testing.assert_array_equal(
        np.load(mask_path), np.load(executor.output_dir / "perception" / "mask_1.npy")
    )


def test_explore_obeys_atomic_action_budget_and_reobserves_every_step(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision({"kind": "explore"})],
        config=ExecutionConfig(max_actions=2, navigation_chunk=5),
    )
    outcome = _run(executor)
    assert outcome.stop_reason == "action_budget"
    assert (outcome.actions, outcome.decisions) == (2, 1)
    assert isinstance(executor.perception, FakePerception)
    assert executor.perception.frames == [0, 1, 2]
    assert tuple(executor.memory.frames) == (0, 1, 2)
    assert isinstance(executor.environment, FakeEnvironment)
    assert len(executor.environment.calls) == 2
    assert all(mask is None for _, mask in executor.environment.calls)


def test_explore_chunk_returns_control_to_policy_without_extra_atomic_actions(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision({"kind": "explore"}), _decision()],
        config=ExecutionConfig(navigation_chunk=2),
    )
    outcome = _run(executor)
    assert outcome.stop_reason == "policy_stop"
    assert (outcome.actions, outcome.decisions) == (2, 2)
    assert isinstance(executor.perception, FakePerception)
    assert executor.perception.frames == [0, 1, 2]


def test_explore_stops_at_failure_budget_after_recording_failed_observation(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision({"kind": "explore"})],
        config=ExecutionConfig(max_failures=1),
        successes=[False],
    )
    outcome = _run(executor)
    assert outcome.stop_reason == "failure_budget"
    assert (outcome.actions, outcome.failures) == (1, 1)
    assert tuple(executor.memory.frames) == (0, 1)


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
            _decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 1}),
            _decision(
                {
                    "kind": "interact",
                    "interaction_action": "PickupObject",
                    "target": {"type": "object", "object_id": 0},
                }
            ),
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
        [_decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 8})],
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
        [_decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 8})],
        config=ExecutionConfig(max_failures=1),
        successes=[False],
    )
    outcome = _run(executor)
    assert (outcome.actions, outcome.failures) == (1, 1)
    assert outcome.stop_reason == "failure_budget"
    assert executor.odometry.agent_position == (0.0, 0.0, 0.0)
    spatial = executor.memory.spatial_map
    assert spatial.cell_state(spatial.world_to_cell(0.0, 0.25)) == "occupied"
    assert tuple(executor.memory.frames) == (0, 1)


def test_failed_westward_move_marks_attempted_cell_without_float_drift(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 1})],
        config=ExecutionConfig(max_failures=1),
        successes=[False],
        yaw_degrees=270.0,
    )
    _run(executor)
    spatial = executor.memory.spatial_map
    assert spatial.cell_state(spatial.world_to_cell(-0.25, 0.0)) == "occupied"


def test_action_budget_reached_on_last_decision_reports_action_limit(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [_decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 8})],
        config=ExecutionConfig(max_actions=1, max_decisions=1),
    )
    outcome = _run(executor)
    assert outcome.actions == 1
    assert outcome.stop_reason == "action_budget"


def test_retrieval_uses_only_seen_frames_and_does_not_move(tmp_path: Path) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision({"kind": "motion", "motion_action": "MoveAhead", "repeat": 1}),
            _decision(
                {"kind": "retrieve", "target": {"type": "object", "object_id": 0}}
            ),
            _decision(),
        ],
    )
    outcome = _run(executor)
    assert outcome.actions == 1
    assert len(executor.memory.frames) == 2
    assert isinstance(executor.runtime, FakeRuntime)
    task = executor.runtime.tasks[-1]
    assert task.context["retrieved_frame_ids"] == [0]
    assert len(task.image_paths) == 3
    assert task.image_paths[-1] == executor.memory.frames[0].rgb_path


def test_retrieval_does_not_duplicate_the_current_rgb_attachment(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision(
                {"kind": "retrieve", "target": {"type": "object", "object_id": 0}}
            ),
            _decision(),
        ],
    )
    outcome = _run(executor)
    assert outcome.actions == 0
    assert isinstance(executor.runtime, FakeRuntime)
    task = executor.runtime.tasks[-1]
    assert task.context["retrieved_frame_ids"] == []
    assert "No additional historical views" in str(task.context["feedback"])
    assert len(task.image_paths) == 2


def test_repeated_retrieval_waits_for_new_observations_without_forcing_actions(
    tmp_path: Path,
) -> None:
    move: dict[str, object] = {
        "kind": "motion",
        "motion_action": "MoveAhead",
        "repeat": 1,
    }
    retrieve: dict[str, object] = {
        "kind": "retrieve",
        "target": {"type": "object", "object_id": 0},
    }
    executor = _executor(
        tmp_path,
        [
            _decision(move),
            _decision(retrieve),
            _decision(retrieve),
            _decision(move),
            _decision(retrieve),
            _decision(),
        ],
    )
    outcome = _run(executor)
    assert outcome.actions == 2
    assert isinstance(executor.environment, FakeEnvironment)
    assert executor.environment.calls == [("MoveAhead", None), ("MoveAhead", None)]
    assert isinstance(executor.runtime, FakeRuntime)
    tasks = executor.runtime.tasks
    assert tasks[2].context["retrieved_frame_ids"] == [0]
    assert tasks[3].context["retrieved_frame_ids"] == []
    assert len(tasks[3].image_paths) == 2
    assert "obtain a new observation" in str(tasks[3].context["feedback"])
    assert tasks[5].context["retrieved_frame_ids"] == [1]
    assert tasks[5].image_paths[-1] == executor.memory.frames[1].rgb_path


def test_unknown_retrieval_target_is_feedback_and_does_not_make_up_evidence(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        [
            _decision(
                {"kind": "retrieve", "target": {"type": "object", "object_id": 999}}
            ),
            _decision(),
        ],
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
