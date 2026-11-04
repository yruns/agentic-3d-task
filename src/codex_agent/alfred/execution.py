"""Closed-loop episode execution; simulator truth stays outside policy inputs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from codex_agent.tasks.base import CodexExecutor

from .dataset import AlfredInstruction
from .geometry import Detection, Observation
from .memory import IncrementalSceneMemory
from .navigation import next_navigation_action
from .odometry import ActionOdometry, camera_intrinsics
from .perception import PerceptionObservation
from .policy import NAVIGATION_ACTIONS, AlfredDecision, AlfredPolicyTask
from .simulator import SimulatorObservation


class ActionEnvironment(Protocol):
    """Actor's simulator port has no terminal scorer or metadata lookup."""

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation: ...


class ImagePerception(Protocol):
    def predict(
        self, rgb_path: Path, *, frame_id: str, horizon_degrees: float, output_dir: Path
    ) -> PerceptionObservation: ...


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit action, failure, model and image budgets for one episode."""

    max_actions: int = 1000
    max_failures: int = 10
    max_decisions: int = 120
    navigation_chunk: int = 12
    retrieved_images: int = 3
    use_step_instructions: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_actions",
            "max_failures",
            "max_decisions",
            "navigation_chunk",
            "retrieved_images",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}={value!r} must be a positive integer")
        if self.retrieved_images > 8:
            raise ValueError("retrieved_images must be <= 8")


@dataclass(frozen=True)
class ExecutionOutcome:
    """Execution accounting; success is determined only by the terminal scorer."""

    stop_reason: str
    actions: int
    failures: int
    decisions: int
    input_tokens: int | None
    output_tokens: int | None
    perception_seconds: float
    policy_seconds: float


class EpisodeExecutor:
    """Own the mutable memory and observation state for one sequential episode."""

    def __init__(
        self,
        *,
        environment: ActionEnvironment,
        perception: ImagePerception,
        runtime: CodexExecutor,
        output_dir: Path,
        config: ExecutionConfig,
    ) -> None:
        self.environment = environment
        self.perception = perception
        self.runtime = runtime
        self.output_dir = output_dir
        self.config = config
        self.memory = IncrementalSceneMemory(output_dir / "memory")
        self.odometry = ActionOdometry()
        self.actions = 0
        self.failures = 0
        self.decisions = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._usage_complete = True
        self.perception_seconds = 0.0
        self.policy_seconds = 0.0
        self._feedback = "Initial observation."
        self._note = ""
        self._recent_actions: list[dict[str, object]] = []
        self._retrieved: tuple[Path, ...] = ()
        self._retrieved_ids: list[int] = []
        self._current: SimulatorObservation | None = None
        self._predicted: PerceptionObservation | None = None

    def run(
        self, instruction: AlfredInstruction, initial: SimulatorObservation
    ) -> ExecutionOutcome:
        """Observe, decide, act and reobserve until stop or a recorded budget limit."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._observe(initial)
        reason = "decision_budget"
        for _ in range(self.config.max_decisions):
            if self.actions >= self.config.max_actions:
                reason = "action_budget"
                break
            if self.failures >= self.config.max_failures:
                reason = "failure_budget"
                break
            decision = self._decide(instruction)
            if decision.kind == "stop":
                reason = "policy_stop"
                break
            self._dispatch(decision)
        if reason != "policy_stop":
            if self.actions >= self.config.max_actions:
                reason = "action_budget"
            elif self.failures >= self.config.max_failures:
                reason = "failure_budget"
        write_json(self.output_dir / "memory.json", self.memory.snapshot())
        self.memory.render_bev(self.output_dir / "final_bev.png")
        return ExecutionOutcome(
            reason,
            self.actions,
            self.failures,
            self.decisions,
            self.input_tokens if self._usage_complete else None,
            self.output_tokens if self._usage_complete else None,
            self.perception_seconds,
            self.policy_seconds,
        )

    def _observe(self, observation: SimulatorObservation) -> None:
        started = time.monotonic()
        prediction = self.perception.predict(
            observation.rgb_path,
            frame_id=str(observation.frame_id),
            horizon_degrees=observation.horizon_degrees,
            output_dir=self.output_dir / "perception",
        )
        self.perception_seconds += time.monotonic() - started
        depth = np.load(prediction.depth_path, allow_pickle=False)
        detections = tuple(
            Detection(item.category, _load_mask(item.mask_path))
            for item in prediction.detections
        )
        self.memory.integrate(
            Observation(
                frame_id=observation.frame_id,
                rgb_path=observation.rgb_path,
                depth_m=depth,
                intrinsics=camera_intrinsics(
                    width=depth.shape[1], height=depth.shape[0]
                ),
                camera_to_world=self.odometry.camera_to_world(
                    yaw_degrees=observation.yaw_degrees,
                    horizon_degrees=observation.horizon_degrees,
                ),
                agent_position=self.odometry.agent_position,
                detections=detections,
            )
        )
        self._current, self._predicted = observation, prediction
        append_jsonl(
            self.output_dir / "observations.jsonl",
            {
                "observation": observation.model_dump(mode="json"),
                "prediction": prediction.model_dump(mode="json"),
                "agent_position": self.odometry.agent_position,
                "detection_objects": self.memory.current_detection_objects,
            },
        )

    def _decide(self, instruction: AlfredInstruction) -> AlfredDecision:
        assert self._current is not None and self._predicted is not None
        self.decisions += 1
        directory = self.output_dir / "decisions" / f"{self.decisions:04d}"
        directory.mkdir(parents=True)
        current_rgb = self._annotate_current(directory / "current.png")
        bev = self.memory.render_bev(directory / "bev.png")
        snapshot = self.memory.snapshot()
        snapshot.pop("free_cells")
        snapshot.pop("occupied_cells")
        context: dict[str, object] = {
            "scene_memory": snapshot,
            "yaw_degrees": self._current.yaw_degrees,
            "horizon_degrees": self._current.horizon_degrees,
            "feedback": self._feedback,
            "previous_note": self._note,
            "recent_actions": self._recent_actions[-12:],
            "retrieved_frame_ids": self._retrieved_ids,
            "remaining_actions": self.config.max_actions - self.actions,
            "remaining_decisions": self.config.max_decisions - self.decisions,
        }
        task = AlfredPolicyTask(
            instruction,
            context,
            (current_rgb, bev, *self._retrieved),
            self.config.use_step_instructions,
        )
        request = task.build_turn_request()
        write_json(
            directory / "request.json",
            {
                "prompt": request.prompt,
                "images": [str(path) for path in request.image_paths],
            },
        )
        started = time.monotonic()
        result = self.runtime.execute(task)
        self.policy_seconds += time.monotonic() - started
        usage = result.turn.metadata.usage
        totals = usage.get("total") if usage else None
        if (
            isinstance(totals, dict)
            and isinstance(totals.get("input_tokens"), int)
            and isinstance(totals.get("output_tokens"), int)
        ):
            self.input_tokens += totals["input_tokens"]
            self.output_tokens += totals["output_tokens"]
        else:
            self._usage_complete = False
        write_json(
            directory / "response.json",
            {
                "decision": result.outcome.model_dump(mode="json"),
                "metadata": result.turn.metadata.as_dict(),
                "raw_response": result.turn.final_response,
            },
        )
        self._note = result.outcome.memory_note
        self._retrieved, self._retrieved_ids = (), []
        return result.outcome

    def _dispatch(self, decision: AlfredDecision) -> None:
        cell = (
            (decision.cell[0], decision.cell[1]) if decision.cell is not None else None
        )
        if decision.kind == "retrieve":
            if (
                decision.object_id is not None
                and decision.object_id not in self.memory.objects
            ):
                self._feedback = (
                    "Rejected retrieval: object ID has never been observed."
                )
                return
            frames = self.memory.retrieve_frames(
                object_ids=() if decision.object_id is None else (decision.object_id,),
                cells=() if cell is None else (cell,),
                limit=self.config.retrieved_images,
            )
            self._retrieved = tuple(frame.rgb_path for frame in frames)
            self._retrieved_ids = [frame.frame_id for frame in frames]
            self._feedback = (
                f"Retrieved observed frames {self._retrieved_ids}; no action performed."
            )
            return
        if decision.kind == "navigate":
            for _ in range(self.config.navigation_chunk):
                assert self._current is not None
                action, feedback = next_navigation_action(
                    self.memory,
                    yaw_degrees=self._current.yaw_degrees,
                    horizon_degrees=self._current.horizon_degrees,
                    object_id=decision.object_id,
                    cell=cell,
                )
                self._feedback = feedback
                if action is None or not self._step(action, object_id=None):
                    break
            return
        assert decision.action is not None
        for _ in range(decision.repeat):
            if not self._step(decision.action, object_id=decision.object_id):
                break

    def _step(self, action: str, *, object_id: int | None) -> bool:
        assert self._current is not None and self._predicted is not None
        if (
            self.actions >= self.config.max_actions
            or self.failures >= self.config.max_failures
        ):
            return False
        mask_path = None
        horizon = self._current.horizon_degrees
        if (action == "LookDown" and horizon >= 60 - 1e-3) or (
            action == "LookUp" and horizon <= -30 + 1e-3
        ):
            self._feedback = "Camera pitch limit reached; select another action."
            return False
        if action not in NAVIGATION_ACTIONS:
            indices = [
                index
                for index, identity in self.memory.current_detection_objects.items()
                if identity == object_id
            ]
            if not indices:
                self._feedback = "Rejected interaction: target has no mask in the current observation."
                return False
            mask_path = self._predicted.detections[indices[0]].mask_path
        previous = self._current
        observation = self.environment.step(action, mask_path=mask_path)
        self.actions += 1
        if not observation.last_action_success:
            self.failures += 1
        self.memory.record_interaction(
            action, object_id, success=observation.last_action_success
        )
        self.odometry.update(
            action,
            success=observation.last_action_success,
            yaw_degrees=previous.yaw_degrees,
        )
        record: dict[str, object] = {
            "index": self.actions,
            "action": action,
            "object_id": object_id,
            "source_frame": previous.frame_id,
            "result_frame": observation.frame_id,
            "success": observation.last_action_success,
            "error": observation.last_action_error,
        }
        append_jsonl(self.output_dir / "actions.jsonl", record)
        self._recent_actions.append(record)
        self._feedback = json.dumps(record)
        self._observe(observation)
        if action == "MoveAhead" and not observation.last_action_success:
            yaw = math.radians(previous.yaw_degrees)
            cell = self.memory.world_to_cell(
                round(self.odometry.x + self.odometry.step_size_m * math.sin(yaw), 9),
                round(self.odometry.y + self.odometry.step_size_m * math.cos(yaw), 9),
            )
            self.memory.record_collision(cell)
        return observation.last_action_success

    def _annotate_current(self, path: Path) -> Path:
        from PIL import Image, ImageDraw

        assert self._current is not None and self._predicted is not None
        with Image.open(self._current.rgb_path) as source:
            image = source.convert("RGB").resize((600, 600))
        drawing = ImageDraw.Draw(image)
        for index, identity in self.memory.current_detection_objects.items():
            item = self._predicted.detections[index]
            box = tuple(float(coordinate) * 2 for coordinate in item.bbox_xyxy)
            drawing.rectangle(box, outline="yellow", width=2)
            drawing.text(
                (box[0], box[1]),
                f"{identity}: {item.category}",
                fill="black",
                stroke_width=1,
                stroke_fill="yellow",
            )
        image.save(path)
        return path


def _load_mask(path: Path) -> np.ndarray:
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 2 or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"predicted mask must be binary and 2D: {path}")
    return np.asarray(mask, dtype=bool)


def write_json(path: Path, payload: object) -> None:
    """Atomically replace one artifact; separate run IDs preserve run history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    pending.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    """Flush each action/observation record before the next external operation."""
    with path.open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
