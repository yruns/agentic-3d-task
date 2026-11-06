"""Closed-loop episode execution driven by a single interactive Codex turn.

``EpisodeExecutor`` owns the simulator connection, perception worker, and
:class:`IncrementalSceneMemory`. Before starting the Codex turn, it performs an
automatic initial panoramic scan to populate the 3D object catalog and 5cm
occupancy grid, starts an in-process :class:`EpisodeToolServer`, and launches a
single :class:`AlfredPolicyTask` turn where Codex orchestrates high-level macro
tools and receives compact ``SceneDelta`` updates.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from codex_agent.errors import CodexResponseError, CodexTurnError
from codex_agent.models import CodexTaskResult
from codex_agent.runtime import CodexAgentRuntime
from codex_agent.tasks.base import CodexExecutor

from .dataset import AlfredInstruction
from .delta import format_visible_objects_now
from .episode_server import EpisodeToolServer
from .exploration import suggest_frontiers
from .geometry import Detection, Observation
from .macro_controller import MacroActionController
from .memory import IncrementalSceneMemory, MappingConfig, MemoryConfig
from .odometry import ActionOdometry, camera_intrinsics
from .perception import PerceptionObservation
from .policy import NAVIGATION_ACTIONS, AlfredEpisodeTurnOutcome, AlfredPolicyTask
from .provenance import file_sha256
from .simulator import SimulatorObservation
from .spatial_map import SpatialMap, SpatialMapConfig


class ActionEnvironment(Protocol):
    """Actor's simulator port has no terminal scorer or metadata lookup."""

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation: ...


class ImagePerception(Protocol):
    def predict(
        self, rgb_path: Path, *, frame_id: str, intrinsics: np.ndarray, output_dir: Path
    ) -> PerceptionObservation: ...


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit action, failure, tool-call, and navigation budgets for one episode."""

    max_actions: int = 1000
    max_failures: int = 10
    max_decisions: int = 40
    navigation_chunk: int = 32
    retrieved_images: int = 3
    use_step_instructions: bool = False
    initial_scan: bool = True

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
    """Own the mutable memory, macro controller, and tool server for one episode."""

    def __init__(
        self,
        *,
        environment: ActionEnvironment,
        perception: ImagePerception,
        runtime: CodexExecutor,
        output_dir: Path,
        config: ExecutionConfig,
        memory_config: MemoryConfig | None = None,
        spatial_config: SpatialMapConfig | None = None,
    ) -> None:
        self.environment = environment
        self.perception = perception
        self.runtime = runtime
        self.output_dir = output_dir
        self.config = config
        self.memory = IncrementalSceneMemory(
            output_dir / "memory",
            memory_config
            or MemoryConfig(mapping=MappingConfig(max_bbox_area_ratio=1.0)),
            spatial_map=SpatialMap(spatial_config),
        )
        self.odometry = ActionOdometry()
        self.macro = MacroActionController(self)
        self.actions = 0
        self.failures = 0
        self.decisions = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._usage_complete = True
        self.perception_seconds = 0.0
        self.policy_seconds = 0.0
        self.finish_requested = False
        self.finish_summary = ""
        self.completed_macros: set[str] = set()
        self._tool_call_summaries: list[str] = []
        self.workspace_dir: Path | None = None
        self._recent_actions: list[dict[str, object]] = []
        self._current: SimulatorObservation | None = None
        self._predicted: PerceptionObservation | None = None
        self.current_instruction: AlfredInstruction | None = None
        self.finish_rejections: int = 0

    @property
    def current_observation(self) -> SimulatorObservation:
        if self._current is None:
            raise RuntimeError("No observation has been integrated yet.")
        return self._current

    def can_step(self) -> bool:
        """Return whether action and failure budgets still permit a simulator step."""
        return (
            not self.finish_requested
            and self.actions < self.config.max_actions
            and self.failures < self.config.max_failures
        )

    def run(
        self, instruction: AlfredInstruction, initial: SimulatorObservation
    ) -> ExecutionOutcome:
        """Perform initial scan, start tool server, and run a single Codex turn."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_instruction = instruction
        self._observe(initial)
        with tempfile.TemporaryDirectory(prefix="alfred_actor_ws_") as tmp_ws:
            self.workspace_dir = Path(tmp_ws).resolve()
            if self.config.initial_scan and self.can_step():
                self.macro.scan_surroundings(pitches=(30.0,))

            initial_rgb, initial_bev = self.render_latest_artifacts(prefix="initial")
            initial_context = self._build_initial_context()
            task = AlfredPolicyTask(
                instruction=instruction,
                episode_dir=self.workspace_dir,
                initial_context=initial_context,
                image_paths=(initial_rgb, initial_bev),
                use_step_instructions=self.config.use_step_instructions,
            )

            decisions_dir = self.output_dir / "decisions" / "0001"
            decisions_dir.mkdir(parents=True, exist_ok=True)
            request = task.build_turn_request()
            write_json(
                decisions_dir / "request.json",
                {
                    "prompt": request.prompt,
                    "images": [str(path) for path in request.image_paths],
                },
            )

            started = time.monotonic()
            orig_cfg = (
                self.runtime.config
                if isinstance(self.runtime, CodexAgentRuntime)
                else None
            )
            if isinstance(self.runtime, CodexAgentRuntime) and orig_cfg is not None:
                self.runtime.config = replace(orig_cfg, project_root=self.workspace_dir)
            result: CodexTaskResult[AlfredEpisodeTurnOutcome] | None = None
            turn_interrupted_error: str | None = None
            max_turn_continuations = 2
            try:
                with EpisodeToolServer(self):
                    for cont_idx in range(max_turn_continuations + 1):
                        try:
                            if cont_idx > 0:
                                time.sleep(2.0 * cont_idx)
                                cont_rgb, cont_bev = self.render_latest_artifacts(
                                    prefix=f"cont{cont_idx}"
                                )
                                cont_context = self._build_initial_context()
                                task = AlfredPolicyTask(
                                    instruction=instruction,
                                    episode_dir=self.workspace_dir,
                                    initial_context=cont_context,
                                    image_paths=(cont_rgb, cont_bev),
                                    use_step_instructions=self.config.use_step_instructions,
                                )
                            turn_res = self.runtime.execute(task)
                            result = turn_res
                            turn_interrupted_error = None
                            usage = turn_res.turn.metadata.usage
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
                            meta_dict = turn_res.turn.metadata.as_dict()
                            attempts_list = meta_dict.get("attempts")
                            was_interrupted = isinstance(attempts_list, list) and any(
                                isinstance(a, dict) and a.get("status") == "interrupted"
                                for a in attempts_list
                            )
                            if (
                                was_interrupted
                                and not self.finish_requested
                                and turn_res.outcome.status == "incomplete"
                                and self.can_step()
                                and self.decisions < self.config.max_decisions
                                and cont_idx < max_turn_continuations
                            ):
                                continue
                            break
                        except (CodexTurnError, CodexResponseError) as exc:
                            turn_interrupted_error = f"{type(exc).__name__}: {exc}"
                            budget_exhausted = (
                                self.finish_requested
                                or not self.can_step()
                                or self.decisions >= self.config.max_decisions
                            )
                            if budget_exhausted:
                                break
                            if (
                                self.decisions == 0
                                and isinstance(exc, CodexResponseError)
                            ) or cont_idx >= max_turn_continuations:
                                raise
            finally:
                if isinstance(self.runtime, CodexAgentRuntime) and orig_cfg is not None:
                    self.runtime.config = orig_cfg
                self.workspace_dir = None
            self.policy_seconds += time.monotonic() - started

        if self.decisions == 0:
            self.decisions = 1

        if result is not None:
            write_json(
                decisions_dir / "response.json",
                {
                    "decision": result.outcome.model_dump(mode="json"),
                    "metadata": result.turn.metadata.as_dict(),
                    "raw_response": result.turn.final_response,
                },
            )
            completed_by_policy = (
                self.finish_requested or result.outcome.status == "completed"
            )
        else:
            self._usage_complete = False
            write_json(
                decisions_dir / "response.json",
                {
                    "decision": {
                        "status": "incomplete",
                        "summary": self.finish_summary
                        or "Turn interrupted at budget limit.",
                        "completed_subgoals": [],
                    },
                    "metadata": {"interrupted_error": turn_interrupted_error},
                    "raw_response": None,
                },
            )
            completed_by_policy = self.finish_requested

        if completed_by_policy:
            reason = "policy_stop"
        elif self.actions >= self.config.max_actions:
            reason = "action_budget"
        elif self.failures >= self.config.max_failures:
            reason = "failure_budget"
        else:
            reason = "decision_budget"

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

    def _build_initial_context(self) -> dict[str, object]:
        obs = self.current_observation
        agent_pos = self.memory.agent_position or (0.0, 0.0, 0.0)
        agent_region = self.memory.world_to_region(agent_pos[0], agent_pos[1])
        region_layout: dict[str, list[str]] = {}
        catalog: list[dict[str, object]] = []
        for obj_id, node in sorted(self.memory.objects.items()):
            reg = self.memory.world_to_region(
                float(node.centroid[0]), float(node.centroid[1])
            )
            reg_key = f"[{reg[0]}, {reg[1]}]"
            region_layout.setdefault(reg_key, []).append(f"#{obj_id} {node.category}")
            dist_m = round(
                float(np.linalg.norm(node.centroid - np.asarray(agent_pos))), 2
            )
            recent_frames = [
                v.frame_id for v in node.views if v.frame_id > node.stale_through_frame
            ][-3:]
            top_evidence = dict(
                sorted(node.category_evidence.items(), key=lambda kv: (-kv[1], kv[0]))[
                    :3
                ]
            )
            catalog_item: dict[str, object] = {
                "object_id": obj_id,
                "category": node.category,
                "category_evidence": top_evidence,
                "region": list(reg),
                "distance_m": dist_m,
                "centroid": [round(float(v), 2) for v in node.centroid],
                "provisional": node.provisional,
                "state": node.state,
                "action_states": {
                    name: st.value for name, st in node.action_states.items()
                },
                "best_frames": recent_frames,
            }
            if node.parent_receptacle_id is not None:
                catalog_item["parent_receptacle_id"] = node.parent_receptacle_id
            catalog.append(catalog_item)
        frontiers = [
            {
                "region": list(c.region),
                "world_xy": [
                    round(c.world_position[0], 2),
                    round(c.world_position[1], 2),
                ],
                "path_cost_m": round(c.path_cost_m, 2),
                "info_gain_m2": round(c.information_gain_m2, 2),
            }
            for c in suggest_frontiers(self.memory, limit=5)
        ]
        held_payload: dict[str, object] | None = None
        if (
            self.memory.held_object_id is not None
            and self.memory.held_object_id in self.memory.objects
        ):
            held_node = self.memory.objects[self.memory.held_object_id]
            held_payload = {
                "object_id": held_node.object_id,
                "category": held_node.category,
            }
        context: dict[str, object] = {
            "agent_state": {
                "position_xy": [round(agent_pos[0], 3), round(agent_pos[1], 3)],
                "region": list(agent_region),
                "yaw_degrees": round(obs.yaw_degrees, 1),
                "horizon_degrees": round(obs.horizon_degrees, 1),
                "held_object": held_payload,
            },
            "scene_region_layout": region_layout,
            "object_catalog": catalog,
            "visible_objects_now": format_visible_objects_now(self.memory),
            "top_frontiers": frontiers,
            "budget": {
                "remaining_decisions": max(
                    0, self.config.max_decisions - self.decisions
                ),
                "remaining_actions": max(0, self.config.max_actions - self.actions),
                "remaining_failures": max(0, self.config.max_failures - self.failures),
            },
        }
        if self._tool_call_summaries:
            context["continuation_note"] = (
                "Resuming interrupted turn in the same episode. Do NOT repeat "
                "completed steps; continue from the current agent_state and held_object."
            )
            context["completed_macros"] = sorted(self.completed_macros)
            context["recent_tool_calls"] = self._tool_call_summaries[-12:]
        return context

    def _observe(self, observation: SimulatorObservation) -> None:
        started = time.monotonic()
        intrinsics = camera_intrinsics()
        prediction = self.perception.predict(
            observation.rgb_path,
            frame_id=str(observation.frame_id),
            intrinsics=intrinsics,
            output_dir=self.output_dir / "perception",
        )
        self.perception_seconds += time.monotonic() - started
        depth = np.load(prediction.depth_path, allow_pickle=False)
        if prediction.depth_model == "simulator_gt":
            gt_file = (
                observation.rgb_path.resolve().parent.parent
                / "depths"
                / f"{observation.frame_id:06d}.npy"
            )
            if gt_file.is_file():
                depth = np.load(gt_file, allow_pickle=False).astype(np.float32)
                np.save(
                    prediction.depth_path,
                    np.asarray(np.clip(depth, 1e-3, 5.0), dtype=np.float32),
                    allow_pickle=False,
                )
            if np.any((depth > 0) & (depth < 4.99)):
                depth = np.where(depth >= 4.99, 0.0, depth)
        with np.load(prediction.features_path, allow_pickle=False) as features:
            detections = tuple(
                Detection(
                    item.category,
                    _load_mask(item.mask_path),
                    item.confidence,
                    features["image_features"][index],
                    features["text_features"][index],
                    bbox_xyxy=item.bbox_xyxy,
                )
                for index, item in enumerate(prediction.detections)
            )
            frame_feature = features["frame_feature"].copy()
        self.memory.integrate(
            Observation(
                frame_id=observation.frame_id,
                rgb_path=observation.rgb_path,
                depth_m=depth,
                intrinsics=intrinsics,
                camera_to_world=self.odometry.camera_to_world(
                    yaw_degrees=observation.yaw_degrees,
                    horizon_degrees=observation.horizon_degrees,
                ),
                agent_position=self.odometry.agent_position,
                detections=detections,
                frame_feature=frame_feature,
            )
        )
        for temp_path in (
            prediction.depth_path,
            prediction.features_path,
            *(item.mask_path for item in prediction.detections),
        ):
            temp_path.unlink(missing_ok=True)
        if prediction.depth_model == "simulator_gt":
            gt_depth_tmp = (
                observation.rgb_path.resolve().parent.parent
                / "depths"
                / f"{observation.frame_id:06d}.npy"
            )
            gt_depth_tmp.unlink(missing_ok=True)
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

    def step_primitive(self, action: str, *, object_id: int | None) -> bool:
        """Execute one primitive simulator action and integrate the resulting frame."""
        assert self._current is not None and self._predicted is not None
        if (
            self.actions >= self.config.max_actions
            or self.failures >= self.config.max_failures
        ):
            return False
        mask_path = None
        mask_record: dict[str, object] | None = None
        horizon = self._current.horizon_degrees
        if (action == "LookDown" and horizon >= 60 - 1e-3) or (
            action == "LookUp" and horizon <= -30 + 1e-3
        ):
            return False
        if action not in NAVIGATION_ACTIONS:
            if object_id is None:
                return False
            mask_selection = self.memory.select_interaction_mask(
                object_id, action=action
            )
            if mask_selection is None:
                return False
            detection_index, mask, mask_source = mask_selection
            mask_path = (
                self.output_dir / "interaction_masks" / f"{self.actions + 1:06d}.npy"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, mask, allow_pickle=False)
            mask_record = {
                "path": str(mask_path),
                "sha256": file_sha256(mask_path),
                "source": mask_source,
                "source_detection_index": detection_index,
            }
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
            "interaction_mask": mask_record,
        }
        append_jsonl(self.output_dir / "actions.jsonl", record)
        self._recent_actions.append(record)
        self._observe(observation)
        if action == "MoveAhead" and not observation.last_action_success:
            yaw = math.radians(previous.yaw_degrees)
            cell = self.memory.spatial_map.world_to_cell(
                round(self.odometry.x + self.odometry.step_size_m * math.sin(yaw), 9),
                round(self.odometry.y + self.odometry.step_size_m * math.cos(yaw), 9),
            )
            self.memory.record_collision(cell)
        return observation.last_action_success

    def publish_tool_artifact(self, path: Path) -> Path:
        """Mirror a rendered tool image into the isolated workspace if active."""
        resolved = path.resolve()
        if self.workspace_dir is not None and self.workspace_dir.is_dir():
            dest = self.workspace_dir / resolved.name
            if dest != resolved:
                shutil.copy2(resolved, dest)
            return dest
        return resolved

    def render_latest_artifacts(self, *, prefix: str = "step") -> tuple[Path, Path]:
        """Render annotated current RGB and current BEV under tool_artifacts/."""
        artifacts_dir = self.output_dir / "tool_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fid = self.memory._latest_frame_id
        rgb_out = artifacts_dir / f"{prefix}_f{fid:04d}_rgb.png"
        bev_out = artifacts_dir / f"{prefix}_f{fid:04d}_bev.png"
        self._annotate_current(rgb_out)
        self.memory.render_bev(bev_out)
        return (
            self.publish_tool_artifact(rgb_out),
            self.publish_tool_artifact(bev_out),
        )

    def _annotate_current(self, path: Path) -> Path:
        from PIL import Image, ImageDraw

        assert self._current is not None and self._predicted is not None
        with Image.open(self._current.rgb_path) as source:
            image = source.convert("RGB").resize((600, 600))
        drawing = ImageDraw.Draw(image)
        for index, identity in self.memory.current_detection_objects.items():
            resolved = self.memory.resolve_object_id(identity)
            item = self._predicted.detections[index]
            box = tuple(float(coordinate) * 2 for coordinate in item.bbox_xyxy)
            drawing.rectangle(box, outline="yellow", width=2)
            drawing.text(
                (box[0], box[1]),
                f"#{resolved}: {item.category}",
                fill="black",
                stroke_width=1,
                stroke_fill="yellow",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
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
