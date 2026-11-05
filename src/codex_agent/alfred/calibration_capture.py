"""Frozen train-only RGB/metric-depth capture, isolated from ALFRED acting.

This offline diagnostic never instantiates an actor or requests task scores.
The private simulator worker is the sole producer of calibration ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .provenance import code_provenance, file_sha256
from .worker import JsonWorker

CAPTURE_PROTOCOL: Literal["alfred_train_four_scenes_three_stations_five_pitches_v1"] = (
    "alfred_train_four_scenes_three_stations_five_pitches_v1"
)
SCENES = ("FloorPlan1", "FloorPlan201", "FloorPlan301", "FloorPlan401")
PITCHES = (-30, 0, 30, 45, 60)
STATION_ACTIONS = (
    (),
    ("RotateRight", "MoveAhead", "MoveAhead"),
    ("RotateLeft", "MoveAhead", "MoveAhead"),
)


class SourceFile(BaseModel):
    """A pinned source file read by the offline capture environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaptureTrajectory(BaseModel):
    """One deterministically selected train trajectory, independent of outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task: str
    scene_id: str
    trajectory_path: Path
    trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("task")
    @classmethod
    def safe_task(cls, task: str) -> str:
        path = PurePosixPath(task)
        if (
            path.is_absolute()
            or len(path.parts) != 2
            or any(part in (".", "..") for part in path.parts)
            or str(path) != task
            or "\\" in task
            or "\x00" in task
        ):
            raise ValueError("task must be a safe relative '<task>/trial' identity")
        return task


class CapturePlan(BaseModel):
    """Immutable sampling and source contract written before any frame capture."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    split: Literal["train"] = "train"
    capture_protocol: Literal[
        "alfred_train_four_scenes_three_stations_five_pitches_v1"
    ] = CAPTURE_PROTOCOL
    seed: Literal[0] = 0
    image_width: Literal[300] = 300
    image_height: Literal[300] = 300
    vertical_fov_degrees: Literal[60] = 60
    pitches_degrees: tuple[int, ...] = PITCHES
    station_actions: tuple[tuple[str, ...], ...] = STATION_ACTIONS
    navigation_pitch_degrees: Literal[0] = 0
    depth_scale_to_meters: float = Field(default=0.001, ge=0.001, le=0.001)
    depth_saturation_m: float = Field(default=5.0, ge=5.0, le=5.0)
    initialization: Literal[
        "official_reset_restore_explicit_fov_initialize_official_init"
    ] = "official_reset_restore_explicit_fov_initialize_official_init"
    data_root: Path
    splits_path: Path
    splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    alfred_root: Path
    ai2thor_root: Path
    unity_executable: Path
    sources: tuple[SourceFile, ...]
    trajectories: tuple[CaptureTrajectory, ...]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code: dict[str, object]

    @model_validator(mode="after")
    def fixed_coverage(self) -> CapturePlan:
        if self.pitches_degrees != PITCHES or self.station_actions != STATION_ACTIONS:
            raise ValueError(
                "camera pitches and navigation stations must match the fixed capture protocol"
            )
        if tuple(item.scene_id for item in self.trajectories) != SCENES:
            raise ValueError(
                "capture plan must contain exactly the four fixed scenes in protocol order"
            )
        if len({item.task for item in self.trajectories}) != 4:
            raise ValueError(
                "capture plan must contain four distinct train trajectories"
            )
        if len({item.name for item in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")
        return self


class CapturedFrame(BaseModel):
    """Calibrated frame record accepted only by offline depth diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    frame_id: str
    scene_id: str
    rgb_path: Path
    gt_depth_path: Path
    intrinsics: list[list[float]]
    pitch_degrees: float
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gt_depth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera_to_world: list[list[float]] | None = None

    @field_validator("camera_to_world")
    @classmethod
    def valid_camera_pose(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        if value is not None and (
            len(value) != 4
            or any(
                len(row) != 4 or not all(math.isfinite(x) for x in row) for row in value
            )
            or value[3] != [0, 0, 0, 1]
        ):
            raise ValueError(
                "camera_to_world must be a finite homogeneous 4x4 transform"
            )
        return value

    @field_validator("intrinsics")
    @classmethod
    def valid_intrinsics(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(
            len(row) != 3 or not all(math.isfinite(x) for x in row) for row in value
        ):
            raise ValueError("intrinsics must be finite 3x3")
        if value[0][0] <= 0 or value[1][1] <= 0 or value[2] != [0, 0, 1]:
            raise ValueError("intrinsics must be a positive pinhole calibration")
        return value


class _CapturedScene(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scene_id: str
    frames: list[CapturedFrame]
    actions_path: Path
    actions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    depth_units: Literal["meters_from_ai2thor_2.1_mm_div_1000"]
    depth_saturation_m: float = Field(ge=5.0, le=5.0)


def _train_tasks(splits_path: Path) -> set[str]:
    raw = json.loads(splits_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), list):
        raise ValueError("splits must contain an explicit train list")
    tasks: set[str] = set()
    for entry in raw["train"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("task"), str):
            raise ValueError("every train entry must contain a task string")
        tasks.add(CaptureTrajectory.safe_task(entry["task"]))
    if not tasks:
        raise ValueError("train split must not be empty")
    return tasks


def _trajectory_path(data_root: Path, task: str) -> Path:
    train_root = (data_root / "train").resolve(strict=True)
    path = (train_root / task / "traj_data.json").resolve(strict=True)
    if not path.is_relative_to(train_root):
        raise ValueError("trajectory must remain inside the train data directory")
    return path


def _selection_hash(task: str) -> str:
    return hashlib.sha256(f"0:train/{task}".encode()).hexdigest()


def _sources(
    alfred_root: Path, ai2thor_root: Path, unity_executable: Path
) -> tuple[SourceFile, ...]:
    locations = {
        "alfred_thor_env": alfred_root / "env/thor_env.py",
        "alfred_constants": alfred_root / "gen/constants.py",
        "ai2thor_server": ai2thor_root / "server.py",
        "ai2thor_controller": ai2thor_root / "controller.py",
        "ai2thor_package": ai2thor_root / "__init__.py",
        "ai2thor_builds": ai2thor_root / "_builds.py",
        "unity_executable": unity_executable,
    }
    return tuple(
        SourceFile(name=name, path=path.resolve(strict=True), sha256=file_sha256(path))
        for name, path in sorted(locations.items())
    )


def _source_digest(
    splits_sha256: str,
    sources: tuple[SourceFile, ...],
    trajectories: tuple[CaptureTrajectory, ...],
) -> str:
    contract = {
        "protocol": CAPTURE_PROTOCOL,
        "split": "train",
        "seed": 0,
        "splits_sha256": splits_sha256,
        "sources": {item.name: item.sha256 for item in sources},
        "trajectories": {item.task: item.trajectory_sha256 for item in trajectories},
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def create_capture_plan(
    *,
    data_root: Path,
    splits_path: Path,
    alfred_root: Path,
    ai2thor_root: Path,
    unity_executable: Path,
    code: dict[str, object],
) -> CapturePlan:
    """Select each scene's lowest seed-0 identity SHA without reading any outcomes.

    Annotation repeats do not change trajectory selection. Candidates are read
    in hash order until all four scenes have their first eligible train entry.
    """
    data_root = data_root.resolve(strict=True)
    splits_path = splits_path.resolve(strict=True)
    selected: dict[str, CaptureTrajectory] = {}
    for task in sorted(_train_tasks(splits_path), key=_selection_hash):
        path = _trajectory_path(data_root, task)
        raw = json.loads(path.read_text(encoding="utf-8"))
        scene_id = raw["scene"]["floor_plan"]
        if scene_id in SCENES and scene_id not in selected:
            selected[scene_id] = CaptureTrajectory(
                task=task,
                scene_id=scene_id,
                trajectory_path=path,
                trajectory_sha256=file_sha256(path),
                selection_sha256=_selection_hash(task),
            )
        if len(selected) == len(SCENES):
            break
    if set(selected) != set(SCENES):
        raise ValueError(
            f"train data does not provide required scenes: {sorted(set(SCENES) - set(selected))}"
        )
    sources = _sources(alfred_root, ai2thor_root, unity_executable)
    trajectories = tuple(selected[scene] for scene in SCENES)
    splits_hash = file_sha256(splits_path)
    return CapturePlan(
        data_root=data_root,
        splits_path=splits_path,
        splits_sha256=splits_hash,
        alfred_root=alfred_root.resolve(strict=True),
        ai2thor_root=ai2thor_root.resolve(strict=True),
        unity_executable=unity_executable.resolve(strict=True),
        sources=sources,
        trajectories=trajectories,
        source_sha256=_source_digest(splits_hash, sources, trajectories),
        code=code,
    )


def write_capture_plan(plan: CapturePlan, path: Path) -> None:
    """Freeze a new plan; existing evidence is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(plan.model_dump_json(indent=2) + "\n")


def validate_capture_plan(plan: CapturePlan) -> None:
    """Check train membership, deterministic selection, and all source hashes."""
    reconstructed = create_capture_plan(
        data_root=plan.data_root,
        splits_path=plan.splits_path,
        alfred_root=plan.alfred_root,
        ai2thor_root=plan.ai2thor_root,
        unity_executable=plan.unity_executable,
        code=plan.code,
    )
    if reconstructed != plan:
        raise ValueError(
            "frozen capture plan differs from current train selection or source hashes"
        )


def run_capture(
    *,
    plan_path: Path,
    output_dir: Path,
    project_root: Path,
    simulator_python: Path,
    x_display: str,
    timeout_seconds: float = 300.0,
) -> Path:
    """Capture the frozen 60-frame protocol only on clean committed project code."""
    current_code = code_provenance(project_root)
    plan = CapturePlan.model_validate_json(plan_path.read_bytes())
    if current_code["executed_code_sha256"] != plan.code.get(
        "executed_code_sha256"
    ) or current_code["branch"] != plan.code.get("branch"):
        raise ValueError(
            "capture code must match the frozen plan's executed source and branch"
        )
    validate_capture_plan(plan)
    if not simulator_python.is_file():
        raise FileNotFoundError(f"simulator interpreter missing: {simulator_python}")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "capture_status.json"
    receipt: dict[str, object] = {
        "status": "running",
        "split": "train",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "code": current_code,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    scenes: list[_CapturedScene] = []
    try:
        for trajectory in plan.trajectories:
            scene_output = output_dir / trajectory.scene_id
            with JsonWorker(
                command=[
                    # Resolving a venv's Python symlink selects the base
                    # interpreter and loses the pinned simulator packages.
                    str(simulator_python.absolute()),
                    "-u",
                    str(Path(__file__).with_name("calibration_capture_worker.py")),
                    "--alfred-root",
                    str(plan.alfred_root),
                    "--ai2thor-root",
                    str(plan.ai2thor_root),
                    "--unity-executable",
                    str(plan.unity_executable),
                    "--output-dir",
                    str(scene_output),
                    "--x-display",
                    x_display.lstrip(":"),
                ],
                cwd=plan.alfred_root,
                env=os.environ,
                log_path=scene_output / "worker.stderr.log",
                timeout_seconds=timeout_seconds,
            ) as worker:
                result = _CapturedScene.model_validate(
                    worker.request(
                        {
                            "command": "capture",
                            "trajectory_path": str(trajectory.trajectory_path),
                            "trajectory_sha256": trajectory.trajectory_sha256,
                            "scene_id": trajectory.scene_id,
                            "sources": {
                                item.name: item.sha256 for item in plan.sources
                            },
                        }
                    )
                )
            _validate_captured_scene(result, trajectory.scene_id, scene_output)
            scenes.append(result)
        validate_capture_plan(plan)
        end_code = code_provenance(project_root)
        if end_code["executed_code_sha256"] != current_code["executed_code_sha256"]:
            raise ValueError("executed source changed during capture")
        frames = [
            frame.model_dump(mode="json", exclude_none=True)
            for scene in scenes
            for frame in scene.frames
        ]
        manifest = {
            "schema_version": 1,
            "split": "train",
            "capture_protocol": CAPTURE_PROTOCOL,
            "source_sha256": plan.source_sha256,
            "frames": frames,
            "provenance": {
                "plan_path": str(plan_path.resolve()),
                "plan_sha256": file_sha256(plan_path),
                "code": current_code,
                "end_code": end_code,
                "sources": [item.model_dump(mode="json") for item in plan.sources],
                "trajectories": [
                    item.model_dump(mode="json") for item in plan.trajectories
                ],
                "depth_units": "meters_from_ai2thor_2.1_mm_div_1000",
                "depth_saturation_m": 5.0,
                "saturated_pixels_are_not_unbounded_true_depth": True,
                "resolution": [300, 300],
                "vertical_fov_degrees": 60.0,
                "station_actions": STATION_ACTIONS,
                "pitches_degrees": PITCHES,
                "navigation_pitch_degrees": 0,
                "action_receipts": [
                    {"path": str(scene.actions_path), "sha256": scene.actions_sha256}
                    for scene in scenes
                ],
                "learned_checkpoints": {},
                "ground_truth_scope": "offline_depth_calibration_only; never actor input",
            },
        }
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        receipt.update(
            status="completed",
            frame_count=len(frames),
            manifest_path=str(manifest_path),
        )
        return manifest_path
    except Exception as exc:
        receipt.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def _validate_captured_scene(
    scene: _CapturedScene, expected_scene: str, output_dir: Path
) -> None:
    import numpy as np
    from PIL import Image

    if scene.scene_id != expected_scene or len(scene.frames) != 15:
        raise ValueError(
            "capture must return exactly 15 frames from the requested scene"
        )
    if len({frame.frame_id for frame in scene.frames}) != 15:
        raise ValueError("capture frame identities must be unique")
    expected_ids = [
        f"{expected_scene}_s{station}_p{pitch:+03d}"
        for station in range(3)
        for pitch in PITCHES
    ]
    if [frame.frame_id for frame in scene.frames] != expected_ids:
        raise ValueError(
            "capture frame identities must preserve scene/station/pitch order"
        )
    if [frame.pitch_degrees for frame in scene.frames] != list(PITCHES) * 3:
        raise ValueError("capture pitches must follow the frozen five-angle schedule")
    for path, expected_hash in [(scene.actions_path, scene.actions_sha256)] + [
        (path, digest)
        for frame in scene.frames
        for path, digest in (
            (frame.rgb_path, frame.rgb_sha256),
            (frame.gt_depth_path, frame.gt_depth_sha256),
        )
    ]:
        if (
            not path.resolve(strict=True).is_relative_to(output_dir.resolve())
            or file_sha256(path) != expected_hash
        ):
            raise ValueError(
                "capture artifact must remain inside its scene directory and match its hash"
            )
    if any(frame.scene_id != expected_scene for frame in scene.frames):
        raise ValueError("frame scene identities must match the requested scene")
    focal = 300 / (2 * math.tan(math.radians(60) / 2))
    expected_intrinsics = np.array([[focal, 0, 149.5], [0, focal, 149.5], [0, 0, 1]])
    for frame in scene.frames:
        if not np.allclose(frame.intrinsics, expected_intrinsics, atol=1e-8):
            raise ValueError(
                "captured intrinsics differ from fixed 300-pixel / 60-degree camera"
            )
        with Image.open(frame.rgb_path) as image:
            if image.size != (300, 300) or image.mode != "RGB":
                raise ValueError("capture RGB must retain 300x300 RGB pixels")
        depth = np.load(frame.gt_depth_path, allow_pickle=False)
        if (
            not isinstance(depth, np.ndarray)
            or depth.shape != (300, 300)
            or depth.dtype != np.float32
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
            or np.any(depth > 5.00001)
        ):
            raise ValueError(
                "capture GT depth must be finite float32 300x300 meters in [0, 5]"
            )
