"""Standalone Python 3.8 worker for offline ALFRED train depth calibration.

No project package, actor, detector, policy, goal checker or reward is imported.
Ground-truth depth and optional camera pose are written only to this diagnostic
worker's output directory. The actor simulator protocol remains separate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

_PITCHES = (-30, 0, 30, 45, 60)
_STATIONS = (
    (),
    ("RotateRight", "MoveAhead", "MoveAhead"),
    ("RotateLeft", "MoveAhead", "MoveAhead"),
)
_RENDER = {
    "renderImage": True,
    "renderDepthImage": True,
    "renderClassImage": False,
    "renderObjectImage": False,
}


def depth_meters(depth_frame: Any) -> Any:
    """Convert pinned AI2-THOR 2.1 millimeters to meters using a fixed scale.

    The pinned server clips depth_frame at 5000 mm. Saturated samples are retained
    as 5 m and declared in the manifest, never reinterpreted as unlimited depth.
    Any is confined to the optional external NumPy/simulator boundary.
    """
    import numpy as np

    raw = np.asarray(depth_frame)
    if (
        raw.ndim != 2
        or raw.shape != (300, 300)
        or not np.isfinite(raw).all()
        or np.any(raw < 0)
        or np.any(raw > 5000.01)
    ):
        raise ValueError(
            "pinned depth_frame must be finite 300x300 millimeters in [0, 5000]"
        )
    return np.asarray(raw / 1000.0, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _horizon(metadata: dict[str, Any]) -> float:
    value = float(metadata["agent"]["cameraHorizon"])
    if not math.isfinite(value):
        raise ValueError("camera horizon must be finite")
    return (value + 180) % 360 - 180


def _camera_pose(metadata: dict[str, Any]) -> list[list[float]] | None:
    """Use an explicit cameraPosition only; never guess camera-height offsets."""
    position = metadata.get("cameraPosition")
    if not isinstance(position, dict):
        return None
    yaw = math.radians(float(metadata["agent"]["rotation"]["y"]))
    pitch = math.radians(_horizon(metadata))
    x, y, z = float(position["x"]), float(position["z"]), float(position["y"])
    if not all(math.isfinite(value) for value in (x, y, z, yaw, pitch)):
        raise ValueError("camera pose metadata must be finite")
    return [
        [
            math.cos(yaw),
            -math.sin(pitch) * math.sin(yaw),
            math.cos(pitch) * math.sin(yaw),
            x,
        ],
        [
            -math.sin(yaw),
            -math.sin(pitch) * math.cos(yaw),
            math.cos(pitch) * math.cos(yaw),
            y,
        ],
        [0.0, -math.cos(pitch), -math.sin(pitch), z],
        [0.0, 0.0, 0.0, 1.0],
    ]


class _CalibrationWorker:
    """Own exactly one simulator scene; Any is restricted to the legacy API."""

    def __init__(
        self,
        alfred_root: Path,
        ai2thor_root: Path,
        output_dir: Path,
        x_display: str,
        unity_executable: Path,
    ) -> None:
        self.alfred_root = alfred_root
        self.ai2thor_root = ai2thor_root
        self.unity_executable = unity_executable
        self.output_dir = output_dir
        self.x_display = x_display
        self.environment: Any = None
        self.started = False
        self.actions: list[dict[str, object]] = []

    def capture(self, request: dict[str, Any]) -> dict[str, object]:
        if self.started:
            raise RuntimeError("a capture worker accepts exactly one scene")
        self.started = True
        if (
            set(request)
            != {
                "command",
                "trajectory_path",
                "trajectory_sha256",
                "scene_id",
                "sources",
            }
            or request["command"] != "capture"
        ):
            raise ValueError("invalid offline capture request fields")
        if request["scene_id"] not in {
            "FloorPlan1",
            "FloorPlan201",
            "FloorPlan301",
            "FloorPlan401",
        }:
            raise ValueError(
                "capture scene must belong to the fixed train calibration protocol"
            )
        self._verify_sources(request["sources"])
        path = Path(request["trajectory_path"]).resolve(strict=True)
        if _sha256(path) != request["trajectory_sha256"]:
            raise ValueError("trajectory hash changed after capture-plan freeze")
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        scene = trajectory["scene"]
        if scene["floor_plan"] != request["scene_id"]:
            raise ValueError("trajectory scene does not match frozen selection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize(scene)
        frames = self._capture_stations(request["scene_id"])
        action_path = self.output_dir / "actions.jsonl"
        return {
            "scene_id": request["scene_id"],
            "frames": frames,
            "actions_path": str(action_path.resolve()),
            "actions_sha256": _sha256(action_path),
            "depth_units": "meters_from_ai2thor_2.1_mm_div_1000",
            "depth_saturation_m": 5.0,
        }

    def _verify_sources(self, expected: dict[str, str]) -> None:
        sources = {
            "alfred_thor_env": self.alfred_root / "env/thor_env.py",
            "alfred_constants": self.alfred_root / "gen/constants.py",
            "ai2thor_server": self.ai2thor_root / "server.py",
            "ai2thor_controller": self.ai2thor_root / "controller.py",
            "ai2thor_package": self.ai2thor_root / "__init__.py",
            "ai2thor_builds": self.ai2thor_root / "_builds.py",
            "unity_executable": self.unity_executable,
        }
        if not isinstance(expected, dict) or set(expected) != set(sources):
            raise ValueError(
                "capture requires the complete frozen simulator source hashes"
            )
        if any(_sha256(path) != expected[name] for name, path in sources.items()):
            raise ValueError("simulator source changed after capture-plan freeze")

    def _initialize(self, scene: dict[str, Any]) -> None:
        sys.path[:0] = [str(self.alfred_root), str(self.alfred_root / "gen")]
        import ai2thor
        from ai2thor import _builds, server
        from env.thor_env import ThorEnv
        from gen import constants

        if (
            ai2thor.__version__ != "2.1.0"
            or _builds.VERSION != "201909061227"
            or server.MAX_DEPTH != 5000
        ):
            raise RuntimeError(
                "calibration requires ai2thor 2.1.0 build 201909061227 with 5000-mm depth saturation"
            )
        if Path(ai2thor.__file__).resolve().parent != self.ai2thor_root.resolve():
            raise RuntimeError(
                "loaded ai2thor package differs from the frozen package source path"
            )
        constants.RENDER_DEPTH_IMAGE = True
        constants.RENDER_IMAGE = True
        constants.RENDER_CLASS_IMAGE = False
        constants.RENDER_OBJECT_IMAGE = False
        self.environment = ThorEnv(
            x_display=self.x_display,
            player_screen_width=300,
            player_screen_height=300,
            build_path=str(self.unity_executable),
        )
        self.environment.reset(
            scene["floor_plan"],
            render_image=True,
            render_depth_image=True,
            render_class_image=False,
            render_object_image=False,
        )
        self._record("reset", {"scene_id": scene["floor_plan"]}, required=True)
        self.environment.restore_scene(
            scene["object_poses"], scene["object_toggles"], scene["dirty_and_empty"]
        )
        self._record("restore_scene", {}, required=True)
        self.environment.step(
            {
                "action": "Initialize",
                "gridSize": 0.25,
                "cameraY": constants.CAMERA_HEIGHT_OFFSET,
                "fieldOfView": 60.0,
                "visibilityDistance": constants.VISIBILITY_DISTANCE,
                "makeAgentsVisible": False,
                **_RENDER,
            }
        )
        self._record(
            "Initialize",
            {"fieldOfView": 60.0, "width": 300, "height": 300},
            required=True,
        )
        self.environment.step(dict(scene["init_action"]))
        self._record(
            "init_action",
            {"action": str(scene["init_action"]["action"])},
            required=True,
        )
        # Deliberately no set_task/get_goal_conditions/reachable-position calls.

    def _capture_stations(self, scene_id: str) -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []
        for station, transitions in enumerate(_STATIONS):
            if transitions:
                self._set_pitch(0)
            for action in transitions:
                request: dict[str, object] = {"action": action, **_RENDER}
                if action == "MoveAhead":
                    request["moveMagnitude"] = 0.25
                self.environment.step(request)
                self._record(action, {"station": station}, required=False)
            for pitch in _PITCHES:
                self._set_pitch(pitch)
                frames.append(self._save_frame(scene_id, station, pitch))
        return frames

    def _set_pitch(self, pitch: int) -> None:
        current = _horizon(self.environment.last_event.metadata)
        self.environment.look_angle(pitch - current, render_settings=_RENDER)
        self._record("SetPitch", {"requested_pitch_degrees": pitch}, required=True)
        if abs(_horizon(self.environment.last_event.metadata) - pitch) > 1e-3:
            raise RuntimeError("camera failed to reach a required calibration pitch")

    def _record(
        self, operation: str, details: dict[str, object], *, required: bool
    ) -> None:
        metadata = self.environment.last_event.metadata
        success = bool(metadata["lastActionSuccess"])
        record = {
            "index": len(self.actions),
            "operation": operation,
            "success": success,
            "error": str(metadata.get("errorMessage", "")),
            "actual_pitch_degrees": _horizon(metadata),
            **details,
        }
        self.actions.append(record)
        with (self.output_dir / "actions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if required and not success:
            raise RuntimeError(
                f"required calibration operation failed: {operation}; receipt preserved"
            )

    def _save_frame(self, scene_id: str, station: int, pitch: int) -> dict[str, object]:
        import numpy as np
        from PIL import Image

        event = self.environment.last_event
        rgb = np.asarray(event.frame)
        if rgb.shape != (300, 300, 3) or rgb.dtype != np.uint8:
            raise ValueError("calibration RGB must be uint8 300x300x3")
        depth = depth_meters(event.depth_frame)
        identity = f"{scene_id}_s{station}_p{pitch:+03d}"
        rgb_path, depth_path = (
            self.output_dir / f"{identity}.png",
            self.output_dir / f"{identity}_depth_m.npy",
        )
        if rgb_path.exists() or depth_path.exists():
            raise FileExistsError("refusing to overwrite captured calibration frames")
        Image.fromarray(rgb).save(str(rgb_path))
        np.save(str(depth_path), depth, allow_pickle=False)
        focal = 300 / (2 * math.tan(math.radians(60) / 2))
        frame: dict[str, object] = {
            "frame_id": identity,
            "scene_id": scene_id,
            "rgb_path": str(rgb_path.resolve()),
            "gt_depth_path": str(depth_path.resolve()),
            "intrinsics": [[focal, 0.0, 149.5], [0.0, focal, 149.5], [0.0, 0.0, 1.0]],
            "pitch_degrees": float(pitch),
            "rgb_sha256": _sha256(rgb_path),
            "gt_depth_sha256": _sha256(depth_path),
        }
        pose = _camera_pose(event.metadata)
        if pose is not None:
            frame["camera_to_world"] = pose
        return frame

    def close(self) -> None:
        if self.environment is not None:
            self.environment.stop()
            self.environment = None


def _serve(worker: _CalibrationWorker, output: TextIO) -> int:
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"command": "close"}:
                return 0
            try:
                if not isinstance(request, dict):
                    raise ValueError("capture request must be a JSON object")
                result = worker.capture(request)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                output.write(
                    json.dumps(
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "message": "offline capture failed; inspect preserved worker log and action receipts",
                        }
                    )
                    + "\n"
                )
                output.flush()
                return 1
            output.write(json.dumps({"ok": True, "result": result}) + "\n")
            output.flush()
    finally:
        worker.close()
    return 0


def main() -> int:
    """Launch the isolated Python 3.8 protocol without importing Unity eagerly."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alfred-root", "ai2thor-root", "output-dir", "unity-executable"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--x-display", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = _CalibrationWorker(
        args.alfred_root.resolve(),
        args.ai2thor_root.resolve(),
        args.output_dir.resolve(),
        args.x_display,
        args.unity_executable.resolve(strict=True),
    )
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(protocol_fd, "w", buffering=1) as output:
        with contextlib.redirect_stdout(sys.stderr):
            return _serve(worker, output)


if __name__ == "__main__":
    raise SystemExit(main())
