"""Python 3.8-compatible worker for the pinned, external ALFRED/AI2-THOR stack.

Run as a file with the explicitly configured legacy interpreter. Do not import
the main project package here. RGB and proprioception form the acting boundary;
segmentation is private and used only by ALFRED's official mask action API.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

_NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)
_INTERACTION_ACTIONS = frozenset(
    {
        "PickupObject",
        "PutObject",
        "OpenObject",
        "CloseObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "SliceObject",
    }
)
_RENDER_SETTINGS = {
    "renderImage": True,
    "renderDepthImage": False,
    "renderClassImage": False,
    "renderObjectImage": True,
}


def verify_simulator_version(version: str, build: str) -> None:
    """Require both the pinned Python API and its original Unity build."""
    if version != "2.1.0" or build != "201909061227":
        raise RuntimeError(
            "ALFRED requires ai2thor==2.1.0 with build 201909061227; "
            f"got version={version!r}, build={build!r}"
        )


def sanitize_action_error(message: str, objects: Sequence[Mapping[str, object]]) -> str:
    """Remove simulator object identifiers, which can encode hidden positions."""
    for item in objects:
        identifier = item.get("objectId")
        if isinstance(identifier, str) and identifier:
            message = message.replace(identifier, "object")
    return re.sub(r"[A-Za-z][A-Za-z0-9_]*(?:\|[^\s,;]+)+", "object", message)


def load_interaction_mask(path: Path, *, shape: tuple[int, int]) -> Any:
    """Validate a predicted binary PNG/NumPy mask at the simulator boundary."""
    import numpy as np
    from PIL import Image

    if path.suffix == ".npy":
        mask = np.load(str(path), allow_pickle=False)
    else:
        with Image.open(str(path)) as image:
            mask = np.asarray(image)
    if mask.shape != shape:
        raise ValueError(
            f"interaction mask shape {mask.shape!r} does not match RGB {shape!r}"
        )
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("interaction mask must be binary (0/1 or 0/255)")
    if not np.any(mask):
        raise ValueError("interaction mask must contain foreground pixels")
    return np.asarray(mask != 0, dtype=np.uint8)


class _SimulatorWorker:
    """Single-episode owner; ``Any`` is confined to the external ThorEnv API."""

    def __init__(self, *, alfred_root: Path, output_dir: Path, x_display: str) -> None:
        self.alfred_root = alfred_root
        self.output_dir = output_dir
        self.x_display = x_display
        self.environment: Any = None
        self.frame_id = 0
        self.trajectory_path: Path | None = None
        self.trajectory_sha256 = ""
        self.scored = False
        self.actions: list[dict[str, object]] = []

    def reset(self, trajectory_path: Path) -> dict[str, object]:
        if self.environment is not None:
            raise RuntimeError("each simulator worker accepts exactly one episode")
        sys.path[:0] = [str(self.alfred_root), str(self.alfred_root / "gen")]
        import ai2thor
        from ai2thor import _builds

        verify_simulator_version(ai2thor.__version__, _builds.VERSION)
        from env.thor_env import ThorEnv
        from gen import constants

        # These are process-local settings in the official simulator wrapper.
        constants.RENDER_DEPTH_IMAGE = False
        constants.RENDER_CLASS_IMAGE = False
        constants.RENDER_OBJECT_IMAGE = True
        constants.RENDER_IMAGE = True
        raw_trajectory = trajectory_path.read_bytes()
        trajectory = json.loads(raw_trajectory)
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be a JSON object")
        scene = trajectory["scene"]
        self.trajectory_path = trajectory_path
        self.trajectory_sha256 = hashlib.sha256(raw_trajectory).hexdigest()
        self.environment = ThorEnv(x_display=self.x_display)
        self.environment.reset(
            scene["floor_plan"],
            render_depth_image=False,
            render_class_image=False,
            render_object_image=True,
        )
        self._require_success("reset")
        self.environment.restore_scene(
            scene["object_poses"], scene["object_toggles"], scene["dirty_and_empty"]
        )
        self._require_success("restore_scene")
        self.environment.step(dict(scene["init_action"]))
        self._require_success("init_action")
        self.environment.set_task(
            trajectory,
            SimpleNamespace(
                reward_config=str(self.alfred_root / "models/config/rewards.json")
            ),
            reward_type="sparse",
        )
        return self._observe(success=True, error="")

    def step(self, action: str, mask_path: Path | None) -> dict[str, object]:
        self._require_active()
        if action not in _NAVIGATION_ACTIONS | _INTERACTION_ACTIONS:
            raise ValueError(f"unsupported ALFRED action: {action!r}")
        if (action in _INTERACTION_ACTIONS) != (mask_path is not None):
            raise ValueError(
                "mask_path must be supplied exactly for interaction actions"
            )
        mask = None
        if mask_path is not None:
            mask = load_interaction_mask(
                mask_path, shape=self.environment.last_event.frame.shape[:2]
            )
        if action in {"LookUp", "LookDown"}:
            # Official look_angle defaults disable segmentation. Supply explicit
            # settings on the same action so the next mask uses the new camera.
            from gen import constants

            angle = constants.AGENT_HORIZON_ADJ * (-1 if action == "LookUp" else 1)
            event = self.environment.look_angle(angle, render_settings=_RENDER_SETTINGS)
            success = bool(event.metadata["lastActionSuccess"])
            error = str(event.metadata["errorMessage"])
            api_action: object = {"action": action}
        else:
            success, _, _, error, api_action = self.environment.va_interact(
                action, interact_mask=mask, smooth_nav=False
            )
        self.frame_id += 1
        self.actions.append(
            {"action": action, "success": bool(success), "api_action": api_action}
        )
        safe_error = sanitize_action_error(
            str(error), self.environment.last_event.metadata["objects"]
        )
        return self._observe(success=bool(success), error=safe_error)

    def score(self) -> dict[str, object]:
        self._require_active()
        success = bool(self.environment.get_goal_satisfied())
        completed, total = self.environment.get_goal_conditions_met()
        if not 0 <= completed <= total or total <= 0:
            raise ValueError("official scorer returned invalid goal condition counts")
        snapshot_path = self.output_dir / "terminal_evaluation.json"
        snapshot = {
            "trajectory_path": str(self.trajectory_path),
            "trajectory_sha256": self.trajectory_sha256,
            "metadata": self.environment.last_event.metadata,
            "cleaned_objects": sorted(self.environment.cleaned_objects),
            "cooled_objects": sorted(self.environment.cooled_objects),
            "heated_objects": sorted(self.environment.heated_objects),
            "actions": self.actions,
            "success": success,
            "completed_goal_conditions": int(completed),
            "total_goal_conditions": int(total),
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        self.scored = True
        return {
            "success": success,
            "completed_goal_conditions": int(completed),
            "total_goal_conditions": int(total),
            "snapshot_path": str(snapshot_path),
        }

    def close(self) -> None:
        if self.environment is not None:
            self.environment.stop()
            self.environment = None

    def _require_active(self) -> None:
        if self.environment is None:
            raise RuntimeError("reset is required before using the simulator")
        if self.scored:
            raise RuntimeError("terminal scoring ended this episode")

    def _require_success(self, operation: str) -> None:
        metadata = self.environment.last_event.metadata
        if not metadata["lastActionSuccess"]:
            raise RuntimeError(f"scene {operation} failed: {metadata['errorMessage']}")

    def _observe(self, *, success: bool, error: str) -> dict[str, object]:
        import numpy as np
        from PIL import Image

        event = self.environment.last_event
        frame = event.frame
        if (
            frame is None
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            raise ValueError("simulator RGB must be a uint8 HxWx3 image")
        frame_dir = self.output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = frame_dir / f"{self.frame_id:06d}.png"
        Image.fromarray(frame).save(str(rgb_path))
        agent = event.metadata["agent"]
        yaw = float(agent["rotation"]["y"])
        horizon = float(agent["cameraHorizon"])
        if not math.isfinite(yaw) or not math.isfinite(horizon):
            raise ValueError("simulator proprioception must be finite")
        return {
            "frame_id": self.frame_id,
            "rgb_path": str(rgb_path),
            "yaw_degrees": yaw,
            "horizon_degrees": horizon,
            "last_action_success": success,
            "last_action_error": error,
        }


def _dispatch(worker: _SimulatorWorker, payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("worker request must be a JSON object")
    command = payload.get("command")
    if command == "reset" and set(payload) == {"command", "trajectory_path"}:
        if not isinstance(payload["trajectory_path"], str):
            raise ValueError("trajectory_path must be a string")
        return worker.reset(Path(payload["trajectory_path"]).resolve(strict=True))
    if command == "step" and set(payload) == {"command", "action", "mask_path"}:
        if not isinstance(payload["action"], str):
            raise ValueError("action must be a string")
        mask_path = payload["mask_path"]
        if mask_path is not None and not isinstance(mask_path, str):
            raise ValueError("mask_path must be a string or null")
        path = Path(mask_path).resolve(strict=True) if mask_path is not None else None
        return worker.step(payload["action"], path)
    if command == "score" and set(payload) == {"command"}:
        return worker.score()
    raise ValueError(f"unknown command or invalid fields: {command!r}")


def _serve(worker: _SimulatorWorker, protocol_output: TextIO) -> int:
    try:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if payload == {"command": "close"}:
                    return 0
                result = _dispatch(worker, payload)
            except Exception as exc:
                # RPC boundary: report failure then exit, never fabricate evidence
                # or continue a corrupted episode after an unexpected failure.
                traceback.print_exc(file=sys.stderr)
                response = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                protocol_output.write(json.dumps(response) + "\n")
                protocol_output.flush()
                return 1
            protocol_output.write(json.dumps({"ok": True, "result": result}) + "\n")
            protocol_output.flush()
    finally:
        worker.close()
    return 0


def main() -> int:
    """Start the protocol without importing the optional simulator at startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfred-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--x-display", required=True)
    args = parser.parse_args()
    worker = _SimulatorWorker(
        alfred_root=args.alfred_root.resolve(strict=True),
        output_dir=args.output_dir.absolute(),
        x_display=args.x_display,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Unity inherits OS descriptors rather than Python's sys.stdout object.
    # Reserve a protocol descriptor and redirect fd 1 before creating children.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(protocol_fd, "w", buffering=1) as protocol_output:
        with contextlib.redirect_stdout(sys.stderr):
            return _serve(worker, protocol_output)


if __name__ == "__main__":
    raise SystemExit(main())
