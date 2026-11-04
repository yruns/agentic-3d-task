"""Action odometry and calibrated camera geometry, without world-state queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ActionOdometry:
    """Episode-relative translation using successful 0.25 m ALFRED moves.

    Heading and camera pitch are proprioceptive measurements. No simulator
    position, reachable-position list or object geometry enters this estimate.
    Camera height is the calibrated standing-agent height used by FILM.
    """

    x: float = 0.0
    y: float = 0.0
    camera_height_m: float = 1.576
    step_size_m: float = 0.25

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y)):
            raise ValueError("odometry x and y must be finite")
        for name in ("camera_height_m", "step_size_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}={value!r} must be finite and positive")

    def update(self, action: str, *, success: bool, yaw_degrees: float) -> None:
        """Advance translation only after a successful forward action."""
        if not math.isfinite(yaw_degrees):
            raise ValueError("yaw_degrees must be finite")
        if success and action == "MoveAhead":
            yaw = math.radians(yaw_degrees)
            self.x = round(self.x + self.step_size_m * math.sin(yaw), 9)
            self.y = round(self.y + self.step_size_m * math.cos(yaw), 9)

    @property
    def agent_position(self) -> tuple[float, float, float]:
        """Ground-plane position in the episode-relative Z-up world."""
        return self.x, self.y, 0.0

    def camera_to_world(
        self, *, yaw_degrees: float, horizon_degrees: float
    ) -> NDArray[np.float64]:
        """Map camera right/down/forward to world X/Y/up; positive pitch is down."""
        if not all(math.isfinite(v) for v in (yaw_degrees, horizon_degrees)):
            raise ValueError("yaw_degrees and horizon_degrees must be finite")
        yaw, pitch = math.radians(yaw_degrees), math.radians(horizon_degrees)
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        forward = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = -math.sin(pitch) * forward - math.cos(pitch) * up
        transform[:3, 2] = math.cos(pitch) * forward - math.sin(pitch) * up
        transform[:3, 3] = self.x, self.y, self.camera_height_m
        return transform


def camera_intrinsics(
    *, width: int = 300, height: int = 300, vertical_fov_degrees: float = 60.0
) -> NDArray[np.float64]:
    """Construct pinhole intrinsics for the ALFRED simulator's calibrated camera."""
    if width < 1 or height < 1 or not 0 < vertical_fov_degrees < 180:
        raise ValueError("positive image dimensions and FOV in (0, 180) are required")
    focal = height / (2 * math.tan(math.radians(vertical_fov_degrees) / 2))
    return np.array(
        [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]]
    )
