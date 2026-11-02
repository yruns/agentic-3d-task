"""Frozen 3D target anchor derived from a confirmed lifted mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import FloatArray, _validate_points_world
from codex_agent.scenefunc3d.backends.motion_priors import motion_size_prior

_MIN_ANCHOR_RADIUS_M = 0.02


@dataclass(frozen=True)
class TargetAnchor:
    """A frozen 3D anchor used as a visibility probe and noise filter."""

    centroid: tuple[float, float, float]
    radius_m: float
    motion_type: str
    seed_frame_id: str
    source_point_count: int


def build_anchor(
    points_world: FloatArray,
    *,
    motion_type: str,
    seed_frame_id: str,
    radius_percentile: float = 95.0,
) -> TargetAnchor:
    """Build a target anchor (centroid + motion-capped robust radius)."""
    points = _validate_points_world(points_world, field_name="anchor_points_world")
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    robust_radius = float(np.percentile(distances, radius_percentile))
    capped_radius = min(robust_radius, motion_size_prior(motion_type).max_radius_m)
    radius_m = max(capped_radius, _MIN_ANCHOR_RADIUS_M)
    return TargetAnchor(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        radius_m=radius_m,
        motion_type=motion_type,
        seed_frame_id=seed_frame_id,
        source_point_count=int(points.shape[0]),
    )


__all__ = ["TargetAnchor", "build_anchor"]
