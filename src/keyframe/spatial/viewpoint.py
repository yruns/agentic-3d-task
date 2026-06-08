"""Viewer-frame geometry for viewpoint-aware spatial constraints.

The executor consults this module when a constraint sets
``reference_frame=ReferenceFrame.VIEWER``. ``resolve_viewer_pose`` picks a
viewer position + facing direction (preferring a real camera frame that looks
at the anchors), and ``viewer_frame_relation_score`` / ``viewer_frame_axis_value``
evaluate directional relations and selection axes in that frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from keyframe.models.scene import SceneObject

#: How the viewer pose was obtained.
ViewerPoseSource = Literal["trajectory", "geometric"]


@dataclass(frozen=True)
class ViewerPose:
    """A resolved viewer pose with an orthonormal (forward, right, up) basis.

    All vectors are in world coordinates; ``up`` is +Z by convention.
    """

    position: NDArray[np.float64]
    forward: NDArray[np.float64]
    right: NDArray[np.float64]
    up: NDArray[np.float64]
    source: ViewerPoseSource


def _to_xyz(value: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.shape == (3,) else None


def viewer_axes(
    forward: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build a horizontal (forward, right, up) orthonormal basis with +Z up."""
    flat = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        raise ValueError("viewer forward is degenerate after horizontal projection")
    flat = flat / norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(flat, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        raise ValueError("viewer right vector is degenerate")
    return flat, right / right_norm, up


def _camera_visibility_score(
    cam_world_pose: NDArray[np.float64],
    anchor_centroids: NDArray[np.float64],
) -> float:
    """Mean cosine between a camera's forward axis and the anchor directions."""
    cam_pos = cam_world_pose[:3, 3]
    cam_forward = -cam_world_pose[:3, 2]
    cam_forward = cam_forward / max(1e-6, float(np.linalg.norm(cam_forward)))

    cosines: list[float] = []
    for centroid in anchor_centroids:
        delta = centroid - cam_pos
        norm = float(np.linalg.norm(delta))
        if norm < 1e-3:
            continue
        cosines.append(float(np.dot(delta / norm, cam_forward)))
    if not cosines:
        return float("-inf")
    return float(np.mean(cosines))


def resolve_viewer_pose(
    anchor_objects: Sequence[SceneObject],
    *,
    all_object_centroids: NDArray[np.float64] | None = None,
    camera_poses: Sequence[NDArray[np.float64]] = (),
    standoff: float = 1.5,
    allow_geometric_fallback: bool = True,
) -> ViewerPose | None:
    """Resolve a viewer pose from facing-anchor objects.

    Trajectory mode (preferred) picks the camera frame whose forward axis best
    aligns with the anchors. The geometric fallback places the viewer
    ``standoff`` metres back from the anchor centroid along the room-center ->
    anchor line. The executor passes ``allow_geometric_fallback=False`` so a
    degenerate trajectory cannot silently fabricate a pose; returns None when
    no pose can be resolved.
    """
    if not anchor_objects:
        return None

    centroids = [
        c for c in (_to_xyz(obj.centroid) for obj in anchor_objects) if c is not None
    ]
    if not centroids:
        return None
    anchor_centroids = np.stack(centroids)
    mean_anchor = anchor_centroids.mean(axis=0)

    if camera_poses:
        best_pose: NDArray[np.float64] | None = None
        best_score = float("-inf")
        for raw_pose in camera_poses:
            pose = np.asarray(raw_pose, dtype=np.float64)
            if pose.shape != (4, 4):
                continue
            score = _camera_visibility_score(pose, anchor_centroids)
            if score > best_score:
                best_score = score
                best_pose = pose
        if best_pose is not None and best_score > -1.0:
            try:
                forward, right, up = viewer_axes(-best_pose[:3, 2])
            except ValueError:
                pass
            else:
                return ViewerPose(
                    position=best_pose[:3, 3].astype(np.float64),
                    forward=forward,
                    right=right,
                    up=up,
                    source="trajectory",
                )

    if not allow_geometric_fallback:
        return None

    if all_object_centroids is None or len(all_object_centroids) == 0:
        room_center = anchor_centroids.mean(axis=0)
    else:
        room_center = np.asarray(all_object_centroids, dtype=np.float64).mean(axis=0)

    forward_world = mean_anchor - room_center
    forward_world[2] = 0.0
    if float(np.linalg.norm(forward_world)) < 1e-3:
        return None
    try:
        forward, right, up = viewer_axes(forward_world)
    except ValueError:
        return None

    return ViewerPose(
        position=(mean_anchor - standoff * forward).astype(np.float64),
        forward=forward,
        right=right,
        up=up,
        source="geometric",
    )


def viewer_frame_relation_score(
    relation: str,
    candidate_centroid: NDArray[np.float64],
    anchor_centroid: NDArray[np.float64],
    pose: ViewerPose,
) -> float:
    """Satisfaction score in [0, 1] for a directional relation in viewer frame."""
    delta = np.asarray(candidate_centroid, dtype=np.float64) - np.asarray(
        anchor_centroid, dtype=np.float64
    )
    right_proj = float(np.dot(delta, pose.right))
    forward_proj = float(np.dot(delta, pose.forward))

    relation_norm = relation.lower().replace(" ", "_")
    if relation_norm == "right_of":
        sign, axis_proj, off_proj = 1.0, right_proj, forward_proj
    elif relation_norm == "left_of":
        sign, axis_proj, off_proj = -1.0, right_proj, forward_proj
    elif relation_norm == "in_front_of":
        sign, axis_proj, off_proj = -1.0, forward_proj, right_proj
    elif relation_norm == "behind":
        sign, axis_proj, off_proj = 1.0, forward_proj, right_proj
    else:
        return 0.0

    signed = sign * axis_proj
    if signed <= 0:
        return 0.0
    score = signed / (signed + abs(off_proj) + 0.5)
    return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0


def viewer_frame_axis_value(
    centroid: NDArray[np.float64],
    pose: ViewerPose,
    axis: Literal["right", "forward"],
) -> float:
    """Project a centroid onto a viewer-frame axis for SelectConstraint sorting."""
    point = np.asarray(centroid, dtype=np.float64)
    if axis == "right":
        return float(np.dot(point, pose.right))
    if axis == "forward":
        return float(np.dot(point, pose.forward))
    raise ValueError(f"unknown viewer axis {axis!r}; expected 'right' or 'forward'")
