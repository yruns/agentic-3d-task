"""Oriented 3D IoU for 9-DOF bounding boxes (EmbodiedScan convention).

Box format ``[cx, cy, cz, dx, dy, dz, alpha, beta, gamma]``:

* ``(cx, cy, cz)`` — center,
* ``(dx, dy, dz)`` — size along each axis,
* ``(alpha, beta, gamma)`` — ZXY Euler angles in radians,
  i.e. ``R = Rz(alpha) @ Rx(beta) @ Ry(gamma)``.

IoU is computed by enumerating the vertices of the intersection polytope (box
corners inside the other box plus edge/face crossings) and taking the convex
hull volume — exact for two oriented boxes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..errors import Nr3dDataError

_BBOX_DOF = 9

# Unit-box corner offsets (s in {-0.5, +0.5} per axis).
_UNIT_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ],
    dtype=np.float64,
)

# 12 edges as (start_corner, end_corner).
_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

# 6 faces as 4 corner indices (counter-clockwise outward).
_BOX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
)


def _as_bbox_array(bbox: Sequence[float], *, name: str) -> np.ndarray:
    values = [float(v) for v in bbox]
    if len(values) != _BBOX_DOF:
        raise Nr3dDataError(
            f"{name} must contain exactly {_BBOX_DOF} floats, got {len(values)}"
        )
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise Nr3dDataError(f"{name} contains non-finite values: {values}")
    return array


def euler_to_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert ZXY Euler angles to a 3x3 rotation matrix (``Rz @ Rx @ Ry``)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.array(
        [
            [ca * cg - sa * sb * sg, -sa * cb, ca * sg + sa * sb * cg],
            [sa * cg + ca * sb * sg, ca * cb, sa * sg - ca * sb * cg],
            [-cb * sg, sb, cb * cg],
        ],
        dtype=np.float64,
    )


def oriented_bbox_to_corners(bbox_9dof: Sequence[float]) -> np.ndarray:
    """Return the ``(8, 3)`` world-space corners of a 9-DOF oriented box."""
    box = _as_bbox_array(bbox_9dof, name="bbox_9dof")
    center = box[0:3]
    size = box[3:6]
    rotation = euler_to_rotation_matrix(box[6], box[7], box[8])
    corners = _UNIT_CORNERS * size
    corners = (rotation @ corners.T).T
    world_corners: np.ndarray = corners + center
    return world_corners


def compute_oriented_iou_3d(bbox1: Sequence[float], bbox2: Sequence[float]) -> float:
    """Return the 3D IoU in ``[0, 1]`` between two 9-DOF oriented boxes."""
    box1 = _as_bbox_array(bbox1, name="bbox1")
    box2 = _as_bbox_array(bbox2, name="bbox2")

    vol1 = abs(float(box1[3] * box1[4] * box1[5]))
    vol2 = abs(float(box2[3] * box2[4] * box2[5]))
    if vol1 < 1e-12 or vol2 < 1e-12:
        return 0.0

    corners1 = oriented_bbox_to_corners(bbox1)
    corners2 = oriented_bbox_to_corners(bbox2)
    intersection_pts = _find_intersection_vertices(corners1, corners2)
    if len(intersection_pts) < 4:
        return 0.0

    inter_vol = _convex_hull_volume(intersection_pts)
    if inter_vol < 1e-12:
        return 0.0

    union = vol1 + vol2 - inter_vol
    return float(inter_vol / max(union, 1e-12))


def _find_intersection_vertices(
    corners1: np.ndarray, corners2: np.ndarray
) -> np.ndarray:
    faces1 = _box_face_planes(corners1)
    faces2 = _box_face_planes(corners2)
    pts: list[np.ndarray] = []

    for corner in corners1:
        if _point_inside_halfspaces(corner, faces2):
            pts.append(corner)
    for corner in corners2:
        if _point_inside_halfspaces(corner, faces1):
            pts.append(corner)

    for i, j in _BOX_EDGES:
        for normal, offset in faces2:
            pt = _edge_plane_intersection(corners1[i], corners1[j], normal, offset)
            if pt is not None and _point_inside_halfspaces(pt, faces2):
                pts.append(pt)
    for i, j in _BOX_EDGES:
        for normal, offset in faces1:
            pt = _edge_plane_intersection(corners2[i], corners2[j], normal, offset)
            if pt is not None and _point_inside_halfspaces(pt, faces1):
                pts.append(pt)

    if not pts:
        return np.empty((0, 3))
    return np.array(pts)


def _box_face_planes(corners: np.ndarray) -> list[tuple[np.ndarray, float]]:
    center = corners.mean(axis=0)
    planes: list[tuple[np.ndarray, float]] = []
    for face in _BOX_FACES:
        p0, p1, p2 = corners[face[0]], corners[face[1]], corners[face[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = float(np.dot(normal, p0))
        if np.dot(normal, center) > offset:
            normal = -normal
            offset = -offset
        planes.append((normal, offset))
    return planes


def _point_inside_halfspaces(
    point: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
    eps: float = 1e-6,
) -> bool:
    return all(np.dot(normal, point) <= offset + eps for normal, offset in planes)


def _edge_plane_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    normal: np.ndarray,
    offset: float,
    eps: float = 1e-8,
) -> np.ndarray | None:
    d0 = float(np.dot(normal, p0)) - offset
    d1 = float(np.dot(normal, p1)) - offset
    if d0 * d1 > eps:
        return None
    denom = d0 - d1
    if abs(denom) < eps:
        return None
    t = d0 / denom
    if t < -eps or t > 1.0 + eps:
        return None
    crossing: np.ndarray = p0 + t * (p1 - p0)
    return crossing


def _convex_hull_volume(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-8) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
    except QhullError:
        # A degenerate (near-coplanar/collinear) point set has no 3D volume.
        return 0.0
    return float(hull.volume)


__all__ = [
    "euler_to_rotation_matrix",
    "oriented_bbox_to_corners",
    "compute_oriented_iou_3d",
]
