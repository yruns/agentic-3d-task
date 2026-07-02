"""Layered, anchor-centric multi-view point fusion for SceneFun3D."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor
from codex_agent.scenefunc3d.backends.lift_3d import FloatArray
from codex_agent.scenefunc3d.backends.motion_priors import motion_size_prior
from codex_agent.scenefunc3d.tools.models import ToolInputError


@dataclass(frozen=True)
class FrameLift:
    """One frame's lifted raw-mesh vertex ids."""

    frame_id: str
    point_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ToolInputError("FrameLift.frame_id must be non-empty")
        if any(index < 0 for index in self.point_indices):
            raise ToolInputError("FrameLift.point_indices must be non-negative")


@dataclass(frozen=True)
class MultiViewLiftBundle:
    """Per-frame lifted vertex ids plus the frozen anchor for one sample."""

    frames: tuple[FrameLift, ...]
    anchor: TargetAnchor
    visibility_counts: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ToolInputError("MultiViewLiftBundle requires at least one frame")


@dataclass(frozen=True)
class FusionParams:
    """Tunable fusion thresholds (calibrated by the offline sweep)."""

    agreement_tau: float = 0.5
    radius_scale: float = 1.0
    cluster_link_eps_m: float = 0.02
    min_cluster_points: int = 10


@dataclass(frozen=True)
class FusedInstance:
    """One fused instance cluster."""

    point_indices: tuple[int, ...]
    confidence: float
    bbox_extent_m: float
    size_prior_ok: bool


@dataclass(frozen=True)
class FusedMask:
    """Final fused mask (primary instance) plus ranked instances for AP."""

    point_indices: tuple[int, ...]
    confidence: float
    instances: tuple[FusedInstance, ...]
    params: FusionParams


def agreement_scores(
    frames: Iterable[FrameLift],
    visibility_counts: Mapping[int, int],
) -> dict[int, float]:
    """Return per-vertex multi-view agreement in [0, 1].

    Agreement is ``hits / visibility`` when a per-vertex visibility count is
    known, else ``hits / n_frames`` over the frames in the bundle. Duplicate
    ids inside one frame count once for that frame.
    """
    frame_list = list(frames)
    n_frames = len(frame_list)
    hit_counts: dict[int, int] = {}
    for frame in frame_list:
        for index in set(frame.point_indices):
            hit_counts[index] = hit_counts.get(index, 0) + 1

    scores: dict[int, float] = {}
    for index, hits in hit_counts.items():
        denominator = visibility_counts.get(index, n_frames)
        if denominator <= 0:
            denominator = n_frames
        scores[index] = min(1.0, hits / denominator)
    return scores


def _neighbor_pairs(coords: FloatArray, eps: float) -> list[tuple[int, int]]:
    """Return index pairs within ``eps`` (scipy KD-tree, numpy fallback)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        deltas = coords[:, None, :] - coords[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas)
        upper = np.triu(distances_sq <= eps * eps, k=1)
        rows, cols = np.nonzero(upper)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    return [(int(a), int(b)) for a, b in pairs]


def connected_components(coords: FloatArray, eps: float) -> list[tuple[int, ...]]:
    """Single-linkage components of ``coords`` at radius ``eps``."""
    n_points = int(coords.shape[0])
    parent = list(range(n_points))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in _neighbor_pairs(coords, eps):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    groups: dict[int, list[int]] = {}
    for node in range(n_points):
        groups.setdefault(find(node), []).append(node)
    return [tuple(sorted(members)) for members in groups.values()]


def anchor_component_indices(
    coords: FloatArray,
    *,
    anchor_centroid: tuple[float, float, float],
    eps: float,
    min_points: int,
) -> tuple[int, ...]:
    """Return the component containing the point nearest the anchor centroid.

    Returns an empty tuple when the chosen component is smaller than
    ``min_points`` (treated as a failed cluster).
    """
    if coords.shape[0] == 0:
        return ()
    centroid = np.asarray(anchor_centroid, dtype=np.float64)
    nearest_row = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))
    for component in connected_components(coords, eps):
        if nearest_row in component:
            if len(component) < min_points:
                return ()
            return component
    return ()


def _bbox_extent_m(coords: FloatArray) -> float:
    if coords.shape[0] == 0:
        return 0.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0)))


def _empty_fused_mask(params: FusionParams) -> FusedMask:
    return FusedMask(point_indices=(), confidence=0.0, instances=(), params=params)


def fuse_multiview_points(
    bundle: MultiViewLiftBundle,
    scene_vertices: FloatArray,
    params: FusionParams,
) -> FusedMask:
    """Fuse per-frame lifted vertex ids into one anchor-centric mask.

    Pipeline: agreement gate -> anchor radius gate -> connected components ->
    keep the anchor component. Confidence is the mean agreement of kept
    vertices; the kept component is also emitted as the primary ranked instance.
    """
    scores = agreement_scores(bundle.frames, bundle.visibility_counts)
    kept_ids = [
        index for index, score in scores.items() if score >= params.agreement_tau
    ]
    if not kept_ids:
        return _empty_fused_mask(params)

    centroid = np.asarray(bundle.anchor.centroid, dtype=np.float64)
    gate_radius = bundle.anchor.radius_m * params.radius_scale
    kept_coords = scene_vertices[kept_ids]
    within_radius = np.linalg.norm(kept_coords - centroid, axis=1) <= gate_radius
    gated_ids = [index for index, keep in zip(kept_ids, within_radius) if keep]
    if not gated_ids:
        return _empty_fused_mask(params)

    gated_coords = scene_vertices[gated_ids]
    component_rows = anchor_component_indices(
        gated_coords,
        anchor_centroid=bundle.anchor.centroid,
        eps=params.cluster_link_eps_m,
        min_points=params.min_cluster_points,
    )
    if not component_rows:
        return _empty_fused_mask(params)

    final_ids = tuple(sorted(gated_ids[row] for row in component_rows))
    confidence = float(np.mean([scores[index] for index in final_ids]))
    extent = _bbox_extent_m(scene_vertices[list(final_ids)])
    size_ok = extent <= motion_size_prior(bundle.anchor.motion_type).max_bbox_extent_m
    instance = FusedInstance(
        point_indices=final_ids,
        confidence=confidence,
        bbox_extent_m=extent,
        size_prior_ok=size_ok,
    )
    return FusedMask(
        point_indices=final_ids,
        confidence=confidence,
        instances=(instance,),
        params=params,
    )


__all__ = [
    "FrameLift",
    "MultiViewLiftBundle",
    "FusionParams",
    "FusedInstance",
    "FusedMask",
    "agreement_scores",
    "connected_components",
    "anchor_component_indices",
    "fuse_multiview_points",
]
