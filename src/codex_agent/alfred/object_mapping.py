"""Deterministic point-cloud and CLIP operations for online object mapping.

These implement the geometry/feature fusion used by our OpenEQA mapper without
its dataset loop, mutable list identities, pickle inputs, or model dependencies.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .geometry import Detection, FloatArray, bound_rows, validated_feature


@dataclass(frozen=True)
class MappingConfig:
    """Metric geometry and CLIP thresholds, serialized with each evaluation."""

    voxel_size_m: float = 0.025
    dbscan_eps_m: float = 0.1
    dbscan_min_points: int = 10
    min_detection_points: int = 16
    max_points_per_object: int = 4096
    association_threshold: float = 1.2
    phys_bias: float = 0.0
    min_mask_confidence: float = 0.25
    min_mask_pixels: int = 25
    max_bbox_area_ratio: float = 0.5
    contained_child_fraction: float = 0.8
    contained_parent_fraction: float = 0.7
    merge_overlap_threshold: float = 0.7
    merge_image_similarity: float = 0.8
    merge_text_similarity: float = 0.8
    maintenance_interval: int = 20
    confirmation_observations: int = 3
    duplicate_mask_iou: float = 0.8
    duplicate_image_similarity: float = 0.9
    reacquire_similarity: float = 0.75
    reacquire_margin_m: float = 0.6
    held_max_camera_distance_m: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = {
            "dbscan_min_points",
            "min_detection_points",
            "max_points_per_object",
            "maintenance_interval",
            "confirmation_observations",
            "min_mask_pixels",
        }
        unit_fields = {
            "merge_overlap_threshold",
            "merge_image_similarity",
            "merge_text_similarity",
            "duplicate_mask_iou",
            "duplicate_image_similarity",
            "reacquire_similarity",
            "min_mask_confidence",
            "max_bbox_area_ratio",
            "contained_child_fraction",
            "contained_parent_fraction",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in integer_fields:
                if type(value) is not int or value < 1:
                    raise ValueError(
                        f"{item.name} must be a positive integer, got {value!r}"
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{item.name} must be finite, got {value!r}")
            elif item.name in unit_fields:
                if not 0 <= value <= 1:
                    raise ValueError(f"{item.name} must be in [0, 1], got {value!r}")
            elif item.name == "phys_bias":
                if not -1 <= value <= 1:
                    raise ValueError("phys_bias must be in [-1, 1]")
            elif value <= 0:
                raise ValueError(f"{item.name} must be positive, got {value!r}")
        if self.association_threshold > 2:
            raise ValueError(
                "association_threshold must not exceed the overlap + cosine maximum 2"
            )
        if self.max_bbox_area_ratio == 0:
            raise ValueError("max_bbox_area_ratio must be positive")


def feature_cosine(first: FloatArray, second: FloatArray) -> float:
    """Compute cosine, including accepted floating-point normalization error."""
    left = validated_feature(first, "first_feature")
    right = validated_feature(second, "second_feature")
    if left.shape != right.shape:
        raise ValueError("feature dimensions must match")
    return float(
        np.clip((left @ right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1, 1)
    )


def mapping_masks(
    detections: Sequence[Detection], config: MappingConfig
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Apply OpenEQA detection filters, then subtract contained instance masks.

    Detector boxes and original masks remain immutable. Returned masks alone
    supply geometry and interactions, preserving the original feature crops.
    """
    retained: dict[int, np.ndarray] = {}
    rejected: dict[int, str] = {}
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox_xyxy
        if detection.confidence < config.min_mask_confidence:
            rejected[index] = "confidence_below_threshold"
        elif int(detection.mask.sum()) < max(10, config.min_mask_pixels):
            rejected[index] = "mask_area_below_threshold"
        elif detection.category.casefold() in {"wall", "floor", "ceiling"}:
            rejected[index] = "background_category"
        elif (x2 - x1) * (y2 - y1) > config.max_bbox_area_ratio * detection.mask.size:
            rejected[index] = "detector_bbox_area_above_threshold"
        else:
            retained[index] = detection.mask.copy()
    for parent in retained:
        parent_box = np.asarray(detections[parent].bbox_xyxy)
        parent_area = float(np.prod(parent_box[2:] - parent_box[:2]))
        for child in retained:
            if parent == child:
                continue
            child_box = np.asarray(detections[child].bbox_xyxy)
            child_area = float(np.prod(child_box[2:] - child_box[:2]))
            intersection = float(
                np.prod(
                    np.maximum(
                        0,
                        np.minimum(parent_box[2:], child_box[2:])
                        - np.maximum(parent_box[:2], child_box[:2]),
                    )
                )
            )
            if (
                intersection / parent_area < config.contained_parent_fraction
                and intersection / child_area > config.contained_child_fraction
            ):
                retained[parent] &= ~retained[child]
    for mask in retained.values():
        mask.setflags(write=False)
    return retained, rejected


def association_similarity(
    overlap: float, visual_similarity: float, phys_bias: float
) -> float:
    """Combine geometric support and CLIP cosine with the OpenEQA sim_sum rule."""
    values = (overlap, visual_similarity, phys_bias)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("association similarities and phys_bias must be finite")
    if (
        not 0 <= overlap <= 1
        or not -1 <= visual_similarity <= 1
        or not -1 <= phys_bias <= 1
    ):
        raise ValueError(
            "overlap must be in [0,1]; cosine and phys_bias must be in [-1,1]"
        )
    return (1 + phys_bias) * overlap + (1 - phys_bias) * visual_similarity


def validate_cloud(points: FloatArray, colors: FloatArray) -> None:
    """Reject malformed, nonfinite or misaligned world point/color arrays."""
    if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
        raise ValueError("points and colors must have the same Nx3 shape")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("points and colors must contain finite values")
    if np.any(colors < 0) or np.any(colors > 1):
        raise ValueError("colors must be in [0, 1]")


def voxel_reduce(
    points: FloatArray, colors: FloatArray, size_m: float
) -> tuple[FloatArray, FloatArray]:
    """Average points/colors on the original cloud-min-minus-half-voxel lattice."""
    validate_cloud(points, colors)
    if not math.isfinite(size_m) or size_m <= 0:
        raise ValueError("size_m must be positive and finite")
    if not len(points):
        return points.copy(), colors.copy()
    _, inverse, counts = np.unique(
        np.floor((points - points.min(axis=0) + size_m / 2) / size_m).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    point_sums = np.zeros((len(counts), 3), dtype=np.float64)
    color_sums = np.zeros_like(point_sums)
    np.add.at(point_sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    return point_sums / counts[:, None], color_sums / counts[:, None]


def largest_density_cluster(
    points: FloatArray, *, eps_m: float, min_points: int
) -> np.ndarray:
    """Return the largest DBSCAN cluster indices; retain a wholly sparse cloud.

    As in the OpenEQA mapper, absence of a core cluster does not erase a small
    newly observed object. No random perturbation is added to the geometry.
    """
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    if (
        not math.isfinite(eps_m)
        or eps_m <= 0
        or type(min_points) is not int
        or min_points < 1
    ):
        raise ValueError("DBSCAN eps_m and min_points must be positive")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    neighbors = cKDTree(points).query_ball_point(points, eps_m)
    labels = np.full(len(points), -1, dtype=np.int64)
    visited = np.zeros(len(points), dtype=bool)
    cluster = 0
    for index in range(len(points)):
        if visited[index]:
            continue
        visited[index] = True
        if len(neighbors[index]) < min_points:
            continue
        labels[index] = cluster
        queue = deque(int(value) for value in neighbors[index])
        queued = set(queue)
        while queue:
            neighbor = queue.popleft()
            if not visited[neighbor]:
                visited[neighbor] = True
                if len(neighbors[neighbor]) >= min_points:
                    for candidate in neighbors[neighbor]:
                        if candidate not in queued:
                            queue.append(int(candidate))
                            queued.add(int(candidate))
            if labels[neighbor] < 0:
                labels[neighbor] = cluster
        cluster += 1
    if cluster == 0:
        return np.arange(len(points), dtype=np.int64)
    counts = np.bincount(labels[labels >= 0])
    selected = np.flatnonzero(labels == int(np.argmax(counts)))
    # Preserve the original mapper's guard against destroying a tiny object.
    return selected if len(selected) >= 5 else np.arange(len(points), dtype=np.int64)


def process_cloud(
    points: FloatArray, colors: FloatArray, config: MappingConfig, *, denoise: bool
) -> tuple[FloatArray, FloatArray]:
    """Voxel-average and denoise before enforcing the explicit final storage cap."""
    points, colors = voxel_reduce(points, colors, config.voxel_size_m)
    if denoise:
        keep = largest_density_cluster(
            points, eps_m=config.dbscan_eps_m, min_points=config.dbscan_min_points
        )
        points, colors = points[keep], colors[keep]
    return bound_rows(points, config.max_points_per_object), bound_rows(
        colors, config.max_points_per_object
    )


def point_overlap(
    observed: FloatArray, existing: FloatArray, distance_m: float
) -> float:
    """Fraction of observed points within one voxel of an existing object.

    This is the directed nearest-point kernel. The mapper separately applies
    the oriented-envelope candidate gate, so the denominator remains explicit.
    """
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive and finite")
    for points in (observed, existing):
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.isfinite(points).all()
        ):
            raise ValueError("point clouds must be finite Nx3 arrays")
    if not len(observed) or not len(existing):
        return 0.0
    if np.any(observed.min(axis=0) > existing.max(axis=0) + distance_m) or np.any(
        existing.min(axis=0) > observed.max(axis=0) + distance_m
    ):
        return 0.0
    distances, _ = cKDTree(existing).query(
        observed, k=1, distance_upper_bound=distance_m
    )
    return float(np.count_nonzero(np.isfinite(distances)) / len(observed))


def _oriented_envelope(points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit hull-PCA axes, using deterministic PCA for degenerate point sets."""
    support = points
    if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 3:
        try:
            support = points[ConvexHull(points).vertices]
        except QhullError:
            # A numerically degenerate hull has no reliable 3D volume. PCA still
            # yields a conservative envelope without perturbing measured points.
            support = points
    centered = support - support.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered / max(1, len(support)))
    axes = axes[:, ::-1]
    coordinates = points @ axes
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    return axes @ ((minimum + maximum) / 2), axes, (maximum - minimum) / 2


def oriented_envelopes_overlap(first: FloatArray, second: FloatArray) -> bool:
    """Check PCA oriented boxes using the separating-axis theorem, without jitter.

    Zero-thickness boxes remain usable for planar detections. Contact within
    numerical tolerance is a candidate; final point-neighbor support still gates
    association. This does not claim bitwise equivalence to legacy 3D IoU.
    """
    for points in (first, second):
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("oriented envelopes require nonempty finite Nx3 clouds")
    first_center, first_axes, first_extent = _oriented_envelope(first)
    second_center, second_axes, second_extent = _oriented_envelope(second)
    axes = [first_axes[:, axis] for axis in range(3)] + [
        second_axes[:, axis] for axis in range(3)
    ]
    axes.extend(
        np.cross(first_axes[:, left], second_axes[:, right])
        for left in range(3)
        for right in range(3)
    )
    offset = second_center - first_center
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length <= 1e-12:
            continue
        direction = axis / length
        first_radius = float(np.abs(first_axes.T @ direction) @ first_extent)
        second_radius = float(np.abs(second_axes.T @ direction) @ second_extent)
        if abs(float(offset @ direction)) > first_radius + second_radius + 1e-9:
            return False
    return True


def fuse_features(
    first: FloatArray, second: FloatArray, first_count: int, second_count: int
) -> FloatArray:
    """Observation-count weighted normalized CLIP fusion, never RGB histograms."""
    left = validated_feature(first, "first_feature")
    right = validated_feature(second, "second_feature")
    if left.shape != right.shape:
        raise ValueError("feature dimensions must match")
    if any(
        type(value) is not int or value < 1 for value in (first_count, second_count)
    ):
        raise ValueError("feature observation counts must be positive integers")
    combined = left * first_count + right * second_count
    norm = float(np.linalg.norm(combined))
    if norm < 1e-8:
        raise ValueError("weighted features cancel and cannot be normalized")
    return np.asarray(combined / norm, dtype=np.float64)
