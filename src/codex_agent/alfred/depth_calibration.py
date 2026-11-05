"""Train-only metric-depth comparison with a predeclared selection rule.

Ground-truth depth is used exclusively for scoring. Predictions are never fitted,
median-scaled, or otherwise aligned to ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .depth_models import validate_intrinsics
from .provenance import file_sha256

SELECTION_RULE: dict[str, object] = {
    "version": 1,
    "minimum_frame_coverage": 0.99,
    "aggregation": "equal-weight macro average of frames",
    "ranking": ["abs_rel", "near_abs_rel", "edge_gradient_mae_m", "rmse_m"],
    "tie_tolerance": 1e-8,
    "near_range_m": [0.3, 2.0],
    "gt_depth_range_m_exclusive": [0.0, 5.0],
    "gt_saturation_m": 5.0,
    "edge_depth_jump_m": 0.1,
    "ground_truth_scale_alignment": False,
    "geometry_metrics": "diagnostic only; excluded from model selection",
}


class CalibrationFrame(BaseModel):
    """One immutable captured training observation with evaluator-only targets."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    frame_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    scene_id: str = Field(min_length=1)
    rgb_path: Path
    gt_depth_path: Path
    intrinsics: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    pitch_degrees: float = Field(ge=-30.001, le=60.001)
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gt_depth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera_to_world: (
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
        | None
    ) = None
    floor_mask_path: Path | None = None
    floor_mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_geometry(self) -> CalibrationFrame:
        validate_intrinsics(np.asarray(self.intrinsics))
        if (self.floor_mask_path is None) != (self.floor_mask_sha256 is None):
            raise ValueError("floor mask path and hash must be provided together")
        if self.camera_to_world is not None:
            pose = np.asarray(self.camera_to_world)
            if (
                not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
            ):
                raise ValueError("camera_to_world must be a rigid 4x4 camera pose")
        return self


class CalibrationManifest(BaseModel):
    """Frozen train capture; extra top-level provenance is retained verbatim."""

    model_config = ConfigDict(extra="allow", frozen=True)
    schema_version: Literal[1]
    split: Literal["train"]
    capture_protocol: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frames: tuple[CalibrationFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_frames(self) -> CalibrationManifest:
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ValueError("calibration frame_id values must be unique")
        return self


def load_calibration_manifest(path: Path) -> CalibrationManifest:
    """Validate every frozen image/depth hash before loading either model."""
    manifest = CalibrationManifest.model_validate_json(path.read_text())
    resolved: list[CalibrationFrame] = []
    for frame in manifest.frames:
        updates: dict[str, Path] = {}
        for field, expected in (
            ("rgb_path", frame.rgb_sha256),
            ("gt_depth_path", frame.gt_depth_sha256),
            ("floor_mask_path", frame.floor_mask_sha256),
        ):
            source = getattr(frame, field)
            if source is None:
                continue
            source = (
                (path.parent / source).resolve()
                if not source.is_absolute()
                else source.resolve()
            )
            if not source.is_file() or file_sha256(source) != expected:
                raise ValueError(
                    f"frozen calibration artifact missing or changed: {source}"
                )
            updates[field] = source
        resolved.append(frame.model_copy(update=updates))
    return manifest.model_copy(update={"frames": tuple(resolved)})


class DepthMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    total_pixels: int = Field(gt=0)
    saturated_gt_pixels: int = Field(ge=0)
    saturation_fraction: float = Field(ge=0, le=1)
    valid_gt_pixels: int = Field(gt=0)
    valid_prediction_pixels: int = Field(ge=0)
    valid_coverage: float = Field(ge=0, le=1)
    abs_rel: float | None = Field(ge=0)
    rmse_m: float | None = Field(ge=0)
    median_scale_ratio: float | None = Field(gt=0)
    near_gt_pixels: int = Field(ge=0)
    near_valid_pixels: int = Field(ge=0)
    near_coverage: float | None = Field(ge=0, le=1)
    near_abs_rel: float | None = Field(ge=0)
    near_rmse_m: float | None = Field(ge=0)
    edge_pairs: int = Field(ge=0)
    edge_gradient_mae_m: float | None = Field(ge=0)


def depth_metrics(
    predicted: NDArray[np.floating[Any]], ground_truth: NDArray[np.floating[Any]]
) -> DepthMetrics:
    """Score raw camera-axis meters, including invalid-prediction coverage loss."""
    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(ground_truth, dtype=np.float64)
    if target.ndim != 2 or not target.size or prediction.shape != target.shape:
        raise ValueError("predicted/ground_truth must be matching nonempty HxW depth")
    valid_gt = np.isfinite(target) & (target > 0) & (target < 5)
    saturated = np.isfinite(target) & (target >= 5)
    if not valid_gt.any():
        raise ValueError("ground_truth has no valid positive depth pixels")
    valid = valid_gt & np.isfinite(prediction) & (prediction > 0)
    near_gt = valid_gt & (target >= 0.3) & (target <= 2.0)
    near = valid & near_gt
    abs_rel, rmse = _depth_error(prediction, target, valid)
    near_abs_rel, near_rmse = _depth_error(prediction, target, near)
    edges: list[NDArray[np.float64]] = []
    for axis in (0, 1):
        target_delta = np.diff(target, axis=axis)
        predicted_delta = np.diff(prediction, axis=axis)
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        edge = valid[tuple(left)] & valid[tuple(right)] & (np.abs(target_delta) >= 0.1)
        edges.append(np.abs(predicted_delta[edge] - target_delta[edge]))
    edge_errors = np.concatenate(edges)
    return DepthMetrics(
        total_pixels=target.size,
        saturated_gt_pixels=int(saturated.sum()),
        saturation_fraction=float(saturated.mean()),
        valid_gt_pixels=int(valid_gt.sum()),
        valid_prediction_pixels=int(valid.sum()),
        valid_coverage=float(valid.sum() / valid_gt.sum()),
        abs_rel=abs_rel,
        rmse_m=rmse,
        median_scale_ratio=(
            float(np.median(prediction[valid] / target[valid])) if valid.any() else None
        ),
        near_gt_pixels=int(near_gt.sum()),
        near_valid_pixels=int(near.sum()),
        near_coverage=float(near.sum() / near_gt.sum()) if near_gt.any() else None,
        near_abs_rel=near_abs_rel,
        near_rmse_m=near_rmse,
        edge_pairs=len(edge_errors),
        edge_gradient_mae_m=float(edge_errors.mean()) if edge_errors.size else None,
    )


def _depth_error(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> tuple[float | None, float | None]:
    if not valid.any():
        return None, None
    residual = prediction[valid] - target[valid]
    return float(np.mean(np.abs(residual) / target[valid])), float(
        np.sqrt(np.mean(residual**2))
    )


def aggregate_depth_metrics(rows: Sequence[DepthMetrics]) -> dict[str, object]:
    """Report macro scores and coverage denominators, retaining unavailable metrics."""
    if not rows:
        raise ValueError("cannot aggregate empty depth comparison")
    result: dict[str, object] = {
        "frames": len(rows),
        "minimum_frame_coverage": min(row.valid_coverage for row in rows),
        "total_pixels": sum(row.total_pixels for row in rows),
        "saturated_gt_pixels": sum(row.saturated_gt_pixels for row in rows),
        "valid_gt_pixels": sum(row.valid_gt_pixels for row in rows),
        "valid_prediction_pixels": sum(row.valid_prediction_pixels for row in rows),
    }
    for name in (
        "saturation_fraction",
        "valid_coverage",
        "abs_rel",
        "rmse_m",
        "median_scale_ratio",
        "near_coverage",
        "near_abs_rel",
        "near_rmse_m",
        "edge_gradient_mae_m",
    ):
        values = [getattr(row, name) for row in rows if getattr(row, name) is not None]
        result[name] = float(np.mean(values)) if values else None
        result[name + "_frames"] = len(values)
    return result


def select_depth_backend(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    """Rank eligible backends by the protocol, with no preference for their names."""
    eligible: dict[str, dict[str, object]] = {}
    excluded: dict[str, str] = {}
    for name, summary in summaries.items():
        coverage = summary.get("minimum_frame_coverage")
        if (
            not isinstance(coverage, (float, int))
            or coverage < 0.99
            or summary.get("abs_rel") is None
        ):
            excluded[name] = "coverage below 0.99 or raw metric accuracy unavailable"
        else:
            eligible[name] = summary
    remaining = sorted(eligible)
    deciding_metric: str | None = None
    for metric in ("abs_rel", "near_abs_rel", "edge_gradient_mae_m", "rmse_m"):
        if len(remaining) <= 1:
            break
        values = {name: eligible[name].get(metric) for name in remaining}
        if any(value is None for value in values.values()):
            continue
        numeric = {
            name: float(value)
            for name, value in values.items()
            if isinstance(value, (int, float))
        }
        if len(numeric) != len(remaining) or not all(
            math.isfinite(value) for value in numeric.values()
        ):
            raise ValueError(f"invalid selection metric: {metric}")
        best = min(numeric.values())
        remaining = [name for name in remaining if numeric[name] <= best + 1e-8]
        deciding_metric = metric
    return {
        "status": (
            "selected"
            if len(remaining) == 1
            else "tie_requires_review" if remaining else "no_eligible_backend"
        ),
        "selected_backend": remaining[0] if len(remaining) == 1 else None,
        "tied_backends": remaining if len(remaining) > 1 else [],
        "deciding_metric": deciding_metric,
        "excluded": excluded,
        "rule": SELECTION_RULE,
    }


def plane_metrics(
    predicted: NDArray[np.floating[Any]],
    ground_truth: NDArray[np.floating[Any]],
    intrinsics: NDArray[np.floating[Any]],
    floor_mask: NDArray[np.bool_] | None,
) -> dict[str, object]:
    """Compare fitted floor planes only when an explicit evaluator mask is present."""
    if floor_mask is None:
        return {"status": "unavailable", "reason": "no frozen floor mask"}
    target, prediction = np.asarray(ground_truth), np.asarray(predicted)
    if floor_mask.shape != target.shape or prediction.shape != target.shape:
        raise ValueError("floor mask/depth shapes differ")
    valid = (
        floor_mask
        & np.isfinite(target)
        & (target > 0)
        & (target < 5)
        & np.isfinite(prediction)
        & (prediction > 0)
    )
    rows, columns = np.nonzero(valid)
    if len(rows) < 50:
        return {"status": "unavailable", "reason": "fewer than 50 valid floor pixels"}
    rays = np.linalg.inv(validate_intrinsics(intrinsics)) @ np.stack(
        (columns, rows, np.ones(len(rows)))
    )
    planes = []
    for depth in (target, prediction):
        points = (rays * depth[rows, columns]).T
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
        normal = vectors[-1]
        planes.append((normal, float(normal @ center)))
    gt_normal, gt_offset = planes[0]
    normal, offset = planes[1]
    sign = 1 if float(normal @ gt_normal) >= 0 else -1
    return {
        "status": "computed",
        "pixels": len(rows),
        "normal_error_degrees": float(
            np.degrees(np.arccos(np.clip(abs(normal @ gt_normal), 0, 1)))
        ),
        "offset_error_m": abs(sign * offset - gt_offset),
    }


def temporal_depth_consistency(
    source_depth: NDArray[np.floating[Any]],
    target_depth: NDArray[np.floating[Any]],
    source_gt: NDArray[np.floating[Any]],
    target_gt: NDArray[np.floating[Any]],
    source: CalibrationFrame,
    target: CalibrationFrame,
) -> dict[str, object]:
    """Measure reprojection residual on GT-visible correspondences; never fit scale."""
    if (
        source.camera_to_world is None
        or target.camera_to_world is None
        or source.scene_id != target.scene_id
    ):
        return {
            "status": "unavailable",
            "reason": "matching scene and two camera poses required",
        }
    ys, xs = np.mgrid[0 : source_depth.shape[0] : 8, 0 : source_depth.shape[1] : 8]
    rows, columns = ys.ravel(), xs.ravel()
    source_values, source_truth = source_depth[rows, columns], source_gt[rows, columns]
    valid = (
        np.isfinite(source_values)
        & (source_values > 0)
        & np.isfinite(source_truth)
        & (source_truth > 0)
        & (source_truth < 5)
    )
    rows, columns = rows[valid], columns[valid]
    rays = np.linalg.inv(validate_intrinsics(np.asarray(source.intrinsics))) @ np.stack(
        (columns, rows, np.ones(len(rows)))
    )
    transform = np.linalg.inv(np.asarray(target.camera_to_world)) @ np.asarray(
        source.camera_to_world
    )
    residuals: list[float] = []
    projected = []
    for values in (source_gt[rows, columns], source_depth[rows, columns]):
        points = transform[:3, :3] @ (rays * values) + transform[:3, 3, None]
        pixels = validate_intrinsics(np.asarray(target.intrinsics)) @ points
        projected.append((points[2], pixels[:2] / np.maximum(pixels[2], 1e-8)))
    height, width = target_depth.shape
    for index in range(len(rows)):
        gt_z, gt_uv = projected[0][0][index], projected[0][1][:, index]
        pred_z, pred_uv = projected[1][0][index], projected[1][1][:, index]
        if (
            gt_z <= 0
            or pred_z <= 0
            or not np.isfinite(gt_uv).all()
            or not np.isfinite(pred_uv).all()
        ):
            continue
        gx, gy = np.rint(gt_uv).astype(int)
        px, py = np.rint(pred_uv).astype(int)
        if not (
            0 <= gx < width
            and 0 <= gy < height
            and 0 <= px < width
            and 0 <= py < height
        ):
            continue
        truth, prediction = float(target_gt[gy, gx]), float(target_depth[py, px])
        if not (
            math.isfinite(truth)
            and 0 < truth < 5
            and math.isfinite(prediction)
            and prediction > 0
        ):
            continue
        if abs(truth - gt_z) <= 0.05:
            residuals.append(abs(prediction - pred_z))
    return {
        "status": "computed" if residuals else "unavailable",
        "visible_correspondences": len(residuals),
        "mae_m": float(np.mean(residuals)) if residuals else None,
        "sampling_stride": 8,
        "gt_visibility_tolerance_m": 0.05,
    }
