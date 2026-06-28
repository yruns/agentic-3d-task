"""Scoring contracts for SceneFunc3D mask evaluation."""

from __future__ import annotations

import json
from collections.abc import Set
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, TypeAlias
from zipfile import BadZipFile

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StringConstraints

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.final_mask_artifacts import FinalMaskArtifactDocument
from codex_agent.scenefunc3d.sample import load_sample, scene_dir_for

from .metrics import MaskMetrics, compute_mask_metrics

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
PointId: TypeAlias = Annotated[int, Field(ge=0, strict=True)]
_PREDICTION_POINT_ID_KEYS = ("point_indices", "point_ids")
_SCENEFUNC3D_TASK_NAME = "scenefunc3d_mask_generation"


class _ScoringAnnotation(BaseModel):
    """Hidden annotation record used only by the scorer."""

    model_config = ConfigDict(extra="ignore")

    annot_id: NonEmptyString
    indices: tuple[PointId, ...] = Field(min_length=1)


class _ScoringAnnotationsFile(BaseModel):
    """SceneFunc3D annotations file with hidden point ids exposed to scoring."""

    model_config = ConfigDict(extra="ignore")

    visit_id: NonEmptyString
    annotations: tuple[_ScoringAnnotation, ...]


class _ScoringResultOutcome(BaseModel):
    """Subset of the runner outcome needed by the scorer."""

    model_config = ConfigDict(extra="ignore")

    mask_artifact_path: NonEmptyString


class _ScoringResultFile(BaseModel):
    """Subset of a SceneFunc3D runner result file needed by the scorer."""

    model_config = ConfigDict(extra="ignore")

    task_name: NonEmptyString
    sample_id: NonEmptyString
    outcome: _ScoringResultOutcome


@dataclass(frozen=True)
class SceneFunc3dScore:
    """SceneFunc3D score for one sample."""

    sample_id: str
    metrics: MaskMetrics
    failure_type: str


def score_point_ids(
    *,
    sample_id: str,
    predicted_ids: Set[int],
    gt_ids: Set[int],
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score predicted point ids against ground truth ids for one sample."""
    return SceneFunc3dScore(
        sample_id=sample_id,
        metrics=compute_mask_metrics(predicted_ids=predicted_ids, gt_ids=gt_ids),
        failure_type=failure_type,
    )


def load_gt_point_ids(data_root: Path, sample_id: str) -> frozenset[int]:
    """Load hidden ground-truth annotation point ids for one sample.

    The agent-visible sample loader intentionally hides annotation ``indices``.
    The scorer reloads the annotation file and only uses the sample's validated
    ``annotation_ids`` links to select GT ids.
    """
    root = Path(data_root)
    sample = load_sample(root, sample_id)
    annotations_path = (
        scene_dir_for(root, sample.visit_id) / f"{sample.visit_id}_annotations.json"
    )
    annotations_file = _load_scoring_annotations(
        annotations_path, expected_visit_id=sample.visit_id
    )
    annotations_by_id = _annotations_by_id(annotations_file.annotations)

    point_ids: set[int] = set()
    for annotation_id in sample.annotation_ids:
        annotation = annotations_by_id.get(annotation_id)
        if annotation is None:
            raise SceneFunc3dDataError(
                "annotation id not found while loading scorer GT: "
                f"annotation_id={annotation_id!r}; sample_id={sample_id!r}; "
                f"annotations_path={annotations_path}"
            )
        point_ids.update(annotation.indices)
    if not point_ids:
        raise SceneFunc3dDataError(
            "ground-truth annotation indices are empty: "
            f"sample_id={sample_id!r}; annotations_path={annotations_path}"
        )
    return frozenset(point_ids)


def load_predicted_point_ids(mask_npz_path: Path) -> frozenset[int]:
    """Load predicted scene point ids from a final mask NPZ artifact.

    Lifted ``points_world`` alone are not the same thing as SceneFunc3D
    annotation indices. The scorer therefore requires an explicit
    ``point_indices`` or ``point_ids`` key until nearest-scene-point assignment is
    implemented.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise SceneFunc3dDataError(
            "loading predicted point ids requires numpy: "
            f"mask_npz_path={mask_npz_path}"
        ) from exc

    try:
        with np.load(mask_npz_path) as archive:
            selected_keys = tuple(
                key for key in _PREDICTION_POINT_ID_KEYS if key in archive.files
            )
            if len(selected_keys) > 1:
                raise SceneFunc3dDataError(
                    "mask NPZ must contain only one predicted point-id key: "
                    f"mask_npz_path={mask_npz_path}; keys={selected_keys}"
                )
            if not selected_keys:
                if "points_world" in archive.files:
                    raise SceneFunc3dDataError(
                        "mask NPZ contains points_world but no predicted point ids; "
                        "points_world cannot be scored against annotation indices "
                        "without nearest-scene-point assignment: "
                        f"mask_npz_path={mask_npz_path}"
                    )
                raise SceneFunc3dDataError(
                    "mask NPZ is missing predicted point ids; expected one of "
                    f"{_PREDICTION_POINT_ID_KEYS}: mask_npz_path={mask_npz_path}"
                )
            point_ids_array = np.asarray(archive[selected_keys[0]])
    except SceneFunc3dDataError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise SceneFunc3dDataError(
            "could not load predicted point-id NPZ: "
            f"mask_npz_path={mask_npz_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if point_ids_array.ndim != 1:
        raise SceneFunc3dDataError(
            "predicted point ids must be a 1D array: "
            f"mask_npz_path={mask_npz_path}; ndim={point_ids_array.ndim}"
        )
    if not np.issubdtype(point_ids_array.dtype, np.integer):
        raise SceneFunc3dDataError(
            "predicted point ids must use an integer dtype: "
            f"mask_npz_path={mask_npz_path}; dtype={point_ids_array.dtype}"
        )
    if bool(np.any(point_ids_array < 0)):
        raise SceneFunc3dDataError(
            "predicted point ids must be non-negative: "
            f"mask_npz_path={mask_npz_path}"
        )
    return frozenset(int(point_id) for point_id in point_ids_array.flat)


def score_mask_npz(
    *,
    data_root: Path,
    sample_id: str,
    mask_npz_path: Path,
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score a mask NPZ containing predicted point ids against hidden GT."""
    return score_point_ids(
        sample_id=sample_id,
        predicted_ids=load_predicted_point_ids(mask_npz_path),
        gt_ids=load_gt_point_ids(data_root, sample_id),
        failure_type=failure_type,
    )


def score_mask_artifact(
    *,
    data_root: Path,
    sample_id: str,
    mask_artifact_path: Path,
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score a single-fragment or fused final mask artifact against hidden GT."""
    artifact_document = _load_final_mask_artifact_document(mask_artifact_path)
    return score_mask_npz(
        data_root=data_root,
        sample_id=sample_id,
        mask_npz_path=_resolve_artifact_member_path(
            artifact_path=mask_artifact_path,
            raw_member_path=artifact_document.mask_npz_path,
        ),
        failure_type=failure_type,
    )


def score_result_file(
    *,
    data_root: Path,
    result_path: Path,
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score a SceneFunc3D runner result JSON against hidden GT."""
    result_file = _load_result_file_for_scoring(result_path)
    if result_file.task_name != _SCENEFUNC3D_TASK_NAME:
        raise SceneFunc3dDataError(
            "runner result task_name is not a SceneFunc3D mask task: "
            f"task_name={result_file.task_name!r}; result_path={result_path}"
        )
    return score_mask_artifact(
        data_root=data_root,
        sample_id=result_file.sample_id,
        mask_artifact_path=_resolve_artifact_member_path(
            artifact_path=result_path,
            raw_member_path=result_file.outcome.mask_artifact_path,
        ),
        failure_type=failure_type,
    )


def _load_scoring_annotations(
    path: Path, *, expected_visit_id: str
) -> _ScoringAnnotationsFile:
    if not path.is_file():
        raise SceneFunc3dDataError(f"SceneFunc3D annotations JSON is missing: {path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(f"{path}: invalid JSON: {exc}") from exc
    try:
        annotations_file = _ScoringAnnotationsFile.model_validate(payload)
    except ValidationError as exc:
        raise SceneFunc3dDataError(f"{path}: failed scorer validation: {exc}") from exc
    if annotations_file.visit_id != expected_visit_id:
        raise SceneFunc3dDataError(
            "annotations visit_id does not match sample visit_id: "
            f"annotations_visit_id={annotations_file.visit_id!r}; "
            f"expected_visit_id={expected_visit_id!r}; path={path}"
        )
    return annotations_file


def _annotations_by_id(
    annotations: tuple[_ScoringAnnotation, ...],
) -> dict[str, _ScoringAnnotation]:
    annotations_by_id: dict[str, _ScoringAnnotation] = {}
    for annotation in annotations:
        if annotation.annot_id in annotations_by_id:
            raise SceneFunc3dDataError(
                "duplicate annotation id in scorer annotations: "
                f"annotation_id={annotation.annot_id!r}"
            )
        annotations_by_id[annotation.annot_id] = annotation
    return annotations_by_id


def _load_result_file_for_scoring(result_path: Path) -> _ScoringResultFile:
    if not result_path.is_file():
        raise SceneFunc3dDataError(f"runner result JSON is missing: {result_path}")
    try:
        payload: object = json.loads(result_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"runner result is not valid JSON: {result_path}"
        ) from exc
    try:
        return _ScoringResultFile.model_validate(payload)
    except ValidationError as exc:
        raise SceneFunc3dDataError(
            "runner result failed scorer validation: "
            f"result_path={result_path}; error={exc}"
        ) from exc


def _load_final_mask_artifact_document(
    artifact_path: Path,
) -> FinalMaskArtifactDocument:
    if not artifact_path.is_file():
        raise SceneFunc3dDataError(
            f"final mask artifact JSON is missing: {artifact_path}"
        )
    try:
        payload: object = json.loads(artifact_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"final mask artifact is not valid JSON: {artifact_path}"
        ) from exc
    try:
        return FinalMaskArtifactDocument.model_validate(payload)
    except ValidationError as exc:
        raise SceneFunc3dDataError(
            "final mask artifact failed scorer validation: "
            f"artifact_path={artifact_path}; error={exc}"
        ) from exc


def _resolve_artifact_member_path(*, artifact_path: Path, raw_member_path: str) -> Path:
    member_path = Path(raw_member_path).expanduser()
    if member_path.is_absolute():
        return member_path.resolve()
    return (artifact_path.parent / member_path).resolve()


__all__ = [
    "SceneFunc3dScore",
    "load_gt_point_ids",
    "load_predicted_point_ids",
    "score_mask_artifact",
    "score_mask_npz",
    "score_point_ids",
    "score_result_file",
]
