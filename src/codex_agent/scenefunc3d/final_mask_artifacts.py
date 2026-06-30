"""Validation for final SceneFunc3D 3D mask artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias, cast
from zipfile import BadZipFile

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.types import StringConstraints

from ..errors import CodexResponseError, SceneFunc3dDataError
from .task import ApprovalAction, validate_fragment_approval_actions

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.backends.lift_3d import FloatArray, IntArray

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class FinalMaskLiftGeometry(BaseModel):
    """3D geometry summary the agent reviewed before accepting a fragment."""

    model_config = ConfigDict(extra="forbid")

    bbox_min_xyz: tuple[float, float, float]
    bbox_max_xyz: tuple[float, float, float]
    bbox_extent_xyz: tuple[float, float, float]
    max_extent_meters: float

    @model_validator(mode="after")
    def validate_geometry_consistency(self) -> FinalMaskLiftGeometry:
        """Validate finite bbox geometry and derived extent fields."""
        _validate_xyz_values("bbox_min_xyz", self.bbox_min_xyz)
        _validate_xyz_values("bbox_max_xyz", self.bbox_max_xyz)
        _validate_xyz_values("bbox_extent_xyz", self.bbox_extent_xyz)
        expected_extent = _extent_xyz_tuple(self.bbox_min_xyz, self.bbox_max_xyz)
        if any(extent_value < 0.0 for extent_value in expected_extent):
            raise ValueError(
                "lift_geometry.bbox_extent_xyz must be non-negative: "
                f"bbox_min_xyz={self.bbox_min_xyz}; "
                f"bbox_max_xyz={self.bbox_max_xyz}; "
                f"expected={expected_extent}"
            )
        if not _xyz_values_close(self.bbox_extent_xyz, expected_extent):
            raise ValueError(
                "lift_geometry.bbox_extent_xyz must match bbox_max_xyz - "
                "bbox_min_xyz: "
                f"bbox_extent_xyz={self.bbox_extent_xyz}; "
                f"expected={expected_extent}"
            )
        expected_max_extent = max(expected_extent)
        if (
            not math.isfinite(self.max_extent_meters)
            or self.max_extent_meters < 0.0
            or not math.isclose(
                self.max_extent_meters,
                expected_max_extent,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "lift_geometry.max_extent_meters must match max(bbox_extent_xyz): "
                f"max_extent_meters={self.max_extent_meters}; "
                f"expected={expected_max_extent}"
            )
        return self


class FinalMaskAcceptedFragment(BaseModel):
    """One accepted 3D fragment recorded in the final mask artifact."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: NonEmptyString
    frame_id: NonEmptyString
    point_count: int = Field(gt=0, strict=True)
    lift_geometry: FinalMaskLiftGeometry
    approval_actions: tuple[ApprovalAction, ...] = Field(min_length=1)
    review_artifacts: FinalMaskReviewArtifacts


class FinalMaskReviewArtifacts(BaseModel):
    """Review artifacts the agent inspected before accepting a fragment."""

    model_config = ConfigDict(extra="forbid")

    molmo_raw_text_path: NonEmptyString
    molmo_overlay_path: NonEmptyString
    sam_contact_sheet_path: NonEmptyString
    sam_candidate_overlay_path: NonEmptyString
    lift_overlay_path: NonEmptyString


class FinalMaskMultiViewAction(str, Enum):
    """Agent decision after reviewing the first lifted 3D seed."""

    STOP = "stop"
    EXPAND = "expand"


class FinalMaskMultiViewDecision(BaseModel):
    """Recorded decision about whether first-lift evidence needed more views."""

    model_config = ConfigDict(extra="forbid")

    seed_fragment_id: NonEmptyString
    action: FinalMaskMultiViewAction
    reason: NonEmptyString
    suggested_frame_ids: tuple[NonEmptyString, ...] = ()
    rejected_suggested_frame_ids: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_suggestion_accounting(self) -> FinalMaskMultiViewDecision:
        """Validate accepted and rejected suggested frame sets are disjoint."""
        overlapping_frame_ids = _overlapping_frame_ids(
            self.suggested_frame_ids,
            self.rejected_suggested_frame_ids,
        )
        if overlapping_frame_ids:
            raise ValueError(
                "multi_view_decision.suggested_frame_ids and "
                "rejected_suggested_frame_ids must not overlap: "
                f"overlapping_frame_ids={overlapping_frame_ids}"
            )
        return self


class FinalMaskArtifactDocument(BaseModel):
    """Strict JSON contract for one fused SceneFunc3D mask artifact."""

    model_config = ConfigDict(extra="forbid")

    accepted_frame_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    accepted_fragments: tuple[FinalMaskAcceptedFragment, ...] = Field(min_length=1)
    multi_view_decision: FinalMaskMultiViewDecision
    mask_npz_path: NonEmptyString
    mask_ply_path: NonEmptyString

    @model_validator(mode="after")
    def validate_multi_view_decision_contract(self) -> FinalMaskArtifactDocument:
        """Validate that the recorded multi-view decision matches final fragments."""
        fragment_ids = tuple(
            fragment.fragment_id for fragment in self.accepted_fragments
        )
        if self.multi_view_decision.seed_fragment_id != fragment_ids[0]:
            raise ValueError(
                "multi_view_decision.seed_fragment_id must match the first accepted "
                "fragment_id: "
                f"seed_fragment_id={self.multi_view_decision.seed_fragment_id}; "
                f"first_fragment_id={fragment_ids[0]}"
            )

        fragment_frame_ids = _accepted_frame_ids_from_fragments(self.accepted_fragments)
        if self.multi_view_decision.action == FinalMaskMultiViewAction.EXPAND:
            _validate_expand_multi_view_decision(
                decision=self.multi_view_decision,
                accepted_frame_ids=fragment_frame_ids,
            )
        elif self.multi_view_decision.suggested_frame_ids:
            raise ValueError(
                "multi_view_decision.suggested_frame_ids must be empty when "
                "action='stop': "
                f"suggested_frame_ids={self.multi_view_decision.suggested_frame_ids}"
            )
        elif len(fragment_frame_ids) > 1:
            raise ValueError(
                "multi_view_decision.action must be 'expand' when the final artifact "
                "contains accepted fragments from multiple frames: "
                f"accepted_frame_ids={fragment_frame_ids}"
            )
        return self


@dataclass(frozen=True)
class ValidatedFinalMaskArtifact:
    """Validated final mask artifact summary."""

    artifact_path: Path
    mask_npz_path: Path
    mask_ply_path: Path
    accepted_frame_ids: tuple[str, ...]
    accepted_fragment_ids: tuple[str, ...]
    multi_view_decision: FinalMaskMultiViewDecision
    point_count: int


@dataclass(frozen=True)
class _LiftOverlaySummary:
    frame_id: str
    candidate_id: str


def validate_final_mask_artifact(
    *,
    artifact_path: Path,
    mask_npz_path: Path,
    mask_ply_path: Path,
    selected_frame_ids: tuple[str, ...],
    accepted_fragment_ids: tuple[str, ...],
    allow_copied_artifact_members: bool = False,
) -> ValidatedFinalMaskArtifact:
    """Validate final artifact JSON, NPZ points, and PLY vertex consistency."""
    artifact_document = _load_final_mask_artifact_document(artifact_path)
    _require_matching_artifact_path(
        artifact_document.mask_npz_path,
        artifact_path=artifact_path,
        expected_path=mask_npz_path,
        field_name="mask_npz_path",
        allow_copied_artifact_members=allow_copied_artifact_members,
    )
    _require_matching_artifact_path(
        artifact_document.mask_ply_path,
        artifact_path=artifact_path,
        expected_path=mask_ply_path,
        field_name="mask_ply_path",
        allow_copied_artifact_members=allow_copied_artifact_members,
    )
    artifact_fragment_ids = tuple(
        fragment.fragment_id for fragment in artifact_document.accepted_fragments
    )
    if artifact_fragment_ids != accepted_fragment_ids:
        raise CodexResponseError(
            "accepted_fragment_ids must match final artifact accepted_fragments: "
            f"response={accepted_fragment_ids}; artifact={artifact_fragment_ids}"
        )
    fragment_frame_ids = _accepted_frame_ids_from_fragments(
        artifact_document.accepted_fragments
    )
    if artifact_document.accepted_frame_ids != fragment_frame_ids:
        raise CodexResponseError(
            "accepted_frame_ids must match final artifact accepted_fragments frame_id "
            f"provenance: accepted_frame_ids={artifact_document.accepted_frame_ids}; "
            f"fragment_frame_ids={fragment_frame_ids}"
        )
    if selected_frame_ids != artifact_document.accepted_frame_ids:
        raise CodexResponseError(
            "selected_frame_ids must match final artifact accepted_frame_ids: "
            f"response={selected_frame_ids}; "
            f"artifact={artifact_document.accepted_frame_ids}"
        )
    artifact_root = _infer_final_artifact_root(artifact_path)
    for fragment in artifact_document.accepted_fragments:
        try:
            validate_fragment_approval_actions(
                fragment.fragment_id, fragment.approval_actions
            )
        except SceneFunc3dDataError as exc:
            raise CodexResponseError(
                "accepted fragment approval_actions failed SceneFunc3D approval "
                f"gate replay: fragment_id={fragment.fragment_id}; "
                f"frame_id={fragment.frame_id}; error={exc}"
            ) from exc
        _validate_fragment_review_artifacts(
            fragment,
            artifact_root=artifact_root,
            allow_copied_artifact_members=allow_copied_artifact_members,
        )
        _validate_fragment_lift_geometry(
            fragment,
            artifact_root=artifact_root,
            final_mask_npz_path=mask_npz_path,
            accepted_fragment_count=len(artifact_document.accepted_fragments),
            allow_copied_artifact_members=allow_copied_artifact_members,
        )

    npz_point_count = validate_points_world_npz(mask_npz_path)
    load_point_indices_npz(mask_npz_path, expected_count=npz_point_count)
    ply_vertex_count = validate_ascii_points_ply(mask_ply_path)
    if ply_vertex_count != npz_point_count:
        raise CodexResponseError(
            "mask_ply_path vertex count must match mask_npz_path point count: "
            f"mask_ply_path={mask_ply_path}; vertices={ply_vertex_count}; "
            f"mask_npz_path={mask_npz_path}; points={npz_point_count}"
        )
    for fragment in artifact_document.accepted_fragments:
        if fragment.point_count > npz_point_count:
            raise CodexResponseError(
                "accepted fragment point_count cannot exceed fused mask point count: "
                f"fragment_id={fragment.fragment_id}; "
                f"fragment_points={fragment.point_count}; "
                f"fused_points={npz_point_count}"
            )
    fragment_point_count_sum = sum(
        fragment.point_count for fragment in artifact_document.accepted_fragments
    )
    if fragment_point_count_sum != npz_point_count:
        raise CodexResponseError(
            "accepted fragment point_count sum must match fused mask point count: "
            f"fragment_points={fragment_point_count_sum}; "
            f"fused_points={npz_point_count}"
        )

    return ValidatedFinalMaskArtifact(
        artifact_path=artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        accepted_frame_ids=artifact_document.accepted_frame_ids,
        accepted_fragment_ids=accepted_fragment_ids,
        multi_view_decision=artifact_document.multi_view_decision,
        point_count=npz_point_count,
    )


def validate_points_world_npz(mask_npz_path: Path) -> int:
    """Validate a lifted/fused mask NPZ and return its world-point count."""
    points_world = load_points_world_npz(mask_npz_path)
    return int(points_world.shape[0])


def load_points_world_npz(mask_npz_path: Path) -> FloatArray:
    """Load and validate a lifted/fused mask NPZ world-point array."""
    try:
        import numpy as np
    except ImportError as exc:
        raise CodexResponseError(
            "mask_npz_path validation requires numpy: " f"mask_npz_path={mask_npz_path}"
        ) from exc

    try:
        with np.load(mask_npz_path) as archive:
            if "points_world" not in archive.files:
                raise CodexResponseError(
                    "mask_npz_path is missing required key 'points_world': "
                    f"mask_npz_path={mask_npz_path}"
                )
            points_world = cast(
                "FloatArray",
                np.asarray(archive["points_world"], dtype=np.float64),
            )
    except CodexResponseError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise CodexResponseError(
            "could not read mask_npz_path as a points_world NPZ: "
            f"mask_npz_path={mask_npz_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise CodexResponseError(
            "mask_npz_path points_world must have shape (N, 3): "
            f"mask_npz_path={mask_npz_path}; shape={points_world.shape}"
        )
    if points_world.shape[0] <= 0:
        raise CodexResponseError(
            "mask_npz_path points_world must contain at least one point: "
            f"mask_npz_path={mask_npz_path}"
        )
    if not bool(np.all(np.isfinite(points_world))):
        raise CodexResponseError(
            "mask_npz_path points_world must contain only finite values: "
            f"mask_npz_path={mask_npz_path}"
        )
    return points_world


def load_point_indices_npz(mask_npz_path: Path, *, expected_count: int) -> IntArray:
    """Load and validate point ids aligned with a lifted/fused mask NPZ."""
    try:
        import numpy as np
    except ImportError as exc:
        raise CodexResponseError(
            "mask_npz_path point_indices validation requires numpy: "
            f"mask_npz_path={mask_npz_path}"
        ) from exc

    try:
        with np.load(mask_npz_path) as archive:
            if "point_indices" not in archive.files:
                raise CodexResponseError(
                    "mask_npz_path is missing required key 'point_indices': "
                    f"mask_npz_path={mask_npz_path}"
                )
            point_indices = np.asarray(archive["point_indices"])
    except CodexResponseError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise CodexResponseError(
            "could not read mask_npz_path point_indices: "
            f"mask_npz_path={mask_npz_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if point_indices.ndim != 1:
        raise CodexResponseError(
            "mask_npz_path point_indices must be a 1D array: "
            f"mask_npz_path={mask_npz_path}; ndim={point_indices.ndim}"
        )
    if point_indices.shape[0] != expected_count:
        raise CodexResponseError(
            "mask_npz_path point_indices count must match points_world count: "
            f"mask_npz_path={mask_npz_path}; indices={point_indices.shape[0]}; "
            f"points={expected_count}"
        )
    if not np.issubdtype(point_indices.dtype, np.integer):
        raise CodexResponseError(
            "mask_npz_path point_indices must contain integer scene point ids: "
            f"mask_npz_path={mask_npz_path}; dtype={point_indices.dtype}"
        )
    if bool(np.any(point_indices < 0)):
        raise CodexResponseError(
            "mask_npz_path point_indices must contain non-negative scene point ids: "
            f"mask_npz_path={mask_npz_path}"
        )
    return cast("IntArray", point_indices.astype(np.int64, copy=False))


def validate_ascii_points_ply(mask_ply_path: Path) -> int:
    """Validate an ASCII point-cloud PLY and return its vertex count."""
    try:
        lines = mask_ply_path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise CodexResponseError(
            "mask_ply_path must be ASCII PLY: " f"mask_ply_path={mask_ply_path}"
        ) from exc
    except OSError as exc:
        raise CodexResponseError(
            "could not read mask_ply_path: "
            f"mask_ply_path={mask_ply_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if len(lines) < 4 or lines[0].strip() != "ply":
        raise CodexResponseError(
            "mask_ply_path must start with a PLY header: "
            f"mask_ply_path={mask_ply_path}"
        )
    if not any(line.strip() == "format ascii 1.0" for line in lines[:4]):
        raise CodexResponseError(
            "mask_ply_path must declare 'format ascii 1.0': "
            f"mask_ply_path={mask_ply_path}"
        )
    vertex_count = _parse_ply_vertex_count(lines, mask_ply_path)
    end_header_index = _find_ply_end_header(lines, mask_ply_path)
    vertex_lines = lines[end_header_index + 1 : end_header_index + 1 + vertex_count]
    if len(vertex_lines) != vertex_count:
        raise CodexResponseError(
            "mask_ply_path does not contain the declared number of vertex rows: "
            f"mask_ply_path={mask_ply_path}; declared={vertex_count}; "
            f"actual={len(vertex_lines)}"
        )
    for row_index, vertex_line in enumerate(vertex_lines):
        _validate_ply_vertex_line(
            vertex_line, mask_ply_path=mask_ply_path, row_index=row_index
        )
    return vertex_count


def _load_final_mask_artifact_document(
    artifact_path: Path,
) -> FinalMaskArtifactDocument:
    try:
        payload: object = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CodexResponseError(
            f"could not read mask_artifact_path: {artifact_path}"
        ) from exc
    except JSONDecodeError as exc:
        raise CodexResponseError(
            f"mask_artifact_path is not valid JSON: {artifact_path}"
        ) from exc
    try:
        return FinalMaskArtifactDocument.model_validate(payload)
    except ValidationError as exc:
        raise CodexResponseError(
            "mask_artifact_path does not match final mask artifact schema: "
            f"mask_artifact_path={artifact_path}; error={exc}"
        ) from exc


def _require_matching_artifact_path(
    raw_path: str,
    *,
    artifact_path: Path,
    expected_path: Path,
    field_name: str,
    allow_copied_artifact_members: bool,
) -> None:
    raw_actual_path = Path(raw_path).expanduser()
    actual_path = raw_actual_path.resolve()
    resolved_expected_path = expected_path.expanduser().resolve()
    if actual_path != resolved_expected_path:
        copied_actual_path = _resolve_copied_final_artifact_member_path(
            artifact_path=artifact_path,
            stale_member_path=raw_actual_path,
            enabled=allow_copied_artifact_members,
        )
        if copied_actual_path == resolved_expected_path:
            return
        raise CodexResponseError(
            f"mask_artifact_path {field_name} must match final response: "
            f"artifact={actual_path}; response={resolved_expected_path}"
        )


def _resolve_copied_final_artifact_member_path(
    *,
    artifact_path: Path,
    stale_member_path: Path,
    enabled: bool,
) -> Path | None:
    if (
        not enabled
        or not stale_member_path.is_absolute()
        or stale_member_path.is_file()
    ):
        return None
    candidates = _existing_copied_final_artifact_member_candidates(
        search_root=artifact_path.parent,
        stale_member_path=stale_member_path,
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise CodexResponseError(
            "copied final artifact member path is ambiguous: "
            f"artifact_path={artifact_path}; "
            f"stale_member_path={stale_member_path}; "
            f"candidates={tuple(str(path) for path in candidates)}"
        )
    return None


def _existing_copied_final_artifact_member_candidates(
    *,
    search_root: Path,
    stale_member_path: Path,
) -> tuple[Path, ...]:
    parent_name = stale_member_path.parent.name
    candidate_paths = [
        search_root / stale_member_path.name,
        search_root / parent_name / stale_member_path.name,
    ]
    return _deduplicate_existing_paths(candidate_paths)


def _deduplicate_existing_paths(candidate_paths: list[Path]) -> tuple[Path, ...]:
    resolved_candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        resolved_path = candidate_path.resolve()
        if resolved_path in seen_paths:
            continue
        resolved_candidates.append(resolved_path)
        seen_paths.add(resolved_path)
    return tuple(sorted(resolved_candidates))


def _accepted_frame_ids_from_fragments(
    accepted_fragments: tuple[FinalMaskAcceptedFragment, ...],
) -> tuple[str, ...]:
    seen_frame_ids: set[str] = set()
    accepted_frame_ids: list[str] = []
    for fragment in accepted_fragments:
        if fragment.frame_id not in seen_frame_ids:
            accepted_frame_ids.append(fragment.frame_id)
            seen_frame_ids.add(fragment.frame_id)
    return tuple(accepted_frame_ids)


def _validate_xyz_values(field_name: str, values: tuple[float, float, float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"lift_geometry.{field_name} must contain finite values")


def _extent_xyz_tuple(
    bbox_min_xyz: tuple[float, float, float],
    bbox_max_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        bbox_max_xyz[0] - bbox_min_xyz[0],
        bbox_max_xyz[1] - bbox_min_xyz[1],
        bbox_max_xyz[2] - bbox_min_xyz[2],
    )


def _xyz_values_close(
    actual_values: tuple[float, float, float],
    expected_values: tuple[float, float, float],
) -> bool:
    return all(
        math.isclose(
            actual_value,
            expected_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for actual_value, expected_value in zip(
            actual_values, expected_values, strict=True
        )
    )


def _validate_fragment_lift_geometry(
    fragment: FinalMaskAcceptedFragment,
    *,
    artifact_root: Path,
    final_mask_npz_path: Path,
    accepted_fragment_count: int,
    allow_copied_artifact_members: bool,
) -> None:
    source_npz_path = _fragment_geometry_source_npz_path(
        fragment,
        artifact_root=artifact_root,
        final_mask_npz_path=final_mask_npz_path,
        accepted_fragment_count=accepted_fragment_count,
        allow_copied_artifact_members=allow_copied_artifact_members,
    )
    if source_npz_path is None:
        return
    expected_geometry = _mask_lift_geometry_from_npz(source_npz_path)
    if not _lift_geometry_matches(fragment.lift_geometry, expected_geometry):
        raise CodexResponseError(
            "accepted fragment lift_geometry must match the validated fragment "
            "mask geometry: "
            f"fragment_id={fragment.fragment_id}; "
            f"source_npz_path={source_npz_path}; "
            f"lift_geometry={fragment.lift_geometry.model_dump(mode='json')}; "
            f"expected={expected_geometry.model_dump(mode='json')}"
        )


def _fragment_geometry_source_npz_path(
    fragment: FinalMaskAcceptedFragment,
    *,
    artifact_root: Path,
    final_mask_npz_path: Path,
    accepted_fragment_count: int,
    allow_copied_artifact_members: bool,
) -> Path | None:
    standard_candidate_id = _candidate_id_from_standard_fragment(fragment)
    if standard_candidate_id is not None:
        standard_npz_path = (
            artifact_root / "fragments" / fragment.fragment_id / "mask_data.npz"
        )
        if standard_npz_path.is_file() or not allow_copied_artifact_members:
            return standard_npz_path
        if accepted_fragment_count == 1:
            return final_mask_npz_path
        return None
    if accepted_fragment_count == 1:
        return final_mask_npz_path
    return None


def _mask_lift_geometry_from_npz(mask_npz_path: Path) -> FinalMaskLiftGeometry:
    points_world = load_points_world_npz(mask_npz_path)
    bbox_min_xyz = _xyz_tuple(cast("FloatArray", points_world.min(axis=0)))
    bbox_max_xyz = _xyz_tuple(cast("FloatArray", points_world.max(axis=0)))
    bbox_extent_xyz = _extent_xyz_tuple(bbox_min_xyz, bbox_max_xyz)
    return FinalMaskLiftGeometry(
        bbox_min_xyz=bbox_min_xyz,
        bbox_max_xyz=bbox_max_xyz,
        bbox_extent_xyz=bbox_extent_xyz,
        max_extent_meters=max(bbox_extent_xyz),
    )


def _xyz_tuple(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _lift_geometry_matches(
    actual_geometry: FinalMaskLiftGeometry,
    expected_geometry: FinalMaskLiftGeometry,
) -> bool:
    return (
        _xyz_values_close(actual_geometry.bbox_min_xyz, expected_geometry.bbox_min_xyz)
        and _xyz_values_close(
            actual_geometry.bbox_max_xyz, expected_geometry.bbox_max_xyz
        )
        and _xyz_values_close(
            actual_geometry.bbox_extent_xyz,
            expected_geometry.bbox_extent_xyz,
        )
        and math.isclose(
            actual_geometry.max_extent_meters,
            expected_geometry.max_extent_meters,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )


def _infer_final_artifact_root(artifact_path: Path) -> Path:
    resolved_artifact_path = artifact_path.expanduser().resolve()
    artifact_parent = resolved_artifact_path.parent
    if (
        resolved_artifact_path.name == "mask_artifact.json"
        and artifact_parent.name == "fused"
    ):
        return artifact_parent.parent
    return artifact_parent


def _validate_expand_multi_view_decision(
    *,
    decision: FinalMaskMultiViewDecision,
    accepted_frame_ids: tuple[str, ...],
) -> None:
    if len(accepted_frame_ids) <= 1:
        raise ValueError(
            "multi_view_decision.action='expand' requires accepted fragments from "
            "multiple frames: "
            f"accepted_frame_ids={accepted_frame_ids}"
        )
    if not decision.suggested_frame_ids:
        raise ValueError(
            "multi_view_decision.suggested_frame_ids must be non-empty when "
            "action='expand'"
        )
    accepted_followup_frame_ids = set(accepted_frame_ids[1:])
    suggested_frame_ids = set(decision.suggested_frame_ids)
    missing_accepted_followup_frame_ids = tuple(
        sorted(accepted_followup_frame_ids - suggested_frame_ids)
    )
    if missing_accepted_followup_frame_ids:
        raise ValueError(
            "multi_view_decision.suggested_frame_ids must include every accepted "
            "follow-up frame: "
            f"suggested_frame_ids={decision.suggested_frame_ids}; "
            f"accepted_followup_frame_ids={tuple(sorted(accepted_followup_frame_ids))}; "
            f"missing_accepted_followup_frame_ids="
            f"{missing_accepted_followup_frame_ids}"
        )


def _overlapping_frame_ids(
    first_frame_ids: tuple[str, ...],
    second_frame_ids: tuple[str, ...],
) -> tuple[str, ...]:
    second_frame_id_set = set(second_frame_ids)
    return tuple(
        frame_id for frame_id in first_frame_ids if frame_id in second_frame_id_set
    )


def _validate_fragment_review_artifacts(
    fragment: FinalMaskAcceptedFragment,
    *,
    artifact_root: Path,
    allow_copied_artifact_members: bool,
) -> None:
    standard_candidate_id = _candidate_id_from_standard_fragment(fragment)
    for field_name, raw_path in _review_artifact_path_items(fragment.review_artifacts):
        artifact_path = Path(raw_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise CodexResponseError(
                "accepted fragment review_artifacts path must exist: "
                f"fragment_id={fragment.fragment_id}; "
                f"frame_id={fragment.frame_id}; "
                f"field={field_name}; path={artifact_path}"
            )
        if field_name == "lift_overlay_path":
            _validate_lift_overlay_provenance(fragment, artifact_path)
        if standard_candidate_id is not None:
            _validate_standard_review_artifact_provenance(
                fragment,
                artifact_path,
                field_name=field_name,
                candidate_id=standard_candidate_id,
                artifact_root=artifact_root,
            )
            if field_name == "lift_overlay_path":
                _validate_standard_lift_overlay_summary(
                    fragment,
                    artifact_path,
                    candidate_id=standard_candidate_id,
                )
                _validate_standard_fragment_mask_artifacts(
                    fragment,
                    artifact_root=artifact_root,
                    allow_copied_artifact_members=allow_copied_artifact_members,
                )


def _candidate_id_from_standard_fragment(
    fragment: FinalMaskAcceptedFragment,
) -> str | None:
    expected_prefix = f"{fragment.frame_id}_"
    if not fragment.fragment_id.startswith(expected_prefix):
        return None
    candidate_id = fragment.fragment_id[len(expected_prefix) :]
    if candidate_id == "":
        return None
    return candidate_id


def _validate_standard_review_artifact_provenance(
    fragment: FinalMaskAcceptedFragment,
    artifact_path: Path,
    *,
    field_name: str,
    candidate_id: str,
    artifact_root: Path,
) -> None:
    expected_path = _expected_standard_review_artifact_path(
        artifact_root,
        fragment,
        field_name=field_name,
        candidate_id=candidate_id,
    )
    if expected_path is None:
        return
    resolved_expected_path = expected_path.expanduser().resolve()
    if artifact_path != resolved_expected_path:
        raise CodexResponseError(
            "accepted fragment review_artifacts path must match the canonical "
            "run-local tool artifact path: "
            f"fragment_id={fragment.fragment_id}; "
            f"frame_id={fragment.frame_id}; "
            f"field={field_name}; expected_path={resolved_expected_path}; "
            f"path={artifact_path}"
        )


def _expected_standard_review_artifact_path(
    artifact_root: Path,
    fragment: FinalMaskAcceptedFragment,
    *,
    field_name: str,
    candidate_id: str,
) -> Path | None:
    if field_name == "molmo_raw_text_path":
        return artifact_root / "molmo" / f"{fragment.frame_id}_raw.txt"
    if field_name == "molmo_overlay_path":
        return artifact_root / "molmo" / f"{fragment.frame_id}_points.jpg"
    if field_name == "sam_contact_sheet_path":
        return artifact_root / "sam" / fragment.frame_id / "contact_sheet.jpg"
    if field_name == "sam_candidate_overlay_path":
        return artifact_root / "sam" / fragment.frame_id / f"{candidate_id}_overlay.jpg"
    if field_name == "lift_overlay_path":
        return artifact_root / "fragments" / fragment.fragment_id / "lift_overlay.txt"
    return None


def _validate_lift_overlay_provenance(
    fragment: FinalMaskAcceptedFragment,
    lift_overlay_path: Path,
) -> None:
    expected_lift_overlay_name = "lift_overlay.txt"
    if lift_overlay_path.name != expected_lift_overlay_name:
        raise CodexResponseError(
            "accepted fragment review_artifacts lift_overlay_path must use the "
            "standard lift overlay filename: "
            f"expected_name={expected_lift_overlay_name!r}; "
            f"fragment_id={fragment.fragment_id}; "
            f"frame_id={fragment.frame_id}; "
            f"path={lift_overlay_path}"
        )
    if lift_overlay_path.parent.name != fragment.fragment_id:
        raise CodexResponseError(
            "accepted fragment review_artifacts lift_overlay_path must come from "
            "the accepted fragment directory: "
            f"fragment_id={fragment.fragment_id}; "
            f"frame_id={fragment.frame_id}; "
            f"path={lift_overlay_path}"
        )


def _validate_standard_lift_overlay_summary(
    fragment: FinalMaskAcceptedFragment,
    lift_overlay_path: Path,
    *,
    candidate_id: str,
) -> None:
    summary = _read_lift_overlay_summary(lift_overlay_path)
    if summary.frame_id != fragment.frame_id:
        raise CodexResponseError(
            "accepted fragment review_artifacts lift_overlay_path frame_id must "
            "match the accepted fragment frame_id: "
            f"fragment_id={fragment.fragment_id}; "
            f"frame_id={fragment.frame_id!r}; "
            f"lift_frame_id={summary.frame_id!r}; "
            f"path={lift_overlay_path}"
        )
    if summary.candidate_id != candidate_id:
        raise CodexResponseError(
            "accepted fragment review_artifacts lift_overlay_path candidate_id must "
            "match the accepted fragment candidate_id: "
            f"fragment_id={fragment.fragment_id}; "
            f"candidate_id={candidate_id!r}; "
            f"lift_candidate_id={summary.candidate_id!r}; "
            f"path={lift_overlay_path}"
        )


def _validate_standard_fragment_mask_artifacts(
    fragment: FinalMaskAcceptedFragment,
    *,
    artifact_root: Path,
    allow_copied_artifact_members: bool,
) -> None:
    fragment_dir = artifact_root / "fragments" / fragment.fragment_id
    mask_npz_path = fragment_dir / "mask_data.npz"
    mask_ply_path = fragment_dir / "lifted_points.ply"
    if (
        allow_copied_artifact_members
        and not mask_npz_path.is_file()
        and not mask_ply_path.is_file()
    ):
        return
    fragment_npz_point_count = validate_points_world_npz(mask_npz_path)
    load_point_indices_npz(mask_npz_path, expected_count=fragment_npz_point_count)
    fragment_ply_vertex_count = validate_ascii_points_ply(mask_ply_path)
    if fragment_ply_vertex_count != fragment_npz_point_count:
        raise CodexResponseError(
            "accepted fragment lifted_points.ply vertex count must match "
            "mask_data.npz point count: "
            f"fragment_id={fragment.fragment_id}; "
            f"mask_npz_path={mask_npz_path}; points={fragment_npz_point_count}; "
            f"mask_ply_path={mask_ply_path}; vertices={fragment_ply_vertex_count}"
        )
    if fragment_npz_point_count != fragment.point_count:
        raise CodexResponseError(
            "accepted fragment point_count must match canonical fragment "
            "mask_data.npz point count: "
            f"fragment_id={fragment.fragment_id}; "
            f"point_count={fragment.point_count}; "
            f"mask_npz_path={mask_npz_path}; points={fragment_npz_point_count}"
        )


def _read_lift_overlay_summary(lift_overlay_path: Path) -> _LiftOverlaySummary:
    try:
        lines = lift_overlay_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CodexResponseError(
            "could not read accepted fragment lift overlay summary: "
            f"lift_overlay_path={lift_overlay_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    fields: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        fields[key.strip()] = value.strip()
    return _LiftOverlaySummary(
        frame_id=fields.get("frame_id", ""),
        candidate_id=fields.get("candidate_id", ""),
    )


def _review_artifact_path_items(
    review_artifacts: FinalMaskReviewArtifacts,
) -> tuple[tuple[str, str], ...]:
    return (
        ("molmo_raw_text_path", review_artifacts.molmo_raw_text_path),
        ("molmo_overlay_path", review_artifacts.molmo_overlay_path),
        ("sam_contact_sheet_path", review_artifacts.sam_contact_sheet_path),
        (
            "sam_candidate_overlay_path",
            review_artifacts.sam_candidate_overlay_path,
        ),
        ("lift_overlay_path", review_artifacts.lift_overlay_path),
    )


def _parse_ply_vertex_count(lines: list[str], mask_ply_path: Path) -> int:
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) == 3 and tokens[0] == "element" and tokens[1] == "vertex":
            try:
                vertex_count = int(tokens[2])
            except ValueError as exc:
                raise CodexResponseError(
                    "mask_ply_path has an invalid vertex count: "
                    f"mask_ply_path={mask_ply_path}; raw_count={tokens[2]!r}"
                ) from exc
            if vertex_count <= 0:
                raise CodexResponseError(
                    "mask_ply_path vertex count must be positive: "
                    f"mask_ply_path={mask_ply_path}; vertex_count={vertex_count}"
                )
            return vertex_count
    raise CodexResponseError(
        "mask_ply_path is missing an element vertex header: "
        f"mask_ply_path={mask_ply_path}"
    )


def _find_ply_end_header(lines: list[str], mask_ply_path: Path) -> int:
    for index, line in enumerate(lines):
        if line.strip() == "end_header":
            return index
    raise CodexResponseError(
        "mask_ply_path is missing end_header: " f"mask_ply_path={mask_ply_path}"
    )


def _validate_ply_vertex_line(
    vertex_line: str,
    *,
    mask_ply_path: Path,
    row_index: int,
) -> None:
    tokens = vertex_line.strip().split()
    if len(tokens) < 3:
        raise CodexResponseError(
            "mask_ply_path vertex row must contain at least XYZ columns: "
            f"mask_ply_path={mask_ply_path}; row_index={row_index}"
        )
    try:
        coordinates = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
    except ValueError as exc:
        raise CodexResponseError(
            "mask_ply_path vertex row contains non-numeric XYZ values: "
            f"mask_ply_path={mask_ply_path}; row_index={row_index}"
        ) from exc
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise CodexResponseError(
            "mask_ply_path vertex row contains non-finite XYZ values: "
            f"mask_ply_path={mask_ply_path}; row_index={row_index}"
        )


__all__ = [
    "FinalMaskAcceptedFragment",
    "FinalMaskArtifactDocument",
    "FinalMaskLiftGeometry",
    "FinalMaskMultiViewAction",
    "FinalMaskMultiViewDecision",
    "FinalMaskReviewArtifacts",
    "ValidatedFinalMaskArtifact",
    "load_point_indices_npz",
    "load_points_world_npz",
    "validate_ascii_points_ply",
    "validate_final_mask_artifact",
    "validate_points_world_npz",
]
