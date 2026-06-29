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


class FinalMaskAcceptedFragment(BaseModel):
    """One accepted 3D fragment recorded in the final mask artifact."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: NonEmptyString
    frame_id: NonEmptyString
    point_count: int = Field(gt=0, strict=True)
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
    point_count: int


def validate_final_mask_artifact(
    *,
    artifact_path: Path,
    mask_npz_path: Path,
    mask_ply_path: Path,
    selected_frame_ids: tuple[str, ...],
    accepted_fragment_ids: tuple[str, ...],
) -> ValidatedFinalMaskArtifact:
    """Validate final artifact JSON, NPZ points, and PLY vertex consistency."""
    artifact_document = _load_final_mask_artifact_document(artifact_path)
    _require_matching_artifact_path(
        artifact_document.mask_npz_path,
        expected_path=mask_npz_path,
        field_name="mask_npz_path",
    )
    _require_matching_artifact_path(
        artifact_document.mask_ply_path,
        expected_path=mask_ply_path,
        field_name="mask_ply_path",
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
        _validate_fragment_review_artifacts(fragment)

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
    expected_path: Path,
    field_name: str,
) -> None:
    actual_path = Path(raw_path).expanduser().resolve()
    resolved_expected_path = expected_path.expanduser().resolve()
    if actual_path != resolved_expected_path:
        raise CodexResponseError(
            f"mask_artifact_path {field_name} must match final response: "
            f"artifact={actual_path}; response={resolved_expected_path}"
        )


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
    if not accepted_followup_frame_ids.intersection(suggested_frame_ids):
        raise ValueError(
            "multi_view_decision.suggested_frame_ids must include at least one "
            "accepted follow-up frame: "
            f"suggested_frame_ids={decision.suggested_frame_ids}; "
            f"accepted_followup_frame_ids={tuple(sorted(accepted_followup_frame_ids))}"
        )


def _validate_fragment_review_artifacts(fragment: FinalMaskAcceptedFragment) -> None:
    for field_name, raw_path in _review_artifact_path_items(fragment.review_artifacts):
        artifact_path = Path(raw_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise CodexResponseError(
                "accepted fragment review_artifacts path must exist: "
                f"fragment_id={fragment.fragment_id}; "
                f"frame_id={fragment.frame_id}; "
                f"field={field_name}; path={artifact_path}"
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
