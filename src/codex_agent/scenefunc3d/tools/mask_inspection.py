"""Mask inspection, multi-view suggestion, and fusion tools."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    StringConstraints,
    ValidationError,
    model_validator,
)
from typing_extensions import NotRequired

from ...errors import CodexResponseError, SceneFunc3dDataError
from ..final_mask_artifacts import (
    FinalMaskArtifactDocument,
    FinalMaskMultiViewAction,
    load_point_indices_npz,
    load_points_world_npz,
    validate_ascii_points_ply,
    validate_points_world_npz,
)
from ..task import ApprovalAction, validate_fragment_approval_actions
from .crop_recommendations import (
    RecommendedCrop as SuggestedCrop,
)
from .crop_recommendations import (
    RecommendedCropPayload,
    recommended_crops_for_matched_objects,
)
from .keyframe_retrieval import (
    KeyframeMatchedObjectPayload,
    VisibleObjectFrameScore,
    visible_object_frame_scores,
)
from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.backends.lift_3d import FloatArray, IntArray

SafePathComponentText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_FRAME_ID_LENGTH = 6


class MaskInspectionPayload(TypedDict):
    """JSON-ready mask inspection result."""

    artifact_path: str | None
    overlay_paths: list[str]
    lifted_point_count: int
    bbox_min_xyz: list[float]
    bbox_max_xyz: list[float]
    bbox_extent_xyz: list[float]
    max_extent_meters: float
    status: str


class SuggestedViewPayload(TypedDict):
    """JSON-ready suggested view."""

    frame_id: str
    reason: str
    rank: int
    has_depth: bool
    has_intrinsics: bool
    has_pose: bool
    view_diversity_score: float
    matched_objects: NotRequired[list[KeyframeMatchedObjectPayload]]
    recommended_crops: NotRequired[list[RecommendedCropPayload]]


class SuggestedViewsPayload(TypedDict):
    """JSON-ready suggested view result."""

    seed_fragment_id: str
    seed_lift_point_count: int
    seed_lift_status: str
    expansion_recommendation: str
    expansion_reason: str
    views: list[SuggestedViewPayload]


class LiftGeometryPayload(TypedDict):
    """JSON-ready 3D geometry summary for one accepted fragment."""

    bbox_min_xyz: list[float]
    bbox_max_xyz: list[float]
    bbox_extent_xyz: list[float]
    max_extent_meters: float


class AcceptedFragmentPayload(TypedDict):
    """JSON-ready accepted mask fragment."""

    fragment_id: str
    frame_id: str
    point_count: int
    lift_geometry: LiftGeometryPayload
    approval_actions: list[str]
    review_artifacts: AcceptedFragmentReviewArtifactsPayload


class AcceptedFragmentReviewArtifactsPayload(TypedDict):
    """JSON-ready review artifact paths for one accepted fragment."""

    molmo_raw_text_path: str
    molmo_overlay_path: str
    sam_contact_sheet_path: str
    sam_candidate_overlay_path: str
    lift_overlay_path: str


class MultiViewDecisionPayload(TypedDict):
    """JSON-ready first-lift multi-view decision."""

    seed_fragment_id: str
    action: str
    reason: str
    suggested_frame_ids: list[str]
    rejected_suggested_frame_ids: list[str]


class FusedMaskPayload(TypedDict):
    """JSON-ready fused mask result."""

    accepted_frame_ids: list[str]
    accepted_fragments: list[AcceptedFragmentPayload]
    multi_view_decision: MultiViewDecisionPayload
    mask_artifact_path: str
    mask_npz_path: str
    mask_ply_path: str


class FusedMaskArtifactFilePayload(TypedDict):
    """JSON payload written as the final evaluable mask artifact."""

    accepted_frame_ids: list[str]
    accepted_fragments: list[AcceptedFragmentPayload]
    multi_view_decision: MultiViewDecisionPayload
    mask_npz_path: str
    mask_ply_path: str


class InspectMaskArtifactArgs(BaseModel):
    """Arguments for inspecting one lifted or fused 3D mask artifact."""

    model_config = ConfigDict(extra="forbid")

    mask_npz_path: FilePath
    mask_ply_path: FilePath
    artifact_path: FilePath | None = None
    overlay_paths: tuple[FilePath, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_review_only_fields(cls, payload: object) -> object:
        """Ignore review context fields that are not inspection inputs."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        overlay_path = values.pop("overlay_path", None)
        if "overlay_paths" not in values and overlay_path is not None:
            values["overlay_paths"] = [overlay_path]
        values.pop("frame_id", None)
        values.pop("candidate_id", None)
        return values


class SuggestAdditionalViewsArgs(BaseModel):
    """Arguments for suggesting extra views after accepting a 3D seed."""

    model_config = ConfigDict(extra="forbid")

    seed_fragment_id: SafePathComponentText
    accepted_frame_id: SafePathComponentText
    seed_mask_npz_path: FilePath
    seed_mask_ply_path: FilePath
    seed_lift_overlay_path: FilePath
    candidate_frame_ids: tuple[SafePathComponentText, ...] = ()
    task_description: NonEmptyText | None = None
    min_seed_point_count: int = Field(default=256, ge=1, strict=True)
    k: int = Field(default=4, ge=1, le=8, strict=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_seed_frame_aliases(cls, payload: object) -> object:
        """Accept common seed-frame wording from agent tool calls."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        if "accepted_frame_id" not in values and "seed_frame_id" in values:
            values["accepted_frame_id"] = values.pop("seed_frame_id")
        else:
            values.pop("seed_frame_id", None)
        if "accepted_frame_id" not in values:
            inferred_frame_id = _infer_frame_id_from_seed_fragment(
                values.get("seed_fragment_id")
            )
            if inferred_frame_id is not None:
                values["accepted_frame_id"] = inferred_frame_id
        values.pop("seed_candidate_id", None)
        values.pop("desc_id", None)
        values.pop("annotation_ids", None)
        return values


def _infer_frame_id_from_seed_fragment(seed_fragment_id: object) -> str | None:
    """Infer the accepted frame id from ``<frame_id>_<candidate_id>`` fragments."""
    if not isinstance(seed_fragment_id, str):
        return None
    candidate_frame_id, separator, _candidate_id = seed_fragment_id.partition("_")
    if separator == "":
        return None
    if _candidate_id == "":
        return None
    if len(candidate_frame_id) != _FRAME_ID_LENGTH:
        return None
    if not candidate_frame_id.isdecimal():
        return None
    return candidate_frame_id


class AcceptedFragmentInput(BaseModel):
    """One accepted lifted fragment to fuse."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: SafePathComponentText
    frame_id: SafePathComponentText
    mask_npz_path: FilePath
    mask_ply_path: FilePath
    approval_actions: tuple[ApprovalAction, ...] = Field(min_length=1)
    review_artifacts: AcceptedFragmentReviewArtifactsInput

    @model_validator(mode="before")
    @classmethod
    def normalize_review_only_fields(cls, payload: object) -> object:
        """Ignore SAM candidate context already encoded in ``fragment_id``."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        values.pop("candidate_id", None)
        return values


class AcceptedFragmentReviewArtifactsInput(BaseModel):
    """Agent-reviewed artifacts supporting one accepted fragment."""

    model_config = ConfigDict(extra="forbid")

    molmo_raw_text_path: FilePath
    molmo_overlay_path: FilePath
    sam_contact_sheet_path: FilePath
    sam_candidate_overlay_path: FilePath
    lift_overlay_path: FilePath

    def to_domain(self) -> AcceptedFragmentReviewArtifacts:
        """Return an immutable domain value for persisted fragment provenance."""
        return AcceptedFragmentReviewArtifacts(
            molmo_raw_text_path=self.molmo_raw_text_path,
            molmo_overlay_path=self.molmo_overlay_path,
            sam_contact_sheet_path=self.sam_contact_sheet_path,
            sam_candidate_overlay_path=self.sam_candidate_overlay_path,
            lift_overlay_path=self.lift_overlay_path,
        )


class MultiViewDecisionInput(BaseModel):
    """Agent decision after inspecting the first lifted 3D seed."""

    model_config = ConfigDict(extra="forbid")

    seed_fragment_id: SafePathComponentText
    action: FinalMaskMultiViewAction
    reason: NonEmptyText
    suggested_frame_ids: tuple[SafePathComponentText, ...] = ()
    rejected_suggested_frame_ids: tuple[SafePathComponentText, ...] = ()

    def to_domain(self) -> MultiViewDecision:
        """Return an immutable domain value for persisted fusion provenance."""
        return MultiViewDecision(
            seed_fragment_id=self.seed_fragment_id,
            action=self.action,
            reason=self.reason,
            suggested_frame_ids=self.suggested_frame_ids,
            rejected_suggested_frame_ids=self.rejected_suggested_frame_ids,
        )


class FuseAcceptedMasksArgs(BaseModel):
    """Arguments for fusing accepted 3D mask fragments."""

    model_config = ConfigDict(extra="forbid")

    fragments: tuple[AcceptedFragmentInput, ...] = Field(min_length=1)
    multi_view_decision: MultiViewDecisionInput

    @model_validator(mode="before")
    @classmethod
    def normalize_fragment_aliases(cls, payload: object) -> object:
        """Accept explicit accepted-fragment wording from agent tool calls."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        if "fragments" not in values and "accepted_fragments" in values:
            values["fragments"] = values.pop("accepted_fragments")
        else:
            values.pop("accepted_fragments", None)
        values.pop("task_description", None)
        return values


@dataclass(frozen=True)
class MaskInspectionResult:
    """Artifact inspection result for agent review."""

    artifact_path: Path | None
    overlay_paths: tuple[Path, ...]
    lifted_point_count: int
    bbox_min_xyz: tuple[float, float, float]
    bbox_max_xyz: tuple[float, float, float]
    bbox_extent_xyz: tuple[float, float, float]
    max_extent_meters: float
    status: str

    def __post_init__(self) -> None:
        """Validate domain invariants for one inspection result."""
        _validate_non_negative_int("lifted_point_count", self.lifted_point_count)
        _validate_xyz_tuple("bbox_min_xyz", self.bbox_min_xyz)
        _validate_xyz_tuple("bbox_max_xyz", self.bbox_max_xyz)
        _validate_xyz_tuple("bbox_extent_xyz", self.bbox_extent_xyz)
        if not math.isfinite(self.max_extent_meters) or self.max_extent_meters < 0.0:
            raise ValueError(
                "max_extent_meters must be finite and non-negative: "
                f"{self.max_extent_meters!r}"
            )
        _validate_non_empty_text("status", self.status)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready inspection result."""
        return {
            "artifact_path": (
                str(self.artifact_path) if self.artifact_path is not None else None
            ),
            "overlay_paths": [str(path) for path in self.overlay_paths],
            "lifted_point_count": self.lifted_point_count,
            "bbox_min_xyz": list(self.bbox_min_xyz),
            "bbox_max_xyz": list(self.bbox_max_xyz),
            "bbox_extent_xyz": list(self.bbox_extent_xyz),
            "max_extent_meters": self.max_extent_meters,
            "status": self.status,
        }


@dataclass(frozen=True)
class SuggestedView:
    """One additional view suggested from an accepted 3D seed."""

    frame_id: str
    reason: str
    rank: int
    has_depth: bool
    has_intrinsics: bool
    has_pose: bool
    view_diversity_score: float
    matched_objects: tuple[KeyframeMatchedObjectPayload, ...] = ()
    recommended_crops: tuple[SuggestedCrop, ...] = ()

    def __post_init__(self) -> None:
        """Validate domain invariants for one suggested view."""
        _validate_non_empty_text("frame_id", self.frame_id)
        _validate_non_empty_text("reason", self.reason)
        _validate_positive_int("rank", self.rank)
        if not math.isfinite(self.view_diversity_score):
            raise ValueError(
                "view_diversity_score must be finite: " f"{self.view_diversity_score!r}"
            )
        if self.view_diversity_score < 0.0:
            raise ValueError(
                "view_diversity_score must be non-negative: "
                f"{self.view_diversity_score!r}"
            )

    def to_payload(self) -> SuggestedViewPayload:
        """Return the JSON-ready suggested view."""
        payload: SuggestedViewPayload = {
            "frame_id": self.frame_id,
            "reason": self.reason,
            "rank": self.rank,
            "has_depth": self.has_depth,
            "has_intrinsics": self.has_intrinsics,
            "has_pose": self.has_pose,
            "view_diversity_score": self.view_diversity_score,
        }
        if self.matched_objects:
            payload["matched_objects"] = list(self.matched_objects)
        if self.recommended_crops:
            payload["recommended_crops"] = [
                crop.to_payload() for crop in self.recommended_crops
            ]
        return payload


class SeedLiftStatus(str, Enum):
    """Coarse geometry status for the first accepted 3D seed."""

    SPARSE = "sparse"
    USABLE = "usable"


@dataclass(frozen=True)
class SuggestedViewsResult:
    """Additional views suggested for multi-view expansion."""

    seed_fragment_id: str
    seed_lift_point_count: int
    seed_lift_status: SeedLiftStatus
    expansion_recommendation: FinalMaskMultiViewAction
    expansion_reason: str
    views: tuple[SuggestedView, ...]

    def __post_init__(self) -> None:
        """Validate domain invariants for a suggested view set."""
        _validate_non_empty_text("seed_fragment_id", self.seed_fragment_id)
        _validate_positive_int("seed_lift_point_count", self.seed_lift_point_count)
        _validate_non_empty_text("expansion_reason", self.expansion_reason)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready suggested view set."""
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "seed_lift_point_count": self.seed_lift_point_count,
            "seed_lift_status": self.seed_lift_status.value,
            "expansion_recommendation": self.expansion_recommendation.value,
            "expansion_reason": self.expansion_reason,
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(frozen=True)
class _CandidateViewGeometry:
    """Geometry signals used to rank one candidate follow-up frame."""

    frame_id: str
    temporal_distance: int
    has_depth: bool
    has_intrinsics: bool
    has_pose: bool
    view_diversity_score: float
    primary_object_match_count: int
    visible_object_score: float
    matched_objects: tuple[KeyframeMatchedObjectPayload, ...]


@dataclass(frozen=True)
class _LiftOverlaySummary:
    """Parsed key fields from a lift overlay summary artifact."""

    frame_id: str
    candidate_id: str


@dataclass(frozen=True)
class AcceptedFragmentReviewArtifacts:
    """Agent-visible evidence artifacts reviewed before accepting a fragment."""

    molmo_raw_text_path: Path
    molmo_overlay_path: Path
    sam_contact_sheet_path: Path
    sam_candidate_overlay_path: Path
    lift_overlay_path: Path

    def to_payload(self) -> AcceptedFragmentReviewArtifactsPayload:
        """Return JSON-ready review artifact paths."""
        return {
            "molmo_raw_text_path": str(self.molmo_raw_text_path),
            "molmo_overlay_path": str(self.molmo_overlay_path),
            "sam_contact_sheet_path": str(self.sam_contact_sheet_path),
            "sam_candidate_overlay_path": str(self.sam_candidate_overlay_path),
            "lift_overlay_path": str(self.lift_overlay_path),
        }


@dataclass(frozen=True)
class AcceptedFragment:
    """One accepted 3D mask fragment."""

    fragment_id: str
    frame_id: str
    point_count: int
    lift_geometry: LiftGeometrySummary
    approval_actions: tuple[ApprovalAction, ...]
    review_artifacts: AcceptedFragmentReviewArtifacts

    def __post_init__(self) -> None:
        """Validate domain invariants for one accepted fragment."""
        _validate_non_empty_text("fragment_id", self.fragment_id)
        _validate_non_empty_text("frame_id", self.frame_id)
        _validate_positive_int("point_count", self.point_count)
        if not self.approval_actions:
            raise ValueError("approval_actions must be non-empty")
        try:
            validate_fragment_approval_actions(self.fragment_id, self.approval_actions)
        except SceneFunc3dDataError as exc:
            raise ValueError(
                "approval_actions must replay the SceneFunc3D approval gates: "
                f"fragment_id={self.fragment_id}; error={exc}"
            ) from exc

    def to_payload(self) -> AcceptedFragmentPayload:
        """Return the JSON-ready accepted fragment."""
        return {
            "fragment_id": self.fragment_id,
            "frame_id": self.frame_id,
            "point_count": self.point_count,
            "lift_geometry": self.lift_geometry.to_payload(),
            "approval_actions": [action.value for action in self.approval_actions],
            "review_artifacts": self.review_artifacts.to_payload(),
        }


@dataclass(frozen=True)
class MultiViewDecision:
    """Agent's first-lift decision about whether to add more views."""

    seed_fragment_id: str
    action: FinalMaskMultiViewAction
    reason: str
    suggested_frame_ids: tuple[str, ...]
    rejected_suggested_frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate domain invariants for a multi-view decision."""
        _validate_non_empty_text("seed_fragment_id", self.seed_fragment_id)
        _validate_non_empty_text("reason", self.reason)
        for frame_id in self.suggested_frame_ids:
            _validate_non_empty_text("suggested_frame_id", frame_id)
        for frame_id in self.rejected_suggested_frame_ids:
            _validate_non_empty_text("rejected_suggested_frame_id", frame_id)

    def to_payload(self) -> MultiViewDecisionPayload:
        """Return the JSON-ready multi-view decision."""
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "action": self.action.value,
            "reason": self.reason,
            "suggested_frame_ids": list(self.suggested_frame_ids),
            "rejected_suggested_frame_ids": list(self.rejected_suggested_frame_ids),
        }


@dataclass(frozen=True)
class FusedMaskResult:
    """Fused mask artifact result."""

    accepted_fragments: tuple[AcceptedFragment, ...]
    multi_view_decision: MultiViewDecision
    mask_artifact_path: Path
    mask_npz_path: Path
    mask_ply_path: Path

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready fused mask result."""
        return {
            "accepted_frame_ids": list(_accepted_frame_ids(self.accepted_fragments)),
            "accepted_fragments": [
                fragment.to_payload() for fragment in self.accepted_fragments
            ],
            "multi_view_decision": self.multi_view_decision.to_payload(),
            "mask_artifact_path": str(self.mask_artifact_path),
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
        }


def inspect_mask_artifact(args: InspectMaskArtifactArgs) -> MaskInspectionResult:
    """Inspect a lifted or fused mask artifact before agent approval."""
    lifted_point_count = _validate_mask_pair(
        mask_npz_path=args.mask_npz_path,
        mask_ply_path=args.mask_ply_path,
    )
    geometry_summary = _mask_geometry_summary(args.mask_npz_path)
    return MaskInspectionResult(
        artifact_path=args.artifact_path,
        overlay_paths=tuple(args.overlay_paths),
        lifted_point_count=lifted_point_count,
        bbox_min_xyz=geometry_summary.bbox_min_xyz,
        bbox_max_xyz=geometry_summary.bbox_max_xyz,
        bbox_extent_xyz=geometry_summary.bbox_extent_xyz,
        max_extent_meters=geometry_summary.max_extent_meters,
        status="valid",
    )


@dataclass(frozen=True)
class LiftGeometrySummary:
    """Axis-aligned 3D extent summary for an inspected or accepted mask."""

    bbox_min_xyz: tuple[float, float, float]
    bbox_max_xyz: tuple[float, float, float]
    bbox_extent_xyz: tuple[float, float, float]
    max_extent_meters: float

    def to_payload(self) -> LiftGeometryPayload:
        """Return JSON-ready geometry metadata."""
        return {
            "bbox_min_xyz": list(self.bbox_min_xyz),
            "bbox_max_xyz": list(self.bbox_max_xyz),
            "bbox_extent_xyz": list(self.bbox_extent_xyz),
            "max_extent_meters": self.max_extent_meters,
        }


def _mask_geometry_summary(mask_npz_path: Path) -> LiftGeometrySummary:
    points_world = load_points_world_npz(mask_npz_path)
    bbox_min_xyz = _xyz_tuple(cast("FloatArray", points_world.min(axis=0)))
    bbox_max_xyz = _xyz_tuple(cast("FloatArray", points_world.max(axis=0)))
    bbox_extent_xyz = _extent_xyz_tuple(bbox_min_xyz, bbox_max_xyz)
    return LiftGeometrySummary(
        bbox_min_xyz=bbox_min_xyz,
        bbox_max_xyz=bbox_max_xyz,
        bbox_extent_xyz=bbox_extent_xyz,
        max_extent_meters=max(bbox_extent_xyz),
    )


def _xyz_tuple(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _extent_xyz_tuple(
    bbox_min_xyz: tuple[float, float, float],
    bbox_max_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        bbox_max_xyz[0] - bbox_min_xyz[0],
        bbox_max_xyz[1] - bbox_min_xyz[1],
        bbox_max_xyz[2] - bbox_min_xyz[2],
    )


def suggest_additional_views(
    tool_scene: SceneFunc3dToolScene, args: SuggestAdditionalViewsArgs
) -> SuggestedViewsResult:
    """Suggest nearby views after a first accepted 3D fragment."""
    seed_lift_point_count = _validate_mask_pair(
        mask_npz_path=args.seed_mask_npz_path,
        mask_ply_path=args.seed_mask_ply_path,
    )
    _validate_seed_lift_overlay(
        args.seed_lift_overlay_path,
        seed_fragment_id=args.seed_fragment_id,
        accepted_frame_id=args.accepted_frame_id,
    )
    _validate_seed_fragment_paths(
        seed_mask_npz_path=args.seed_mask_npz_path,
        seed_mask_ply_path=args.seed_mask_ply_path,
        seed_lift_overlay_path=args.seed_lift_overlay_path,
        seed_fragment_id=args.seed_fragment_id,
    )
    available_frame_ids = set(tool_scene.rgb_frame_ids)
    if args.accepted_frame_id not in available_frame_ids:
        raise ToolInputError(
            "accepted_frame_id must exist in the SceneFunc3D scene: "
            f"accepted_frame_id={args.accepted_frame_id}; "
            f"scene_root={tool_scene.scene_root}"
        )
    candidate_frame_ids = (
        args.candidate_frame_ids
        if args.candidate_frame_ids
        else tool_scene.rgb_frame_ids
    )
    missing_candidate_ids = tuple(
        frame_id
        for frame_id in candidate_frame_ids
        if frame_id not in available_frame_ids
    )
    if missing_candidate_ids:
        raise ToolInputError(
            "candidate_frame_ids must exist in the SceneFunc3D scene: "
            f"missing={missing_candidate_ids}; scene_root={tool_scene.scene_root}"
        )

    accepted_camera_center = _camera_center_for_frame(
        tool_scene, args.accepted_frame_id
    )
    visible_score_by_frame_id = _visible_object_scores_by_frame_id(
        tool_scene, args.task_description
    )
    candidate_geometries = tuple(
        _candidate_view_geometry(
            tool_scene,
            frame_id=frame_id,
            accepted_frame_id=args.accepted_frame_id,
            accepted_camera_center=accepted_camera_center,
            visible_object_match=visible_score_by_frame_id.get(frame_id),
        )
        for frame_id in candidate_frame_ids
        if frame_id != args.accepted_frame_id
    )
    seed_lift_status = _seed_lift_status(
        seed_lift_point_count,
        min_seed_point_count=args.min_seed_point_count,
    )
    expansion_recommendation = _expansion_recommendation(
        seed_lift_status,
        candidate_view_count=len(candidate_geometries),
    )
    expansion_reason = _expansion_reason(
        seed_lift_point_count=seed_lift_point_count,
        min_seed_point_count=args.min_seed_point_count,
        seed_lift_status=seed_lift_status,
        candidate_view_count=len(candidate_geometries),
    )
    if expansion_recommendation is FinalMaskMultiViewAction.STOP:
        return SuggestedViewsResult(
            seed_fragment_id=args.seed_fragment_id,
            seed_lift_point_count=seed_lift_point_count,
            seed_lift_status=seed_lift_status,
            expansion_recommendation=expansion_recommendation,
            expansion_reason=expansion_reason,
            views=(),
        )

    ranked_candidates = sorted(candidate_geometries, key=_candidate_rank_key)
    views = tuple(
        _suggested_view_from_geometry(
            candidate,
            rank=rank,
            seed_lift_point_count=seed_lift_point_count,
            min_seed_point_count=args.min_seed_point_count,
            seed_lift_status=seed_lift_status,
            task_description=args.task_description,
        )
        for rank, candidate in enumerate(ranked_candidates[: args.k], start=1)
    )
    return SuggestedViewsResult(
        seed_fragment_id=args.seed_fragment_id,
        seed_lift_point_count=seed_lift_point_count,
        seed_lift_status=seed_lift_status,
        expansion_recommendation=expansion_recommendation,
        expansion_reason=expansion_reason,
        views=views,
    )


def fuse_accepted_masks(
    args: FuseAcceptedMasksArgs, *, out_dir: Path
) -> FusedMaskResult:
    """Fuse accepted lifted fragments into one final mask artifact."""
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    _validate_unique_fragment_ids(args.fragments)
    accepted_fragments: list[AcceptedFragment] = []
    fragment_points: list[FloatArray] = []
    fragment_point_indices: list[IntArray] = []
    for fragment in args.fragments:
        _validate_accepted_fragment_approval_actions(
            fragment.fragment_id, fragment.approval_actions
        )
        point_count = _validate_mask_pair(
            mask_npz_path=fragment.mask_npz_path,
            mask_ply_path=fragment.mask_ply_path,
        )
        fragment_points.append(load_points_world_npz(fragment.mask_npz_path))
        fragment_point_indices.append(
            load_point_indices_npz(
                fragment.mask_npz_path,
                expected_count=point_count,
            )
        )
        accepted_fragments.append(
            AcceptedFragment(
                fragment_id=fragment.fragment_id,
                frame_id=fragment.frame_id,
                point_count=point_count,
                lift_geometry=_mask_geometry_summary(fragment.mask_npz_path),
                approval_actions=fragment.approval_actions,
                review_artifacts=fragment.review_artifacts.to_domain(),
            )
        )

    multi_view_decision = args.multi_view_decision.to_domain()
    fused_points = cast("FloatArray", np.concatenate(fragment_points, axis=0))
    fused_point_indices = cast(
        "IntArray", np.concatenate(fragment_point_indices, axis=0)
    )
    fused_dir = out_dir / "fused"
    mask_npz_path = write_lift_npz(
        fused_dir / "mask_data.npz",
        fused_points,
        point_indices=fused_point_indices,
    )
    mask_ply_path = write_lift_ply(fused_dir / "lifted_points.ply", fused_points)
    mask_artifact_path = _write_fused_mask_artifact(
        fused_dir / "mask_artifact.json",
        accepted_fragments=tuple(accepted_fragments),
        multi_view_decision=multi_view_decision,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
    )
    return FusedMaskResult(
        accepted_fragments=tuple(accepted_fragments),
        multi_view_decision=multi_view_decision,
        mask_artifact_path=mask_artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
    )


def _validate_mask_pair(*, mask_npz_path: Path, mask_ply_path: Path) -> int:
    try:
        npz_point_count = validate_points_world_npz(mask_npz_path)
        load_point_indices_npz(mask_npz_path, expected_count=npz_point_count)
        ply_vertex_count = validate_ascii_points_ply(mask_ply_path)
    except CodexResponseError as exc:
        raise ToolInputError(str(exc)) from exc
    if npz_point_count != ply_vertex_count:
        raise ToolInputError(
            "mask npz point count must match mask ply vertex count: "
            f"mask_npz_path={mask_npz_path}; points={npz_point_count}; "
            f"mask_ply_path={mask_ply_path}; vertices={ply_vertex_count}"
        )
    return npz_point_count


def _validate_unique_fragment_ids(
    fragments: tuple[AcceptedFragmentInput, ...],
) -> None:
    seen_fragment_ids: set[str] = set()
    for fragment in fragments:
        if fragment.fragment_id in seen_fragment_ids:
            raise ToolInputError(
                f"duplicate accepted fragment_id={fragment.fragment_id!r}"
            )
        seen_fragment_ids.add(fragment.fragment_id)


def _validate_accepted_fragment_approval_actions(
    fragment_id: str, approval_actions: tuple[ApprovalAction, ...]
) -> None:
    try:
        validate_fragment_approval_actions(fragment_id, approval_actions)
    except SceneFunc3dDataError as exc:
        raise ToolInputError(
            "accepted fragment approval_actions failed SceneFunc3D approval gate "
            f"replay: fragment_id={fragment_id}; error={exc}"
        ) from exc


def _write_fused_mask_artifact(
    artifact_path: Path,
    *,
    accepted_fragments: tuple[AcceptedFragment, ...],
    multi_view_decision: MultiViewDecision,
    mask_npz_path: Path,
    mask_ply_path: Path,
) -> Path:
    payload: FusedMaskArtifactFilePayload = {
        "accepted_frame_ids": list(_accepted_frame_ids(accepted_fragments)),
        "accepted_fragments": [
            fragment.to_payload() for fragment in accepted_fragments
        ],
        "multi_view_decision": multi_view_decision.to_payload(),
        "mask_npz_path": str(mask_npz_path),
        "mask_ply_path": str(mask_ply_path),
    }
    try:
        FinalMaskArtifactDocument.model_validate(payload)
    except ValidationError as exc:
        raise ToolInputError(
            "fused mask artifact payload failed final schema validation: "
            f"artifact_path={artifact_path}; error={exc}"
        ) from exc
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ToolInputError(
            "could not write fused mask artifact: "
            f"artifact_path={artifact_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return artifact_path


def _accepted_frame_ids(
    accepted_fragments: tuple[AcceptedFragment, ...],
) -> tuple[str, ...]:
    seen_frame_ids: set[str] = set()
    accepted_frame_ids: list[str] = []
    for fragment in accepted_fragments:
        if fragment.frame_id not in seen_frame_ids:
            accepted_frame_ids.append(fragment.frame_id)
            seen_frame_ids.add(fragment.frame_id)
    return tuple(accepted_frame_ids)


def _candidate_view_geometry(
    tool_scene: SceneFunc3dToolScene,
    *,
    frame_id: str,
    accepted_frame_id: str,
    accepted_camera_center: tuple[float, float, float] | None,
    visible_object_match: VisibleObjectFrameScore | None,
) -> _CandidateViewGeometry:
    from codex_agent.scenefunc3d.backends.frame_assets import (
        resolve_available_frame_geometry_assets,
    )

    available_assets = resolve_available_frame_geometry_assets(tool_scene, frame_id)
    candidate_camera_center = (
        _load_camera_center(available_assets.pose_path)
        if available_assets.pose_path is not None
        else None
    )
    return _CandidateViewGeometry(
        frame_id=frame_id,
        temporal_distance=_temporal_distance(frame_id, accepted_frame_id),
        has_depth=available_assets.depth_path is not None,
        has_intrinsics=available_assets.intrinsics_path is not None,
        has_pose=available_assets.pose_path is not None,
        view_diversity_score=_view_diversity_score(
            accepted_camera_center, candidate_camera_center
        ),
        visible_object_score=(
            visible_object_match.score if visible_object_match is not None else 0.0
        ),
        primary_object_match_count=(
            visible_object_match.primary_object_match_count
            if visible_object_match is not None
            else 0
        ),
        matched_objects=(
            visible_object_match.matched_objects
            if visible_object_match is not None
            else ()
        ),
    )


def _candidate_rank_key(
    candidate: _CandidateViewGeometry,
) -> tuple[int, int, float, float, int, str]:
    complete_geometry_score = int(
        candidate.has_depth and candidate.has_intrinsics and candidate.has_pose
    )
    return (
        -complete_geometry_score,
        -candidate.primary_object_match_count,
        -candidate.visible_object_score,
        -candidate.view_diversity_score,
        candidate.temporal_distance,
        candidate.frame_id,
    )


def _suggested_view_from_geometry(
    candidate: _CandidateViewGeometry,
    *,
    rank: int,
    seed_lift_point_count: int,
    min_seed_point_count: int,
    seed_lift_status: SeedLiftStatus,
    task_description: str | None,
) -> SuggestedView:
    return SuggestedView(
        frame_id=candidate.frame_id,
        reason=_suggested_view_reason(
            candidate=candidate,
            seed_lift_point_count=seed_lift_point_count,
            min_seed_point_count=min_seed_point_count,
            seed_lift_status=seed_lift_status,
        ),
        rank=rank,
        has_depth=candidate.has_depth,
        has_intrinsics=candidate.has_intrinsics,
        has_pose=candidate.has_pose,
        view_diversity_score=candidate.view_diversity_score,
        matched_objects=candidate.matched_objects,
        recommended_crops=recommended_crops_for_matched_objects(
            candidate.matched_objects,
            task_description,
        ),
    )


def _visible_object_scores_by_frame_id(
    tool_scene: SceneFunc3dToolScene, task_description: str | None
) -> dict[str, VisibleObjectFrameScore]:
    if task_description is None:
        return {}
    return {
        frame_score.frame_id: frame_score
        for frame_score in visible_object_frame_scores(tool_scene, task_description)
    }


def _temporal_distance(frame_id: str, accepted_frame_id: str) -> int:
    frame_number = _frame_number(frame_id)
    accepted_number = _frame_number(accepted_frame_id)
    if frame_number is None or accepted_number is None:
        return 10**12
    return abs(frame_number - accepted_number)


def _seed_lift_status(
    seed_lift_point_count: int, *, min_seed_point_count: int
) -> SeedLiftStatus:
    if seed_lift_point_count < min_seed_point_count:
        return SeedLiftStatus.SPARSE
    return SeedLiftStatus.USABLE


def _expansion_recommendation(
    seed_lift_status: SeedLiftStatus,
    *,
    candidate_view_count: int,
) -> FinalMaskMultiViewAction:
    if seed_lift_status is SeedLiftStatus.SPARSE:
        return FinalMaskMultiViewAction.EXPAND
    if candidate_view_count > 0:
        return FinalMaskMultiViewAction.EXPAND
    return FinalMaskMultiViewAction.STOP


def _expansion_reason(
    *,
    seed_lift_point_count: int,
    min_seed_point_count: int,
    seed_lift_status: SeedLiftStatus,
    candidate_view_count: int,
) -> str:
    if seed_lift_status is SeedLiftStatus.SPARSE:
        return (
            "seed_geometry_sparse: "
            f"point_count={seed_lift_point_count} below "
            f"min_seed_point_count={min_seed_point_count}; "
            "additional views are recommended"
        )
    if candidate_view_count > 0:
        return (
            "seed_geometry_usable: "
            f"point_count={seed_lift_point_count} meets "
            f"min_seed_point_count={min_seed_point_count}; "
            f"{candidate_view_count} additional candidate views are available; "
            "multi-view expansion is recommended"
        )
    return (
        "seed_geometry_usable: "
        f"point_count={seed_lift_point_count} meets "
        f"min_seed_point_count={min_seed_point_count}; "
        "no additional candidate views are available"
    )


def _suggested_view_reason(
    *,
    candidate: _CandidateViewGeometry,
    seed_lift_point_count: int,
    min_seed_point_count: int,
    seed_lift_status: SeedLiftStatus,
) -> str:
    geometry_reason = _candidate_geometry_reason(candidate)
    visible_object_reason = _candidate_visible_object_reason(candidate)
    view_reason = f"view_diversity_score={candidate.view_diversity_score:.3f}"
    temporal_reason = f"temporal_distance={candidate.temporal_distance}"
    if seed_lift_status is SeedLiftStatus.SPARSE:
        middle_reasons = _join_suggestion_reasons(
            visible_object_reason,
            geometry_reason,
            view_reason,
            temporal_reason,
        )
        return (
            "seed_geometry_sparse: "
            f"point_count={seed_lift_point_count} below "
            f"min_seed_point_count={min_seed_point_count}; "
            f"{middle_reasons}"
        )
    middle_reasons = _join_suggestion_reasons(
        visible_object_reason,
        geometry_reason,
        view_reason,
        temporal_reason,
    )
    return (
        "seed_geometry_usable: "
        f"point_count={seed_lift_point_count} meets "
        f"min_seed_point_count={min_seed_point_count}; "
        f"{middle_reasons}"
    )


def _candidate_visible_object_reason(candidate: _CandidateViewGeometry) -> str | None:
    if not candidate.matched_objects:
        return None
    return f"visible_object_query_match: score={candidate.visible_object_score:.3f}"


def _join_suggestion_reasons(*reasons: str | None) -> str:
    return "; ".join(reason for reason in reasons if reason is not None)


def _candidate_geometry_reason(candidate: _CandidateViewGeometry) -> str:
    fields = (
        ("depth", candidate.has_depth),
        ("intrinsics", candidate.has_intrinsics),
        ("pose", candidate.has_pose),
    )
    available_fields = tuple(field_name for field_name, exists in fields if exists)
    missing_fields = tuple(field_name for field_name, exists in fields if not exists)
    if not missing_fields:
        return "candidate has depth, intrinsics, and pose"
    if not available_fields:
        return "candidate is missing depth, intrinsics, and pose"
    return (
        "candidate has "
        f"{_join_field_names(available_fields)}; "
        f"missing {_join_field_names(missing_fields)}"
    )


def _join_field_names(field_names: tuple[str, ...]) -> str:
    if len(field_names) == 1:
        return field_names[0]
    if len(field_names) == 2:
        return f"{field_names[0]} and {field_names[1]}"
    return f"{', '.join(field_names[:-1])}, and {field_names[-1]}"


def _camera_center_for_frame(
    tool_scene: SceneFunc3dToolScene, frame_id: str
) -> tuple[float, float, float] | None:
    from codex_agent.scenefunc3d.backends.frame_assets import (
        resolve_available_frame_geometry_assets,
    )

    available_assets = resolve_available_frame_geometry_assets(tool_scene, frame_id)
    if available_assets.pose_path is None:
        return None
    return _load_camera_center(available_assets.pose_path)


def _load_camera_center(pose_path: Path) -> tuple[float, float, float]:
    try:
        raw_values = tuple(
            float(value) for value in pose_path.read_text(encoding="utf-8").split()
        )
    except OSError as exc:
        raise ToolInputError(
            "could not read camera pose for view suggestion: "
            f"pose_path={pose_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse camera pose for view suggestion: "
            f"pose_path={pose_path}; error_type={exc.__class__.__name__}"
        ) from exc
    if len(raw_values) != 16:
        raise ToolInputError(
            "camera pose for view suggestion must contain 16 values: "
            f"pose_path={pose_path}; actual={len(raw_values)}"
        )
    return (raw_values[3], raw_values[7], raw_values[11])


def _view_diversity_score(
    accepted_camera_center: tuple[float, float, float] | None,
    candidate_camera_center: tuple[float, float, float] | None,
) -> float:
    if accepted_camera_center is None or candidate_camera_center is None:
        return 0.0
    return float(math.dist(accepted_camera_center, candidate_camera_center))


def _validate_seed_lift_overlay(
    overlay_path: Path, *, seed_fragment_id: str, accepted_frame_id: str
) -> None:
    summary = _read_lift_overlay_summary(overlay_path)
    if summary.frame_id != accepted_frame_id:
        raise ToolInputError(
            "seed lift overlay frame_id must match accepted_frame_id: "
            f"seed_lift_overlay_path={overlay_path}; "
            f"overlay_frame_id={summary.frame_id!r}; "
            f"accepted_frame_id={accepted_frame_id!r}"
        )
    expected_seed_fragment_id = f"{summary.frame_id}_{summary.candidate_id}"
    if summary.candidate_id == "" or seed_fragment_id != expected_seed_fragment_id:
        raise ToolInputError(
            "seed lift overlay candidate_id must match seed_fragment_id: "
            f"seed_lift_overlay_path={overlay_path}; "
            f"candidate_id={summary.candidate_id!r}; "
            f"seed_fragment_id={seed_fragment_id!r}; "
            f"expected_seed_fragment_id={expected_seed_fragment_id!r}"
        )


def _validate_seed_fragment_paths(
    *,
    seed_mask_npz_path: Path,
    seed_mask_ply_path: Path,
    seed_lift_overlay_path: Path,
    seed_fragment_id: str,
) -> None:
    if (
        seed_mask_npz_path.parent != seed_mask_ply_path.parent
        or seed_mask_npz_path.parent != seed_lift_overlay_path.parent
    ):
        raise ToolInputError(
            "seed_mask_npz_path, seed_mask_ply_path, and seed_lift_overlay_path "
            "must come from the same seed fragment directory: "
            f"seed_mask_npz_path={seed_mask_npz_path}; "
            f"seed_mask_ply_path={seed_mask_ply_path}; "
            f"seed_lift_overlay_path={seed_lift_overlay_path}"
        )
    if seed_mask_npz_path.parent.name != seed_fragment_id:
        raise ToolInputError(
            "seed artifact directory name must match seed_fragment_id: "
            f"seed_fragment_id={seed_fragment_id!r}; "
            f"artifact_dir={seed_mask_npz_path.parent}"
        )
    expected_names = (
        ("seed_mask_npz_path", seed_mask_npz_path, "mask_data.npz"),
        ("seed_mask_ply_path", seed_mask_ply_path, "lifted_points.ply"),
        ("seed_lift_overlay_path", seed_lift_overlay_path, "lift_overlay.txt"),
    )
    for field_name, path, expected_name in expected_names:
        if path.name != expected_name:
            raise ToolInputError(
                f"{field_name} must use the standard lift artifact filename "
                f"{expected_name!r}: path={path}"
            )


def _read_lift_overlay_summary(overlay_path: Path) -> _LiftOverlaySummary:
    try:
        lines = overlay_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ToolInputError(
            "could not read seed lift overlay summary: "
            f"seed_lift_overlay_path={overlay_path}; "
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


def _frame_number(frame_id: str) -> int | None:
    try:
        return int(frame_id)
    except ValueError:
        return None


def _validate_non_empty_text(field_name: str, field_value: str) -> None:
    if field_value.strip() == "":
        raise ValueError(f"{field_name} must be non-empty")


def _validate_non_negative_int(field_name: str, field_value: int) -> None:
    if field_value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_xyz_tuple(
    field_name: str, field_value: tuple[float, float, float]
) -> None:
    if any(not math.isfinite(value) for value in field_value):
        raise ValueError(f"{field_name} must contain finite coordinates")


def _validate_positive_int(field_name: str, field_value: int) -> None:
    if field_value <= 0:
        raise ValueError(f"{field_name} must be positive")


__all__ = [
    "AcceptedFragment",
    "AcceptedFragmentInput",
    "AcceptedFragmentPayload",
    "AcceptedFragmentReviewArtifacts",
    "AcceptedFragmentReviewArtifactsInput",
    "AcceptedFragmentReviewArtifactsPayload",
    "FuseAcceptedMasksArgs",
    "FusedMaskPayload",
    "FusedMaskResult",
    "InspectMaskArtifactArgs",
    "LiftGeometryPayload",
    "LiftGeometrySummary",
    "MaskInspectionPayload",
    "MaskInspectionResult",
    "MultiViewDecision",
    "MultiViewDecisionInput",
    "MultiViewDecisionPayload",
    "RecommendedCropPayload",
    "SeedLiftStatus",
    "SuggestedCrop",
    "SuggestAdditionalViewsArgs",
    "SuggestedView",
    "SuggestedViewPayload",
    "SuggestedViewsPayload",
    "SuggestedViewsResult",
    "fuse_accepted_masks",
    "inspect_mask_artifact",
    "suggest_additional_views",
]
