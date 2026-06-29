"""Mask inspection, multi-view suggestion, and fusion tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    StringConstraints,
    ValidationError,
)

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


class MaskInspectionPayload(TypedDict):
    """JSON-ready mask inspection result."""

    artifact_path: str | None
    overlay_paths: list[str]
    lifted_point_count: int
    status: str


class SuggestedViewPayload(TypedDict):
    """JSON-ready suggested view."""

    frame_id: str
    reason: str
    rank: int


class SuggestedViewsPayload(TypedDict):
    """JSON-ready suggested view result."""

    seed_fragment_id: str
    views: list[SuggestedViewPayload]


class AcceptedFragmentPayload(TypedDict):
    """JSON-ready accepted mask fragment."""

    fragment_id: str
    frame_id: str
    point_count: int
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


class SuggestAdditionalViewsArgs(BaseModel):
    """Arguments for suggesting extra views after accepting a 3D seed."""

    model_config = ConfigDict(extra="forbid")

    seed_fragment_id: SafePathComponentText
    accepted_frame_id: SafePathComponentText
    candidate_frame_ids: tuple[SafePathComponentText, ...] = ()
    k: int = Field(default=4, ge=1, le=8, strict=True)


class AcceptedFragmentInput(BaseModel):
    """One accepted lifted fragment to fuse."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: SafePathComponentText
    frame_id: SafePathComponentText
    mask_npz_path: FilePath
    mask_ply_path: FilePath
    approval_actions: tuple[ApprovalAction, ...] = Field(min_length=1)
    review_artifacts: AcceptedFragmentReviewArtifactsInput


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

    def to_domain(self) -> MultiViewDecision:
        """Return an immutable domain value for persisted fusion provenance."""
        return MultiViewDecision(
            seed_fragment_id=self.seed_fragment_id,
            action=self.action,
            reason=self.reason,
            suggested_frame_ids=self.suggested_frame_ids,
        )


class FuseAcceptedMasksArgs(BaseModel):
    """Arguments for fusing accepted 3D mask fragments."""

    model_config = ConfigDict(extra="forbid")

    fragments: tuple[AcceptedFragmentInput, ...] = Field(min_length=1)
    multi_view_decision: MultiViewDecisionInput


@dataclass(frozen=True)
class MaskInspectionResult:
    """Artifact inspection result for agent review."""

    artifact_path: Path | None
    overlay_paths: tuple[Path, ...]
    lifted_point_count: int
    status: str

    def __post_init__(self) -> None:
        """Validate domain invariants for one inspection result."""
        _validate_non_negative_int("lifted_point_count", self.lifted_point_count)
        _validate_non_empty_text("status", self.status)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready inspection result."""
        return {
            "artifact_path": (
                str(self.artifact_path) if self.artifact_path is not None else None
            ),
            "overlay_paths": [str(path) for path in self.overlay_paths],
            "lifted_point_count": self.lifted_point_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class SuggestedView:
    """One additional view suggested from an accepted 3D seed."""

    frame_id: str
    reason: str
    rank: int

    def __post_init__(self) -> None:
        """Validate domain invariants for one suggested view."""
        _validate_non_empty_text("frame_id", self.frame_id)
        _validate_non_empty_text("reason", self.reason)
        _validate_positive_int("rank", self.rank)

    def to_payload(self) -> SuggestedViewPayload:
        """Return the JSON-ready suggested view."""
        return {"frame_id": self.frame_id, "reason": self.reason, "rank": self.rank}


@dataclass(frozen=True)
class SuggestedViewsResult:
    """Additional views suggested for multi-view expansion."""

    seed_fragment_id: str
    views: tuple[SuggestedView, ...]

    def __post_init__(self) -> None:
        """Validate domain invariants for a suggested view set."""
        _validate_non_empty_text("seed_fragment_id", self.seed_fragment_id)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready suggested view set."""
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "views": [view.to_payload() for view in self.views],
        }


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

    def __post_init__(self) -> None:
        """Validate domain invariants for a multi-view decision."""
        _validate_non_empty_text("seed_fragment_id", self.seed_fragment_id)
        _validate_non_empty_text("reason", self.reason)
        for frame_id in self.suggested_frame_ids:
            _validate_non_empty_text("suggested_frame_id", frame_id)

    def to_payload(self) -> MultiViewDecisionPayload:
        """Return the JSON-ready multi-view decision."""
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "action": self.action.value,
            "reason": self.reason,
            "suggested_frame_ids": list(self.suggested_frame_ids),
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
    return MaskInspectionResult(
        artifact_path=args.artifact_path,
        overlay_paths=tuple(args.overlay_paths),
        lifted_point_count=lifted_point_count,
        status="valid",
    )


def suggest_additional_views(
    tool_scene: SceneFunc3dToolScene, args: SuggestAdditionalViewsArgs
) -> SuggestedViewsResult:
    """Suggest nearby views after a first accepted 3D fragment."""
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
    ranked_frame_ids = sorted(
        (
            frame_id
            for frame_id in candidate_frame_ids
            if frame_id != args.accepted_frame_id
        ),
        key=lambda frame_id: _view_rank_key(frame_id, args.accepted_frame_id),
    )
    views = tuple(
        SuggestedView(
            frame_id=frame_id,
            reason=f"nearest temporal neighbor to accepted_frame_id={args.accepted_frame_id}",
            rank=rank,
        )
        for rank, frame_id in enumerate(ranked_frame_ids[: args.k], start=1)
    )
    return SuggestedViewsResult(seed_fragment_id=args.seed_fragment_id, views=views)


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


def _view_rank_key(frame_id: str, accepted_frame_id: str) -> tuple[int, str]:
    frame_number = _frame_number(frame_id)
    accepted_number = _frame_number(accepted_frame_id)
    if frame_number is None or accepted_number is None:
        return (10**12, frame_id)
    return (abs(frame_number - accepted_number), frame_id)


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
    "MaskInspectionPayload",
    "MaskInspectionResult",
    "MultiViewDecision",
    "MultiViewDecisionInput",
    "MultiViewDecisionPayload",
    "SuggestAdditionalViewsArgs",
    "SuggestedView",
    "SuggestedViewPayload",
    "SuggestedViewsPayload",
    "SuggestedViewsResult",
    "fuse_accepted_masks",
    "inspect_mask_artifact",
    "suggest_additional_views",
]
