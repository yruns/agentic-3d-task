"""SceneFunc3D mask-generation runner for single samples and batches."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, Literal, NoReturn, TypeAlias, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.types import StringConstraints

from ..config import CodexAgentConfig
from ..errors import (
    CodexAgentError,
    CodexResponseError,
    CodexTurnError,
    SceneFunc3dDataError,
)
from ..json_extraction import extract_json_object
from ..models import CodexTurnMetadata, CodexTurnRequest
from ..tasks.base import CodexExecutor
from .backends.config import HttpHeader, load_backend_settings
from .evaluation.payloads import SceneFunc3dScorePayload, score_to_payload
from .evaluation.scorer import SceneFunc3dScore, score_result_file
from .failure_artifacts import (
    SceneFunc3dFailureArtifact,
    SceneFunc3dFailureStage,
    SceneFunc3dFailureTurnPayload,
    write_failure_artifact,
)
from .final_mask_artifacts import (
    FinalMaskAcceptedFragment,
    FinalMaskArtifactDocument,
    FinalMaskMultiViewAction,
    FinalMaskMultiViewDecision,
    ValidatedFinalMaskArtifact,
    validate_final_mask_artifact,
)
from .playbook import SCENEFUNC3D_TOOL_NAMES, SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, list_sample_ids, load_sample, scene_dir_for
from .servers.schemas import HealthResponse
from .tool_context import SceneFunc3dToolContext, write_tool_context
from .tools.mask_artifacts import (
    SceneFunc3dCompletedRunSummary,
    SceneFunc3dRunCompletionEvent,
    SceneFunc3dRunMultiViewDecisionPayload,
    append_run_completion_event,
    artifact_paths_for,
    write_completed_run_summary,
)
from .tools.scene_context import SceneFunc3dToolScene

TASK_NAME = "scenefunc3d_mask_generation"
DEFAULT_TOOL_CLI_MODULE = "codex_agent.scenefunc3d.tools"
DEFAULT_TOOL_CONTEXT_FILENAME = "tool_context.json"
SCENEFUNC3D_ALLOWED_TOOL_NAMES = SCENEFUNC3D_TOOL_NAMES

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_POINT_MATCH_ABS_TOLERANCE_PX = 1e-6


class SceneFunc3dMaskOutcomePayload(TypedDict):
    """JSON-ready payload for a completed SceneFunc3D mask run."""

    mask_artifact_path: str
    mask_npz_path: str
    mask_ply_path: str
    selected_frame_ids: list[str]
    accepted_fragment_ids: list[str]
    confidence: float
    uncertainties: list[str]


class CodexTurnMetadataPayload(TypedDict):
    """JSON-ready metadata payload for the Codex turn that produced a result."""

    turn_id: str | None
    status: str | None
    duration_ms: int | None
    usage: object | None
    input_tokens: int | None
    cached_input_tokens: int | None
    reasoning_summary: str | None
    run_home: str | None
    attempts: list[object]


class SceneFunc3dRunResultPayload(TypedDict):
    """JSON-ready durable result for one SceneFunc3D sample run."""

    task_name: str
    sample_id: str
    outcome: SceneFunc3dMaskOutcomePayload
    artifact: SceneFunc3dRunArtifactPayload
    turn: CodexTurnMetadataPayload


class SceneFunc3dRunArtifactPayload(TypedDict):
    """JSON-ready validated artifact summary for one SceneFunc3D result."""

    accepted_frame_ids: list[str]
    accepted_fragment_ids: list[str]
    final_point_count: int
    multi_view_decision: SceneFunc3dRunMultiViewDecisionPayload


class SceneFunc3dCliRunPayload(TypedDict):
    """Default CLI payload after running one SceneFunc3D sample."""

    result_path: str


class SceneFunc3dCliRunAndScorePayload(TypedDict):
    """CLI payload after running and scoring one SceneFunc3D sample."""

    result_path: str
    score: SceneFunc3dScorePayload


class SceneFunc3dBatchMeanMetricsPayload(TypedDict):
    """JSON-ready aggregate metrics over a SceneFunc3D batch."""

    iou: float
    precision: float
    recall: float
    f1: float


class SceneFunc3dBatchSampleResultPayload(TypedDict):
    """JSON-ready result row for one SceneFunc3D batch sample."""

    sample_id: str
    status: str
    failure_stage: str
    result_path: str | None
    failure_path: str | None
    score: SceneFunc3dScorePayload | None
    error: str


class SceneFunc3dBatchRunSummaryPayload(TypedDict):
    """JSON-ready summary for a SceneFunc3D batch run."""

    task_name: str
    scoring_enabled: bool
    sample_count: int
    completed_count: int
    failed_count: int
    scored_count: int
    mean_metrics: SceneFunc3dBatchMeanMetricsPayload | None
    results: list[SceneFunc3dBatchSampleResultPayload]


class SceneFunc3dCliBatchRunPayload(TypedDict):
    """CLI payload after running a SceneFunc3D batch."""

    summary_path: str
    summary: SceneFunc3dBatchRunSummaryPayload


SceneFunc3dBatchSampleStatus: TypeAlias = Literal["completed", "failed"]
SceneFunc3dBatchFailureStage: TypeAlias = Literal["", "run", "score"]


class SceneFunc3dMaskDecision(BaseModel):
    """Strict final JSON contract for one SceneFunc3D mask-generation turn."""

    model_config = ConfigDict(extra="forbid")

    mask_artifact_path: NonEmptyString = Field(
        description=(
            "Final mask artifact JSON path returned by "
            "fuse_accepted_masks.mask_artifact_path. Do not use fragment review "
            "artifacts such as lift_overlay.txt."
        )
    )
    mask_npz_path: NonEmptyString = Field(
        description=(
            "Final fused mask NPZ path returned by fuse_accepted_masks.mask_npz_path. "
            "Do not use an individual fragment mask_data.npz."
        )
    )
    mask_ply_path: NonEmptyString = Field(
        description=(
            "Final fused mask PLY path returned by fuse_accepted_masks.mask_ply_path. "
            "Do not use an individual fragment lifted_points.ply."
        )
    )
    selected_frame_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    accepted_fragment_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: tuple[str, ...]


class _FuseToolResult(BaseModel):
    """Strict result payload for a successful ``fuse_accepted_masks`` event."""

    model_config = ConfigDict(extra="forbid")

    accepted_frame_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    accepted_fragments: tuple[FinalMaskAcceptedFragment, ...] = Field(min_length=1)
    multi_view_decision: FinalMaskMultiViewDecision
    mask_artifact_path: NonEmptyString
    mask_npz_path: NonEmptyString
    mask_ply_path: NonEmptyString

    def model_post_init(self, __context: object) -> None:
        """Validate the fused-result contract shared with final artifacts."""
        fragment_frame_ids = _accepted_frame_ids_from_fuse_fragments(
            self.accepted_fragments
        )
        if self.accepted_frame_ids != fragment_frame_ids:
            raise ValueError(
                "accepted_frame_ids must match accepted_fragments frame_id "
                "provenance: "
                f"accepted_frame_ids={self.accepted_frame_ids}; "
                f"fragment_frame_ids={fragment_frame_ids}"
            )
        FinalMaskArtifactDocument.model_validate(
            {
                "accepted_frame_ids": self.accepted_frame_ids,
                "accepted_fragments": self.accepted_fragments,
                "multi_view_decision": self.multi_view_decision,
                "mask_npz_path": self.mask_npz_path,
                "mask_ply_path": self.mask_ply_path,
            }
        )


class _FuseToolEvent(BaseModel):
    """Strict JSONL event for a completed ``fuse_accepted_masks`` call."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["tool_completed"]
    tool_name: Literal["fuse_accepted_masks"]
    status: Literal["success"]
    args: Mapping[str, object]
    result: _FuseToolResult
    error: str


class _ToolPointAliasPayload(TypedDict):
    """Strict point payload built from compact tool-event aliases."""

    x_px: float
    y_px: float
    source: str
    label: str


class _ToolPointPayload(BaseModel):
    """One point prompt exchanged between Molmo and SAM."""

    model_config = ConfigDict(extra="ignore")

    x_px: float = Field(ge=0.0, allow_inf_nan=False)
    y_px: float = Field(ge=0.0, allow_inf_nan=False)
    source: str = ""
    label: str = ""


class _EvidenceImageFrameResult(BaseModel):
    """One evidence image returned by ``view_frame`` or ``view_crop``."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    image_path: NonEmptyString


class _EvidenceImageToolResult(BaseModel):
    """Fields needed to prove one Molmo call used a viewed evidence image."""

    model_config = ConfigDict(extra="ignore")

    frames: tuple[_EvidenceImageFrameResult, ...] = Field(min_length=1)


class _MolmoPointToolArgs(BaseModel):
    """Fields needed to prove one Molmo call used a prior evidence image."""

    model_config = ConfigDict(extra="ignore")

    image_path: NonEmptyString


class _MolmoPointToolResult(BaseModel):
    """Fields needed to prove one accepted fragment's Molmo provenance."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    points: tuple[_ToolPointPayload, ...] = Field(min_length=1)
    raw_text_path: NonEmptyString
    overlay_path: NonEmptyString


class _SamCandidateToolResult(BaseModel):
    """Fields needed to prove one accepted fragment's SAM candidate provenance."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: NonEmptyString
    mask_npz_path: NonEmptyString
    overlay_path: NonEmptyString


class _SamMaskToolResult(BaseModel):
    """Fields needed to prove one accepted fragment's SAM provenance."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    candidates: tuple[_SamCandidateToolResult, ...] = Field(min_length=1)
    contact_sheet_path: NonEmptyString


class _SamMaskToolArgs(BaseModel):
    """Fields needed to prove one SAM call used Molmo point prompts."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    image_path: NonEmptyString
    points: tuple[_ToolPointPayload, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_point_aliases(cls, payload: object) -> object:
        """Accept the compact point aliases supported by the SAM tool."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        if "points" not in values and "point" in values:
            values["points"] = (values.pop("point"),)
        if "points" not in values and "point_xy" in values:
            values["points"] = (
                _tool_point_alias_payload_from_xy(
                    values.pop("point_xy"),
                    raw_source=values.get("point_source"),
                    raw_label=values.get("point_label"),
                ),
            )
        return values


class _LiftMaskToolResult(BaseModel):
    """Fields needed to prove one accepted fragment's 3D lift provenance."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    candidate_id: NonEmptyString
    mask_npz_path: NonEmptyString
    mask_ply_path: NonEmptyString
    overlay_path: NonEmptyString


class _LiftMaskToolArgs(BaseModel):
    """Fields needed to prove one lift call used the accepted SAM candidate."""

    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    candidate_id: NonEmptyString
    mask_path: NonEmptyString = Field(
        validation_alias=AliasChoices("mask_path", "mask_npz_path")
    )


class _InspectMaskToolArgs(BaseModel):
    """Fields needed to prove agent reviewed one lifted 3D mask artifact."""

    model_config = ConfigDict(extra="ignore")

    mask_npz_path: NonEmptyString
    mask_ply_path: NonEmptyString
    overlay_paths: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_overlay_path_alias(cls, payload: object) -> object:
        """Accept the single-overlay alias supported by the inspect tool."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        overlay_path = values.get("overlay_path")
        if overlay_path is None:
            overlay_path = values.get("lift_overlay_path")
        if "overlay_paths" not in values and overlay_path is not None:
            values["overlay_paths"] = (overlay_path,)
        return values


class _InspectMaskToolResult(BaseModel):
    """Fields needed to prove a 3D mask inspection returned geometry summary."""

    model_config = ConfigDict(extra="ignore")

    overlay_paths: tuple[NonEmptyString, ...] = ()
    lifted_point_count: int = Field(ge=1)
    bbox_min_xyz: tuple[float, float, float]
    bbox_max_xyz: tuple[float, float, float]
    bbox_extent_xyz: tuple[float, float, float]
    max_extent_meters: float = Field(ge=0.0, allow_inf_nan=False)
    status: NonEmptyString


class _SuggestAdditionalViewsToolArgs(BaseModel):
    """Fields needed to prove multi-view suggestions used the inspected seed."""

    model_config = ConfigDict(extra="ignore")

    seed_fragment_id: NonEmptyString
    accepted_frame_id: NonEmptyString
    seed_mask_npz_path: NonEmptyString
    seed_mask_ply_path: NonEmptyString
    seed_lift_overlay_path: NonEmptyString

    @model_validator(mode="before")
    @classmethod
    def normalize_seed_frame_aliases(cls, payload: object) -> object:
        """Mirror the tool args model so runner validation accepts real events."""
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
        return values


def _infer_frame_id_from_seed_fragment(seed_fragment_id: object) -> str | None:
    if not isinstance(seed_fragment_id, str):
        return None
    frame_id, separator, _candidate_id = seed_fragment_id.partition("_")
    if not separator or not frame_id:
        return None
    return frame_id


@dataclass(frozen=True)
class _ParsedFuseToolEvent:
    """A validated fuse event with source location for diagnostics."""

    line_number: int
    event: _FuseToolEvent


@dataclass(frozen=True)
class _ParsedMolmoPointToolEvent:
    """A validated Molmo point event with source location for diagnostics."""

    line_number: int
    args: _MolmoPointToolArgs
    result: _MolmoPointToolResult


@dataclass(frozen=True)
class _ParsedEvidenceImageToolEvent:
    """A validated evidence-image event with source location."""

    line_number: int
    tool_name: str
    result: _EvidenceImageToolResult


@dataclass(frozen=True)
class _ParsedSamMaskToolEvent:
    """A validated SAM mask event with source location for diagnostics."""

    line_number: int
    args: _SamMaskToolArgs
    result: _SamMaskToolResult


@dataclass(frozen=True)
class _MatchedSamMaskToolEvent:
    """A matched SAM event and the accepted candidate's 2D mask path."""

    line_number: int
    candidate_mask_npz_path: str


@dataclass(frozen=True)
class _ParsedLiftMaskToolEvent:
    """A validated 3D lift event with source location for diagnostics."""

    line_number: int
    args: _LiftMaskToolArgs
    result: _LiftMaskToolResult


@dataclass(frozen=True)
class _ParsedInspectMaskToolEvent:
    """A validated 3D mask inspection event with source location."""

    line_number: int
    args: _InspectMaskToolArgs
    result: _InspectMaskToolResult


@dataclass(frozen=True)
class _ValidatedStandardFragmentToolChain:
    """Validated upstream event line numbers for one accepted standard fragment."""

    fragment_id: str
    frame_id: str
    lift_line_number: int
    inspect_line_number: int
    mask_npz_path: Path
    mask_ply_path: Path
    lift_overlay_path: Path


@dataclass(frozen=True)
class _SuccessfulToolResultEvent:
    """A successful generic tool event result and source location."""

    line_number: int
    tool_name: str
    args: Mapping[object, object]
    result: Mapping[object, object]


@dataclass(frozen=True)
class SceneFunc3dMaskOutcome:
    """Parsed SceneFunc3D mask-generation output."""

    mask_artifact_path: Path
    mask_npz_path: Path
    mask_ply_path: Path
    selected_frame_ids: tuple[str, ...]
    accepted_fragment_ids: tuple[str, ...]
    confidence: float
    uncertainties: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_artifact_path", Path(self.mask_artifact_path))
        object.__setattr__(self, "mask_npz_path", Path(self.mask_npz_path))
        object.__setattr__(self, "mask_ply_path", Path(self.mask_ply_path))
        object.__setattr__(
            self,
            "selected_frame_ids",
            _require_non_empty_string_tuple(
                self.selected_frame_ids,
                field_name="selected_frame_ids",
            ),
        )
        object.__setattr__(
            self,
            "accepted_fragment_ids",
            _require_non_empty_string_tuple(
                self.accepted_fragment_ids,
                field_name="accepted_fragment_ids",
            ),
        )
        object.__setattr__(
            self,
            "uncertainties",
            tuple(str(item) for item in self.uncertainties),
        )
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_payload(self) -> SceneFunc3dMaskOutcomePayload:
        """Return a JSON-serializable payload for ``result.json``."""
        return {
            "mask_artifact_path": str(self.mask_artifact_path),
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "selected_frame_ids": list(self.selected_frame_ids),
            "accepted_fragment_ids": list(self.accepted_fragment_ids),
            "confidence": self.confidence,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class SceneFunc3dRunnerConfig:
    """Configuration for a SceneFunc3D single-sample run."""

    dataset_root: Path
    output_dir: Path
    backend_config_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "backend_config_path", Path(self.backend_config_path))


@dataclass(frozen=True)
class SceneFunc3dBatchSampleResult:
    """The run and score status for one SceneFunc3D batch sample."""

    sample_id: str
    status: SceneFunc3dBatchSampleStatus
    failure_stage: SceneFunc3dBatchFailureStage
    result_path: Path | None
    score: SceneFunc3dScore | None
    failure_path: Path | None = None
    error: str = ""

    def to_payload(self) -> SceneFunc3dBatchSampleResultPayload:
        """Return a JSON-serializable batch result row."""
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "result_path": (
                str(self.result_path) if self.result_path is not None else None
            ),
            "failure_path": (
                str(self.failure_path) if self.failure_path is not None else None
            ),
            "score": score_to_payload(self.score) if self.score is not None else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class SceneFunc3dBatchRunSummary:
    """Aggregate metrics over one SceneFunc3D batch run."""

    scoring_enabled: bool
    sample_count: int
    completed_count: int
    failed_count: int
    scored_count: int
    mean_iou: float | None
    mean_precision: float | None
    mean_recall: float | None
    mean_f1: float | None
    results: tuple[SceneFunc3dBatchSampleResult, ...]

    def to_payload(self) -> SceneFunc3dBatchRunSummaryPayload:
        """Return the JSON-ready representation stored in ``evaluation_summary``."""
        return {
            "task_name": TASK_NAME,
            "scoring_enabled": self.scoring_enabled,
            "sample_count": self.sample_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "scored_count": self.scored_count,
            "mean_metrics": self._mean_metrics_payload(),
            "results": [result.to_payload() for result in self.results],
        }

    def _mean_metrics_payload(self) -> SceneFunc3dBatchMeanMetricsPayload | None:
        if (
            self.mean_iou is None
            or self.mean_precision is None
            or self.mean_recall is None
            or self.mean_f1 is None
        ):
            return None
        return {
            "iou": self.mean_iou,
            "precision": self.mean_precision,
            "recall": self.mean_recall,
            "f1": self.mean_f1,
        }


class SceneFunc3dMaskTask:
    """A Codex task for producing one SceneFunc3D 3D mask artifact."""

    def __init__(
        self,
        *,
        sample: SceneFunc3dSample,
        scene_root: Path,
        output_dir: Path,
        backend_config_path: Path,
        tool_context_path: Path,
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
    ) -> None:
        self.sample = sample
        self.scene_root = Path(scene_root)
        self.output_dir = Path(output_dir)
        self.backend_config_path = Path(backend_config_path)
        self.tool_context_path = Path(tool_context_path)
        self.tool_cli_module = tool_cli_module

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        self._require_tool_context_file()
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=SceneFunc3dMaskDecision.model_json_schema(),
            skills=(),
            image_paths=(),
        )

    def _require_tool_context_file(self) -> None:
        if self.tool_context_path.is_file():
            return
        raise SceneFunc3dDataError(
            "SceneFunc3D tool context is missing before prompt generation: "
            f"path={self.tool_context_path}"
        )

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except CodexResponseError:
            return False
        return True

    def parse_response(self, response_text: str) -> SceneFunc3dMaskOutcome:
        decision = self._parse_decision(response_text)
        mask_artifact_path = self._validate_artifact_path(
            decision.mask_artifact_path,
            field_name="mask_artifact_path",
            suffix=".json",
        )
        mask_npz_path = self._validate_artifact_path(
            decision.mask_npz_path,
            field_name="mask_npz_path",
            suffix=".npz",
        )
        mask_ply_path = self._validate_artifact_path(
            decision.mask_ply_path,
            field_name="mask_ply_path",
            suffix=".ply",
        )
        selected_frame_ids = tuple(decision.selected_frame_ids)
        accepted_fragment_ids = tuple(decision.accepted_fragment_ids)
        self._validate_selected_frame_ids(selected_frame_ids)
        validate_final_mask_artifact(
            artifact_path=mask_artifact_path,
            mask_npz_path=mask_npz_path,
            mask_ply_path=mask_ply_path,
            selected_frame_ids=selected_frame_ids,
            accepted_fragment_ids=accepted_fragment_ids,
        )
        return SceneFunc3dMaskOutcome(
            mask_artifact_path=mask_artifact_path,
            mask_npz_path=mask_npz_path,
            mask_ply_path=mask_ply_path,
            selected_frame_ids=selected_frame_ids,
            accepted_fragment_ids=accepted_fragment_ids,
            confidence=decision.confidence,
            uncertainties=tuple(decision.uncertainties),
        )

    def validate_outcome(
        self, outcome: SceneFunc3dMaskOutcome
    ) -> SceneFunc3dMaskOutcome:
        """Revalidate a runtime outcome before writing durable result metadata."""
        validated_outcome = self.parse_response(
            json.dumps(outcome.to_payload(), ensure_ascii=False)
        )
        _validate_outcome_against_fuse_tool_event(
            validated_outcome, self.output_dir / "events.jsonl"
        )
        return validated_outcome

    def _parse_decision(self, response_text: str) -> SceneFunc3dMaskDecision:
        payload = extract_json_object(response_text)
        try:
            return SceneFunc3dMaskDecision.model_validate(payload)
        except ValidationError as exc:
            raise CodexResponseError(
                f"Codex response does not match the SceneFunc3D mask schema: {exc}"
            ) from exc

    def _validate_artifact_path(
        self,
        raw_path: str,
        *,
        field_name: str,
        suffix: str,
    ) -> Path:
        output_root = self.output_dir.expanduser().resolve()
        artifact_path = Path(raw_path).expanduser().resolve()
        try:
            artifact_path.relative_to(output_root)
        except ValueError as exc:
            raise CodexResponseError(
                f"{field_name} must be inside the sample output directory: "
                f"path={artifact_path}; output_dir={output_root}"
            ) from exc
        if artifact_path.suffix != suffix:
            if field_name == "mask_artifact_path":
                raise CodexResponseError(
                    "mask_artifact_path must be the .json path returned by "
                    "fuse_accepted_masks.mask_artifact_path; do not use review "
                    "artifacts such as lift_overlay.txt: "
                    f"path={artifact_path}"
                )
            raise CodexResponseError(
                f"{field_name} must have suffix {suffix!r}: path={artifact_path}"
            )
        if not artifact_path.is_file():
            raise CodexResponseError(f"{field_name} does not exist: {artifact_path}")
        return artifact_path

    def _validate_selected_frame_ids(self, selected_frame_ids: tuple[str, ...]) -> None:
        try:
            tool_scene = SceneFunc3dToolScene.load(self.scene_root)
        except SceneFunc3dDataError as exc:
            raise CodexResponseError(
                f"could not validate selected_frame_ids against scene_root: "
                f"{self.scene_root}"
            ) from exc
        available_frame_ids = set(tool_scene.rgb_frame_ids)
        missing_frame_ids = tuple(
            frame_id
            for frame_id in selected_frame_ids
            if frame_id not in available_frame_ids
        )
        if missing_frame_ids:
            raise CodexResponseError(
                "selected_frame_ids must exist in the SceneFunc3D scene: "
                f"missing={missing_frame_ids}; scene_root={self.scene_root}"
            )

    def _build_prompt(self) -> str:
        schema = SceneFunc3dMaskDecision.model_json_schema()
        return (
            build_prompt(self.sample) + "\n\nRuntime paths:\n"
            f"- tool_context: {self.tool_context_path}\n"
            "\nHow to run a tool (in the shell):\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--context {self.tool_context_path} --args '<json>'\n"
            "- For molmo_point, sam_mask, lift_mask_to_3d, "
            "inspect_mask_artifact, suggest_additional_views, and "
            "fuse_accepted_masks, set yield_time_ms=30000 on the shell call so "
            "the command can finish and return JSON. If the shell reports "
            "Process running with session ID, wait on that same session until "
            "it exits and returns JSON; do not start another SceneFunc3D tool "
            "while a previous tool command is still running.\n"
            "- Tool outputs are JSON. Open any returned image_path with "
            "view_image before relying on visual evidence.\n"
            "- Use the Molmo point and SAM candidates approval gates in the "
            "playbook before accepting mask fragments.\n"
            "\n" + _hard_limits_section() + "\n"
            "\nFinal JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        )


def build_prompt(sample: SceneFunc3dSample) -> str:
    """Build the prompt prefix for one SceneFunc3D sample."""
    context_json = json.dumps(sample.agent_context, ensure_ascii=False, indent=2)
    return (
        f"{SCENEFUNC3D_TOOLS_PLAYBOOK}\n\n"
        "Task context:\n"
        f"{context_json}\n\n"
        "Generate a SceneFunc3D 3D mask artifact for this task."
    )


def _hard_limits_section() -> str:
    """Return prompt limits that keep the turn focused on SceneFunc3D tools."""
    return (
        "Hard limits:\n"
        "- You are NOT exploring or editing a codebase. Everything needed for "
        "this single SceneFunc3D case is in this message and the CLI tools "
        "above.\n"
        "- Do NOT read, cat, sed, head, grep, rg, or open any SKILL.md, "
        "AGENTS.md, README, docs, or source file, and do NOT list or search the "
        "repository.\n"
        "- Use ONLY these SceneFunc3D tools plus view_image: "
        f"{', '.join(SCENEFUNC3D_ALLOWED_TOOL_NAMES)}.\n"
        "- Never re-run a tool with identical arguments and never re-view an "
        "image you have already seen; if a result is empty or errors, change "
        "frame, crop, prompt, or mask candidate instead."
    )


def load_runner_sample(
    config: SceneFunc3dRunnerConfig, sample_id: str
) -> SceneFunc3dSample:
    """Load the sample a future runtime turn will solve."""
    _ = config.output_dir
    _ = config.backend_config_path
    return load_sample(config.dataset_root, sample_id)


def check_sidecar_health(backend_config_path: Path) -> None:
    """Verify configured Molmo and SAM sidecars expose healthy ``/health``."""
    settings = load_backend_settings(Path(backend_config_path))
    molmo_health = _fetch_sidecar_health(
        settings.molmo_url,
        service_name="molmo",
        timeout_seconds=settings.request_timeout_seconds,
        request_headers=settings.request_headers,
    )
    sam_health = _fetch_sidecar_health(
        settings.sam_url,
        service_name="sam",
        timeout_seconds=settings.request_timeout_seconds,
        request_headers=settings.request_headers,
    )
    _require_model_loaded(molmo_health, service_name="molmo")
    _require_model_loaded(sam_health, service_name="sam")


def run_single_sample(
    config: SceneFunc3dRunnerConfig,
    *,
    sample_id: str,
    executor: CodexExecutor,
    check_sidecars: bool = True,
) -> Path:
    """Run one SceneFunc3D sample, write ``result.json``, and return its path."""
    if check_sidecars:
        check_sidecar_health(config.backend_config_path)

    sample = load_runner_sample(config, sample_id)
    artifact_paths = artifact_paths_for(
        config.output_dir, sample.visit_id, sample.desc_id
    )
    sample_output_dir = artifact_paths.root
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_run_events(artifact_paths.events_jsonl)
    tool_context_path = write_tool_context(
        sample_output_dir / DEFAULT_TOOL_CONTEXT_FILENAME,
        SceneFunc3dToolContext(
            sample_id=sample.sample_id,
            scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
            backend_config_path=config.backend_config_path,
            out_dir=sample_output_dir,
        ),
    )
    task = SceneFunc3dMaskTask(
        sample=sample,
        scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
        output_dir=sample_output_dir,
        backend_config_path=config.backend_config_path,
        tool_context_path=tool_context_path,
    )
    turn_metadata: CodexTurnMetadata | None = None
    try:
        result = executor.execute(task)
        turn_metadata = result.turn.metadata
        validated_outcome = task.validate_outcome(result.outcome)
        validated_artifact = _validate_outcome_artifact(validated_outcome)
    except CodexTurnError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="codex_turn",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    except CodexResponseError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="response_validation",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    except CodexAgentError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="run",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    result_path = sample_output_dir / "result.json"
    result_payload: SceneFunc3dRunResultPayload = {
        "task_name": result.task_name,
        "sample_id": sample_id,
        "outcome": validated_outcome.to_payload(),
        "artifact": _artifact_payload(validated_artifact),
        "turn": _metadata_payload(result.turn.metadata),
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = write_completed_run_summary(
        artifact_paths,
        _completed_run_summary(
            sample=sample,
            outcome=validated_outcome,
            artifact=validated_artifact,
        ),
    )
    append_run_completion_event(
        artifact_paths,
        SceneFunc3dRunCompletionEvent(
            sample_id=sample_id,
            result_path=result_path,
            summary_path=summary_path,
            mask_artifact_path=validated_outcome.mask_artifact_path,
        ),
    )
    return result_path


def run_samples(
    config: SceneFunc3dRunnerConfig,
    *,
    sample_ids: Sequence[str],
    executor: CodexExecutor,
    check_sidecars: bool = True,
    score: bool = True,
    continue_on_error: bool = True,
) -> SceneFunc3dBatchRunSummary:
    """Run and optionally score multiple SceneFunc3D samples sequentially."""
    normalized_sample_ids = _normalized_batch_sample_ids(sample_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if check_sidecars:
        check_sidecar_health(config.backend_config_path)

    results: list[SceneFunc3dBatchSampleResult] = []
    for sample_id in normalized_sample_ids:
        try:
            result_path = run_single_sample(
                config,
                sample_id=sample_id,
                executor=executor,
                check_sidecars=False,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failure_path = _existing_failure_artifact_path(config, sample_id)
            results.append(
                SceneFunc3dBatchSampleResult(
                    sample_id=sample_id,
                    status="failed",
                    failure_stage="run",
                    result_path=None,
                    score=None,
                    failure_path=failure_path,
                    error=_format_batch_error(exc),
                )
            )
            continue

        sample_score: SceneFunc3dScore | None = None
        if score:
            try:
                sample_score = score_result_file(
                    data_root=config.dataset_root, result_path=result_path
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(
                    SceneFunc3dBatchSampleResult(
                        sample_id=sample_id,
                        status="failed",
                        failure_stage="score",
                        result_path=result_path,
                        score=None,
                        error=_format_batch_error(exc),
                    )
                )
                continue
        results.append(
            SceneFunc3dBatchSampleResult(
                sample_id=sample_id,
                status="completed",
                failure_stage="",
                result_path=result_path,
                score=sample_score,
            )
        )

    summary = _summarize_batch_results(tuple(results), scoring_enabled=score)
    _write_batch_summary(config.output_dir / "evaluation_summary.json", summary)
    return summary


def _normalized_batch_sample_ids(sample_ids: Sequence[str]) -> tuple[str, ...]:
    normalized_sample_ids = tuple(sample_id.strip() for sample_id in sample_ids)
    if not normalized_sample_ids:
        raise ValueError("sample_ids must not be empty")
    if any(not sample_id for sample_id in normalized_sample_ids):
        raise ValueError("sample_ids must not contain empty values")
    seen_sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    for sample_id in normalized_sample_ids:
        if sample_id in seen_sample_ids:
            duplicate_sample_ids.append(sample_id)
        seen_sample_ids.add(sample_id)
    if duplicate_sample_ids:
        raise ValueError(
            f"duplicate SceneFunc3D sample ids: {tuple(duplicate_sample_ids)}"
        )
    return normalized_sample_ids


def _write_sample_failure_artifact(
    *,
    sample: SceneFunc3dSample,
    failure_path: Path,
    failure_stage: SceneFunc3dFailureStage,
    exc: CodexAgentError,
    events_path: Path,
    tool_context_path: Path,
    turn_metadata: CodexTurnMetadata | None,
) -> Path:
    return write_failure_artifact(
        failure_path,
        SceneFunc3dFailureArtifact(
            task_name=TASK_NAME,
            sample_id=sample.sample_id,
            failure_stage=failure_stage,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            events_path=str(events_path),
            tool_context_path=str(tool_context_path),
            turn=_failure_turn_payload(turn_metadata),
        ),
    )


def _failure_turn_payload(
    metadata: CodexTurnMetadata | None,
) -> SceneFunc3dFailureTurnPayload:
    if metadata is None:
        return SceneFunc3dFailureTurnPayload()
    return SceneFunc3dFailureTurnPayload.model_validate(_metadata_payload(metadata))


def _existing_failure_artifact_path(
    config: SceneFunc3dRunnerConfig, sample_id: str
) -> Path | None:
    try:
        sample = load_runner_sample(config, sample_id)
    except CodexAgentError:
        return None
    failure_path = (
        artifact_paths_for(config.output_dir, sample.visit_id, sample.desc_id).root
        / "failure.json"
    )
    if failure_path.is_file():
        return failure_path
    return None


def _summarize_batch_results(
    results: tuple[SceneFunc3dBatchSampleResult, ...],
    *,
    scoring_enabled: bool,
) -> SceneFunc3dBatchRunSummary:
    sample_count = len(results)
    completed_count = sum(1 for result in results if result.status == "completed")
    failed_count = sample_count - completed_count
    scores = tuple(result.score for result in results if result.score is not None)
    return SceneFunc3dBatchRunSummary(
        scoring_enabled=scoring_enabled,
        sample_count=sample_count,
        completed_count=completed_count,
        failed_count=failed_count,
        scored_count=len(scores),
        mean_iou=_mean_score_iou(scores),
        mean_precision=_mean_score_precision(scores),
        mean_recall=_mean_score_recall(scores),
        mean_f1=_mean_score_f1(scores),
        results=results,
    )


def _write_batch_summary(
    summary_path: Path, summary: SceneFunc3dBatchRunSummary
) -> Path:
    summary_path.write_text(
        json.dumps(summary.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _mean_score_iou(scores: tuple[SceneFunc3dScore, ...]) -> float | None:
    if not scores:
        return None
    return sum(score.metrics.iou for score in scores) / len(scores)


def _mean_score_precision(scores: tuple[SceneFunc3dScore, ...]) -> float | None:
    if not scores:
        return None
    return sum(score.metrics.precision for score in scores) / len(scores)


def _mean_score_recall(scores: tuple[SceneFunc3dScore, ...]) -> float | None:
    if not scores:
        return None
    return sum(score.metrics.recall for score in scores) / len(scores)


def _mean_score_f1(scores: tuple[SceneFunc3dScore, ...]) -> float | None:
    if not scores:
        return None
    return sum(score.metrics.f1 for score in scores) / len(scores)


def _format_batch_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:480]}"


def _initialize_run_events(events_path: Path) -> None:
    """Start a fresh per-run tool event log for one sample output directory."""
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
    except OSError as exc:
        raise SceneFunc3dDataError(
            "could not initialize SceneFunc3D tool event log: "
            f"events_path={events_path}; error_type={exc.__class__.__name__}"
        ) from exc


def _validate_outcome_artifact(
    outcome: SceneFunc3dMaskOutcome,
) -> ValidatedFinalMaskArtifact:
    return validate_final_mask_artifact(
        artifact_path=outcome.mask_artifact_path,
        mask_npz_path=outcome.mask_npz_path,
        mask_ply_path=outcome.mask_ply_path,
        selected_frame_ids=outcome.selected_frame_ids,
        accepted_fragment_ids=outcome.accepted_fragment_ids,
    )


def _validate_outcome_against_fuse_tool_event(
    outcome: SceneFunc3dMaskOutcome, events_path: Path
) -> None:
    """Ensure the final mask outcome was produced by this run's fuse tool call."""
    artifact_document = _load_mask_artifact_document_for_provenance(
        outcome.mask_artifact_path
    )
    matching_fuse_event = _last_matching_fuse_tool_event(
        outcome,
        artifact_document=artifact_document,
        events_path=events_path,
    )
    if matching_fuse_event is not None:
        standard_fragment_chains = _validate_standard_fragment_upstream_tool_events(
            artifact_document,
            artifact_root=_infer_output_root_from_outcome(outcome),
            events_path=events_path,
            fuse_line_number=matching_fuse_event.line_number,
        )
        seed_standard_chain = _seed_standard_fragment_tool_chain(
            artifact_document.multi_view_decision,
            standard_fragment_chains=standard_fragment_chains,
        )
        if seed_standard_chain is not None:
            _reject_seed_suggest_events_before_inspection(
                seed_standard_chain,
                events_path=events_path,
            )
        _validate_multiview_decision_against_tool_events(
            artifact_document.multi_view_decision,
            events_path,
            before_line_number=matching_fuse_event.line_number,
            after_line_number=(
                seed_standard_chain.inspect_line_number
                if seed_standard_chain is not None
                else 0
            ),
            seed_chain=seed_standard_chain,
        )
        return
    raise CodexResponseError(
        "final SceneFunc3D mask outcome must match a successful "
        "fuse_accepted_masks tool event from this run: "
        f"mask_artifact_path={outcome.mask_artifact_path}; "
        f"mask_npz_path={outcome.mask_npz_path}; "
        f"mask_ply_path={outcome.mask_ply_path}; "
        f"selected_frame_ids={outcome.selected_frame_ids}; "
        f"accepted_fragment_ids={outcome.accepted_fragment_ids}; "
        f"events_path={events_path}"
    )


def _last_matching_fuse_tool_event(
    outcome: SceneFunc3dMaskOutcome,
    *,
    artifact_document: FinalMaskArtifactDocument,
    events_path: Path,
) -> _ParsedFuseToolEvent | None:
    matching_event: _ParsedFuseToolEvent | None = None
    for parsed_event in _successful_fuse_tool_events(events_path):
        if _fuse_event_matches_outcome(
            parsed_event,
            outcome,
            artifact_document=artifact_document,
            events_path=events_path,
        ):
            matching_event = parsed_event
    return matching_event


def _infer_output_root_from_outcome(outcome: SceneFunc3dMaskOutcome) -> Path:
    resolved_artifact_path = outcome.mask_artifact_path.expanduser().resolve()
    if (
        resolved_artifact_path.name == "mask_artifact.json"
        and resolved_artifact_path.parent.name == "fused"
    ):
        return resolved_artifact_path.parent.parent
    return resolved_artifact_path.parent


def _load_mask_artifact_document_for_provenance(
    artifact_path: Path,
) -> FinalMaskArtifactDocument:
    try:
        artifact_text = artifact_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read mask_artifact_path while validating fuse provenance: "
            f"mask_artifact_path={artifact_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    try:
        artifact_payload: object = json.loads(artifact_text)
    except JSONDecodeError as exc:
        raise CodexResponseError(
            "mask_artifact_path is not valid JSON while validating fuse "
            f"provenance: mask_artifact_path={artifact_path}; error={exc}"
        ) from exc
    try:
        return FinalMaskArtifactDocument.model_validate(artifact_payload)
    except ValidationError as exc:
        raise CodexResponseError(
            "mask_artifact_path does not match final mask artifact schema while "
            f"validating fuse provenance: mask_artifact_path={artifact_path}; "
            f"error={exc}"
        ) from exc


def _successful_fuse_tool_events(events_path: Path) -> tuple[_ParsedFuseToolEvent, ...]:
    if not events_path.is_file():
        raise CodexResponseError(
            "SceneFunc3D tool event log is missing while validating final artifact "
            f"provenance: events_path={events_path}; "
            "tool_name=fuse_accepted_masks"
        )
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFunc3D tool event log while validating final "
            "artifact provenance: "
            f"events_path={events_path}; error_type={exc.__class__.__name__}"
        ) from exc
    events: list[_ParsedFuseToolEvent] = []
    for line_number, raw_line in enumerate(event_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        event = _load_event_payload(
            line, events_path=events_path, line_number=line_number
        )
        if event.get("tool_name") != "fuse_accepted_masks":
            continue
        if event.get("status") != "success":
            continue
        events.append(
            _parse_fuse_tool_event(
                event,
                events_path=events_path,
                line_number=line_number,
            )
        )
    return tuple(events)


def _parse_fuse_tool_event(
    event: Mapping[object, object], *, events_path: Path, line_number: int
) -> _ParsedFuseToolEvent:
    try:
        fuse_event = _FuseToolEvent.model_validate(event)
    except ValidationError as exc:
        raise CodexResponseError(
            "successful fuse_accepted_masks tool event does not match the expected "
            "tool_completed result schema: "
            f"events_path={events_path}; line={line_number}; error={exc}"
        ) from exc
    return _ParsedFuseToolEvent(line_number=line_number, event=fuse_event)


def _fuse_event_matches_outcome(
    parsed_event: _ParsedFuseToolEvent,
    outcome: SceneFunc3dMaskOutcome,
    *,
    artifact_document: FinalMaskArtifactDocument,
    events_path: Path,
) -> bool:
    event_result = parsed_event.event.result
    return (
        _event_path_matches(
            event_result.mask_artifact_path,
            outcome.mask_artifact_path,
            field_name="mask_artifact_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        )
        and _event_path_matches(
            event_result.mask_npz_path,
            outcome.mask_npz_path,
            field_name="mask_npz_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        )
        and _event_path_matches(
            event_result.mask_ply_path,
            outcome.mask_ply_path,
            field_name="mask_ply_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        )
        and event_result.accepted_frame_ids == outcome.selected_frame_ids
        and _fuse_fragment_ids(event_result.accepted_fragments)
        == outcome.accepted_fragment_ids
        and _fuse_event_matches_artifact_document(event_result, artifact_document)
    )


def _fuse_event_matches_artifact_document(
    event_result: _FuseToolResult,
    artifact_document: FinalMaskArtifactDocument,
) -> bool:
    return (
        event_result.accepted_frame_ids == artifact_document.accepted_frame_ids
        and event_result.accepted_fragments == artifact_document.accepted_fragments
        and event_result.multi_view_decision == artifact_document.multi_view_decision
        and event_result.mask_npz_path == artifact_document.mask_npz_path
        and event_result.mask_ply_path == artifact_document.mask_ply_path
    )


def _validate_standard_fragment_upstream_tool_events(
    artifact_document: FinalMaskArtifactDocument,
    *,
    artifact_root: Path,
    events_path: Path,
    fuse_line_number: int,
) -> tuple[_ValidatedStandardFragmentToolChain, ...]:
    evidence_image_events = _successful_evidence_image_tool_events(events_path)
    molmo_events = _successful_molmo_point_tool_events(events_path)
    sam_events = _successful_sam_mask_tool_events(events_path)
    lift_events = _successful_lift_mask_tool_events(events_path)
    inspect_events = _successful_inspect_mask_tool_events(events_path)
    validated_chains: list[_ValidatedStandardFragmentToolChain] = []
    for fragment in artifact_document.accepted_fragments:
        candidate_id = _standard_fragment_candidate_id(fragment)
        if candidate_id is None:
            continue
        validated_chains.append(
            _require_matching_standard_fragment_tool_chain(
                fragment,
                candidate_id=candidate_id,
                evidence_image_events=evidence_image_events,
                molmo_events=molmo_events,
                sam_events=sam_events,
                lift_events=lift_events,
                inspect_events=inspect_events,
                artifact_root=artifact_root,
                events_path=events_path,
                fuse_line_number=fuse_line_number,
            )
        )
    return tuple(validated_chains)


def _seed_standard_fragment_tool_chain(
    decision: FinalMaskMultiViewDecision,
    *,
    standard_fragment_chains: tuple[_ValidatedStandardFragmentToolChain, ...],
) -> _ValidatedStandardFragmentToolChain | None:
    for chain in standard_fragment_chains:
        if chain.fragment_id == decision.seed_fragment_id:
            return chain
    return None


def _reject_seed_suggest_events_before_inspection(
    seed_chain: _ValidatedStandardFragmentToolChain,
    *,
    events_path: Path,
) -> None:
    if not events_path.is_file():
        return
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFunc3D tool events while validating "
            "pre-inspection multi-view ordering: "
            f"events_path={events_path}; error_type={exc.__class__.__name__}"
        ) from exc
    for line_number, raw_line in enumerate(event_lines, start=1):
        if line_number <= seed_chain.lift_line_number:
            continue
        if line_number >= seed_chain.inspect_line_number:
            continue
        line = raw_line.strip()
        if not line:
            continue
        event = _load_event_payload(
            line, events_path=events_path, line_number=line_number
        )
        if event.get("tool_name") != "suggest_additional_views":
            continue
        if event.get("status") != "success":
            continue
        if event.get("event_type") != "tool_completed":
            raise CodexResponseError(
                "successful suggest_additional_views event must have "
                "event_type='tool_completed': "
                f"events_path={events_path}; line={line_number}"
            )
        result = _require_mapping_field(
            event,
            "result",
            events_path=events_path,
            line_number=line_number,
        )
        if result.get("seed_fragment_id") != seed_chain.fragment_id:
            continue
        raise CodexResponseError(
            "suggest_additional_views for a standard seed fragment must occur "
            "after inspect_mask_artifact has reviewed the lifted 3D seed: "
            f"seed_fragment_id={seed_chain.fragment_id}; "
            f"suggest_line={line_number}; "
            f"lift_line={seed_chain.lift_line_number}; "
            f"inspect_line={seed_chain.inspect_line_number}; "
            f"events_path={events_path}"
        )


def _require_matching_standard_fragment_tool_chain(
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    evidence_image_events: tuple[_ParsedEvidenceImageToolEvent, ...],
    molmo_events: tuple[_ParsedMolmoPointToolEvent, ...],
    sam_events: tuple[_ParsedSamMaskToolEvent, ...],
    lift_events: tuple[_ParsedLiftMaskToolEvent, ...],
    inspect_events: tuple[_ParsedInspectMaskToolEvent, ...],
    artifact_root: Path,
    events_path: Path,
    fuse_line_number: int,
) -> _ValidatedStandardFragmentToolChain:
    matching_molmo_events = _matching_molmo_point_events(
        fragment,
        evidence_image_events=evidence_image_events,
        molmo_events=molmo_events,
        events_path=events_path,
        before_line_number=fuse_line_number,
    )
    if not matching_molmo_events:
        _raise_missing_molmo_point_event(
            fragment, events_path=events_path, before_line_number=fuse_line_number
        )

    last_matching_sam_event: _MatchedSamMaskToolEvent | None = None
    last_matching_lift_event: _ParsedLiftMaskToolEvent | None = None
    latest_validated_chain: _ValidatedStandardFragmentToolChain | None = None
    for molmo_event in matching_molmo_events:
        sam_event = _matching_sam_mask_event(
            fragment,
            candidate_id=candidate_id,
            molmo_image_path=molmo_event.args.image_path,
            molmo_points=molmo_event.result.points,
            sam_events=sam_events,
            events_path=events_path,
            after_line_number=molmo_event.line_number,
            before_line_number=fuse_line_number,
        )
        if sam_event is None:
            continue
        last_matching_sam_event = sam_event
        lift_events_for_sam = _matching_lift_mask_events(
            fragment,
            candidate_id=candidate_id,
            sam_candidate_mask_npz_path=sam_event.candidate_mask_npz_path,
            lift_events=lift_events,
            artifact_root=artifact_root,
            events_path=events_path,
            after_line_number=sam_event.line_number,
            before_line_number=fuse_line_number,
        )
        for lift_event in lift_events_for_sam:
            if (
                last_matching_lift_event is None
                or lift_event.line_number > last_matching_lift_event.line_number
            ):
                last_matching_lift_event = lift_event
            inspect_event = _matching_inspect_mask_artifact_event(
                fragment,
                inspect_events=inspect_events,
                artifact_root=artifact_root,
                events_path=events_path,
                after_line_number=lift_event.line_number,
                before_line_number=fuse_line_number,
            )
            if inspect_event is not None:
                validated_chain = _ValidatedStandardFragmentToolChain(
                    fragment_id=fragment.fragment_id,
                    frame_id=fragment.frame_id,
                    lift_line_number=lift_event.line_number,
                    inspect_line_number=inspect_event.line_number,
                    mask_npz_path=_standard_fragment_mask_npz_path(
                        fragment, artifact_root=artifact_root
                    ),
                    mask_ply_path=_standard_fragment_mask_ply_path(
                        fragment, artifact_root=artifact_root
                    ),
                    lift_overlay_path=Path(fragment.review_artifacts.lift_overlay_path),
                )
                if (
                    latest_validated_chain is None
                    or validated_chain.lift_line_number
                    > latest_validated_chain.lift_line_number
                ):
                    latest_validated_chain = validated_chain

    if last_matching_lift_event is not None:
        if (
            latest_validated_chain is not None
            and latest_validated_chain.lift_line_number
            == last_matching_lift_event.line_number
        ):
            return latest_validated_chain
        _raise_missing_inspect_mask_artifact_event(
            fragment,
            artifact_root=artifact_root,
            events_path=events_path,
            after_line_number=last_matching_lift_event.line_number,
            before_line_number=fuse_line_number,
        )
    elif last_matching_sam_event is not None:
        _raise_missing_lift_mask_event(
            fragment,
            candidate_id=candidate_id,
            sam_candidate_mask_npz_path=last_matching_sam_event.candidate_mask_npz_path,
            artifact_root=artifact_root,
            events_path=events_path,
            after_line_number=last_matching_sam_event.line_number,
            before_line_number=fuse_line_number,
        )
    _raise_missing_sam_mask_event(
        fragment,
        candidate_id=candidate_id,
        events_path=events_path,
        after_line_number=matching_molmo_events[-1].line_number,
        before_line_number=fuse_line_number,
    )


def _standard_fragment_candidate_id(
    fragment: FinalMaskAcceptedFragment,
) -> str | None:
    expected_prefix = f"{fragment.frame_id}_"
    if not fragment.fragment_id.startswith(expected_prefix):
        return None
    candidate_id = fragment.fragment_id[len(expected_prefix) :]
    if candidate_id == "":
        return None
    return candidate_id


def _standard_fragment_mask_npz_path(
    fragment: FinalMaskAcceptedFragment, *, artifact_root: Path
) -> Path:
    return artifact_root / "fragments" / fragment.fragment_id / "mask_data.npz"


def _standard_fragment_mask_ply_path(
    fragment: FinalMaskAcceptedFragment, *, artifact_root: Path
) -> Path:
    return artifact_root / "fragments" / fragment.fragment_id / "lifted_points.ply"


def _matching_molmo_point_events(
    fragment: FinalMaskAcceptedFragment,
    *,
    evidence_image_events: tuple[_ParsedEvidenceImageToolEvent, ...],
    molmo_events: tuple[_ParsedMolmoPointToolEvent, ...],
    events_path: Path,
    before_line_number: int,
) -> tuple[_ParsedMolmoPointToolEvent, ...]:
    matching_events: list[_ParsedMolmoPointToolEvent] = []
    for parsed_event in molmo_events:
        if parsed_event.line_number >= before_line_number:
            continue
        args = parsed_event.args
        result = parsed_event.result
        if result.frame_id != fragment.frame_id:
            continue
        if not _molmo_uses_prior_evidence_image(
            args,
            frame_id=result.frame_id,
            evidence_image_events=evidence_image_events,
            events_path=events_path,
            molmo_line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            result.raw_text_path,
            Path(fragment.review_artifacts.molmo_raw_text_path),
            tool_name="molmo_point",
            field_name="result.raw_text_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            result.overlay_path,
            Path(fragment.review_artifacts.molmo_overlay_path),
            tool_name="molmo_point",
            field_name="result.overlay_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        matching_events.append(parsed_event)
    return tuple(matching_events)


def _raise_missing_molmo_point_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    events_path: Path,
    before_line_number: int,
) -> NoReturn:
    raise CodexResponseError(
        "accepted standard fragment must be backed by a successful molmo_point "
        "tool event from this run, using an image_path returned earlier by "
        "view_frame or view_crop, before fuse_accepted_masks in the required "
        "Molmo point -> SAM mask -> lift_mask_to_3d order: "
        f"fragment_id={fragment.fragment_id}; frame_id={fragment.frame_id}; "
        f"molmo_raw_text_path={fragment.review_artifacts.molmo_raw_text_path}; "
        f"molmo_overlay_path={fragment.review_artifacts.molmo_overlay_path}; "
        f"fuse_line={before_line_number}; "
        f"events_path={events_path}"
    )


def _molmo_uses_prior_evidence_image(
    molmo_args: _MolmoPointToolArgs,
    *,
    frame_id: str,
    evidence_image_events: tuple[_ParsedEvidenceImageToolEvent, ...],
    events_path: Path,
    molmo_line_number: int,
) -> bool:
    for evidence_event in evidence_image_events:
        if evidence_event.line_number >= molmo_line_number:
            continue
        for frame in evidence_event.result.frames:
            if frame.frame_id != frame_id:
                continue
            if _tool_event_path_matches(
                molmo_args.image_path,
                Path(frame.image_path),
                tool_name="molmo_point",
                field_name="args.image_path",
                events_path=events_path,
                line_number=molmo_line_number,
            ):
                return True
    return False


def _matching_sam_mask_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    molmo_image_path: str,
    molmo_points: tuple[_ToolPointPayload, ...],
    sam_events: tuple[_ParsedSamMaskToolEvent, ...],
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> _MatchedSamMaskToolEvent | None:
    for parsed_event in sam_events:
        if (
            parsed_event.line_number <= after_line_number
            or parsed_event.line_number >= before_line_number
        ):
            continue
        args = parsed_event.args
        result = parsed_event.result
        if args.frame_id != fragment.frame_id:
            continue
        if not _tool_event_path_matches(
            args.image_path,
            Path(molmo_image_path),
            tool_name="sam_mask",
            field_name="args.image_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _sam_points_match_molmo_points(args.points, molmo_points):
            continue
        if result.frame_id != fragment.frame_id:
            continue
        if not _tool_event_path_matches(
            result.contact_sheet_path,
            Path(fragment.review_artifacts.sam_contact_sheet_path),
            tool_name="sam_mask",
            field_name="result.contact_sheet_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        candidate_mask_npz_path = _sam_event_matching_candidate_mask_npz_path(
            parsed_event,
            fragment,
            candidate_id=candidate_id,
            events_path=events_path,
        )
        if candidate_mask_npz_path is not None:
            return _MatchedSamMaskToolEvent(
                line_number=parsed_event.line_number,
                candidate_mask_npz_path=candidate_mask_npz_path,
            )
    return None


def _raise_missing_sam_mask_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> NoReturn:
    raise CodexResponseError(
        "accepted standard fragment must be backed by a successful sam_mask "
        "tool event from this run before fuse_accepted_masks in the required "
        "Molmo point -> SAM mask -> lift_mask_to_3d order, and the SAM point "
        "prompt and image_path must come from the matched Molmo point output: "
        f"fragment_id={fragment.fragment_id}; frame_id={fragment.frame_id}; "
        f"candidate_id={candidate_id}; "
        f"sam_contact_sheet_path={fragment.review_artifacts.sam_contact_sheet_path}; "
        f"sam_candidate_overlay_path="
        f"{fragment.review_artifacts.sam_candidate_overlay_path}; "
        f"after_line={after_line_number}; fuse_line={before_line_number}; "
        f"events_path={events_path}"
    )


def _sam_event_matching_candidate_mask_npz_path(
    parsed_event: _ParsedSamMaskToolEvent,
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    events_path: Path,
) -> str | None:
    for candidate in parsed_event.result.candidates:
        if candidate.candidate_id != candidate_id:
            continue
        if _tool_event_path_matches(
            candidate.overlay_path,
            Path(fragment.review_artifacts.sam_candidate_overlay_path),
            tool_name="sam_mask",
            field_name=f"result.candidates[{candidate_id}].overlay_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            return candidate.mask_npz_path
    return None


def _sam_points_match_molmo_points(
    sam_points: tuple[_ToolPointPayload, ...],
    molmo_points: tuple[_ToolPointPayload, ...],
) -> bool:
    return all(
        any(
            _point_coordinates_match(sam_point, molmo_point)
            for molmo_point in molmo_points
        )
        for sam_point in sam_points
    )


def _point_coordinates_match(left: _ToolPointPayload, right: _ToolPointPayload) -> bool:
    return math.isclose(
        left.x_px,
        right.x_px,
        rel_tol=0.0,
        abs_tol=_POINT_MATCH_ABS_TOLERANCE_PX,
    ) and math.isclose(
        left.y_px,
        right.y_px,
        rel_tol=0.0,
        abs_tol=_POINT_MATCH_ABS_TOLERANCE_PX,
    )


def _tool_point_alias_payload_from_xy(
    raw_point_xy: object,
    *,
    raw_source: object,
    raw_label: object,
) -> _ToolPointAliasPayload:
    if not isinstance(raw_point_xy, Sequence) or isinstance(raw_point_xy, str | bytes):
        raise ValueError("point_xy must be a two-item numeric sequence")
    if len(raw_point_xy) != 2:
        raise ValueError("point_xy must contain exactly two coordinates")
    return {
        "x_px": _coerce_tool_point_coordinate(raw_point_xy[0], "point_xy[0]"),
        "y_px": _coerce_tool_point_coordinate(raw_point_xy[1], "point_xy[1]"),
        "source": raw_source if isinstance(raw_source, str) else "",
        "label": raw_label if isinstance(raw_label, str) else "",
    }


def _coerce_tool_point_coordinate(raw_coordinate: object, field_name: str) -> float:
    if isinstance(raw_coordinate, bool) or not isinstance(raw_coordinate, int | float):
        raise ValueError(f"{field_name} must be numeric")
    coordinate = float(raw_coordinate)
    if not math.isfinite(coordinate):
        raise ValueError(f"{field_name} must be finite")
    return coordinate


def _matching_lift_mask_events(
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    sam_candidate_mask_npz_path: str,
    lift_events: tuple[_ParsedLiftMaskToolEvent, ...],
    artifact_root: Path,
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> tuple[_ParsedLiftMaskToolEvent, ...]:
    expected_mask_npz_path = (
        artifact_root / "fragments" / fragment.fragment_id / ("mask_data.npz")
    )
    expected_mask_ply_path = (
        artifact_root / "fragments" / fragment.fragment_id / "lifted_points.ply"
    )
    matching_events: list[_ParsedLiftMaskToolEvent] = []
    for parsed_event in lift_events:
        if (
            parsed_event.line_number <= after_line_number
            or parsed_event.line_number >= before_line_number
        ):
            continue
        args = parsed_event.args
        result = parsed_event.result
        if result.frame_id != fragment.frame_id or result.candidate_id != candidate_id:
            continue
        if args.frame_id != fragment.frame_id or args.candidate_id != candidate_id:
            continue
        if not _tool_event_path_matches(
            args.mask_path,
            Path(sam_candidate_mask_npz_path),
            tool_name="lift_mask_to_3d",
            field_name="args.mask_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            result.mask_npz_path,
            expected_mask_npz_path,
            tool_name="lift_mask_to_3d",
            field_name="result.mask_npz_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            result.mask_ply_path,
            expected_mask_ply_path,
            tool_name="lift_mask_to_3d",
            field_name="result.mask_ply_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            result.overlay_path,
            Path(fragment.review_artifacts.lift_overlay_path),
            tool_name="lift_mask_to_3d",
            field_name="result.overlay_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        matching_events.append(parsed_event)
    return tuple(matching_events)


def _raise_missing_lift_mask_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    candidate_id: str,
    sam_candidate_mask_npz_path: str,
    artifact_root: Path,
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> NoReturn:
    expected_mask_npz_path = (
        artifact_root / "fragments" / fragment.fragment_id / ("mask_data.npz")
    )
    expected_mask_ply_path = (
        artifact_root / "fragments" / fragment.fragment_id / "lifted_points.ply"
    )
    raise CodexResponseError(
        "accepted standard fragment must be backed by a successful "
        "lift_mask_to_3d tool event from this run before fuse_accepted_masks in "
        "the required Molmo point -> SAM mask -> lift_mask_to_3d order: "
        f"fragment_id={fragment.fragment_id}; frame_id={fragment.frame_id}; "
        f"candidate_id={candidate_id}; mask_npz_path={expected_mask_npz_path}; "
        f"mask_ply_path={expected_mask_ply_path}; "
        f"sam_candidate_mask_npz_path={sam_candidate_mask_npz_path}; "
        f"lift_overlay_path={fragment.review_artifacts.lift_overlay_path}; "
        f"after_line={after_line_number}; fuse_line={before_line_number}; "
        f"events_path={events_path}"
    )


def _matching_inspect_mask_artifact_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    inspect_events: tuple[_ParsedInspectMaskToolEvent, ...],
    artifact_root: Path,
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> _ParsedInspectMaskToolEvent | None:
    expected_mask_npz_path = (
        artifact_root / "fragments" / fragment.fragment_id / "mask_data.npz"
    )
    expected_mask_ply_path = (
        artifact_root / "fragments" / fragment.fragment_id / "lifted_points.ply"
    )
    expected_lift_overlay_path = Path(fragment.review_artifacts.lift_overlay_path)
    for parsed_event in inspect_events:
        if (
            parsed_event.line_number <= after_line_number
            or parsed_event.line_number >= before_line_number
        ):
            continue
        args = parsed_event.args
        if not _tool_event_path_matches(
            args.mask_npz_path,
            expected_mask_npz_path,
            tool_name="inspect_mask_artifact",
            field_name="args.mask_npz_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_matches(
            args.mask_ply_path,
            expected_mask_ply_path,
            tool_name="inspect_mask_artifact",
            field_name="args.mask_ply_path",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        if not _tool_event_path_sequence_contains(
            args.overlay_paths,
            expected_lift_overlay_path,
            tool_name="inspect_mask_artifact",
            field_name="args.overlay_paths",
            events_path=events_path,
            line_number=parsed_event.line_number,
        ):
            continue
        return parsed_event
    return None


def _raise_missing_inspect_mask_artifact_event(
    fragment: FinalMaskAcceptedFragment,
    *,
    artifact_root: Path,
    events_path: Path,
    after_line_number: int,
    before_line_number: int,
) -> NoReturn:
    expected_mask_npz_path = (
        artifact_root / "fragments" / fragment.fragment_id / "mask_data.npz"
    )
    expected_mask_ply_path = (
        artifact_root / "fragments" / fragment.fragment_id / "lifted_points.ply"
    )
    raise CodexResponseError(
        "accepted standard fragment must be backed by a successful "
        "inspect_mask_artifact tool event from this run after lift_mask_to_3d "
        "and before fuse_accepted_masks, and the inspection must use the lifted "
        "mask_npz_path, mask_ply_path, and lift overlay reviewed by the agent: "
        f"fragment_id={fragment.fragment_id}; frame_id={fragment.frame_id}; "
        f"mask_npz_path={expected_mask_npz_path}; "
        f"mask_ply_path={expected_mask_ply_path}; "
        f"lift_overlay_path={fragment.review_artifacts.lift_overlay_path}; "
        f"after_lift_line={after_line_number}; fuse_line={before_line_number}; "
        f"events_path={events_path}"
    )


def _successful_molmo_point_tool_events(
    events_path: Path,
) -> tuple[_ParsedMolmoPointToolEvent, ...]:
    events: list[_ParsedMolmoPointToolEvent] = []
    for event in _successful_tool_result_events(
        events_path, tool_names={"molmo_point"}
    ):
        try:
            args = _MolmoPointToolArgs.model_validate(event.args)
            result = _MolmoPointToolResult.model_validate(event.result)
        except ValidationError as exc:
            raise CodexResponseError(
                "successful molmo_point tool event does not match the expected "
                "args/result schema: "
                f"events_path={events_path}; line={event.line_number}; error={exc}"
            ) from exc
        events.append(
            _ParsedMolmoPointToolEvent(
                line_number=event.line_number, args=args, result=result
            )
        )
    return tuple(events)


def _successful_evidence_image_tool_events(
    events_path: Path,
) -> tuple[_ParsedEvidenceImageToolEvent, ...]:
    events: list[_ParsedEvidenceImageToolEvent] = []
    for event in _successful_tool_result_events(
        events_path, tool_names={"view_frame", "view_crop"}
    ):
        try:
            result = _EvidenceImageToolResult.model_validate(event.result)
        except ValidationError as exc:
            raise CodexResponseError(
                "successful evidence image tool event does not match the expected "
                "result schema: "
                f"tool_name={event.tool_name}; events_path={events_path}; "
                f"line={event.line_number}; error={exc}"
            ) from exc
        events.append(
            _ParsedEvidenceImageToolEvent(
                line_number=event.line_number,
                tool_name=event.tool_name,
                result=result,
            )
        )
    return tuple(events)


def _successful_sam_mask_tool_events(
    events_path: Path,
) -> tuple[_ParsedSamMaskToolEvent, ...]:
    events: list[_ParsedSamMaskToolEvent] = []
    for event in _successful_tool_result_events(events_path, tool_names={"sam_mask"}):
        try:
            args = _SamMaskToolArgs.model_validate(event.args)
            result = _SamMaskToolResult.model_validate(event.result)
        except ValidationError as exc:
            raise CodexResponseError(
                "successful sam_mask tool event does not match the expected "
                "args/result schema: "
                f"events_path={events_path}; line={event.line_number}; error={exc}"
            ) from exc
        events.append(
            _ParsedSamMaskToolEvent(
                line_number=event.line_number, args=args, result=result
            )
        )
    return tuple(events)


def _successful_lift_mask_tool_events(
    events_path: Path,
) -> tuple[_ParsedLiftMaskToolEvent, ...]:
    events: list[_ParsedLiftMaskToolEvent] = []
    for event in _successful_tool_result_events(
        events_path, tool_names={"lift_mask_to_3d"}
    ):
        try:
            args = _LiftMaskToolArgs.model_validate(event.args)
            result = _LiftMaskToolResult.model_validate(event.result)
        except ValidationError as exc:
            raise CodexResponseError(
                "successful lift_mask_to_3d tool event does not match the expected "
                "result schema: "
                f"events_path={events_path}; line={event.line_number}; error={exc}"
            ) from exc
        events.append(
            _ParsedLiftMaskToolEvent(
                line_number=event.line_number, args=args, result=result
            )
        )
    return tuple(events)


def _successful_inspect_mask_tool_events(
    events_path: Path,
) -> tuple[_ParsedInspectMaskToolEvent, ...]:
    events: list[_ParsedInspectMaskToolEvent] = []
    for event in _successful_tool_result_events(
        events_path, tool_names={"inspect_mask_artifact"}
    ):
        try:
            args = _InspectMaskToolArgs.model_validate(event.args)
            result = _InspectMaskToolResult.model_validate(event.result)
        except ValidationError as exc:
            raise CodexResponseError(
                "successful inspect_mask_artifact tool event does not match the "
                "expected args/result schema: "
                f"events_path={events_path}; line={event.line_number}; error={exc}"
            ) from exc
        events.append(
            _ParsedInspectMaskToolEvent(
                line_number=event.line_number,
                args=args,
                result=result,
            )
        )
    return tuple(events)


def _successful_tool_result_events(
    events_path: Path, *, tool_names: set[str]
) -> tuple[_SuccessfulToolResultEvent, ...]:
    if not events_path.is_file():
        return ()
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFunc3D tool events while validating upstream "
            "tool provenance: "
            f"events_path={events_path}; error_type={exc.__class__.__name__}"
        ) from exc
    events: list[_SuccessfulToolResultEvent] = []
    for line_number, raw_line in enumerate(event_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        event = _load_event_payload(
            line, events_path=events_path, line_number=line_number
        )
        tool_name = event.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in tool_names:
            continue
        if event.get("status") != "success":
            continue
        if event.get("event_type") != "tool_completed":
            raise CodexResponseError(
                "successful SceneFunc3D upstream tool event must have "
                "event_type='tool_completed': "
                f"tool_name={tool_name}; events_path={events_path}; "
                f"line={line_number}"
            )
        args = _require_mapping_field(
            event,
            "args",
            events_path=events_path,
            line_number=line_number,
        )
        result = _require_mapping_field(
            event,
            "result",
            events_path=events_path,
            line_number=line_number,
        )
        events.append(
            _SuccessfulToolResultEvent(
                line_number=line_number,
                tool_name=tool_name,
                args=args,
                result=result,
            )
        )
    return tuple(events)


def _event_path_matches(
    raw_path: str,
    expected_path: Path,
    *,
    field_name: str,
    events_path: Path,
    line_number: int,
) -> bool:
    try:
        event_path = Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise CodexResponseError(
            "successful fuse_accepted_masks tool event contains an invalid path: "
            f"field={field_name}; events_path={events_path}; line={line_number}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return event_path == expected_path.expanduser().resolve()


def _tool_event_path_matches(
    raw_path: str,
    expected_path: Path,
    *,
    tool_name: str,
    field_name: str,
    events_path: Path,
    line_number: int,
) -> bool:
    try:
        event_path = Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise CodexResponseError(
            "successful SceneFunc3D tool event contains an invalid path: "
            f"tool_name={tool_name}; field={field_name}; "
            f"events_path={events_path}; line={line_number}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return event_path == expected_path.expanduser().resolve()


def _tool_event_path_sequence_contains(
    raw_paths: Sequence[str],
    expected_path: Path,
    *,
    tool_name: str,
    field_name: str,
    events_path: Path,
    line_number: int,
) -> bool:
    return any(
        _tool_event_path_matches(
            raw_path,
            expected_path,
            tool_name=tool_name,
            field_name=f"{field_name}[{index}]",
            events_path=events_path,
            line_number=line_number,
        )
        for index, raw_path in enumerate(raw_paths)
    )


def _accepted_frame_ids_from_fuse_fragments(
    accepted_fragments: tuple[FinalMaskAcceptedFragment, ...],
) -> tuple[str, ...]:
    seen_frame_ids: set[str] = set()
    accepted_frame_ids: list[str] = []
    for fragment in accepted_fragments:
        if fragment.frame_id not in seen_frame_ids:
            accepted_frame_ids.append(fragment.frame_id)
            seen_frame_ids.add(fragment.frame_id)
    return tuple(accepted_frame_ids)


def _fuse_fragment_ids(
    accepted_fragments: tuple[FinalMaskAcceptedFragment, ...],
) -> tuple[str, ...]:
    return tuple(fragment.fragment_id for fragment in accepted_fragments)


def _validate_multiview_decision_against_tool_events(
    decision: FinalMaskMultiViewDecision,
    events_path: Path,
    *,
    before_line_number: int,
    after_line_number: int = 0,
    seed_chain: _ValidatedStandardFragmentToolChain | None = None,
) -> None:
    """Validate final multi-view decision against recorded tool suggestions."""
    if decision.action is FinalMaskMultiViewAction.EXPAND:
        _validate_expand_decision_against_tool_events(
            decision,
            events_path,
            before_line_number=before_line_number,
            after_line_number=after_line_number,
            seed_chain=seed_chain,
        )
        return
    suggested_frame_ids = _expand_suggested_frame_ids_from_events(
        events_path,
        seed_fragment_id=decision.seed_fragment_id,
        after_line_number=after_line_number,
        before_line_number=before_line_number,
        seed_chain=seed_chain,
    )
    if not suggested_frame_ids:
        return
    rejected_frame_ids = set(decision.rejected_suggested_frame_ids)
    missing_frame_ids = tuple(
        frame_id
        for frame_id in suggested_frame_ids
        if frame_id not in rejected_frame_ids
    )
    if missing_frame_ids:
        raise CodexResponseError(
            "multi_view_decision.action='stop' must account for every frame from "
            "successful suggest_additional_views expand results in "
            "rejected_suggested_frame_ids: "
            f"missing_rejected_suggested_frame_ids={missing_frame_ids}; "
            f"events_path={events_path}"
        )


def _validate_expand_decision_against_tool_events(
    decision: FinalMaskMultiViewDecision,
    events_path: Path,
    *,
    before_line_number: int,
    after_line_number: int,
    seed_chain: _ValidatedStandardFragmentToolChain | None,
) -> None:
    suggested_frame_ids = _expand_suggested_frame_ids_from_events(
        events_path,
        seed_fragment_id=decision.seed_fragment_id,
        after_line_number=after_line_number,
        before_line_number=before_line_number,
        seed_chain=seed_chain,
    )
    suggested_frame_id_set = set(suggested_frame_ids)
    missing_frame_ids = tuple(
        frame_id
        for frame_id in decision.suggested_frame_ids
        if frame_id not in suggested_frame_id_set
    )
    if missing_frame_ids:
        raise CodexResponseError(
            "multi_view_decision.action='expand' must be backed by a successful "
            "suggest_additional_views expand result for the seed fragment after "
            "inspect_mask_artifact and before fuse_accepted_masks: "
            f"seed_fragment_id={decision.seed_fragment_id}; "
            f"missing_suggested_frame_ids={missing_frame_ids}; "
            f"after_inspect_line={after_line_number}; "
            f"fuse_line={before_line_number}; "
            f"events_path={events_path}"
        )
    _require_all_expand_suggestions_accounted_for(
        suggested_frame_ids,
        accepted_suggested_frame_ids=decision.suggested_frame_ids,
        rejected_suggested_frame_ids=decision.rejected_suggested_frame_ids,
        events_path=events_path,
    )


def _require_suggest_event_uses_standard_seed(
    raw_args: Mapping[object, object],
    *,
    seed_chain: _ValidatedStandardFragmentToolChain | None,
    events_path: Path,
    line_number: int,
) -> None:
    if seed_chain is None:
        return
    try:
        args = _SuggestAdditionalViewsToolArgs.model_validate(raw_args)
    except ValidationError as exc:
        raise CodexResponseError(
            "successful suggest_additional_views event for a standard seed "
            "fragment must include the inspected seed artifact paths: "
            f"seed_fragment_id={seed_chain.fragment_id}; "
            f"events_path={events_path}; line={line_number}; error={exc}"
        ) from exc
    if args.seed_fragment_id != seed_chain.fragment_id:
        raise CodexResponseError(
            "successful suggest_additional_views event args.seed_fragment_id must "
            "match the inspected standard seed fragment: "
            f"args_seed_fragment_id={args.seed_fragment_id}; "
            f"seed_fragment_id={seed_chain.fragment_id}; "
            f"events_path={events_path}; line={line_number}"
        )
    if args.accepted_frame_id != seed_chain.frame_id:
        raise CodexResponseError(
            "successful suggest_additional_views event args.accepted_frame_id must "
            "match the inspected standard seed frame: "
            f"args_accepted_frame_id={args.accepted_frame_id}; "
            f"seed_frame_id={seed_chain.frame_id}; "
            f"events_path={events_path}; line={line_number}"
        )
    _require_tool_event_path_match(
        args.seed_mask_npz_path,
        seed_chain.mask_npz_path,
        tool_name="suggest_additional_views",
        field_name="args.seed_mask_npz_path",
        events_path=events_path,
        line_number=line_number,
    )
    _require_tool_event_path_match(
        args.seed_mask_ply_path,
        seed_chain.mask_ply_path,
        tool_name="suggest_additional_views",
        field_name="args.seed_mask_ply_path",
        events_path=events_path,
        line_number=line_number,
    )
    _require_tool_event_path_match(
        args.seed_lift_overlay_path,
        seed_chain.lift_overlay_path,
        tool_name="suggest_additional_views",
        field_name="args.seed_lift_overlay_path",
        events_path=events_path,
        line_number=line_number,
    )


def _require_tool_event_path_match(
    raw_path: str,
    expected_path: Path,
    *,
    tool_name: str,
    field_name: str,
    events_path: Path,
    line_number: int,
) -> None:
    if _tool_event_path_matches(
        raw_path,
        expected_path,
        tool_name=tool_name,
        field_name=field_name,
        events_path=events_path,
        line_number=line_number,
    ):
        return
    raise CodexResponseError(
        "successful SceneFunc3D tool event path does not match the validated "
        "provenance chain: "
        f"tool_name={tool_name}; field={field_name}; "
        f"actual={raw_path}; expected={expected_path}; "
        f"events_path={events_path}; line={line_number}"
    )


def _require_all_expand_suggestions_accounted_for(
    suggested_frame_ids: tuple[str, ...],
    *,
    accepted_suggested_frame_ids: tuple[str, ...],
    rejected_suggested_frame_ids: tuple[str, ...],
    events_path: Path,
) -> None:
    accounted_frame_ids = set(accepted_suggested_frame_ids) | set(
        rejected_suggested_frame_ids
    )
    missing_frame_ids = tuple(
        frame_id
        for frame_id in suggested_frame_ids
        if frame_id not in accounted_frame_ids
    )
    if not missing_frame_ids:
        return
    raise CodexResponseError(
        "multi_view_decision.action='expand' must account for every frame from "
        "successful suggest_additional_views expand results in either "
        "suggested_frame_ids or rejected_suggested_frame_ids: "
        f"missing_rejected_suggested_frame_ids={missing_frame_ids}; "
        f"events_path={events_path}"
    )


def _expand_suggested_frame_ids_from_events(
    events_path: Path,
    *,
    seed_fragment_id: str,
    after_line_number: int = 0,
    before_line_number: int,
    seed_chain: _ValidatedStandardFragmentToolChain | None = None,
) -> tuple[str, ...]:
    """Return unique frames suggested by successful expand recommendations."""
    if not events_path.is_file():
        return ()
    frame_ids: list[str] = []
    seen_frame_ids: set[str] = set()
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFunc3D tool events for multi-view validation: "
            f"events_path={events_path}; error_type={exc.__class__.__name__}"
        ) from exc
    for line_number, raw_line in enumerate(event_lines, start=1):
        if line_number <= after_line_number:
            continue
        if line_number >= before_line_number:
            continue
        line = raw_line.strip()
        if not line:
            continue
        event = _load_event_payload(
            line, events_path=events_path, line_number=line_number
        )
        if event.get("tool_name") != "suggest_additional_views":
            continue
        if event.get("status") != "success":
            continue
        if event.get("event_type") != "tool_completed":
            raise CodexResponseError(
                "successful suggest_additional_views event must have "
                "event_type='tool_completed': "
                f"events_path={events_path}; line={line_number}"
            )
        result = _require_mapping_field(
            event,
            "result",
            events_path=events_path,
            line_number=line_number,
        )
        if result.get("seed_fragment_id") != seed_fragment_id:
            continue
        args = _require_mapping_field(
            event,
            "args",
            events_path=events_path,
            line_number=line_number,
        )
        _require_suggest_event_uses_standard_seed(
            args,
            seed_chain=seed_chain,
            events_path=events_path,
            line_number=line_number,
        )
        if (
            result.get("expansion_recommendation")
            != FinalMaskMultiViewAction.EXPAND.value
        ):
            continue
        views = result.get("views")
        if not isinstance(views, list):
            raise CodexResponseError(
                "successful suggest_additional_views expand event must include "
                "result.views as a JSON array: "
                f"events_path={events_path}; line={line_number}"
            )
        for view_index, view_item in enumerate(views):
            view = _require_json_mapping(
                view_item,
                events_path=events_path,
                line_number=line_number,
                field_name=f"result.views[{view_index}]",
            )
            frame_id = view.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id.strip():
                raise CodexResponseError(
                    "successful suggest_additional_views expand event view must "
                    "include a non-empty frame_id: "
                    f"events_path={events_path}; line={line_number}; "
                    f"view_index={view_index}"
                )
            normalized_frame_id = frame_id.strip()
            if normalized_frame_id not in seen_frame_ids:
                frame_ids.append(normalized_frame_id)
                seen_frame_ids.add(normalized_frame_id)
    return tuple(frame_ids)


def _load_event_payload(
    line: str, *, events_path: Path, line_number: int
) -> Mapping[object, object]:
    try:
        parsed = json.loads(line)
    except JSONDecodeError as exc:
        raise CodexResponseError(
            "SceneFunc3D tool event line is not valid JSON: "
            f"events_path={events_path}; line={line_number}; error={exc}"
        ) from exc
    return _require_json_mapping(
        parsed,
        events_path=events_path,
        line_number=line_number,
        field_name="event",
    )


def _require_mapping_field(
    payload: Mapping[object, object],
    field_name: str,
    *,
    events_path: Path,
    line_number: int,
) -> Mapping[object, object]:
    return _require_json_mapping(
        payload.get(field_name),
        events_path=events_path,
        line_number=line_number,
        field_name=field_name,
    )


def _require_json_mapping(
    value: object,
    *,
    events_path: Path,
    line_number: int,
    field_name: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise CodexResponseError(
            "SceneFunc3D tool event field must be a JSON object: "
            f"field={field_name}; events_path={events_path}; line={line_number}"
        )
    return value


def _completed_run_summary(
    *,
    sample: SceneFunc3dSample,
    outcome: SceneFunc3dMaskOutcome,
    artifact: ValidatedFinalMaskArtifact,
) -> SceneFunc3dCompletedRunSummary:
    return SceneFunc3dCompletedRunSummary(
        sample_id=sample.sample_id,
        visit_id=sample.visit_id,
        desc_id=sample.desc_id,
        task_description=sample.task_description,
        selected_frame_ids=outcome.selected_frame_ids,
        accepted_fragment_ids=outcome.accepted_fragment_ids,
        mask_artifact_path=outcome.mask_artifact_path,
        mask_npz_path=outcome.mask_npz_path,
        mask_ply_path=outcome.mask_ply_path,
        confidence=outcome.confidence,
        uncertainties=outcome.uncertainties,
        multi_view_decision=artifact.multi_view_decision,
        final_point_count=artifact.point_count,
    )


def _artifact_payload(
    artifact: ValidatedFinalMaskArtifact,
) -> SceneFunc3dRunArtifactPayload:
    """Return validated final artifact details stored alongside ``result.json``."""
    return {
        "accepted_frame_ids": list(artifact.accepted_frame_ids),
        "accepted_fragment_ids": list(artifact.accepted_fragment_ids),
        "final_point_count": artifact.point_count,
        "multi_view_decision": {
            "seed_fragment_id": artifact.multi_view_decision.seed_fragment_id,
            "action": artifact.multi_view_decision.action.value,
            "reason": artifact.multi_view_decision.reason,
            "suggested_frame_ids": list(
                artifact.multi_view_decision.suggested_frame_ids
            ),
            "rejected_suggested_frame_ids": list(
                artifact.multi_view_decision.rejected_suggested_frame_ids
            ),
        },
    }


def build_arg_parser(
    *, prog: str = "codex_agent.scenefunc3d.runner"
) -> argparse.ArgumentParser:
    """Build the SceneFunc3D runner CLI parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run SceneFunc3D mask-generation cases with Codex.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="SceneFunc3D dataset root containing per-visit scene directories.",
    )
    sample_source = parser.add_mutually_exclusive_group(required=True)
    sample_source.add_argument(
        "--sample-id",
        help="SceneFunc3D sample id in '<visit_id>::<desc_id>' form.",
    )
    sample_source.add_argument(
        "--sample-ids-path",
        type=Path,
        help="JSON file containing a list of SceneFunc3D sample ids.",
    )
    sample_source.add_argument(
        "--all-samples",
        action="store_true",
        help="Run all samples discovered under --dataset-root.",
    )
    parser.add_argument(
        "--backend-config",
        required=True,
        type=Path,
        help="TOML file configuring local Molmo and SAM sidecar backends.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the sample result.json and tool artifacts are written.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score the written result.json against hidden GT and include metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run SceneFunc3D samples from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_from_args(args)


def run_from_args(args: argparse.Namespace) -> int:
    """Run SceneFunc3D samples from parsed command-line arguments."""
    config = SceneFunc3dRunnerConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend_config_path=args.backend_config,
    )
    return _run_from_config_and_args(config, args)


def _run_from_config_and_args(
    config: SceneFunc3dRunnerConfig, args: argparse.Namespace
) -> int:
    """Run SceneFunc3D samples with validated runner config and parsed arguments."""
    sample_ids = _sample_ids_from_args(args, data_root=config.dataset_root)
    executor = _build_executor()
    if len(sample_ids) == 1 and _namespace_optional_str(args, "sample_id") is not None:
        result_path = run_single_sample(
            config,
            sample_id=sample_ids[0],
            executor=executor,
            check_sidecars=True,
        )
        if _namespace_bool(args, "score"):
            payload: (
                SceneFunc3dCliRunPayload
                | SceneFunc3dCliRunAndScorePayload
                | SceneFunc3dCliBatchRunPayload
            ) = _run_and_score_payload(
                data_root=config.dataset_root, result_path=result_path
            )
        else:
            payload = _run_payload(result_path)
    else:
        summary = run_samples(
            config,
            sample_ids=sample_ids,
            executor=executor,
            check_sidecars=True,
            score=_namespace_bool(args, "score"),
        )
        payload = _batch_run_payload(
            config.output_dir / "evaluation_summary.json", summary
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _sample_ids_from_args(
    args: argparse.Namespace, *, data_root: Path
) -> tuple[str, ...]:
    try:
        sample_id = _namespace_optional_str(args, "sample_id")
        if sample_id is not None:
            return _normalized_batch_sample_ids((sample_id,))
        sample_ids_path = _namespace_optional_path(args, "sample_ids_path")
        if sample_ids_path is not None:
            return _normalized_batch_sample_ids(_load_sample_ids_file(sample_ids_path))
        if _namespace_bool(args, "all_samples"):
            return _normalized_batch_sample_ids(list_sample_ids(data_root))
    except ValueError as exc:
        raise SceneFunc3dDataError(str(exc)) from exc
    raise SceneFunc3dDataError("one SceneFunc3D sample source is required")


def _load_sample_ids_file(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise SceneFunc3dDataError(f"SceneFunc3D sample ids file is missing: {path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"SceneFunc3D sample ids file is not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, list):
        raise SceneFunc3dDataError(
            f"SceneFunc3D sample ids file must contain a JSON array: {path}"
        )
    sample_ids: list[str] = []
    for item_index, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise SceneFunc3dDataError(
                "SceneFunc3D sample ids file entries must be non-empty strings: "
                f"path={path}; index={item_index}"
            )
        sample_ids.append(item.strip())
    return tuple(sample_ids)


def _build_executor() -> CodexExecutor:
    from ..cli.runtime_preflight import preflight_codex_runtime
    from ..runtime import CodexAgentRuntime

    config = _tool_writable_runtime_config(CodexAgentConfig.from_env())
    preflight_codex_runtime(config)
    return CodexAgentRuntime(config)


def _tool_writable_runtime_config(config: CodexAgentConfig) -> CodexAgentConfig:
    """Return SceneFunc3D runtime defaults suitable for shell tool execution."""
    if config.sandbox == "full_access":
        return config
    return replace(config, sandbox="workspace_write", sandbox_network_access=True)


def _run_payload(result_path: Path) -> SceneFunc3dCliRunPayload:
    return {"result_path": str(result_path)}


def _run_and_score_payload(
    *, data_root: Path, result_path: Path
) -> SceneFunc3dCliRunAndScorePayload:
    return {
        "result_path": str(result_path),
        "score": score_to_payload(
            score_result_file(data_root=data_root, result_path=result_path)
        ),
    }


def _batch_run_payload(
    summary_path: Path, summary: SceneFunc3dBatchRunSummary
) -> SceneFunc3dCliBatchRunPayload:
    return {
        "summary_path": str(summary_path),
        "summary": summary.to_payload(),
    }


def _fetch_sidecar_health(
    base_url: str,
    *,
    service_name: str,
    timeout_seconds: float,
    request_headers: tuple[HttpHeader, ...],
) -> HealthResponse:
    endpoint = _health_endpoint(base_url)
    request = Request(
        endpoint,
        headers=_health_request_headers(request_headers),
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health check failed: HTTP {exc.code}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health check timed out: {endpoint}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health check failed: {endpoint}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health check failed: {endpoint}"
        ) from exc

    payload = _decode_health_payload(response_body, service_name=service_name)
    try:
        return HealthResponse.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health response failed validation"
        ) from exc


def _health_request_headers(request_headers: tuple[HttpHeader, ...]) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    for header in request_headers:
        headers[header.name] = header.value
    return headers


def _decode_health_payload(response_body: bytes, *, service_name: str) -> object:
    try:
        return json.loads(response_body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health response is not UTF-8"
        ) from exc
    except JSONDecodeError as exc:
        raise RuntimeError(
            f"{service_name} sidecar health response is not JSON"
        ) from exc


def _require_model_loaded(response: HealthResponse, *, service_name: str) -> None:
    if not response.model_loaded:
        raise RuntimeError(
            f"{service_name} sidecar is not ready: model_loaded=false; "
            f"model_name={response.model_name}"
        )


def _metadata_payload(metadata: CodexTurnMetadata) -> CodexTurnMetadataPayload:
    return {
        "turn_id": metadata.turn_id,
        "status": metadata.status,
        "duration_ms": metadata.duration_ms,
        "usage": dict(metadata.usage) if metadata.usage is not None else None,
        "input_tokens": metadata.input_tokens,
        "cached_input_tokens": metadata.cached_input_tokens,
        "reasoning_summary": metadata.reasoning_summary,
        "run_home": metadata.run_home,
        "attempts": [dict(attempt) for attempt in metadata.attempts],
    }


def _health_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/health"


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value: object = getattr(args, name)
    if not isinstance(value, bool):
        raise TypeError(f"argparse field {name!r} must be a bool")
    return value


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value: object = getattr(args, name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise TypeError(f"argparse field {name!r} must be a Path or None")
    return value


def _namespace_optional_str(args: argparse.Namespace, name: str) -> str | None:
    value: object = getattr(args, name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"argparse field {name!r} must be a string or None")
    return value


def _require_non_empty_string_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    stripped_values = tuple(value.strip() for value in values)
    if any(not value for value in stripped_values):
        raise ValueError(f"{field_name} must not contain empty values")
    return stripped_values


__all__ = [
    "TASK_NAME",
    "DEFAULT_TOOL_CLI_MODULE",
    "SCENEFUNC3D_ALLOWED_TOOL_NAMES",
    "SceneFunc3dMaskDecision",
    "CodexTurnMetadataPayload",
    "SceneFunc3dMaskOutcome",
    "SceneFunc3dMaskOutcomePayload",
    "SceneFunc3dMaskTask",
    "SceneFunc3dRunnerConfig",
    "SceneFunc3dBatchSampleResult",
    "SceneFunc3dBatchSampleResultPayload",
    "SceneFunc3dBatchRunSummary",
    "SceneFunc3dBatchRunSummaryPayload",
    "SceneFunc3dRunArtifactPayload",
    "SceneFunc3dRunResultPayload",
    "build_arg_parser",
    "build_prompt",
    "check_sidecar_health",
    "load_runner_sample",
    "main",
    "run_from_args",
    "run_samples",
    "run_single_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
