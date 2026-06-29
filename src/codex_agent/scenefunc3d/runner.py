"""SceneFunc3D mask-generation runner for single samples and batches."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StringConstraints

from ..config import CodexAgentConfig
from ..errors import CodexResponseError, SceneFunc3dDataError
from ..json_extraction import extract_json_object
from ..models import CodexTurnMetadata, CodexTurnRequest
from ..tasks.base import CodexExecutor
from .backends.config import load_backend_settings
from .evaluation.payloads import SceneFunc3dScorePayload, score_to_payload
from .evaluation.scorer import SceneFunc3dScore, score_result_file
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
SCENEFUNC3D_ALLOWED_TOOL_NAMES = SCENEFUNC3D_TOOL_NAMES

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


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

    mask_artifact_path: NonEmptyString
    mask_npz_path: NonEmptyString
    mask_ply_path: NonEmptyString
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


@dataclass(frozen=True)
class _ParsedFuseToolEvent:
    """A validated fuse event with source location for diagnostics."""

    line_number: int
    event: _FuseToolEvent


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
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
    ) -> None:
        self.sample = sample
        self.scene_root = Path(scene_root)
        self.output_dir = Path(output_dir)
        self.backend_config_path = Path(backend_config_path)
        self.tool_cli_module = tool_cli_module

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=SceneFunc3dMaskDecision.model_json_schema(),
            skills=(),
            image_paths=(),
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
        validated_artifact = validate_final_mask_artifact(
            artifact_path=mask_artifact_path,
            mask_npz_path=mask_npz_path,
            mask_ply_path=mask_ply_path,
            selected_frame_ids=selected_frame_ids,
            accepted_fragment_ids=accepted_fragment_ids,
        )
        _validate_multiview_decision_against_tool_events(
            validated_artifact, self.output_dir / "events.jsonl"
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
            f"- scene_root: {self.scene_root}\n"
            f"- backend_config: {self.backend_config_path}\n"
            f"- out_dir: {self.output_dir}\n"
            "\nHow to run a tool (in the shell):\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--scene-root {self.scene_root} "
            f"--backend-config {self.backend_config_path} "
            f"--out-dir {self.output_dir} --args '<json>'\n"
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
    )
    sam_health = _fetch_sidecar_health(
        settings.sam_url,
        service_name="sam",
        timeout_seconds=settings.request_timeout_seconds,
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
    task = SceneFunc3dMaskTask(
        sample=sample,
        scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
        output_dir=sample_output_dir,
        backend_config_path=config.backend_config_path,
    )
    result = executor.execute(task)
    validated_outcome = task.validate_outcome(result.outcome)
    validated_artifact = _validate_outcome_artifact(validated_outcome)
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
            results.append(
                SceneFunc3dBatchSampleResult(
                    sample_id=sample_id,
                    status="failed",
                    failure_stage="run",
                    result_path=None,
                    score=None,
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
    if any(
        _fuse_event_matches_outcome(
            parsed_event,
            outcome,
            artifact_document=artifact_document,
            events_path=events_path,
        )
        for parsed_event in _successful_fuse_tool_events(events_path)
    ):
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
    artifact: ValidatedFinalMaskArtifact, events_path: Path
) -> None:
    """Validate final multi-view decision against recorded tool suggestions."""
    if artifact.multi_view_decision.action is not FinalMaskMultiViewAction.STOP:
        return
    suggested_frame_ids = _expand_suggested_frame_ids_from_events(events_path)
    if not suggested_frame_ids:
        return
    rejected_frame_ids = set(artifact.multi_view_decision.rejected_suggested_frame_ids)
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


def _expand_suggested_frame_ids_from_events(events_path: Path) -> tuple[str, ...]:
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
        result = _require_mapping_field(
            event,
            "result",
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SceneFunc3D runner CLI parser."""
    parser = argparse.ArgumentParser(
        prog="codex_agent.scenefunc3d.runner",
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
        "--skip-sidecar-health-check",
        action="store_true",
        help="Skip Molmo/SAM /health checks before launching Codex.",
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
    config = SceneFunc3dRunnerConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend_config_path=args.backend_config,
    )
    sample_ids = _sample_ids_from_args(args, data_root=config.dataset_root)
    executor = _build_executor()
    if len(sample_ids) == 1 and _namespace_optional_str(args, "sample_id") is not None:
        result_path = run_single_sample(
            config,
            sample_id=sample_ids[0],
            executor=executor,
            check_sidecars=not args.skip_sidecar_health_check,
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
            check_sidecars=not args.skip_sidecar_health_check,
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
    from ..runtime import CodexAgentRuntime

    return CodexAgentRuntime(_tool_writable_runtime_config(CodexAgentConfig.from_env()))


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
) -> HealthResponse:
    endpoint = _health_endpoint(base_url)
    request = Request(
        endpoint,
        headers={"Accept": "application/json"},
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
    "run_samples",
    "run_single_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
