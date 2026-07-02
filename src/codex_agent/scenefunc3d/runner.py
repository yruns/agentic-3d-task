"""SceneFunc3D mask-generation runner for single samples and batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from ..config import CodexAgentConfig
from ..errors import (
    CodexAgentError,
    CodexResponseError,
    CodexTurnError,
    SceneFunc3dDataError,
)
from ..models import CodexTurnMetadata, CodexTurnRequest
from ..tasks.base import CodexExecutor
from .backends.anchor import build_anchor
from .backends.config import HttpHeader, load_backend_settings
from .backends.frame_assets import resolve_frame_geometry_assets
from .backends.fusion import FusionParams
from .backends.lift_3d import load_scene_mesh_vertices
from .evaluation.payloads import SceneFunc3dScorePayload, score_to_payload
from .evaluation.scorer import SceneFunc3dScore, score_result_file
from .failure_artifacts import (
    SceneFunc3dFailureArtifact,
    SceneFunc3dFailureStage,
    SceneFunc3dFailureTurnPayload,
    write_failure_artifact,
)
from .final_mask_artifacts import load_points_world_npz
from .pipeline import run_anchor_multiview_pipeline
from .pipeline_backends import SidecarFrameProposer
from .pipeline_result import (
    SceneFunc3dPipelineOutcome,
    write_pipeline_fused_mask,
    write_pipeline_result,
)
from .playbook import SCENEFUNC3D_TOOL_NAMES, SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, list_sample_ids, load_sample, scene_dir_for
from .seed_contract import (
    SceneFunc3dSeedOutcome,
    SeedConfirmation,
    parse_seed_confirmation,
)
from .seed_recovery import recover_seed_sam_candidate
from .servers.schemas import HealthResponse
from .tool_context import SceneFunc3dToolContext, write_tool_context
from .tools.mask_artifacts import (
    MaskArtifactPaths,
    SceneFunc3dRunCompletionEvent,
    append_run_completion_event,
    artifact_paths_for,
)
from .tools.mask_lifting import LiftMaskArgs, LiftMaskResult, lift_mask_to_3d
from .tools.scene_context import SceneFunc3dToolScene

TASK_NAME = "scenefunc3d_mask_generation"
DEFAULT_TOOL_CLI_MODULE = "codex_agent.scenefunc3d.tools"
DEFAULT_TOOL_CONTEXT_FILENAME = "tool_context.json"
SCENEFUNC3D_ALLOWED_TOOL_NAMES = SCENEFUNC3D_TOOL_NAMES
_PIPELINE_FRAME_CAP = 30
_PIPELINE_DEPTH_TOLERANCE = 0.25


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
            output_schema=SeedConfirmation.model_json_schema(),
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

    def parse_response(self, response_text: str) -> SceneFunc3dSeedOutcome:
        confirmation = parse_seed_confirmation(response_text)
        seed_candidate = recover_seed_sam_candidate(
            self.output_dir / "events.jsonl",
            seed_frame_id=confirmation.seed_frame_id,
            seed_candidate_id=confirmation.seed_candidate_id,
        )
        return SceneFunc3dSeedOutcome(
            seed_frame_id=confirmation.seed_frame_id,
            seed_candidate_id=confirmation.seed_candidate_id,
            seed_fragment_id=confirmation.seed_fragment_id,
            affordance_concept=confirmation.affordance_concept,
            approval_actions=confirmation.approval_actions,
            uncertainties=confirmation.uncertainties,
            seed_sam_mask_npz_path=seed_candidate.mask_npz_path,
        )

    def _build_prompt(self) -> str:
        schema = SeedConfirmation.model_json_schema()
        return (
            build_prompt(self.sample) + "\n\nRuntime paths:\n"
            f"- tool_context: {self.tool_context_path}\n"
            "\nHow to run a tool (in the shell):\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--context {self.tool_context_path} --args '<json>'\n"
            "- For molmo_point and sam_mask, set yield_time_ms=30000 on the shell "
            "call so the command can finish and return JSON. If the shell reports "
            "Process running with session ID, wait on that same session until it "
            "exits and returns JSON; do not start another SceneFun3D tool while a "
            "previous tool command is still running.\n"
            "- Tool outputs are JSON. Open any returned image_path with view_image "
            "before relying on visual evidence.\n"
            "- Use the Molmo point and SAM candidate approval gates in the playbook "
            "before confirming the seed on the original-image SAM overlay.\n"
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
        "frame, prompt, or mask candidate instead."
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
        pipeline_outcome = _run_pipeline_for_seed(
            sample=sample,
            scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
            output_dir=sample_output_dir,
            backend_config_path=config.backend_config_path,
            seed_outcome=result.outcome,
            artifact_paths=artifact_paths,
        )
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
    result_path = write_pipeline_result(
        sample_output_dir / "result.json",
        pipeline_outcome,
        sample_id=sample_id,
        turn_metadata=result.turn.metadata,
    )
    append_run_completion_event(
        artifact_paths,
        SceneFunc3dRunCompletionEvent(
            sample_id=sample_id,
            result_path=result_path,
            summary_path=result_path,
            mask_artifact_path=pipeline_outcome.mask_npz_path,
        ),
    )
    return result_path


def _run_pipeline_for_seed(
    *,
    sample: SceneFunc3dSample,
    scene_root: Path,
    output_dir: Path,
    backend_config_path: Path,
    seed_outcome: SceneFunc3dSeedOutcome,
    artifact_paths: MaskArtifactPaths,
) -> SceneFunc3dPipelineOutcome:
    """Deterministically lift the confirmed 2D seed, then run the multi-view pipeline."""
    scene = SceneFunc3dToolScene.load(scene_root)
    scene_vertices = load_scene_mesh_vertices(scene.raw_mesh_path)
    seed_lift = _lift_seed_fragment(
        scene=scene, seed_outcome=seed_outcome, out_dir=output_dir
    )
    seed_points = load_points_world_npz(seed_lift.mask_npz_path)
    anchor = build_anchor(
        seed_points,
        motion_type=_anchor_motion_type(sample),
        seed_frame_id=seed_outcome.seed_frame_id,
    )
    proposer = SidecarFrameProposer(
        scene=scene,
        out_dir=output_dir,
        backend_config_path=backend_config_path,
        affordance_concept=seed_outcome.affordance_concept,
        task_description=sample.task_description,
    )
    result = run_anchor_multiview_pipeline(
        anchor=anchor,
        scene=scene,
        scene_vertices=scene_vertices,
        proposer=proposer,
        params=FusionParams(),
        frame_cap=_PIPELINE_FRAME_CAP,
        depth_tolerance=_PIPELINE_DEPTH_TOLERANCE,
    )
    mask_npz_path, mask_ply_path = write_pipeline_fused_mask(
        artifact_paths.fused_dir,
        point_indices=result.fused.point_indices,
        scene_vertices=scene_vertices,
    )
    return SceneFunc3dPipelineOutcome(
        seed=seed_outcome,
        seed_point_count=seed_lift.lifted_point_count,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        result=result,
    )


def _lift_seed_fragment(
    *,
    scene: SceneFunc3dToolScene,
    seed_outcome: SceneFunc3dSeedOutcome,
    out_dir: Path,
) -> LiftMaskResult:
    """Lift the confirmed 2D SAM mask to a raw-mesh fragment (deterministic)."""
    assets = resolve_frame_geometry_assets(scene, seed_outcome.seed_frame_id)
    return lift_mask_to_3d(
        LiftMaskArgs(
            frame_id=seed_outcome.seed_frame_id,
            candidate_id=seed_outcome.seed_candidate_id,
            mask_path=seed_outcome.seed_sam_mask_npz_path,
            depth_path=assets.depth_path,
            intrinsics_path=assets.intrinsics_path,
            pose_path=assets.pose_path,
        ),
        out_dir=out_dir,
        raw_mesh_path=scene.raw_mesh_path,
    )


def _anchor_motion_type(sample: SceneFunc3dSample) -> str:
    """Return the seed anchor's motion type (first motion hint, else empty)."""
    if sample.motion_hints:
        return sample.motion_hints[0].motion_type
    return ""


def run_samples(
    config: SceneFunc3dRunnerConfig,
    *,
    sample_ids: Sequence[str],
    executor: CodexExecutor,
    check_sidecars: bool = True,
    score: bool = True,
    continue_on_error: bool = True,
    workers: int = 1,
) -> SceneFunc3dBatchRunSummary:
    """Run and optionally score multiple SceneFunc3D samples."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    normalized_sample_ids = _normalized_batch_sample_ids(sample_ids)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if check_sidecars:
        check_sidecar_health(config.backend_config_path)

    if not continue_on_error or workers == 1 or len(normalized_sample_ids) == 1:
        results = tuple(
            _run_batch_sample(
                config,
                sample_id=sample_id,
                executor=executor,
                score=score,
                continue_on_error=continue_on_error,
            )
            for sample_id in normalized_sample_ids
        )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_batch_sample,
                    config,
                    sample_id=sample_id,
                    executor=executor,
                    score=score,
                    continue_on_error=continue_on_error,
                )
                for sample_id in normalized_sample_ids
            ]
            results = tuple(future.result() for future in futures)

    summary = _summarize_batch_results(results, scoring_enabled=score)
    _write_batch_summary(config.output_dir / "evaluation_summary.json", summary)
    return summary


def _run_batch_sample(
    config: SceneFunc3dRunnerConfig,
    *,
    sample_id: str,
    executor: CodexExecutor,
    score: bool,
    continue_on_error: bool,
) -> SceneFunc3dBatchSampleResult:
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
        return SceneFunc3dBatchSampleResult(
            sample_id=sample_id,
            status="failed",
            failure_stage="run",
            result_path=None,
            score=None,
            failure_path=failure_path,
            error=_format_batch_error(exc),
        )

    sample_score: SceneFunc3dScore | None = None
    if score:
        try:
            sample_score = score_result_file(
                data_root=config.dataset_root, result_path=result_path
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            return SceneFunc3dBatchSampleResult(
                sample_id=sample_id,
                status="failed",
                failure_stage="score",
                result_path=result_path,
                score=None,
                error=_format_batch_error(exc),
            )
    return SceneFunc3dBatchSampleResult(
        sample_id=sample_id,
        status="completed",
        failure_stage="",
        result_path=result_path,
        score=sample_score,
    )


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
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Number of concurrent SceneFunc3D samples to run for batch inputs.",
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
            workers=_namespace_int(args, "workers"),
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


def _namespace_int(args: argparse.Namespace, name: str) -> int:
    value: object = getattr(args, name)
    if not isinstance(value, int):
        raise TypeError(f"argparse field {name!r} must be an int")
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


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"workers must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"workers must be a positive integer, got {value}"
        )
    return value


__all__ = [
    "TASK_NAME",
    "DEFAULT_TOOL_CLI_MODULE",
    "SCENEFUNC3D_ALLOWED_TOOL_NAMES",
    "CodexTurnMetadataPayload",
    "SceneFunc3dMaskTask",
    "SceneFunc3dSeedOutcome",
    "SceneFunc3dRunnerConfig",
    "SceneFunc3dBatchSampleResult",
    "SceneFunc3dBatchSampleResultPayload",
    "SceneFunc3dBatchRunSummary",
    "SceneFunc3dBatchRunSummaryPayload",
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
