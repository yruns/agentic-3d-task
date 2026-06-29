"""Single-case SceneFunc3D mask-generation runner."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, TypeAlias, TypedDict
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
from .evaluation.scorer import score_result_file
from .final_mask_artifacts import (
    ValidatedFinalMaskArtifact,
    validate_final_mask_artifact,
)
from .playbook import SCENEFUNC3D_TOOL_NAMES, SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, load_sample, scene_dir_for
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
        return self.parse_response(json.dumps(outcome.to_payload(), ensure_ascii=False))

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
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SceneFunc3D single-case runner CLI parser."""
    parser = argparse.ArgumentParser(
        prog="codex_agent.scenefunc3d.runner",
        description="Run one SceneFunc3D mask-generation case with Codex.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="SceneFunc3D dataset root containing per-visit scene directories.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="SceneFunc3D sample id in '<visit_id>::<desc_id>' form.",
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
    """Run one SceneFunc3D sample from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SceneFunc3dRunnerConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend_config_path=args.backend_config,
    )
    executor = _build_executor()
    result_path = run_single_sample(
        config,
        sample_id=args.sample_id,
        executor=executor,
        check_sidecars=not args.skip_sidecar_health_check,
    )
    if _namespace_bool(args, "score"):
        payload: SceneFunc3dCliRunPayload | SceneFunc3dCliRunAndScorePayload = (
            _run_and_score_payload(
                data_root=config.dataset_root, result_path=result_path
            )
        )
    else:
        payload = _run_payload(result_path)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


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
    "SceneFunc3dRunArtifactPayload",
    "SceneFunc3dRunResultPayload",
    "build_arg_parser",
    "build_prompt",
    "check_sidecar_health",
    "load_runner_sample",
    "main",
    "run_single_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
