"""Single-case SceneFunc3D mask-generation runner."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, TypeAlias, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StringConstraints

from ..errors import CodexResponseError
from ..json_extraction import extract_json_object
from ..models import CodexTurnMetadata, CodexTurnRequest
from ..tasks.base import CodexExecutor
from .backends.config import load_backend_settings
from .playbook import SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, load_sample, safe_sample_id, scene_dir_for
from .servers.schemas import HealthResponse

TASK_NAME = "scenefunc3d_mask_generation"
DEFAULT_TOOL_CLI_MODULE = "codex_agent.scenefunc3d.tools"

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
    turn: CodexTurnMetadataPayload


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
        return SceneFunc3dMaskOutcome(
            mask_artifact_path=mask_artifact_path,
            mask_npz_path=mask_npz_path,
            mask_ply_path=mask_ply_path,
            selected_frame_ids=tuple(decision.selected_frame_ids),
            accepted_fragment_ids=tuple(decision.accepted_fragment_ids),
            confidence=decision.confidence,
            uncertainties=tuple(decision.uncertainties),
        )

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
            "\nYou are NOT exploring or editing a codebase. Everything needed "
            "for this single SceneFunc3D case is in this message and the CLI "
            "tools above.\n"
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
    sample_output_dir = config.output_dir / safe_sample_id(sample_id)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    task = SceneFunc3dMaskTask(
        sample=sample,
        scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
        output_dir=sample_output_dir,
        backend_config_path=config.backend_config_path,
    )
    result = executor.execute(task)
    result_path = sample_output_dir / "result.json"
    result_payload: SceneFunc3dRunResultPayload = {
        "task_name": result.task_name,
        "sample_id": sample_id,
        "outcome": result.outcome.to_payload(),
        "turn": _metadata_payload(result.turn.metadata),
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one SceneFunc3D sample from the command line."""
    from ..config import CodexAgentConfig
    from ..runtime import CodexAgentRuntime

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SceneFunc3dRunnerConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend_config_path=args.backend_config,
    )
    executor = CodexAgentRuntime(CodexAgentConfig.from_env())
    result_path = run_single_sample(
        config,
        sample_id=args.sample_id,
        executor=executor,
        check_sidecars=not args.skip_sidecar_health_check,
    )
    print(json.dumps({"result_path": str(result_path)}, ensure_ascii=False))
    return 0


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
    "SceneFunc3dMaskDecision",
    "CodexTurnMetadataPayload",
    "SceneFunc3dMaskOutcome",
    "SceneFunc3dMaskOutcomePayload",
    "SceneFunc3dMaskTask",
    "SceneFunc3dRunnerConfig",
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
