"""Structured failure artifacts for SceneFunc3D sample runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..errors import SceneFunc3dDataError

SceneFunc3dFailureStage: TypeAlias = Literal["codex_turn", "response_validation", "run"]


class SceneFunc3dFailureTurnPayload(BaseModel):
    """Nullable Codex turn metadata stored with a failed SceneFunc3D run."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    usage: JsonValue = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_summary: str | None = None
    run_home: str | None = None
    attempts: tuple[JsonValue, ...] = Field(default_factory=tuple)


class SceneFunc3dFailureArtifact(BaseModel):
    """Durable JSON artifact describing one failed SceneFunc3D sample run."""

    model_config = ConfigDict(extra="forbid")

    task_name: str
    sample_id: str
    status: Literal["failed"] = "failed"
    failure_stage: SceneFunc3dFailureStage
    error_type: str
    error_message: str
    events_path: str
    tool_context_path: str
    turn: SceneFunc3dFailureTurnPayload = Field(
        default_factory=SceneFunc3dFailureTurnPayload
    )


def write_failure_artifact(path: Path, artifact: SceneFunc3dFailureArtifact) -> Path:
    """Write a SceneFunc3D failure artifact as UTF-8 JSON with a trailing newline."""
    normalized_path = Path(path)
    try:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SceneFunc3dDataError(
            "could not write SceneFunc3D failure artifact: "
            f"path={normalized_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return normalized_path


__all__ = [
    "SceneFunc3dFailureArtifact",
    "SceneFunc3dFailureStage",
    "SceneFunc3dFailureTurnPayload",
    "write_failure_artifact",
]
