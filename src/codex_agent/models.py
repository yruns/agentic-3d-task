"""Task-agnostic data models for one Codex Agent SDK turn.

These types are deliberately free of any NR3D / visual-grounding concepts so the
same :class:`codex_agent.runtime.CodexAgentRuntime` can drive any future task
(QA, navigation, manipulation, ...). Task-specific shapes live in the task
packages (e.g. ``codex_agent.nr3d``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from .errors import CodexConfigError

OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True)
class CodexSkill:
    """A Codex skill (a ``SKILL.md`` file) attached to a turn."""

    name: str
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.name:
            raise CodexConfigError("CodexSkill.name must be a non-empty string")


@dataclass(frozen=True)
class CodexTurnRequest:
    """Everything needed to issue one Codex turn.

    Attributes:
        prompt: The user-turn text.
        output_schema: JSON schema the model is asked to satisfy.
        skills: Skills to attach (loaded before the prompt).
        image_paths: Local images attached to the turn (e.g. a BEV render).
    """

    prompt: str
    output_schema: Mapping[str, Any]
    skills: Sequence[CodexSkill] = field(default_factory=tuple)
    image_paths: Sequence[Path] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise CodexConfigError("CodexTurnRequest.prompt must be non-empty")
        if not self.output_schema:
            raise CodexConfigError("CodexTurnRequest.output_schema must be non-empty")
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(
            self, "image_paths", tuple(Path(p) for p in self.image_paths)
        )


@dataclass(frozen=True)
class CodexTurnMetadata:
    """Bookkeeping for a completed Codex turn (no model output payload)."""

    turn_id: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    usage: Mapping[str, Any] | None = None
    run_home: str | None = None
    attempts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the metadata."""
        return {
            "turn_id": self.turn_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "usage": dict(self.usage) if self.usage is not None else None,
            "run_home": self.run_home,
            "attempts": [dict(attempt) for attempt in self.attempts],
        }


@dataclass(frozen=True)
class CodexTurnResult:
    """The raw final response text plus metadata for one Codex turn."""

    final_response: str
    metadata: CodexTurnMetadata


@dataclass(frozen=True)
class CodexTaskResult(Generic[OutcomeT]):
    """A parsed task outcome bundled with the turn that produced it."""

    task_name: str
    outcome: OutcomeT
    turn: CodexTurnResult


__all__ = [
    "CodexSkill",
    "CodexTurnRequest",
    "CodexTurnMetadata",
    "CodexTurnResult",
    "CodexTaskResult",
    "OutcomeT",
]
