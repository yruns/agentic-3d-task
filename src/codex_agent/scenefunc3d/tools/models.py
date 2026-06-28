"""Shared types for the SceneFunc3D agent tool layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...errors import CodexAgentError


class ToolInputError(CodexAgentError):
    """Raised when SceneFunc3D tool arguments are invalid but recoverable."""


@runtime_checkable
class ToolPayload(Protocol):
    """A tool result that can render itself as a JSON-serializable mapping."""

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-ready dict at the CLI boundary."""
        ...


__all__ = ["ToolInputError", "ToolPayload"]
