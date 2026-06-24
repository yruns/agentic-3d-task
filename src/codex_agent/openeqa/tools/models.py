"""Shared types for the OpenEQA agent tool layer.

``ToolPayload`` is the common contract every tool result satisfies: a typed,
frozen result object that serializes itself to a JSON-ready ``dict`` at the CLI
boundary. ``ToolInputError`` marks agent-recoverable problems (an unknown frame
id, an empty query, a missing object id) that should be reported back to the
model as a normal tool result rather than crashing the turn.

These mirror the NR3D tool primitives (``codex_agent.nr3d.tools.models``); the
two task families keep parallel, decoupled tool layers on purpose.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ...errors import CodexAgentError


class ToolInputError(CodexAgentError):
    """Raised when tool arguments are invalid but the agent can recover.

    The dispatcher converts these into an ``{"error": ...}`` JSON result on
    stdout (exit code 0) so the agent can read the message and adjust its next
    call instead of treating it as a hard environment failure.
    """


@runtime_checkable
class ToolPayload(Protocol):
    """A tool result that can render itself as a JSON-serializable mapping."""

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-ready dict (the serialization boundary for the CLI)."""
        ...


__all__ = ["ToolInputError", "ToolPayload"]
