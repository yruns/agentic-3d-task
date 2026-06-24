"""Exception hierarchy for the ``codex_agent`` package.

All errors raised by this package derive from :class:`CodexAgentError` so
callers can catch the whole family with a single ``except`` clause while still
being able to distinguish configuration problems from turn/parse failures.
"""

from __future__ import annotations


class CodexAgentError(Exception):
    """Base error for every failure originating in ``codex_agent``."""


class CodexConfigError(CodexAgentError):
    """Raised when runtime configuration or the environment is invalid.

    Examples: a missing ``CODEX_HOME`` directory, an unsupported sandbox mode,
    or the optional ``openai-codex`` dependency not being installed.
    """


class CodexTurnError(CodexAgentError):
    """Raised when a Codex SDK turn fails to produce a usable response."""


class CodexResponseError(CodexAgentError):
    """Raised when a Codex response cannot be parsed into the expected shape."""


class Nr3dDataError(CodexAgentError):
    """Raised when prepared NR3D scene/sample artifacts are missing or malformed."""


class OpenEqaDataError(CodexAgentError):
    """Raised when OpenEQA question or scene assets are missing or malformed."""


class OpenEqaJudgeError(CodexAgentError):
    """Raised when the OpenEQA LLM-as-judge cannot produce a usable score."""


__all__ = [
    "CodexAgentError",
    "CodexConfigError",
    "CodexTurnError",
    "CodexResponseError",
    "Nr3dDataError",
    "OpenEqaDataError",
    "OpenEqaJudgeError",
]
