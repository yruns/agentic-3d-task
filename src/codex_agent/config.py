"""Runtime configuration for :class:`codex_agent.runtime.CodexAgentRuntime`.

The Codex Agent SDK talks to the model through a Codex *model provider* that is
configured in ``$CODEX_HOME/config.toml`` (for this repository, a local
ModelHub adapter). API keys therefore live inside that adapter, **not** in this
config object — keeping secrets out of the Python process and out of git.

All environment-variable reads are centralized in :meth:`CodexAgentConfig.from_env`
so business code never reaches into ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import CodexConfigError

SandboxMode = Literal["read_only", "workspace_write", "full_access"]

_VALID_SANDBOX_MODES: frozenset[str] = frozenset(
    {"read_only", "workspace_write", "full_access"}
)

#: Repository root, derived from this file's location (``src/codex_agent/config.py``).
DEFAULT_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_HOME: Path = DEFAULT_PROJECT_ROOT / ".codex-home"
DEFAULT_CODEX_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_CODEX_MODEL_PROVIDER = "modelhub_adapter"
DEFAULT_PREFIX_CACHE_SESSION_ID = "codex_agent"

#: Characters allowed in a ModelHub prefix-cache session id (sent as an HTTP
#: header value); anything else is replaced with ``_``.
_SESSION_ID_ALLOWED_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)
_SESSION_ID_MAX_LENGTH = 128


def sanitize_session_id(value: str) -> str:
    """Return a header-safe ModelHub session id.

    Replaces disallowed characters with ``_``, trims separator characters from
    the ends, and caps the length. Falls back to the default id when the input
    reduces to an empty string.
    """
    raw = str(value or DEFAULT_PREFIX_CACHE_SESSION_ID).strip()
    safe = "".join(ch if ch in _SESSION_ID_ALLOWED_CHARS else "_" for ch in raw)
    safe = safe.strip("._:-")[:_SESSION_ID_MAX_LENGTH]
    return safe or DEFAULT_PREFIX_CACHE_SESSION_ID


@dataclass(frozen=True)
class CodexAgentConfig:
    """Immutable runtime configuration for the Codex Agent SDK driver.

    Attributes:
        project_root: Working directory passed to Codex (and the trusted
            project path in ``config.toml``).
        codex_home: ``CODEX_HOME`` base directory holding ``config.toml`` and
            per-run isolated state.
        model: Codex model name.
        model_provider: Codex model-provider key defined in ``config.toml``.
        sandbox: Codex sandbox mode for the turn. ``read_only`` is correct for
            prompt-only tasks that never touch the filesystem.
        enable_prefix_cache: Send ModelHub prefix-cache + log-id headers so the
            adapter can reuse the prompt prefix across turns.
        prefix_cache_session_id: Stable session id used for prefix caching.
        keep_run_home: Keep the per-turn ``CODEX_HOME`` copy on disk for
            debugging instead of deleting it after the turn.
    """

    project_root: Path = DEFAULT_PROJECT_ROOT
    codex_home: Path = DEFAULT_CODEX_HOME
    model: str = DEFAULT_CODEX_MODEL
    model_provider: str = DEFAULT_CODEX_MODEL_PROVIDER
    sandbox: SandboxMode = "read_only"
    enable_prefix_cache: bool = True
    prefix_cache_session_id: str = DEFAULT_PREFIX_CACHE_SESSION_ID
    keep_run_home: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "codex_home", Path(self.codex_home))
        if not self.model:
            raise CodexConfigError("model must be a non-empty string")
        if not self.model_provider:
            raise CodexConfigError("model_provider must be a non-empty string")
        if self.sandbox not in _VALID_SANDBOX_MODES:
            raise CodexConfigError(
                f"sandbox={self.sandbox!r} is invalid; "
                f"expected one of {sorted(_VALID_SANDBOX_MODES)}"
            )
        object.__setattr__(
            self,
            "prefix_cache_session_id",
            sanitize_session_id(self.prefix_cache_session_id),
        )

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
        codex_home: str | Path | None = None,
    ) -> CodexAgentConfig:
        """Build a config from ``CODEX_AGENT_*`` environment variables.

        Explicit ``project_root`` / ``codex_home`` arguments win over the
        environment, which in turn wins over the packaged defaults.
        """
        resolved_root = Path(
            project_root
            or os.environ.get("CODEX_AGENT_PROJECT_ROOT")
            or DEFAULT_PROJECT_ROOT
        )
        resolved_home = Path(
            codex_home or os.environ.get("CODEX_HOME") or resolved_root / ".codex-home"
        )
        return cls(
            project_root=resolved_root,
            codex_home=resolved_home,
            model=os.environ.get("CODEX_AGENT_MODEL", DEFAULT_CODEX_MODEL),
            model_provider=os.environ.get(
                "CODEX_AGENT_MODEL_PROVIDER", DEFAULT_CODEX_MODEL_PROVIDER
            ),
            sandbox=_sandbox_from_env(os.environ.get("CODEX_AGENT_SANDBOX")),
            enable_prefix_cache=_env_bool(
                os.environ.get("CODEX_AGENT_ENABLE_PREFIX_CACHE"), default=True
            ),
            prefix_cache_session_id=os.environ.get(
                "CODEX_AGENT_PREFIX_CACHE_SESSION_ID",
                DEFAULT_PREFIX_CACHE_SESSION_ID,
            ),
            keep_run_home=_env_bool(
                os.environ.get("CODEX_AGENT_KEEP_RUN_HOME"), default=False
            ),
        )


def _sandbox_from_env(raw: str | None) -> SandboxMode:
    if raw is None or not raw.strip():
        return "read_only"
    normalized = raw.strip().lower().replace("-", "_")
    aliases: dict[str, SandboxMode] = {
        "read_only": "read_only",
        "readonly": "read_only",
        "workspace": "workspace_write",
        "workspace_write": "workspace_write",
        "full_access": "full_access",
        "danger_full_access": "full_access",
    }
    if normalized not in aliases:
        raise CodexConfigError(
            f"CODEX_AGENT_SANDBOX={raw!r} is invalid; expected one of "
            "read_only, workspace_write, or full_access"
        )
    return aliases[normalized]


def _env_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


__all__ = [
    "CodexAgentConfig",
    "SandboxMode",
    "sanitize_session_id",
    "DEFAULT_PROJECT_ROOT",
    "DEFAULT_CODEX_HOME",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_MODEL_PROVIDER",
    "DEFAULT_PREFIX_CACHE_SESSION_ID",
]
