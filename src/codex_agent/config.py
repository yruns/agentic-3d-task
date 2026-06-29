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
ReasoningEffort = Literal["", "minimal", "low", "medium", "high"]
#: Reasoning-summary verbosity. ``""`` means "do not request a summary" (the
#: model/provider default governs); the others map to the Codex SDK's
#: ``ReasoningSummary`` (``none`` explicitly disables summaries upstream).
ReasoningSummary = Literal["", "auto", "concise", "detailed", "none"]

_VALID_SANDBOX_MODES: frozenset[str] = frozenset(
    {"read_only", "workspace_write", "full_access"}
)
_VALID_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"", "minimal", "low", "medium", "high"}
)
_VALID_REASONING_SUMMARIES: frozenset[str] = frozenset(
    {"", "auto", "concise", "detailed", "none"}
)

#: Repository root, derived from this file's location (``src/codex_agent/config.py``).
DEFAULT_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_HOME: Path = DEFAULT_PROJECT_ROOT / ".codex-home"
DEFAULT_CODEX_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_CODEX_MODEL_PROVIDER = "modelhub_adapter"
DEFAULT_PREFIX_CACHE_SESSION_ID = "codex_agent"
#: Default reasoning effort when neither the CLI nor the environment overrides
#: it. ``medium`` is a deliberate, reproducible default; an empty string would
#: instead follow the model's own (possibly shifting) default.
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"

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
            prompt-only tasks that never touch the filesystem; ``workspace_write``
            is required for tool-using turns that shell out and write annotated
            images under the workspace.
        sandbox_network_access: When ``True`` and ``sandbox`` is
            ``workspace_write``, allow sandboxed commands to reach the network
            (needed by ``keyframe_selector``, which calls the parsing LLM).
        reasoning_effort: Reasoning effort passed to the model per turn (via the
            SDK ``effort`` argument, not a ``config.toml`` override). Defaults to
            ``medium`` (:data:`DEFAULT_REASONING_EFFORT`); set ``""`` to instead
            follow the model's own default.
        reasoning_summary: Reasoning-summary verbosity passed to the model per
            turn (via the SDK ``summary`` argument). ``""`` requests nothing
            (provider default); ``auto`` / ``concise`` / ``detailed`` ask the
            model to emit a human-readable summary of its reasoning, which the
            runtime captures into the turn metadata for tracing/debugging.
        enable_prefix_cache: Send ModelHub prefix-cache + log-id headers so the
            adapter can reuse the prompt prefix across turns.
        prefix_cache_session_id: Stable session id used for prefix caching.
        turn_timeout_s: Wall-clock budget for one agentic turn. ``0`` disables
            the budget; a positive value interrupts a turn that exceeds it so a
            runaway tool loop cannot stall the run, after which a finalization
            re-ask collects the answer from the evidence already gathered.
        max_tool_calls: Maximum tool actions (shell commands, image views, MCP /
            dynamic tool calls) the model may take in one turn before the runtime
            interrupts it. ``0`` disables the cap. This is the reliable backstop
            for the degenerate re-read loop that the wall-clock budget could not
            stop (the SDK ``turn_interrupt`` lands cleanly between the quick
            shell steps of a loop). A finalization re-ask then collects the
            answer from the evidence already gathered.
        max_repeated_tool_calls: Interrupt the turn once the *same* tool action
            (identical command / image path / tool+args) has occurred this many
            times. ``0`` disables it. Catches the pathological "run the identical
            command forever" rut directly, independent of the total cap.
        model_context_window: When positive, override Codex's
            ``model_context_window`` for the (custom) model name. Codex falls
            back to a conservative default for unrecognised model names; setting
            the real window is config hygiene. ``0`` leaves Codex's default.
        restrict_skills_to_project: When ``True`` the turn only exposes the
            project's own skills (passed via ``SkillInput`` / discovered under
            the repo ``.agents/skills``). Ambient skills are removed two ways:
            the app-server runs with an isolated ``HOME`` so user/global skills
            (``~/.agents/skills``, ``~/.codex/superpowers``) are not discovered,
            and the bundled ``.system`` skills (skill-creator, using-superpowers,
            ...) are disabled via the ``skills/config/write`` RPC. This stops the
            model from compulsively re-reading dozens of irrelevant ``SKILL.md``
            files (see ``docs/benchmark/nr3d`` trace analysis).
        keep_run_home: Keep the per-turn ``CODEX_HOME`` copy on disk for
            debugging instead of deleting it after the turn.
        copy_auth_file: Copy ``auth.json`` from the base ``CODEX_HOME`` into the
            per-turn run home. This is off by default because the ModelHub
            adapter path keeps secrets outside Codex run homes; enable it only
            for explicit runs that use the standard Codex ``openai`` provider.
    """

    project_root: Path = DEFAULT_PROJECT_ROOT
    codex_home: Path = DEFAULT_CODEX_HOME
    model: str = DEFAULT_CODEX_MODEL
    model_provider: str = DEFAULT_CODEX_MODEL_PROVIDER
    sandbox: SandboxMode = "read_only"
    sandbox_network_access: bool = False
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT
    reasoning_summary: ReasoningSummary = ""
    enable_prefix_cache: bool = True
    prefix_cache_session_id: str = DEFAULT_PREFIX_CACHE_SESSION_ID
    turn_timeout_s: float = 0.0
    max_tool_calls: int = 0
    max_repeated_tool_calls: int = 0
    model_context_window: int = 0
    restrict_skills_to_project: bool = True
    keep_run_home: bool = False
    copy_auth_file: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "codex_home", Path(self.codex_home))
        if not self.model:
            raise CodexConfigError("model must be a non-empty string")
        if not self.model_provider:
            raise CodexConfigError("model_provider must be a non-empty string")
        if self.reasoning_effort not in _VALID_REASONING_EFFORTS:
            raise CodexConfigError(
                f"reasoning_effort={self.reasoning_effort!r} is invalid; expected "
                f"one of {sorted(_VALID_REASONING_EFFORTS)}"
            )
        if self.reasoning_summary not in _VALID_REASONING_SUMMARIES:
            raise CodexConfigError(
                f"reasoning_summary={self.reasoning_summary!r} is invalid; expected "
                f"one of {sorted(_VALID_REASONING_SUMMARIES)}"
            )
        if self.turn_timeout_s < 0:
            raise CodexConfigError(
                f"turn_timeout_s must be non-negative, got {self.turn_timeout_s}"
            )
        if self.max_tool_calls < 0:
            raise CodexConfigError(
                f"max_tool_calls must be non-negative, got {self.max_tool_calls}"
            )
        if self.max_repeated_tool_calls < 0:
            raise CodexConfigError(
                "max_repeated_tool_calls must be non-negative, got "
                f"{self.max_repeated_tool_calls}"
            )
        if self.model_context_window < 0:
            raise CodexConfigError(
                "model_context_window must be non-negative, got "
                f"{self.model_context_window}"
            )
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
            sandbox_network_access=_env_bool(
                os.environ.get("CODEX_AGENT_SANDBOX_NETWORK"), default=False
            ),
            reasoning_effort=_reasoning_effort_from_env(
                os.environ.get("CODEX_AGENT_REASONING_EFFORT"),
                default=DEFAULT_REASONING_EFFORT,
            ),
            reasoning_summary=_reasoning_summary_from_env(
                os.environ.get("CODEX_AGENT_REASONING_SUMMARY")
            ),
            enable_prefix_cache=_env_bool(
                os.environ.get("CODEX_AGENT_ENABLE_PREFIX_CACHE"), default=True
            ),
            prefix_cache_session_id=os.environ.get(
                "CODEX_AGENT_PREFIX_CACHE_SESSION_ID",
                DEFAULT_PREFIX_CACHE_SESSION_ID,
            ),
            turn_timeout_s=_env_float(
                os.environ.get("CODEX_AGENT_TURN_TIMEOUT_S"), default=0.0
            ),
            max_tool_calls=_env_int(
                os.environ.get("CODEX_AGENT_MAX_TOOL_CALLS"), default=0
            ),
            max_repeated_tool_calls=_env_int(
                os.environ.get("CODEX_AGENT_MAX_REPEATED_TOOL_CALLS"), default=0
            ),
            model_context_window=_env_int(
                os.environ.get("CODEX_AGENT_MODEL_CONTEXT_WINDOW"), default=0
            ),
            restrict_skills_to_project=_env_bool(
                os.environ.get("CODEX_AGENT_RESTRICT_SKILLS"), default=True
            ),
            keep_run_home=_env_bool(
                os.environ.get("CODEX_AGENT_KEEP_RUN_HOME"), default=False
            ),
            copy_auth_file=_env_bool(
                os.environ.get("CODEX_AGENT_COPY_AUTH"), default=False
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


def _reasoning_effort_from_env(
    raw: str | None, *, default: ReasoningEffort = ""
) -> ReasoningEffort:
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized not in _VALID_REASONING_EFFORTS:
        raise CodexConfigError(
            f"CODEX_AGENT_REASONING_EFFORT={raw!r} is invalid; expected one of "
            f"{sorted(_VALID_REASONING_EFFORTS - {''})}"
        )
    # normalized is one of the valid literals here.
    return normalized  # type: ignore[return-value]


def _reasoning_summary_from_env(raw: str | None) -> ReasoningSummary:
    if raw is None or not raw.strip():
        return ""
    normalized = raw.strip().lower()
    if normalized not in _VALID_REASONING_SUMMARIES:
        raise CodexConfigError(
            f"CODEX_AGENT_REASONING_SUMMARY={raw!r} is invalid; expected one of "
            f"{sorted(_VALID_REASONING_SUMMARIES - {''})}"
        )
    # normalized is one of the valid literals here.
    return normalized  # type: ignore[return-value]


def _env_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(raw: str | None, *, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise CodexConfigError(
            f"CODEX_AGENT_TURN_TIMEOUT_S={raw!r} must be a number"
        ) from exc


def _env_int(raw: str | None, *, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise CodexConfigError(f"expected an integer, got {raw!r}") from exc


__all__ = [
    "CodexAgentConfig",
    "SandboxMode",
    "ReasoningEffort",
    "ReasoningSummary",
    "sanitize_session_id",
    "DEFAULT_PROJECT_ROOT",
    "DEFAULT_CODEX_HOME",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_MODEL_PROVIDER",
    "DEFAULT_PREFIX_CACHE_SESSION_ID",
    "DEFAULT_REASONING_EFFORT",
]
