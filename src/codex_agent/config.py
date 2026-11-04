"""Runtime configuration for :class:`codex_agent.runtime.CodexAgentRuntime`.

The Codex Agent SDK talks to the model through a Codex *model provider* that is
configured in ``$CODEX_HOME/config.toml`` (by default, ``codex_adapter``).
Upstream API keys live only in the adapter registry, never in client headers.
An optional local adapter token is distinct from those upstream credentials.

All environment-variable reads are centralized in :meth:`CodexAgentConfig.from_env`
so business code never reaches into ``os.environ`` directly.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

from .errors import CodexConfigError

SandboxMode = Literal["read_only", "workspace_write", "full_access"]
ReasoningEffort = Literal["", "provider-default", "minimal", "low", "medium", "high"]
#: Reasoning-summary verbosity. ``""`` means "do not request a summary" (the
#: model/provider default governs); the others map to the Codex SDK's
#: ``ReasoningSummary`` (``none`` explicitly disables summaries upstream).
ReasoningSummary = Literal["", "auto", "concise", "detailed", "none"]

_VALID_SANDBOX_MODES: frozenset[str] = frozenset(
    {"read_only", "workspace_write", "full_access"}
)
_VALID_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"", "provider-default", "minimal", "low", "medium", "high"}
)
_VALID_REASONING_SUMMARIES: frozenset[str] = frozenset(
    {"", "auto", "concise", "detailed", "none"}
)

#: Repository root, derived from this file's location (``src/codex_agent/config.py``).
DEFAULT_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_HOME: Path = DEFAULT_PROJECT_ROOT / ".codex-home"
DEFAULT_CODEX_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_CODEX_MODEL_PROVIDER = "codex_adapter"
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
            ``medium`` (:data:`DEFAULT_REASONING_EFFORT`); set ``""`` or
            ``"provider-default"`` explicitly to use the provider's own default.
        reasoning_summary: Reasoning-summary verbosity passed to the model per
            turn (via the SDK ``summary`` argument). ``""`` requests nothing
            (provider default); ``auto`` / ``concise`` / ``detailed`` ask the
            model to emit a human-readable summary of its reasoning, which the
            runtime captures into the turn metadata for tracing/debugging.
        enable_prefix_cache: Enable legacy ModelHub cache/log headers. The new
            adapter owns cache policy; its routing UUID is always sent.
        prefix_cache_session_id: Legacy trace label, never a routing identity.
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
        adapter_token: Optional raw local ``CODEX_ADAPTER_TOKEN``, excluded from
            repr and converted into a Bearer header by the runtime. Not an
            upstream model API key.
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
    adapter_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "codex_home", Path(self.codex_home))
        _validate_config(self)
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
            **_runtime_environment(os.environ),
            **_provider_environment(os.environ),
        )


class _RuntimeEnvironment(TypedDict):
    sandbox: SandboxMode
    sandbox_network_access: bool
    turn_timeout_s: float
    max_tool_calls: int
    max_repeated_tool_calls: int
    model_context_window: int
    restrict_skills_to_project: bool
    keep_run_home: bool
    copy_auth_file: bool


class _ProviderEnvironment(TypedDict):
    model: str
    model_provider: str
    reasoning_effort: ReasoningEffort
    reasoning_summary: ReasoningSummary
    enable_prefix_cache: bool
    prefix_cache_session_id: str
    adapter_token: str


def _runtime_environment(env: Mapping[str, str]) -> _RuntimeEnvironment:
    return {
        "sandbox": _sandbox_from_env(env.get("CODEX_AGENT_SANDBOX")),
        "sandbox_network_access": _env_bool(
            env.get("CODEX_AGENT_SANDBOX_NETWORK"), default=False
        ),
        "turn_timeout_s": _env_float(
            env.get("CODEX_AGENT_TURN_TIMEOUT_S"), default=0.0
        ),
        "max_tool_calls": _env_int(env.get("CODEX_AGENT_MAX_TOOL_CALLS"), default=0),
        "max_repeated_tool_calls": _env_int(
            env.get("CODEX_AGENT_MAX_REPEATED_TOOL_CALLS"), default=0
        ),
        "model_context_window": _env_int(
            env.get("CODEX_AGENT_MODEL_CONTEXT_WINDOW"), default=0
        ),
        "restrict_skills_to_project": _env_bool(
            env.get("CODEX_AGENT_RESTRICT_SKILLS"), default=True
        ),
        "keep_run_home": _env_bool(env.get("CODEX_AGENT_KEEP_RUN_HOME"), default=False),
        "copy_auth_file": _env_bool(env.get("CODEX_AGENT_COPY_AUTH"), default=False),
    }


def _provider_environment(env: Mapping[str, str]) -> _ProviderEnvironment:
    return {
        "model": env.get("CODEX_AGENT_MODEL", DEFAULT_CODEX_MODEL),
        "model_provider": env.get(
            "CODEX_AGENT_MODEL_PROVIDER", DEFAULT_CODEX_MODEL_PROVIDER
        ),
        "reasoning_effort": _reasoning_effort_from_env(
            env.get("CODEX_AGENT_REASONING_EFFORT"), default=DEFAULT_REASONING_EFFORT
        ),
        "reasoning_summary": _reasoning_summary_from_env(
            env.get("CODEX_AGENT_REASONING_SUMMARY")
        ),
        "enable_prefix_cache": _env_bool(
            env.get("CODEX_AGENT_ENABLE_PREFIX_CACHE"), default=True
        ),
        "prefix_cache_session_id": env.get(
            "CODEX_AGENT_PREFIX_CACHE_SESSION_ID", DEFAULT_PREFIX_CACHE_SESSION_ID
        ),
        "adapter_token": env.get("CODEX_ADAPTER_TOKEN", ""),
    }


def _validate_config(config: CodexAgentConfig) -> None:
    for name, value in (
        ("model", config.model),
        ("model_provider", config.model_provider),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CodexConfigError(f"{name} must be a non-empty string")
    for name, value, allowed in (
        ("sandbox", config.sandbox, _VALID_SANDBOX_MODES),
        ("reasoning_effort", config.reasoning_effort, _VALID_REASONING_EFFORTS),
        ("reasoning_summary", config.reasoning_summary, _VALID_REASONING_SUMMARIES),
    ):
        if value not in allowed:
            raise CodexConfigError(
                f"{name}={value!r} is invalid; expected one of {sorted(allowed)}"
            )
    for numeric_name, numeric_value in (
        ("turn_timeout_s", config.turn_timeout_s),
        ("max_tool_calls", config.max_tool_calls),
        ("max_repeated_tool_calls", config.max_repeated_tool_calls),
        ("model_context_window", config.model_context_window),
    ):
        if numeric_value < 0 or not math.isfinite(numeric_value):
            raise CodexConfigError(
                f"{numeric_name} must be finite and non-negative, got {numeric_value}"
            )
    if config.adapter_token and not re.fullmatch(
        r"[A-Za-z0-9._~+/-]+=*", config.adapter_token
    ):
        raise CodexConfigError(
            "adapter_token must be a raw Bearer token without whitespace or a Bearer prefix"
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
    if raw is None:
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
