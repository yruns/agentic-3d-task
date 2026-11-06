"""``CodexAgentRuntime`` — a task-agnostic driver for the Codex Agent SDK.

The runtime owns everything that is specific to *running a Codex turn* and
nothing about any particular task:

* isolating ``CODEX_HOME`` per turn (the Codex app-server keeps SQLite state in
  ``CODEX_HOME``; sharing one home across thread workers races its migrations),
* assigning a fresh adapter routing UUID to each independent SDK thread,
* attaching skills / prompt / images and requesting a structured answer,
* issuing a single finalization re-ask when the answer is unusable.

Task logic is injected through the :class:`codex_agent.tasks.base.CodexTask`
protocol, so the same runtime can drive NR3D visual grounding today and other
task families later.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from openai_codex import (
    Codex,
    CodexConfig,
    InputItem,
    LocalImageInput,
    Sandbox,
    SkillInput,
    TextInput,
    Thread,
    TurnResult,
)
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionThreadItem,
    DynamicToolCallThreadItem,
    ImageViewThreadItem,
    ItemCompletedNotification,
    McpToolCallThreadItem,
    MessagePhase,
    ReasoningEffort,
    ReasoningSummary,
    ReasoningThreadItem,
    SkillsConfigWriteResponse,
    ThreadItem,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStatus,
)
from openai_codex.models import Notification

from .config import CodexAgentConfig
from .errors import CodexConfigError, CodexTurnError
from .models import (
    CodexTaskResult,
    CodexTurnMetadata,
    CodexTurnRequest,
    CodexTurnResult,
)
from .provider_catalog import prepare_provider_catalog
from .provider_environment import (
    ISOLATED_HOME_DIRNAME as _ISOLATED_HOME_DIRNAME,
)
from .provider_environment import (
    MODELHUB_EXTRA_HEADER_ENV,
    MODELHUB_LOGID_ENV,
    build_provider_overrides,
    build_turn_environment,
)
from .tasks.base import CodexTask, OutcomeT_co

#: Files copied from the base ``CODEX_HOME`` into each per-turn run home.
_RUN_HOME_SEED_FILES: tuple[str, ...] = (
    "config.toml",
    "installation_id",
    ".personality_migration",
)
_PRESERVED_RUN_HOME_PRUNE_DIRS: tuple[str, ...] = (".tmp", "tmp")
_PRESERVED_RUN_HOME_PRUNE_FILE_PATTERNS: tuple[str, ...] = (
    "*.sqlite",
    "*.sqlite-wal",
    "*.sqlite-shm",
)

#: Bundled ``.system`` skills the Codex app-server installs into every run home.
#: They are irrelevant to focused task agents and are disabled (via the
#: ``skills/config/write`` RPC) when ``restrict_skills_to_project`` is set, so the
#: model is not tempted to read them. HOME isolation removes the much larger
#: user/global skill set; these five are the only ones that live inside CODEX_HOME.
_BUNDLED_SYSTEM_SKILLS: tuple[str, ...] = (
    "imagegen",
    "openai-docs",
    "plugin-creator",
    "skill-creator",
    "skill-installer",
)

#: Tool actions allowed *after* an interrupt request before the runtime gives up
#: draining the stream. A cleanly-interrupted turn emits ``turn/completed`` within
#: a step or two; this caps the rare case of a server that ignores the interrupt.
_POST_INTERRUPT_GRACE_ACTIONS = 8


class _TurnHandle(Protocol):
    """The slice of :class:`openai_codex.TurnHandle` the streaming guard drives.

    Declaring the structural interface (interrupt + notification stream) instead
    of the concrete ``TurnHandle`` keeps the guard exercisable with a lightweight
    fake handle in tests while still binding to the real SDK handle in
    production, where ``thread.turn(...)`` returns a ``TurnHandle``.
    """

    def interrupt(self) -> object: ...

    def stream(self) -> Iterator[Notification]: ...


@dataclass
class _ToolCallLoopGuard:
    """Bound a single turn's tool actions to break degenerate re-read loops.

    Counts tool actions (shell commands, image views, MCP / dynamic tool calls)
    as the turn streams and signals an interrupt when either the running total or
    any single repeated action crosses its threshold. A threshold of ``0``
    disables that check; ``active`` is ``False`` only when both are disabled.
    """

    max_tool_calls: int
    max_repeated_calls: int
    total: int = 0
    _counts: dict[str, int] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.max_tool_calls > 0 or self.max_repeated_calls > 0

    def observe(self, signature: str) -> str | None:
        """Record one tool action; return an interrupt reason when tripped."""
        self.total += 1
        repeats = self._counts.get(signature, 0) + 1
        self._counts[signature] = repeats
        if self.max_tool_calls > 0 and self.total > self.max_tool_calls:
            return f"exceeded max_tool_calls={self.max_tool_calls}"
        if self.max_repeated_calls > 0 and repeats >= self.max_repeated_calls:
            return (
                f"repeated an identical tool action {repeats} times "
                f"(max_repeated_tool_calls={self.max_repeated_calls})"
            )
        return None


@dataclass
class _TurnInterruptState:
    """One-shot interrupt flag shared between the stream loop and a watchdog."""

    interrupted: bool = False
    reason: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class _GuardedTurnResult:
    """A ``TurnResult``-shaped value produced by the streaming guard path.

    Exposes exactly the attributes :meth:`CodexAgentRuntime._build_metadata`
    reads, so the guarded path and the plain ``thread.run`` fast path are
    interchangeable downstream.
    """

    final_response: str | None
    id: str | None = None
    status: TurnStatus | None = None
    error: TurnError | None = None
    duration_ms: int | None = None
    usage: ThreadTokenUsage | None = None
    reasoning_summary: str | None = None


class CodexAgentRuntime:
    """Drive single Codex Agent SDK turns and parse them into task outcomes."""

    def __init__(self, config: CodexAgentConfig | None = None) -> None:
        self.config = config or CodexAgentConfig()

    def execute(self, task: CodexTask[OutcomeT_co]) -> CodexTaskResult[OutcomeT_co]:
        """Run ``task`` end to end: build a turn, run it, parse the outcome."""
        request = task.build_turn_request()
        turn = self.run_turn(request, response_validator=task.is_valid_response)
        outcome = task.parse_response(turn.final_response)
        return CodexTaskResult(task_name=task.task_name, outcome=outcome, turn=turn)

    def run_turn(
        self,
        request: CodexTurnRequest,
        *,
        response_validator: Callable[[str], bool] | None = None,
        max_finalization_retries: int = 1,
    ) -> CodexTurnResult:
        """Run one independent SDK thread, sharing its UUID across all attempts.

        Capability/configuration failures are rejected before SDK startup. Tool
        continuations and finalization keep one routing identity even when the
        adapter cache is disabled; another run_turn always gets a fresh UUID.
        """
        if max_finalization_retries < 0:
            raise CodexConfigError("max_finalization_retries must be non-negative")
        self._validate_request(request)
        run_home = self._prepare_run_home()
        session_id = str(uuid.uuid4())
        try:
            catalog_overrides = prepare_provider_catalog(self.config, run_home, request)
            with Codex(
                config=CodexConfig(
                    config_overrides=self._build_config_overrides() + catalog_overrides,
                    cwd=str(self.config.project_root),
                    env=self._build_turn_env(run_home, session_id=session_id),
                )
            ) as client:
                attempts = self._run_thread(
                    client,
                    request,
                    response_validator=response_validator,
                    max_finalization_retries=max_finalization_retries,
                )
        finally:
            self._cleanup_run_home(run_home)
        result = attempts[-1]
        if result.final_response is None:
            raise CodexTurnError(
                "Codex turn completed without a final_response; "
                f"status={_status_str(result.status)} error={result.error!r}"
            )
        return CodexTurnResult(
            final_response=str(result.final_response),
            metadata=self._build_metadata(
                result, attempts, run_home, session_id=session_id
            ),
        )

    def _run_thread(
        self,
        client: Codex,
        request: CodexTurnRequest,
        *,
        response_validator: Callable[[str], bool] | None,
        max_finalization_retries: int,
    ) -> list[TurnResult | _GuardedTurnResult]:
        if self.config.restrict_skills_to_project:
            self._disable_ambient_system_skills(client)
        # Set sandbox mode once. A per-turn SDK Sandbox enum would silently
        # replace the configured network-access grant with its default False.
        thread = client.thread_start(
            model=self.config.model,
            model_provider=self.config.model_provider,
            sandbox=self._sandbox(),
            cwd=str(self.config.project_root),
        )
        schema = dict(request.output_schema)
        attempts = [
            self._run_turn_bounded(
                thread, self._build_turn_input(request), output_schema=schema
            )
        ]
        for retry in range(max_finalization_retries):
            result = attempts[-1]
            if response_validator is None or self._response_ok(
                result.final_response, response_validator
            ):
                break
            if _status_str(result.status) == "failed" and result.final_response is None:
                # A hard transport/session failure (e.g. 409/429/upstream_rejected)
                # must not trigger a "stop calling tools" finalization prompt on a
                # broken session; surface CodexTurnError so callers can resume on
                # a fresh session with tool execution intact.
                break
            logger.info(
                "codex turn response unusable; finalization re-ask {}/{}",
                retry + 1,
                max_finalization_retries,
            )
            prompt = self._finalization_prompt(result.final_response, schema)
            final_attempt = self._run_turn_bounded(
                thread, [TextInput(prompt)], output_schema=schema
            )
            for _backoff_idx in range(3):
                if (
                    _status_str(final_attempt.status) == "failed"
                    and final_attempt.error is not None
                    and "409" in str(final_attempt.error)
                ):
                    time.sleep(2.0 * (_backoff_idx + 1))
                    final_attempt = self._run_turn_bounded(
                        thread, [TextInput(prompt)], output_schema=schema
                    )
                else:
                    break
            attempts.append(final_attempt)
        return attempts

    def _run_turn_bounded(
        self,
        thread: Thread,
        turn_input: list[InputItem],
        *,
        output_schema: Mapping[str, Any],
    ) -> TurnResult | _GuardedTurnResult:
        """Run normally or stream/interrupt when tool or time limits are set.

        Context windowing can make agents repeat evidence-gathering tools. A
        bounded partial result still permits finalization on the same thread.
        Sandbox policy stays on the thread to preserve its network-access grant.
        """
        cwd = str(self.config.project_root)
        schema = dict(output_schema)
        # Reasoning effort / summary are passed as explicit SDK turn arguments
        # (not config.toml overrides) so they are visible at the call site and
        # scoped to this turn. ``None`` keeps the model/provider default.
        effort = self._effort()
        summary = self._summary()
        guard = _ToolCallLoopGuard(
            max_tool_calls=self.config.max_tool_calls,
            max_repeated_calls=self.config.max_repeated_tool_calls,
        )
        timeout = self.config.turn_timeout_s
        if not guard.active and timeout <= 0:
            return thread.run(
                turn_input,
                cwd=cwd,
                output_schema=schema,
                effort=effort,
                summary=summary,
            )
        handle = thread.turn(
            turn_input,
            cwd=cwd,
            output_schema=schema,
            effort=effort,
            summary=summary,
        )
        return self._consume_guarded_turn(handle, guard=guard, timeout=timeout)

    def _consume_guarded_turn(
        self, handle: _TurnHandle, *, guard: _ToolCallLoopGuard, timeout: float
    ) -> _GuardedTurnResult:
        """Consume a turn's event stream, interrupting it when a bound trips.

        Returns a result shaped like the SDK's ``TurnResult`` (the attributes
        :meth:`_build_metadata` reads). After an interrupt the model usually
        emits ``turn/completed`` within a step or two; a bounded post-interrupt
        grace caps the drain so a server that ignores the interrupt cannot stall
        the run forever.
        """
        state = _TurnInterruptState()
        watchdog: threading.Timer | None = None
        if timeout > 0:
            watchdog = threading.Timer(
                timeout,
                lambda: self._interrupt_turn(
                    handle, state, f"exceeded turn_timeout_s={timeout:g}"
                ),
            )
            watchdog.daemon = True
            watchdog.start()

        items: list[ThreadItem] = []
        usage: ThreadTokenUsage | None = None
        turn: Turn | None = None
        post_interrupt_actions = 0
        stream = handle.stream()
        try:
            for event in stream:
                payload = event.payload
                if isinstance(payload, TurnCompletedNotification):
                    turn = payload.turn
                    continue
                if isinstance(payload, ThreadTokenUsageUpdatedNotification):
                    usage = payload.token_usage
                    continue
                if not isinstance(payload, ItemCompletedNotification):
                    continue
                item = payload.item
                items.append(item)
                signature = _tool_action_signature(item)
                if signature is None:
                    continue
                if state.interrupted:
                    post_interrupt_actions += 1
                    if post_interrupt_actions > _POST_INTERRUPT_GRACE_ACTIONS:
                        self._interrupt_turn(handle, state, state.reason)
                        raise CodexTurnError(
                            "codex turn kept calling tools after interrupt "
                            f"({post_interrupt_actions} actions); abandoning turn "
                            f"[{state.reason}]"
                        )
                    continue
                reason = guard.observe(signature)
                if reason is not None:
                    self._interrupt_turn(handle, state, reason)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            _close_stream(stream)

        if turn is None:
            raise CodexTurnError(
                "codex turn stream ended without a turn/completed event"
            )
        if state.interrupted:
            logger.warning(
                "codex turn interrupted ({}); finalizing from evidence gathered "
                "in {} tool actions",
                state.reason,
                guard.total,
            )
        return _GuardedTurnResult(
            final_response=_final_response_from_items(items),
            id=turn.id,
            status=turn.status,
            error=turn.error,
            duration_ms=turn.duration_ms,
            usage=usage,
            reasoning_summary=_reasoning_summary_from_items(items),
        )

    def _interrupt_turn(
        self, handle: _TurnHandle, state: _TurnInterruptState, reason: str
    ) -> None:
        """Request a one-shot interrupt of an active turn (thread-safe)."""
        with state.lock:
            if state.interrupted:
                return
            state.interrupted = True
            state.reason = reason
        logger.warning("interrupting codex turn: {}", reason)
        try:
            handle.interrupt()
        except Exception as exc:  # interruption is best-effort
            logger.warning("codex turn interrupt request failed: {}", exc)

    def _validate_request(self, request: CodexTurnRequest) -> None:
        for skill in request.skills:
            if not skill.path.exists():
                raise CodexConfigError(
                    f"Codex skill file is missing: {skill.name} at {skill.path}"
                )
        for image in request.image_paths:
            if not image.exists():
                raise CodexConfigError(f"Codex turn image is missing: {image}")

    def _build_turn_input(self, request: CodexTurnRequest) -> list[InputItem]:
        items: list[InputItem] = [
            SkillInput(name=skill.name, path=str(skill.path))
            for skill in request.skills
        ]
        items.append(TextInput(request.prompt))
        items.extend(LocalImageInput(path=str(path)) for path in request.image_paths)
        return items

    def _prepare_run_home(self) -> Path:
        """Create an isolated ``CODEX_HOME`` copy for one app-server process."""
        home = self.config.codex_home
        if not home.exists():
            raise CodexConfigError(f"CODEX_HOME does not exist: {home}")
        if not (home / "config.toml").exists():
            raise CodexConfigError(
                f"CODEX_HOME is missing config.toml: {home / 'config.toml'}"
            )
        run_home = home / "runs" / uuid.uuid4().hex
        if run_home.exists():
            shutil.rmtree(run_home)
        run_home.mkdir(parents=True, exist_ok=True)
        seed_files = list(_RUN_HOME_SEED_FILES)
        if self.config.copy_auth_file:
            auth_path = home / "auth.json"
            if not auth_path.is_file():
                raise CodexConfigError(
                    "CODEX_AGENT_COPY_AUTH is enabled but CODEX_HOME is missing "
                    f"auth.json: {auth_path}"
                )
            seed_files.append("auth.json")
        for name in seed_files:
            source = home / name
            if source.exists():
                shutil.copy2(source, run_home / name)
        if self.config.restrict_skills_to_project:
            (run_home / _ISOLATED_HOME_DIRNAME).mkdir(parents=True, exist_ok=True)
        return run_home

    def _cleanup_run_home(self, run_home: Path) -> None:
        if self.config.keep_run_home:
            _prune_preserved_run_home(run_home)
            return
        runs_root = self.config.codex_home / "runs"
        try:
            run_home.relative_to(runs_root)
        except ValueError:
            return
        _remove_tree_best_effort(run_home)

    def _build_turn_env(
        self, run_home: Path, *, session_id: str | None = None
    ) -> dict[str, str]:
        return build_turn_environment(self.config, run_home, session_id=session_id)

    def _build_config_overrides(self) -> tuple[str, ...]:
        return build_provider_overrides(self.config)

    def _effort(self) -> ReasoningEffort | None:
        """Per-turn reasoning effort, or ``None`` to use the model's default."""
        if self.config.reasoning_effort in {"", "provider-default"}:
            return None
        return ReasoningEffort(self.config.reasoning_effort)

    def _summary(self) -> ReasoningSummary | None:
        """Per-turn reasoning-summary request, or ``None`` for the default.

        ``ReasoningSummary`` is a Pydantic ``RootModel`` over
        ``auto | concise | detailed | none``; ``model_validate`` coerces the
        config string literal into it.
        """
        if not self.config.reasoning_summary:
            return None
        return ReasoningSummary.model_validate(self.config.reasoning_summary)

    def _disable_ambient_system_skills(self, client: Codex) -> None:
        """Disable the bundled ``.system`` skills for this turn.

        HOME isolation (see :meth:`_build_turn_env`) stops the app-server from
        discovering user/global skills, but it still installs a few bundled
        ``.system`` skills into the run home. We disable them through the
        official ``skills/config/write`` RPC so the model's skill catalog is
        limited to the project's own skills. This is best-effort hardening:
        a failure is logged but never aborts the turn (HOME isolation already
        removes the bulk of the ambient skills).
        """
        raw_client = client._client
        for name in _BUNDLED_SYSTEM_SKILLS:
            try:
                raw_client.request(
                    "skills/config/write",
                    {"enabled": False, "name": name},
                    response_model=SkillsConfigWriteResponse,
                )
            except Exception as exc:  # hardening RPC; degrade with visibility
                logger.warning("could not disable system skill {}: {}", name, exc)

    def _sandbox(self) -> Sandbox:
        # SandboxMode literals match the Sandbox enum member names exactly.
        return Sandbox[self.config.sandbox]

    def _finalization_prompt(
        self, previous_response: str | None, output_schema: Mapping[str, Any]
    ) -> str:
        base = (
            "Stop gathering evidence now. Do NOT run any more tools or shell "
            "commands. Using only the evidence already present in this thread, "
            "return one JSON object that matches the requested output schema as "
            "your entire reply — no markdown, prose, or tool commands."
        )
        required = _required_keys(output_schema)
        if required:
            base += f"\nThe JSON object must include these keys: {required}."
        preview = _truncate(previous_response or "", 600)
        if preview:
            return f"{base}\nPrevious non-JSON response: {preview}"
        return base

    def _build_metadata(
        self,
        result: TurnResult | _GuardedTurnResult,
        attempts: Sequence[TurnResult | _GuardedTurnResult],
        run_home: Path,
        *,
        session_id: str,
    ) -> CodexTurnMetadata:
        input_tokens, cached_input_tokens = _extract_token_counts(result.usage)
        return CodexTurnMetadata(
            turn_id=result.id,
            status=_status_str(result.status),
            duration_ms=result.duration_ms,
            usage=_dump_usage(result.usage),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_summary=_result_reasoning_summary(result),
            run_home=str(run_home),
            attempts=tuple(
                {
                    "turn_id": item.id,
                    "status": _status_str(item.status),
                    "duration_ms": item.duration_ms,
                    "has_final_response": bool(item.final_response),
                    "adapter_session_id": session_id,
                }
                for item in attempts
            ),
        )

    @staticmethod
    def _response_ok(text: str | None, validator: Callable[[str], bool]) -> bool:
        if not text:
            return False
        return bool(validator(text))


def _remove_tree_best_effort(path: Path, *, attempts: int = 4) -> None:
    """Delete a per-turn run home, tolerating the Codex app-server cleanup race.

    The Codex app-server clones plugins into ``<run_home>/.tmp`` and may still be
    finalizing that directory as the turn returns, which makes a single
    ``shutil.rmtree`` fail with ``Directory not empty``. Removing the scratch
    home is housekeeping, so retry briefly and then log instead of failing an
    otherwise-successful turn.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if attempt == attempts - 1:
                logger.warning(
                    "could not remove codex run home {} after {} attempts: {}",
                    path,
                    attempts,
                    exc,
                )
                return
            time.sleep(0.25 * (attempt + 1))


def _prune_preserved_run_home(run_home: Path) -> None:
    """Keep trace files while dropping bulky app-server scratch and state."""
    for dirname in _PRESERVED_RUN_HOME_PRUNE_DIRS:
        _remove_tree_best_effort(run_home / dirname)
    for pattern in _PRESERVED_RUN_HOME_PRUNE_FILE_PATTERNS:
        for path in run_home.glob(pattern):
            _remove_file_best_effort(path)


def _remove_file_best_effort(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("could not remove codex run-home file {}: {}", path, exc)


def _tool_action_signature(item: ThreadItem) -> str | None:
    """Return a stable signature for a tool-action thread item, else ``None``.

    The loop guard uses this both to count tool actions and to detect identical
    repeats. Recognises shell commands, image views, and MCP / dynamic tool
    calls via the discriminated ``ThreadItem`` union; messages, reasoning, plans,
    and other non-action items return ``None``.
    """
    root = item.root
    if isinstance(root, CommandExecutionThreadItem):
        return f"cmd:{' '.join(root.command.split())}"
    if isinstance(root, ImageViewThreadItem):
        # ``path`` is an ``AbsolutePathBuf`` (RootModel[str]); unwrap to the str.
        return f"img:{root.path.root}"
    if isinstance(root, McpToolCallThreadItem):
        return f"mcp:{root.server}:{root.tool}:{_compact_json(root.arguments)}"
    if isinstance(root, DynamicToolCallThreadItem):
        return f"dyn:{root.tool}:{_compact_json(root.arguments)}"
    return None


def _final_response_from_items(items: Sequence[ThreadItem]) -> str | None:
    """Extract the model's final answer text from completed thread items.

    Mirrors the SDK's own ``agentMessage`` handling: prefer the most recent
    ``final_answer`` phase message, else fall back to the most recent phase-less
    agent message.
    """
    last_unknown_phase: str | None = None
    for item in reversed(list(items)):
        root = item.root
        if not isinstance(root, AgentMessageThreadItem):
            continue
        if root.phase == MessagePhase.final_answer:
            return str(root.text)
        if root.phase is None and last_unknown_phase is None:
            last_unknown_phase = str(root.text)
    return last_unknown_phase


def _reasoning_summary_from_items(items: Sequence[ThreadItem]) -> str | None:
    """Join any reasoning-summary text emitted across a turn's items.

    The model produces a summary only when the turn requested one (see
    :meth:`CodexAgentRuntime._summary`); each ``ReasoningThreadItem`` then
    carries a list of summary fragments. Fragments are concatenated in order so
    callers get the full reasoning narrative. Returns ``None`` when no summary
    was produced (the common case when summaries are not requested).
    """
    parts: list[str] = []
    for item in items:
        root = item.root
        if not isinstance(root, ReasoningThreadItem):
            continue
        for fragment in root.summary or []:
            text = fragment.strip()
            if text:
                parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def _result_reasoning_summary(result: TurnResult | _GuardedTurnResult) -> str | None:
    """Read the reasoning summary from either turn-result shape.

    The guarded path computes it while streaming (stored on
    :class:`_GuardedTurnResult`); the plain ``thread.run`` fast path exposes the
    raw ``items`` on the SDK's ``TurnResult``.
    """
    if isinstance(result, _GuardedTurnResult):
        return result.reasoning_summary
    return _reasoning_summary_from_items(result.items)


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _close_stream(stream: Iterator[Notification]) -> None:
    # Turn streams are generators (closable); guard the rare non-generator case.
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:  # cleanup must not mask the turn's own outcome
        logger.debug("closing codex turn stream failed: {}", exc)


def _required_keys(output_schema: Mapping[str, Any]) -> list[str]:
    required = output_schema.get("required")
    if isinstance(required, list):
        return [str(key) for key in required]
    return []


def _status_str(status: TurnStatus | None) -> str | None:
    if status is None:
        return None
    return str(status.value)


def _extract_token_counts(
    usage: ThreadTokenUsage | None,
) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, cached_input_tokens)`` from a Codex usage object.

    Reads ``usage.last`` (a ``TokenUsageBreakdown``); returns ``(None, None)``
    when the adapter/model did not report usage for the turn.
    """
    if usage is None:
        return None, None
    last = usage.last
    input_tokens = last.input_tokens
    cached_input_tokens = last.cached_input_tokens
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(cached_input_tokens) if cached_input_tokens is not None else None,
    )


def _dump_usage(usage: ThreadTokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return dict(usage.model_dump(mode="json", exclude_none=True))


def _truncate(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


__all__ = [
    "CodexAgentRuntime",
    "MODELHUB_EXTRA_HEADER_ENV",
    "MODELHUB_LOGID_ENV",
]
