"""Tests for the in-turn loop guard: signatures, stream consumption, interrupt.

The guard streams a turn's events and interrupts it when tool actions exceed a
total or repeated cap. These tests drive it with a fake Codex client whose turn
handles yield real ``openai_codex`` notifications, so the guard is exercised
against the SDK's actual wire models, plus a drift test that ties the guard's
discriminators to those models.
"""

from __future__ import annotations

import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from openai_codex import CodexConfig, TurnResult
from openai_codex.generated.v2_all import (
    AbsolutePathBuf,
    AgentMessageThreadItem,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    DynamicToolCallStatus,
    DynamicToolCallThreadItem,
    ImageViewThreadItem,
    ItemCompletedNotification,
    McpToolCallStatus,
    McpToolCallThreadItem,
    MessagePhase,
    ReasoningThreadItem,
    ThreadItem,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
    TurnStatus,
)
from openai_codex.models import Notification

import codex_agent.runtime as runtime_module
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexTurnError
from codex_agent.models import CodexTurnRequest
from codex_agent.runtime import (
    CodexAgentRuntime,
    _final_response_from_items,
    _tool_action_signature,
    _ToolCallLoopGuard,
    _TurnInterruptState,
)

# --------------------------------------------------------------------------- #
# Real-model builders
# --------------------------------------------------------------------------- #


def _cmd(command: str) -> ThreadItem:
    return ThreadItem(
        root=CommandExecutionThreadItem(
            command=command,
            command_actions=[],
            cwd=AbsolutePathBuf("/tmp"),
            id="cmd",
            status=CommandExecutionStatus.completed,
            type="commandExecution",
        )
    )


def _image(path: str) -> ThreadItem:
    return ThreadItem(
        root=ImageViewThreadItem(id="img", path=AbsolutePathBuf(path), type="imageView")
    )


def _mcp(server: str, tool: str, arguments: Any) -> ThreadItem:
    return ThreadItem(
        root=McpToolCallThreadItem(
            arguments=arguments,
            id="mcp",
            server=server,
            status=McpToolCallStatus.completed,
            tool=tool,
            type="mcpToolCall",
        )
    )


def _dyn(tool: str, arguments: Any) -> ThreadItem:
    return ThreadItem(
        root=DynamicToolCallThreadItem(
            arguments=arguments,
            id="dyn",
            status=DynamicToolCallStatus.completed,
            tool=tool,
            type="dynamicToolCall",
        )
    )


def _reasoning() -> ThreadItem:
    return ThreadItem(root=ReasoningThreadItem(id="reasoning", type="reasoning"))


def _msg(text: str, *, phase: MessagePhase | None = None) -> ThreadItem:
    return ThreadItem(
        root=AgentMessageThreadItem(
            id="msg", text=text, phase=phase, type="agentMessage"
        )
    )


def _usage(input_tokens: int, cached_input_tokens: int) -> ThreadTokenUsage:
    breakdown = TokenUsageBreakdown(
        cached_input_tokens=cached_input_tokens,
        input_tokens=input_tokens,
        output_tokens=0,
        reasoning_output_tokens=0,
        total_tokens=input_tokens,
    )
    return ThreadTokenUsage(last=breakdown, total=breakdown)


def _turn(*, status: TurnStatus = TurnStatus.completed, duration_ms: int = 9) -> Turn:
    return Turn(id="turn-1", items=[], status=status, duration_ms=duration_ms)


def _run_result(final_response: str | None) -> TurnResult:
    return TurnResult(
        id="turn-run",
        status=TurnStatus.completed,
        error=None,
        started_at=None,
        completed_at=None,
        duration_ms=3,
        final_response=final_response,
        items=[],
        usage=None,
    )


def _item_event(item: ThreadItem) -> Notification:
    return Notification(
        method="item/completed",
        payload=ItemCompletedNotification(
            completed_at_ms=0, item=item, thread_id="th", turn_id="tn"
        ),
    )


def _usage_event(usage: ThreadTokenUsage) -> Notification:
    return Notification(
        method="thread/tokenUsage/updated",
        payload=ThreadTokenUsageUpdatedNotification(
            thread_id="th", token_usage=usage, turn_id="tn"
        ),
    )


def _completed_event(turn: Turn | None = None) -> Notification:
    return Notification(
        method="turn/completed",
        payload=TurnCompletedNotification(thread_id="th", turn=turn or _turn()),
    )


# --------------------------------------------------------------------------- #
# Fake streaming Codex client
# --------------------------------------------------------------------------- #


class _Handle:
    """A fake ``TurnHandle`` whose ``stream`` yields scripted notifications."""

    def __init__(
        self, events: list[Notification], *, loop_item: ThreadItem | None = None
    ) -> None:
        self._events = list(events)
        self._loop_item = loop_item
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1

    def stream(self) -> Iterator[Notification]:
        yield from self._events
        if self._loop_item is not None:
            # Simulate a server that ignores the interrupt and keeps emitting the
            # identical tool action. Bounded so a guard bug fails fast (the guard
            # must abandon the turn long before this) instead of hanging.
            for _ in range(1000):
                yield _item_event(self._loop_item)
            yield _completed_event()


class _Thread:
    def __init__(
        self,
        *,
        handles: list[_Handle] | None = None,
        results: list[TurnResult] | None = None,
    ) -> None:
        self._handles = list(handles or [])
        self._results = list(results or [])
        self.turn_calls = 0
        self.run_calls = 0

    def turn(self, items: list[Any], **_: Any) -> _Handle:
        self.turn_calls += 1
        if not self._handles:
            raise AssertionError("fake thread ran out of handles")
        return self._handles.pop(0)

    def run(self, items: list[Any], **_: Any) -> TurnResult:
        self.run_calls += 1
        if not self._results:
            raise AssertionError("fake thread ran out of run results")
        return self._results.pop(0)


class _RawClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, params: dict[str, Any], **_: Any) -> None:
        self.requests.append({"method": method, "params": params})
        return None


class _Client:
    def __init__(self, thread: _Thread) -> None:
        self.thread = thread
        self._client = _RawClient()

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def thread_start(self, **_: Any) -> _Thread:
        return self.thread


def _runtime(
    tmp_path: Path,
    codex_home: Path,
    thread: _Thread,
    monkeypatch: pytest.MonkeyPatch,
    **config_kwargs: Any,
) -> CodexAgentRuntime:
    agent_config = CodexAgentConfig(
        project_root=tmp_path,
        codex_home=codex_home,
        restrict_skills_to_project=False,
        enable_prefix_cache=False,
        **config_kwargs,
    )
    runtime = CodexAgentRuntime(agent_config)

    def _factory(*, config: CodexConfig) -> _Client:
        return _Client(thread)

    monkeypatch.setattr(runtime_module, "Codex", _factory)
    return runtime


def _request() -> CodexTurnRequest:
    return CodexTurnRequest(
        prompt="solve it",
        output_schema={"type": "object", "required": ["proposal_id"]},
    )


def _has_proposal(text: str) -> bool:
    return "proposal_id" in text


# --------------------------------------------------------------------------- #
# _ToolCallLoopGuard
# --------------------------------------------------------------------------- #


def test_guard_inactive_when_both_caps_zero() -> None:
    assert _ToolCallLoopGuard(max_tool_calls=0, max_repeated_calls=0).active is False
    assert _ToolCallLoopGuard(max_tool_calls=1, max_repeated_calls=0).active is True
    assert _ToolCallLoopGuard(max_tool_calls=0, max_repeated_calls=1).active is True


def test_guard_total_cap_trips_after_limit() -> None:
    guard = _ToolCallLoopGuard(max_tool_calls=3, max_repeated_calls=0)
    assert guard.observe("a") is None
    assert guard.observe("b") is None
    assert guard.observe("c") is None
    reason = guard.observe("d")
    assert reason is not None and "max_tool_calls=3" in reason
    assert guard.total == 4


def test_guard_repeated_cap_trips_on_identical_action() -> None:
    guard = _ToolCallLoopGuard(max_tool_calls=0, max_repeated_calls=3)
    assert guard.observe("same") is None
    assert guard.observe("same") is None
    reason = guard.observe("same")
    assert reason is not None and "max_repeated_tool_calls=3" in reason


def test_guard_repeated_cap_counts_per_action() -> None:
    guard = _ToolCallLoopGuard(max_tool_calls=0, max_repeated_calls=3)
    # Distinct actions interleaved never reach the per-action threshold of 3.
    for signature in ("a", "b", "a", "b"):
        assert guard.observe(signature) is None
    # The third occurrence of 'a' trips the per-action cap.
    assert guard.observe("a") is not None


# --------------------------------------------------------------------------- #
# _tool_action_signature / _final_response_from_items
# --------------------------------------------------------------------------- #


def test_signature_for_command_normalizes_whitespace() -> None:
    assert _tool_action_signature(_cmd("sed  -n   SKILL.md")) == "cmd:sed -n SKILL.md"


def test_signature_for_image_unwraps_root_path() -> None:
    assert _tool_action_signature(_image("/x/frame.png")) == "img:/x/frame.png"


def test_signature_for_mcp_and_dynamic_tool_calls() -> None:
    assert _tool_action_signature(_mcp("s", "t", {"b": 1})) == 'mcp:s:t:{"b": 1}'
    assert _tool_action_signature(_dyn("t", {"a": 2})) == 'dyn:t:{"a": 2}'


def test_signature_none_for_non_action_items() -> None:
    assert _tool_action_signature(_msg("hi")) is None
    assert _tool_action_signature(_reasoning()) is None


def test_final_response_prefers_final_answer_phase() -> None:
    items = [
        _msg("thinking", phase=MessagePhase.commentary),
        _msg('{"proposal_id": 3}', phase=MessagePhase.final_answer),
        _msg("trailing commentary", phase=MessagePhase.commentary),
    ]
    assert _final_response_from_items(items) == '{"proposal_id": 3}'


def test_final_response_falls_back_to_phaseless_message() -> None:
    items = [_cmd("ls"), _msg("plain answer")]
    assert _final_response_from_items(items) == "plain answer"


def test_final_response_none_when_no_message() -> None:
    assert _final_response_from_items([_cmd("ls")]) is None


# --------------------------------------------------------------------------- #
# _interrupt_turn idempotency
# --------------------------------------------------------------------------- #


def test_interrupt_turn_is_one_shot() -> None:
    runtime = CodexAgentRuntime(CodexAgentConfig())
    handle = _Handle([])
    state = _TurnInterruptState()
    runtime._interrupt_turn(handle, state, "first")
    runtime._interrupt_turn(handle, state, "second")
    assert handle.interrupts == 1
    assert state.interrupted is True
    assert state.reason == "first"


# --------------------------------------------------------------------------- #
# Integration through run_turn (guarded streaming path)
# --------------------------------------------------------------------------- #


def test_fast_path_used_when_no_caps(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thread = _Thread(results=[_run_result('{"proposal_id": 1}')])
    runtime = _runtime(tmp_path, fake_codex_home, thread, monkeypatch)
    result = runtime.run_turn(_request())
    assert result.final_response == '{"proposal_id": 1}'
    assert thread.run_calls == 1
    assert thread.turn_calls == 0


def test_guarded_turn_under_budget_does_not_interrupt(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = _Handle(
        [
            _item_event(_cmd("python -m codex_agent.nr3d.tools inspect_proposal")),
            _item_event(_msg('{"proposal_id": 5}', phase=MessagePhase.final_answer)),
            _usage_event(_usage(input_tokens=1000, cached_input_tokens=400)),
            _completed_event(),
        ]
    )
    thread = _Thread(handles=[handle])
    runtime = _runtime(
        tmp_path,
        fake_codex_home,
        thread,
        monkeypatch,
        max_tool_calls=30,
        max_repeated_tool_calls=4,
    )
    result = runtime.run_turn(_request(), response_validator=_has_proposal)
    assert result.final_response == '{"proposal_id": 5}'
    assert handle.interrupts == 0
    assert thread.turn_calls == 1
    assert result.metadata.input_tokens == 1000
    assert result.metadata.cached_input_tokens == 400


def test_repeated_cap_interrupts_then_finalizes_from_evidence(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Four identical commands trip the repeated cap; the model then emits its
    # final answer (server honored the interrupt) in the same turn.
    loop_cmd = _cmd("sed -n 1,80p SKILL.md")
    handle = _Handle(
        [_item_event(loop_cmd) for _ in range(4)]
        + [
            _item_event(_msg('{"proposal_id": 2}', phase=MessagePhase.final_answer)),
            _completed_event(),
        ]
    )
    thread = _Thread(handles=[handle])
    runtime = _runtime(
        tmp_path,
        fake_codex_home,
        thread,
        monkeypatch,
        max_repeated_tool_calls=4,
    )
    result = runtime.run_turn(_request(), response_validator=_has_proposal)
    assert handle.interrupts >= 1
    assert result.final_response == '{"proposal_id": 2}'
    assert thread.turn_calls == 1


def test_total_cap_interrupt_triggers_finalization_reask(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first turn loops past the total cap without ever emitting JSON; the
    # runtime finalizes with a re-ask that returns a usable answer.
    looping = _Handle(
        [_item_event(_cmd(f"echo step-{i}")) for i in range(4)] + [_completed_event()]
    )
    finalizer = _Handle(
        [
            _item_event(_msg('{"proposal_id": 8}', phase=MessagePhase.final_answer)),
            _completed_event(),
        ]
    )
    thread = _Thread(handles=[looping, finalizer])
    runtime = _runtime(tmp_path, fake_codex_home, thread, monkeypatch, max_tool_calls=3)
    result = runtime.run_turn(_request(), response_validator=_has_proposal)
    assert looping.interrupts >= 1
    assert result.final_response == '{"proposal_id": 8}'
    assert thread.turn_calls == 2


def test_server_ignoring_interrupt_is_abandoned(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server keeps emitting the identical command after the interrupt; the
    # runtime gives up after the post-interrupt grace and tears down the turn.
    loop_cmd = _cmd("sed -n 1,80p SKILL.md")
    handle = _Handle([_item_event(loop_cmd) for _ in range(4)], loop_item=loop_cmd)
    thread = _Thread(handles=[handle])
    runtime = _runtime(
        tmp_path, fake_codex_home, thread, monkeypatch, max_repeated_tool_calls=4
    )
    with pytest.raises(CodexTurnError):
        runtime.run_turn(_request(), response_validator=_has_proposal)
    assert handle.interrupts >= 1


# --------------------------------------------------------------------------- #
# SDK drift guard — ties signatures to the real openai_codex models
# --------------------------------------------------------------------------- #


def _literal_default(model: Any, field_name: str) -> Any:
    annotation = model.model_fields[field_name].annotation
    args = typing.get_args(annotation)
    return args[0] if args else annotation


def test_sdk_thread_item_discriminators_match_signatures() -> None:
    v2 = pytest.importorskip("openai_codex.generated.v2_all")
    assert _literal_default(v2.CommandExecutionThreadItem, "type") == "commandExecution"
    assert _literal_default(v2.ImageViewThreadItem, "type") == "imageView"
    assert _literal_default(v2.McpToolCallThreadItem, "type") == "mcpToolCall"
    assert _literal_default(v2.DynamicToolCallThreadItem, "type") == "dynamicToolCall"
    assert _literal_default(v2.AgentMessageThreadItem, "type") == "agentMessage"

    assert "command" in v2.CommandExecutionThreadItem.model_fields
    assert "path" in v2.ImageViewThreadItem.model_fields
    for field_name in ("server", "tool", "arguments"):
        assert field_name in v2.McpToolCallThreadItem.model_fields
    for field_name in ("tool", "arguments"):
        assert field_name in v2.DynamicToolCallThreadItem.model_fields
    for field_name in ("text", "phase"):
        assert field_name in v2.AgentMessageThreadItem.model_fields
    assert v2.MessagePhase.final_answer.value == "final_answer"
