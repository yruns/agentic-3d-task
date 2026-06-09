"""Tests for the in-turn loop guard: signatures, stream consumption, interrupt.

The guard streams a turn's events and interrupts it when tool actions exceed a
total or repeated cap. These tests drive it with a fake in-process Codex SDK
module that yields scripted notifications, plus a drift test that ties the
guard's wire discriminators to the real ``openai_codex`` models.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexTurnError
from codex_agent.models import CodexTurnRequest
from codex_agent.runtime import (
    CodexAgentRuntime,
    _final_response_from_items,
    _tool_action_signature,
    _ToolCallLoopGuard,
    _TurnInterruptState,
    _unwrap_root,
)

# --------------------------------------------------------------------------- #
# Fake streaming Codex SDK
# --------------------------------------------------------------------------- #


@dataclass
class _Root:
    type: str
    command: str = ""
    path: Any = ""
    server: str = ""
    tool: str = ""
    arguments: Any = None
    text: str = ""
    phase: Any = None


@dataclass
class _Item:
    root: _Root


@dataclass
class _Phase:
    value: str


@dataclass
class _Event:
    method: str
    payload: Any


@dataclass
class _ItemPayload:
    item: Any


@dataclass
class _UsagePayload:
    token_usage: Any


@dataclass
class _TurnPayload:
    turn: Any


@dataclass
class _Turn:
    id: str = "turn-1"
    status: str = "completed"
    error: Any = None
    duration_ms: int = 9


@dataclass
class _Breakdown:
    input_tokens: int
    cached_input_tokens: int


@dataclass
class _Usage:
    last: _Breakdown


@dataclass
class _RunResult:
    final_response: str | None
    id: str = "turn-run"
    status: str = "completed"
    error: Any = None
    duration_ms: int = 3
    usage: Any = None


def _cmd(command: str) -> _Item:
    return _Item(_Root(type="commandExecution", command=command))


def _msg(text: str, *, phase: str | None = None) -> _Item:
    return _Item(
        _Root(type="agentMessage", text=text, phase=_Phase(phase) if phase else None)
    )


def _item_event(item: _Item) -> _Event:
    return _Event("item/completed", _ItemPayload(item))


def _usage_event(usage: Any) -> _Event:
    return _Event("thread/tokenUsage/updated", _UsagePayload(usage))


def _completed_event(turn: _Turn | None = None) -> _Event:
    return _Event("turn/completed", _TurnPayload(turn or _Turn()))


class _Handle:
    """A fake ``TurnHandle`` whose ``stream`` yields scripted events."""

    def __init__(self, events: list[_Event], *, loop_item: _Item | None = None) -> None:
        self._events = list(events)
        self._loop_item = loop_item
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1

    def stream(self) -> Any:
        yield from self._events
        if self._loop_item is not None:
            # Simulate a server that ignores the interrupt and keeps emitting the
            # identical tool action. Bounded so a guard bug fails fast (the guard
            # must abandon the turn long before this) instead of hanging.
            for _ in range(1000):
                yield _item_event(self._loop_item)
            yield _completed_event()


class _Sandbox:
    read_only = "read_only"
    workspace_write = "workspace_write"
    full_access = "full_access"


class _Thread:
    def __init__(
        self,
        *,
        handles: list[_Handle] | None = None,
        results: list[_RunResult] | None = None,
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

    def run(self, items: list[Any], **_: Any) -> _RunResult:
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
        self._thread = thread
        self._client = _RawClient()

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def thread_start(self, **_: Any) -> _Thread:
        return self._thread


class _Module:
    def __init__(self, thread: _Thread) -> None:
        self._thread = thread
        self.Sandbox = _Sandbox

    def Codex(self, *, config: dict[str, Any]) -> _Client:
        return _Client(self._thread)

    def CodexConfig(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def TextInput(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def SkillInput(self, *, name: str, path: str) -> dict[str, Any]:
        return {"type": "skill", "name": name, "path": path}

    def LocalImageInput(self, *, path: str) -> dict[str, Any]:
        return {"type": "image", "path": path}


def _runtime(
    tmp_path: Path,
    codex_home: Path,
    thread: _Thread,
    monkeypatch: pytest.MonkeyPatch,
    **config_kwargs: Any,
) -> CodexAgentRuntime:
    config = CodexAgentConfig(
        project_root=tmp_path,
        codex_home=codex_home,
        restrict_skills_to_project=False,
        enable_prefix_cache=False,
        **config_kwargs,
    )
    runtime = CodexAgentRuntime(config)
    monkeypatch.setattr(runtime, "_import_codex", lambda: _Module(thread))
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
# _tool_action_signature / _final_response_from_items / _unwrap_root
# --------------------------------------------------------------------------- #


def test_signature_for_command_normalizes_whitespace() -> None:
    item = _Item(_Root(type="commandExecution", command="sed  -n   SKILL.md"))
    assert _tool_action_signature(item) == "cmd:sed -n SKILL.md"


def test_signature_for_image_unwraps_root_path() -> None:
    @dataclass
    class _AbsPath:
        root: str

    item = _Item(_Root(type="imageView", path=_AbsPath(root="/x/frame.png")))
    assert _tool_action_signature(item) == "img:/x/frame.png"


def test_signature_for_mcp_and_dynamic_tool_calls() -> None:
    mcp = _Item(_Root(type="mcpToolCall", server="s", tool="t", arguments={"b": 1}))
    assert _tool_action_signature(mcp) == 'mcp:s:t:{"b": 1}'
    dyn = _Item(_Root(type="dynamicToolCall", tool="t", arguments={"a": 2}))
    assert _tool_action_signature(dyn) == 'dyn:t:{"a": 2}'


def test_signature_none_for_non_action_items() -> None:
    assert _tool_action_signature(_msg("hi")) is None
    assert _tool_action_signature(_Item(_Root(type="reasoning"))) is None


def test_final_response_prefers_final_answer_phase() -> None:
    items = [
        _msg("thinking", phase="commentary"),
        _msg('{"proposal_id": 3}', phase="final_answer"),
        _msg("trailing commentary", phase="commentary"),
    ]
    assert _final_response_from_items(items) == '{"proposal_id": 3}'


def test_final_response_falls_back_to_phaseless_message() -> None:
    items = [_cmd("ls"), _msg("plain answer")]
    assert _final_response_from_items(items) == "plain answer"


def test_final_response_none_when_no_message() -> None:
    assert _final_response_from_items([_cmd("ls")]) is None


def test_unwrap_root_returns_inner_string() -> None:
    @dataclass
    class _Wrapped:
        root: str

    assert _unwrap_root(_Wrapped(root="/abs/path")) == "/abs/path"
    assert _unwrap_root("plain") == "plain"


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
    thread = _Thread(results=[_RunResult(final_response='{"proposal_id": 1}')])
    runtime = _runtime(tmp_path, fake_codex_home, thread, monkeypatch)
    result = runtime.run_turn(_request())
    assert result.final_response == '{"proposal_id": 1}'
    assert thread.run_calls == 1
    assert thread.turn_calls == 0


def test_guarded_turn_under_budget_does_not_interrupt(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = _Usage(last=_Breakdown(input_tokens=1000, cached_input_tokens=400))
    handle = _Handle(
        [
            _item_event(_cmd("python -m codex_agent.nr3d.tools inspect_proposal")),
            _item_event(_msg('{"proposal_id": 5}', phase="final_answer")),
            _usage_event(usage),
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
            _item_event(_msg('{"proposal_id": 2}', phase="final_answer")),
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
            _item_event(_msg('{"proposal_id": 8}', phase="final_answer")),
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
