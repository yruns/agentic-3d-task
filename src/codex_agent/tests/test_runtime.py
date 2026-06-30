"""Unit tests for CodexAgentRuntime using a fake in-process Codex client.

Only :class:`openai_codex.Codex` (which would spawn the app-server subprocess)
is faked; the runtime uses the real ``CodexConfig`` / ``TextInput`` / ``Sandbox``
and emits real ``TurnResult`` values, so these tests bind to the SDK's typed
surface instead of ad-hoc stand-ins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest
from openai_codex import CodexConfig, InputItem, Sandbox, TurnResult
from openai_codex.generated.v2_all import (
    ReasoningEffort,
    ReasoningSummary,
    ReasoningThreadItem,
    ThreadItem,
    ThreadTokenUsage,
    TokenUsageBreakdown,
    TurnStatus,
)

import codex_agent.runtime as runtime_module
from codex_agent.config import (
    CodexAgentConfig,
    SandboxMode,
)
from codex_agent.config import (
    ReasoningEffort as ConfigReasoningEffort,
)
from codex_agent.config import (
    ReasoningSummary as ConfigReasoningSummary,
)
from codex_agent.errors import CodexConfigError, CodexTurnError
from codex_agent.models import CodexSkill, CodexTurnRequest
from codex_agent.runtime import CodexAgentRuntime, _remove_tree_best_effort


def _usage(input_tokens: int, cached_input_tokens: int) -> ThreadTokenUsage:
    breakdown = TokenUsageBreakdown(
        cached_input_tokens=cached_input_tokens,
        input_tokens=input_tokens,
        output_tokens=0,
        reasoning_output_tokens=0,
        total_tokens=input_tokens,
    )
    return ThreadTokenUsage(last=breakdown, total=breakdown)


def _turn_result(
    final_response: str | None,
    *,
    status: TurnStatus = TurnStatus.completed,
    duration_ms: int = 5,
    usage: ThreadTokenUsage | None = None,
    items: list[ThreadItem] | None = None,
) -> TurnResult:
    return TurnResult(
        id="turn-1",
        status=status,
        error=None,
        started_at=None,
        completed_at=None,
        duration_ms=duration_ms,
        final_response=final_response,
        items=items if items is not None else [],
        usage=usage,
    )


def _reasoning_item(*summary: str) -> ThreadItem:
    return ThreadItem(
        root=ReasoningThreadItem(id="r", type="reasoning", summary=list(summary))
    )


@dataclass(frozen=True)
class _ThreadRunCall:
    items: tuple[InputItem, ...]
    cwd: str | None
    sandbox: Sandbox | None
    effort: ReasoningEffort | None
    summary: ReasoningSummary | None


class _SkillConfigWriteParams(TypedDict):
    enabled: bool
    name: str


@dataclass(frozen=True)
class _RawRequest:
    method: str
    params: _SkillConfigWriteParams


@dataclass(frozen=True)
class _ThreadStartCall:
    model: str
    model_provider: str
    sandbox: Sandbox
    cwd: str


class _FakeThread:
    def __init__(self, results: list[TurnResult]) -> None:
        self._results = list(results)
        self.calls: list[_ThreadRunCall] = []

    def run(
        self,
        items: list[InputItem],
        *,
        cwd: str | None = None,
        output_schema: Mapping[str, object] | None = None,
        sandbox: Sandbox | None = None,
        effort: ReasoningEffort | None = None,
        summary: ReasoningSummary | None = None,
    ) -> TurnResult:
        _ = output_schema
        self.calls.append(
            _ThreadRunCall(
                items=tuple(items),
                cwd=cwd,
                sandbox=sandbox,
                effort=effort,
                summary=summary,
            )
        )
        if not self._results:
            raise AssertionError("fake thread ran out of results")
        return self._results.pop(0)


class _FakeRawClient:
    """Stand-in for the low-level CodexClient that records RPC calls."""

    def __init__(self) -> None:
        self.requests: list[_RawRequest] = []

    def request(
        self,
        method: str,
        params: _SkillConfigWriteParams,
        *,
        response_model: type[object] | None = None,
    ) -> None:
        _ = response_model
        self.requests.append(_RawRequest(method=method, params=params))
        return None


class _FakeClient:
    """Stands in for ``openai_codex.Codex``; records config and RPC traffic."""

    def __init__(self, thread: _FakeThread) -> None:
        self.thread = thread
        self._client = _FakeRawClient()
        self.config: CodexConfig | None = None
        self.thread_start_call: _ThreadStartCall | None = None

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def thread_start(
        self, *, model: str, model_provider: str, sandbox: Sandbox, cwd: str
    ) -> _FakeThread:
        self.thread_start_call = _ThreadStartCall(
            model=model,
            model_provider=model_provider,
            sandbox=sandbox,
            cwd=cwd,
        )
        return self.thread

    @property
    def config_overrides(self) -> tuple[str, ...]:
        assert self.config is not None
        return tuple(self.config.config_overrides)

    @property
    def env(self) -> dict[str, str]:
        assert self.config is not None and self.config.env is not None
        return dict(self.config.env)


@dataclass
class _Fake:
    runtime: CodexAgentRuntime
    client: _FakeClient


def _install(
    tmp_path: Path,
    codex_home: Path,
    results: list[TurnResult],
    monkeypatch: pytest.MonkeyPatch,
    *,
    sandbox: SandboxMode = "read_only",
    sandbox_network_access: bool = False,
    restrict_skills_to_project: bool = True,
    reasoning_effort: ConfigReasoningEffort = "medium",
    reasoning_summary: ConfigReasoningSummary = "",
    copy_auth_file: bool = False,
    keep_run_home: bool = False,
) -> _Fake:
    agent_config = CodexAgentConfig(
        project_root=tmp_path,
        codex_home=codex_home,
        sandbox=sandbox,
        sandbox_network_access=sandbox_network_access,
        restrict_skills_to_project=restrict_skills_to_project,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        copy_auth_file=copy_auth_file,
        keep_run_home=keep_run_home,
    )
    runtime = CodexAgentRuntime(agent_config)
    client = _FakeClient(_FakeThread(results))

    def _factory(*, config: CodexConfig) -> _FakeClient:
        client.config = config
        return client

    monkeypatch.setattr(runtime_module, "Codex", _factory)
    return _Fake(runtime=runtime, client=client)


def _request() -> CodexTurnRequest:
    return CodexTurnRequest(prompt="solve it", output_schema={"type": "object"})


def test_run_turn_returns_final_response(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path, fake_codex_home, [_turn_result('{"proposal_id": 3}')], monkeypatch
    )
    result = fake.runtime.run_turn(_request())
    assert result.final_response == '{"proposal_id": 3}'
    assert result.metadata.status == "completed"
    assert result.metadata.duration_ms == 5
    assert len(fake.client.thread.calls) == 1


def test_run_turn_issues_finalization_reask(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [
            _turn_result("I cannot answer in JSON"),
            _turn_result('{"proposal_id": 3}'),
        ],
        monkeypatch,
    )
    result = fake.runtime.run_turn(
        _request(), response_validator=lambda text: "proposal_id" in text
    )
    assert result.final_response == '{"proposal_id": 3}'
    assert len(fake.client.thread.calls) == 2
    assert len(result.metadata.attempts) == 2


def test_run_turn_respects_zero_retries(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path, fake_codex_home, [_turn_result("still not json")], monkeypatch
    )
    result = fake.runtime.run_turn(
        _request(),
        response_validator=lambda text: "proposal_id" in text,
        max_finalization_retries=0,
    )
    assert result.final_response == "still not json"
    assert len(fake.client.thread.calls) == 1


def test_run_turn_none_response_raises(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result(None, status=TurnStatus.failed)],
        monkeypatch,
    )
    with pytest.raises(CodexTurnError):
        fake.runtime.run_turn(_request())


def test_run_turn_cleans_up_run_home(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(tmp_path, fake_codex_home, [_turn_result('{"ok": 1}')], monkeypatch)
    fake.runtime.run_turn(_request())
    runs_dir = fake_codex_home / "runs"
    assert not list(runs_dir.iterdir())


def test_run_turn_does_not_copy_auth_by_default(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (fake_codex_home / "auth.json").write_text("{}", encoding="utf-8")
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        keep_run_home=True,
    )

    result = fake.runtime.run_turn(_request())

    assert result.metadata.run_home is not None
    assert not (Path(result.metadata.run_home) / "auth.json").exists()


def test_run_turn_copies_auth_when_enabled(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (fake_codex_home / "auth.json").write_text(
        '{"auth_mode":"chatgpt"}\n', encoding="utf-8"
    )
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        copy_auth_file=True,
        keep_run_home=True,
    )

    result = fake.runtime.run_turn(_request())

    assert result.metadata.run_home is not None
    assert (Path(result.metadata.run_home) / "auth.json").read_text(
        encoding="utf-8"
    ) == '{"auth_mode":"chatgpt"}\n'


def test_run_turn_requires_auth_file_when_copy_enabled(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        copy_auth_file=True,
    )

    with pytest.raises(CodexConfigError, match="auth.json"):
        fake.runtime.run_turn(_request())


def test_prefix_cache_headers_present(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(tmp_path, fake_codex_home, [_turn_result('{"ok": 1}')], monkeypatch)
    fake.runtime.run_turn(_request())
    assert "CODEX_AGENT_MODELHUB_EXTRA_HEADER" in fake.client.env
    assert "CODEX_HOME" in fake.client.env
    assert any(
        "env_http_headers" in override for override in fake.client.config_overrides
    )


def test_workspace_write_network_override(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        sandbox="workspace_write",
        sandbox_network_access=True,
    )
    fake.runtime.run_turn(_request())
    assert "sandbox_workspace_write.network_access=true" in fake.client.config_overrides


def test_sandbox_mode_set_on_thread_not_overridden_per_turn(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sandbox mode is set once on the thread; the per-turn call must NOT pass
    # a sandbox, or the SDK sends a default WorkspaceWriteSandboxPolicy with
    # network_access=False that overrides the config network grant (which would
    # silently break tools that shell out to the network, e.g. keyframe_selector).
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        sandbox="workspace_write",
        sandbox_network_access=True,
    )
    fake.runtime.run_turn(_request())
    assert fake.client.thread_start_call is not None
    assert fake.client.thread_start_call.sandbox == Sandbox.workspace_write
    assert fake.client.thread.calls and all(
        call.sandbox is None for call in fake.client.thread.calls
    )


def test_restrict_skills_isolates_home_and_disables_system_skills(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(tmp_path, fake_codex_home, [_turn_result('{"ok": 1}')], monkeypatch)
    fake.runtime.run_turn(_request())
    # HOME is isolated under the per-turn run home so user/global skills are hidden.
    assert fake.client.env["HOME"].endswith("/" + runtime_module._ISOLATED_HOME_DIRNAME)
    assert fake.client.env["HOME"].startswith(str(fake_codex_home / "runs"))
    # Bundled .system skills are disabled via the skills/config/write RPC.
    writes = [
        request
        for request in fake.client._client.requests
        if request.method == "skills/config/write"
    ]
    assert {request.params["name"] for request in writes} == set(
        runtime_module._BUNDLED_SYSTEM_SKILLS
    )
    assert all(request.params["enabled"] is False for request in writes)


def test_no_skill_restriction_keeps_real_home(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        restrict_skills_to_project=False,
    )
    fake.runtime.run_turn(_request())
    assert "HOME" not in fake.client.env
    assert not fake.client._client.requests


def test_no_network_override_for_read_only(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        sandbox="read_only",
        sandbox_network_access=True,
    )
    fake.runtime.run_turn(_request())
    assert not any("network_access" in o for o in fake.client.config_overrides)


def test_missing_codex_home_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path, tmp_path / "missing-home", [_turn_result("{}")], monkeypatch
    )
    with pytest.raises(CodexConfigError):
        fake.runtime.run_turn(_request())


def test_missing_skill_file_raises(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(tmp_path, fake_codex_home, [_turn_result("{}")], monkeypatch)
    request = CodexTurnRequest(
        prompt="hi",
        output_schema={"type": "object"},
        skills=(CodexSkill(name="x", path=tmp_path / "nope.md"),),
    )
    with pytest.raises(CodexConfigError):
        fake.runtime.run_turn(request)


def test_execute_runs_task_end_to_end(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path, fake_codex_home, [_turn_result("ok-payload")], monkeypatch
    )

    class _Task:
        task_name = "demo"

        def build_turn_request(self) -> CodexTurnRequest:
            return CodexTurnRequest(prompt="go", output_schema={"type": "object"})

        def is_valid_response(self, response_text: str) -> bool:
            return "ok" in response_text

        def parse_response(self, response_text: str) -> str:
            return response_text.upper()

    result = fake.runtime.execute(_Task())
    assert result.task_name == "demo"
    assert result.outcome == "OK-PAYLOAD"
    assert result.turn.final_response == "ok-payload"


def test_run_turn_captures_cache_tokens(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}', usage=_usage(20000, 7040))],
        monkeypatch,
    )
    result = fake.runtime.run_turn(_request())
    assert result.metadata.input_tokens == 20000
    assert result.metadata.cached_input_tokens == 7040
    assert result.metadata.cache_ratio == pytest.approx(0.352)


def test_cache_ratio_none_without_usage(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path, fake_codex_home, [_turn_result('{"ok": 1}', usage=None)], monkeypatch
    )
    result = fake.runtime.run_turn(_request())
    assert result.metadata.cached_input_tokens is None
    assert result.metadata.cache_ratio is None


def test_default_effort_is_medium_summary_not_requested(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default config requests medium effort (sent as the SDK ``effort`` arg,
    # not a config.toml override) and no summary.
    fake = _install(tmp_path, fake_codex_home, [_turn_result('{"ok": 1}')], monkeypatch)
    fake.runtime.run_turn(_request())
    call = fake.client.thread.calls[0]
    assert call.effort == ReasoningEffort.medium
    assert call.summary is None
    assert not any("model_reasoning" in o for o in fake.client.config_overrides)


def test_empty_effort_follows_model_default(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit empty string means "follow the model's own default": no effort
    # is sent to the SDK.
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        reasoning_effort="",
    )
    fake.runtime.run_turn(_request())
    assert fake.client.thread.calls[0].effort is None


def test_effort_and_summary_passed_at_call_site(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Configured effort/summary travel as SDK turn arguments, NOT as a
    # model_reasoning_effort config.toml override.
    fake = _install(
        tmp_path,
        fake_codex_home,
        [_turn_result('{"ok": 1}')],
        monkeypatch,
        reasoning_effort="high",
        reasoning_summary="auto",
    )
    fake.runtime.run_turn(_request())
    call = fake.client.thread.calls[0]
    assert call.effort == ReasoningEffort.high
    assert call.summary == ReasoningSummary.model_validate("auto")
    assert not any("model_reasoning_effort" in o for o in fake.client.config_overrides)


def test_reasoning_summary_captured_into_metadata(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        tmp_path,
        fake_codex_home,
        [
            _turn_result(
                '{"ok": 1}',
                items=[_reasoning_item("Looked at frame 3.", "Chair is on the left.")],
            )
        ],
        monkeypatch,
        reasoning_summary="auto",
    )
    result = fake.runtime.run_turn(_request())
    assert result.metadata.reasoning_summary == (
        "Looked at frame 3.\n\nChair is on the left."
    )


def test_reasoning_summary_none_when_absent(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(tmp_path, fake_codex_home, [_turn_result('{"ok": 1}')], monkeypatch)
    result = fake.runtime.run_turn(_request())
    assert result.metadata.reasoning_summary is None
    assert result.metadata.as_dict()["reasoning_summary"] is None


def test_remove_tree_best_effort_deletes_dir(tmp_path: Path) -> None:
    target = tmp_path / "run-home"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "f.txt").write_text("hi", encoding="utf-8")
    _remove_tree_best_effort(target)
    assert not target.exists()


def test_remove_tree_best_effort_tolerates_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = 0

    def _boom(path: Path) -> None:
        nonlocal call_count
        _ = path
        call_count += 1
        raise OSError(66, "Directory not empty")

    monkeypatch.setattr(runtime_module.shutil, "rmtree", _boom)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    # Housekeeping must never raise even if removal never succeeds.
    _remove_tree_best_effort(tmp_path / "missing", attempts=3)
    assert call_count == 3
