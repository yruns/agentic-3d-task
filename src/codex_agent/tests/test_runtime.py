"""Unit tests for CodexAgentRuntime using a fake in-process Codex SDK module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import codex_agent.runtime as runtime_module
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError, CodexTurnError
from codex_agent.models import CodexSkill, CodexTurnRequest
from codex_agent.runtime import CodexAgentRuntime, _remove_tree_best_effort


@dataclass
class _FakeResult:
    final_response: str | None
    id: str = "turn-1"
    status: str = "completed"
    duration_ms: int = 5
    usage: Any = None
    error: Any = None


class _FakeSandbox:
    read_only = "read_only"
    workspace_write = "workspace_write"
    full_access = "full_access"


class _FakeThread:
    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        items: list[Any],
        *,
        cwd: str | None = None,
        output_schema: dict[str, Any] | None = None,
        sandbox: Any = None,
    ) -> _FakeResult:
        self.calls.append({"items": items, "cwd": cwd, "sandbox": sandbox})
        if not self._results:
            raise AssertionError("fake thread ran out of results")
        return self._results.pop(0)


class _FakeClient:
    def __init__(self, thread: _FakeThread) -> None:
        self._thread = thread
        self.thread_start_kwargs: dict[str, Any] | None = None

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def thread_start(self, **kwargs: Any) -> _FakeThread:
        self.thread_start_kwargs = kwargs
        return self._thread


class _FakeCodexModule:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.thread = _FakeThread(results)
        self.client = _FakeClient(self.thread)
        self.Sandbox = _FakeSandbox
        self.config_overrides: tuple[str, ...] = ()
        self.env: dict[str, str] = {}

    def Codex(self, *, config: dict[str, Any]) -> _FakeClient:
        self.config_overrides = config["config_overrides"]
        self.env = config["env"]
        return self.client

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
    fake: _FakeCodexModule,
    monkeypatch: pytest.MonkeyPatch,
) -> CodexAgentRuntime:
    config = CodexAgentConfig(project_root=tmp_path, codex_home=codex_home)
    runtime = CodexAgentRuntime(config)
    monkeypatch.setattr(runtime, "_import_codex", lambda: fake)
    return runtime


def _request() -> CodexTurnRequest:
    return CodexTurnRequest(prompt="solve it", output_schema={"type": "object"})


def test_run_turn_returns_final_response(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response='{"proposal_id": 3}')])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    result = runtime.run_turn(_request())
    assert result.final_response == '{"proposal_id": 3}'
    assert result.metadata.status == "completed"
    assert result.metadata.duration_ms == 5
    assert len(fake.thread.calls) == 1


def test_run_turn_issues_finalization_reask(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule(
        [
            _FakeResult(final_response="I cannot answer in JSON"),
            _FakeResult(final_response='{"proposal_id": 3}'),
        ]
    )
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    result = runtime.run_turn(
        _request(), response_validator=lambda text: "proposal_id" in text
    )
    assert result.final_response == '{"proposal_id": 3}'
    assert len(fake.thread.calls) == 2
    assert len(result.metadata.attempts) == 2


def test_run_turn_respects_zero_retries(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response="still not json")])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    result = runtime.run_turn(
        _request(),
        response_validator=lambda text: "proposal_id" in text,
        max_finalization_retries=0,
    )
    assert result.final_response == "still not json"
    assert len(fake.thread.calls) == 1


def test_run_turn_none_response_raises(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response=None, status="failed")])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    with pytest.raises(CodexTurnError):
        runtime.run_turn(_request())


def test_run_turn_cleans_up_run_home(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response='{"ok": 1}')])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    runtime.run_turn(_request())
    runs_dir = fake_codex_home / "runs"
    assert not list(runs_dir.iterdir())


def test_prefix_cache_headers_present(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response='{"ok": 1}')])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    runtime.run_turn(_request())
    assert "CODEX_AGENT_MODELHUB_EXTRA_HEADER" in fake.env
    assert "CODEX_HOME" in fake.env
    assert any("env_http_headers" in override for override in fake.config_overrides)


def test_missing_codex_home_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response="{}")])
    runtime = _runtime(tmp_path, tmp_path / "missing-home", fake, monkeypatch)
    with pytest.raises(CodexConfigError):
        runtime.run_turn(_request())


def test_missing_skill_file_raises(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response="{}")])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)
    request = CodexTurnRequest(
        prompt="hi",
        output_schema={"type": "object"},
        skills=(CodexSkill(name="x", path=tmp_path / "nope.md"),),
    )
    with pytest.raises(CodexConfigError):
        runtime.run_turn(request)


def test_execute_runs_task_end_to_end(
    tmp_path: Path, fake_codex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCodexModule([_FakeResult(final_response="ok-payload")])
    runtime = _runtime(tmp_path, fake_codex_home, fake, monkeypatch)

    class _Task:
        task_name = "demo"

        def build_turn_request(self) -> CodexTurnRequest:
            return CodexTurnRequest(prompt="go", output_schema={"type": "object"})

        def is_valid_response(self, response_text: str) -> bool:
            return "ok" in response_text

        def parse_response(self, response_text: str) -> str:
            return response_text.upper()

    result = runtime.execute(_Task())
    assert result.task_name == "demo"
    assert result.outcome == "OK-PAYLOAD"
    assert result.turn.final_response == "ok-payload"


def test_remove_tree_best_effort_deletes_dir(tmp_path: Path) -> None:
    target = tmp_path / "run-home"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "f.txt").write_text("hi", encoding="utf-8")
    _remove_tree_best_effort(target)
    assert not target.exists()


def test_remove_tree_best_effort_tolerates_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def _boom(path: Any) -> None:
        calls["n"] += 1
        raise OSError(66, "Directory not empty")

    monkeypatch.setattr(runtime_module.shutil, "rmtree", _boom)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    # Housekeeping must never raise even if removal never succeeds.
    _remove_tree_best_effort(tmp_path / "missing", attempts=3)
    assert calls["n"] == 3
