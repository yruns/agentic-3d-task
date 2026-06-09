"""Unit tests for CodexAgentConfig and session-id sanitization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codex_agent.config import CodexAgentConfig, sanitize_session_id
from codex_agent.errors import CodexConfigError


def test_defaults_are_usable() -> None:
    config = CodexAgentConfig()
    assert config.model
    assert config.model_provider
    assert config.sandbox == "read_only"
    assert config.enable_prefix_cache is True


def test_invalid_sandbox_raises() -> None:
    with pytest.raises(CodexConfigError):
        CodexAgentConfig(sandbox="banana")  # type: ignore[arg-type]


def test_empty_model_raises() -> None:
    with pytest.raises(CodexConfigError):
        CodexAgentConfig(model="")


def test_str_paths_are_coerced() -> None:
    config = CodexAgentConfig.from_env(
        project_root="/tmp/x", codex_home="/tmp/x/.codex-home"
    )
    assert isinstance(config.project_root, Path)
    assert isinstance(config.codex_home, Path)


def test_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_AGENT_MODEL", "custom-model")
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "workspace-write")
    monkeypatch.setenv("CODEX_AGENT_ENABLE_PREFIX_CACHE", "0")
    config = CodexAgentConfig.from_env(project_root="/tmp/proj")
    assert config.model == "custom-model"
    assert config.sandbox == "workspace_write"
    assert config.enable_prefix_cache is False
    assert config.codex_home == Path("/tmp/proj/.codex-home")


def test_from_env_rejects_bad_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "nope")
    with pytest.raises(CodexConfigError):
        CodexAgentConfig.from_env()


def test_loop_cap_defaults_are_off() -> None:
    config = CodexAgentConfig()
    assert config.max_tool_calls == 0
    assert config.max_repeated_tool_calls == 0
    assert config.model_context_window == 0


@pytest.mark.parametrize(
    "field",
    ["max_tool_calls", "max_repeated_tool_calls", "model_context_window"],
)
def test_negative_int_fields_raise(field: str) -> None:
    kwargs: dict[str, Any] = {field: -1}
    with pytest.raises(CodexConfigError):
        CodexAgentConfig(**kwargs)


def test_from_env_reads_loop_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_AGENT_MAX_TOOL_CALLS", "30")
    monkeypatch.setenv("CODEX_AGENT_MAX_REPEATED_TOOL_CALLS", "4")
    monkeypatch.setenv("CODEX_AGENT_MODEL_CONTEXT_WINDOW", "1000000")
    config = CodexAgentConfig.from_env()
    assert config.max_tool_calls == 30
    assert config.max_repeated_tool_calls == 4
    assert config.model_context_window == 1_000_000


def test_from_env_rejects_non_integer_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_AGENT_MAX_TOOL_CALLS", "lots")
    with pytest.raises(CodexConfigError):
        CodexAgentConfig.from_env()


def test_session_id_sanitization() -> None:
    assert sanitize_session_id("hello world!") == "hello_world"
    assert sanitize_session_id("") == "codex_agent"
    assert sanitize_session_id("....") == "codex_agent"
    assert len(sanitize_session_id("x" * 500)) <= 128
