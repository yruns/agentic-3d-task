"""Unit tests for the NR3D CLI wiring (config + skill resolution by mode)."""

from __future__ import annotations

import pytest

from codex_agent.cli.run_nr3d import _build_config, _resolve_skill


def test_build_config_tools_enables_workspace_write_and_network() -> None:
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=None, reasoning_effort=None
    )
    assert config.sandbox == "workspace_write"
    assert config.sandbox_network_access is True
    assert config.turn_timeout_s == 0.0
    assert config.reasoning_effort == ""


def test_build_config_default_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    monkeypatch.delenv("CODEX_AGENT_TURN_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CODEX_AGENT_REASONING_EFFORT", raising=False)
    config = _build_config(
        model=None, sandbox=None, tools=False, turn_timeout=None, reasoning_effort=None
    )
    assert config.sandbox == "read_only"
    assert config.sandbox_network_access is False
    assert config.turn_timeout_s == 0.0
    assert config.reasoning_effort == ""


def test_build_config_explicit_sandbox_with_tools_keeps_network() -> None:
    config = _build_config(
        model=None,
        sandbox="workspace_write",
        tools=True,
        turn_timeout=None,
        reasoning_effort=None,
    )
    assert config.sandbox == "workspace_write"
    assert config.sandbox_network_access is True


def test_build_config_explicit_turn_timeout_overrides() -> None:
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=42.0, reasoning_effort=None
    )
    assert config.turn_timeout_s == 42.0


def test_build_config_explicit_reasoning_effort_overrides() -> None:
    config = _build_config(
        model=None,
        sandbox=None,
        tools=True,
        turn_timeout=None,
        reasoning_effort="high",
    )
    assert config.reasoning_effort == "high"


def test_resolve_skill_tools_uses_tools_skill() -> None:
    skill = _resolve_skill(skill_path=None, no_skill=False, tools=True)
    assert skill is not None
    assert skill.name == "nr3d-codex-tools"
    assert skill.path.exists()


def test_resolve_skill_prompt_only_uses_default_skill() -> None:
    skill = _resolve_skill(skill_path=None, no_skill=False, tools=False)
    assert skill is not None
    assert skill.name == "nr3d-codex-sdk"


def test_resolve_skill_no_skill_returns_none() -> None:
    assert _resolve_skill(skill_path=None, no_skill=True, tools=True) is None
