"""Unit tests for the NR3D CLI wiring (config + skill resolution by mode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_agent.cli import run_nr3d
from codex_agent.cli.run_nr3d import (
    _DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS,
    _DEFAULT_TOOLS_MAX_TOOL_CALLS,
    _build_config,
    _resolve_skill,
    main,
)
from codex_agent.config import CodexAgentConfig
from codex_agent.evaluation.nr3d_runner import Nr3dRunSummary
from codex_agent.tasks.base import CodexExecutor


def test_build_config_tools_enables_workspace_write_and_network() -> None:
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=None, reasoning_effort=None
    )
    assert config.sandbox == "workspace_write"
    assert config.sandbox_network_access is True
    assert config.turn_timeout_s == 0.0
    assert config.reasoning_effort == "medium"


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
    assert config.reasoning_effort == "medium"


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


def test_build_config_reasoning_effort_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_AGENT_REASONING_EFFORT", "low")
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=None, reasoning_effort=None
    )
    assert config.reasoning_effort == "low"


def test_build_config_reasoning_summary_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AGENT_REASONING_SUMMARY", raising=False)
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=None, reasoning_effort=None
    )
    assert config.reasoning_summary == ""


def test_build_config_explicit_reasoning_summary_overrides() -> None:
    config = _build_config(
        model=None,
        sandbox=None,
        tools=True,
        turn_timeout=None,
        reasoning_effort=None,
        reasoning_summary="auto",
    )
    assert config.reasoning_summary == "auto"


def test_build_config_tools_sets_loop_caps() -> None:
    config = _build_config(
        model=None, sandbox=None, tools=True, turn_timeout=None, reasoning_effort=None
    )
    assert config.max_tool_calls == _DEFAULT_TOOLS_MAX_TOOL_CALLS
    assert config.max_repeated_tool_calls == _DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS


def test_build_config_no_tools_leaves_caps_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AGENT_MAX_TOOL_CALLS", raising=False)
    monkeypatch.delenv("CODEX_AGENT_MAX_REPEATED_TOOL_CALLS", raising=False)
    config = _build_config(
        model=None, sandbox=None, tools=False, turn_timeout=None, reasoning_effort=None
    )
    assert config.max_tool_calls == 0
    assert config.max_repeated_tool_calls == 0


def test_build_config_explicit_caps_override() -> None:
    config = _build_config(
        model=None,
        sandbox=None,
        tools=True,
        turn_timeout=None,
        reasoning_effort=None,
        max_tool_calls=12,
        max_repeated_tool_calls=2,
    )
    assert config.max_tool_calls == 12
    assert config.max_repeated_tool_calls == 2


def test_build_config_zero_cap_override_disables() -> None:
    config = _build_config(
        model=None,
        sandbox=None,
        tools=True,
        turn_timeout=None,
        reasoning_effort=None,
        max_tool_calls=0,
    )
    assert config.max_tool_calls == 0
    # The unspecified repeated cap still falls back to the tools default.
    assert config.max_repeated_tool_calls == _DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS


def test_build_config_negative_cap_rejected() -> None:
    with pytest.raises(ValueError):
        _build_config(
            model=None,
            sandbox=None,
            tools=True,
            turn_timeout=None,
            reasoning_effort=None,
            max_tool_calls=-1,
        )


def test_resolve_skill_tools_default_inlines_no_skill() -> None:
    # Tool mode inlines the playbook into the prompt and attaches no skill file.
    assert _resolve_skill(skill_path=None, no_skill=False, tools=True) is None


def test_resolve_skill_tools_explicit_path_is_honored(tmp_path: Path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("legacy tools skill", encoding="utf-8")
    skill = _resolve_skill(skill_path=skill_file, no_skill=False, tools=True)
    assert skill is not None
    assert skill.name == "nr3d-codex-tools"
    assert skill.path == skill_file


def test_resolve_skill_prompt_only_uses_no_skill_by_default() -> None:
    assert _resolve_skill(skill_path=None, no_skill=False, tools=False) is None


def test_resolve_skill_rejects_explicit_skill_without_tools(tmp_path: Path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("tools skill", encoding="utf-8")

    with pytest.raises(ValueError, match="--skill-path requires --tools"):
        _resolve_skill(skill_path=skill_file, no_skill=False, tools=False)


def test_main_exits_when_skill_path_is_passed_without_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_ids = tmp_path / "sample_ids.json"
    sample_ids.write_text('["sample-1"]', encoding="utf-8")
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("tools skill", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--sample-ids",
                str(sample_ids),
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "out"),
                "--skill-path",
                str(skill_file),
            ]
        )

    assert exc_info.value.code == 2
    assert "--skill-path requires --tools" in capsys.readouterr().err


def test_resolve_skill_no_skill_returns_none() -> None:
    assert _resolve_skill(skill_path=None, no_skill=True, tools=True) is None


def test_main_preflights_runtime_config_before_constructing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_ids_path = tmp_path / "sample_ids.json"
    sample_ids_path.write_text('["scene0000_00-0"]', encoding="utf-8")
    call_order: list[str] = []

    def fake_preflight(config: CodexAgentConfig) -> None:
        assert isinstance(config, CodexAgentConfig)
        call_order.append("preflight")

    class FakeRuntime:
        def __init__(self, config: CodexAgentConfig) -> None:
            assert isinstance(config, CodexAgentConfig)
            call_order.append("runtime")

    def fake_run_samples(
        *,
        sample_ids: tuple[str, ...],
        data_root: Path,
        output_dir: Path,
        runtime: CodexExecutor,
        pack_name: str,
        skill: object,
        workers: int,
        sample_retries: int,
        tools_enabled: bool,
    ) -> Nr3dRunSummary:
        _ = (
            sample_ids,
            data_root,
            output_dir,
            runtime,
            pack_name,
            skill,
            workers,
            sample_retries,
            tools_enabled,
        )
        return Nr3dRunSummary(n=1, mean_iou=0.0, acc_025=0.0, acc_050=0.0)

    monkeypatch.setattr(
        run_nr3d, "preflight_codex_runtime", fake_preflight, raising=False
    )
    monkeypatch.setattr(run_nr3d, "CodexAgentRuntime", FakeRuntime)
    monkeypatch.setattr(run_nr3d, "run_samples", fake_run_samples)

    exit_code = run_nr3d.main(
        [
            "--sample-ids",
            str(sample_ids_path),
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert call_order == ["preflight", "runtime"]
