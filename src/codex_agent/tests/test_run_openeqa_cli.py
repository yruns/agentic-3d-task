"""Unit tests for the OpenEQA CLI: argument parsing, selection, wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codex_agent.cli import run_openeqa
from codex_agent.config import CodexAgentConfig
from codex_agent.evaluation.openeqa_runner import OpenEqaRunSummary
from codex_agent.openeqa.question import load_questions
from codex_agent.openeqa.scene import filter_questions_with_local_scenes
from codex_agent.tests.conftest import OpenEqaFixture


def _available(fixture: OpenEqaFixture) -> tuple:
    return filter_questions_with_local_scenes(
        load_questions(fixture.questions_path), fixture.data_root
    )


def test_parser_requires_core_arguments() -> None:
    parser = run_openeqa.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_select_questions_with_fold(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    available = _available(openeqa_fixture)
    fold = tmp_path / "fold.json"
    fold.write_text(json.dumps(["q-scannet-2"]), encoding="utf-8")
    selected = run_openeqa._select_questions(
        available, question_ids_path=fold, limit=None
    )
    assert [q.question_id for q in selected] == ["q-scannet-2"]


def test_select_questions_limit_only(openeqa_fixture: OpenEqaFixture) -> None:
    available = _available(openeqa_fixture)
    selected = run_openeqa._select_questions(available, question_ids_path=None, limit=1)
    assert len(selected) == 1
    assert selected[0].question_id == "q-scannet-1"


def _build_config(**overrides: Any) -> CodexAgentConfig:
    """Call ``_build_config`` with prompt-only defaults plus any overrides."""
    kwargs: dict[str, Any] = {
        "model": None,
        "sandbox": None,
        "tools": False,
        "turn_timeout": None,
        "reasoning_effort": None,
        "reasoning_summary": None,
        "max_tool_calls": None,
        "max_repeated_tool_calls": None,
    }
    kwargs.update(overrides)
    return run_openeqa._build_config(**kwargs)


def test_build_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_AGENT_MODEL", raising=False)
    config = _build_config(model="custom-model", reasoning_effort="high")
    assert isinstance(config, CodexAgentConfig)
    assert config.model == "custom-model"
    assert config.reasoning_effort == "high"


def test_build_config_prompt_only_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    config = _build_config()
    assert config.sandbox == "read_only"
    assert config.max_tool_calls == 0


def test_build_config_tools_enables_workspace_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    monkeypatch.delenv("CODEX_AGENT_MAX_TOOL_CALLS", raising=False)
    config = _build_config(tools=True)
    assert config.sandbox == "workspace_write"
    assert config.sandbox_network_access is True
    assert config.max_tool_calls == run_openeqa._DEFAULT_TOOLS_MAX_TOOL_CALLS
    assert (
        config.max_repeated_tool_calls
        == run_openeqa._DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS
    )


def test_build_config_explicit_sandbox_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    config = _build_config(tools=True, sandbox="read_only")
    assert config.sandbox == "read_only"


def test_resolve_skill_missing_path_returns_none(tmp_path: Path) -> None:
    assert run_openeqa._resolve_skill(tmp_path / "missing.md", tools=False) is None


def test_resolve_skill_existing_path(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# skill", encoding="utf-8")
    skill = run_openeqa._resolve_skill(skill_path, tools=False)
    assert skill is not None
    assert skill.name == "openeqa-qa"


def test_resolve_skill_tools_default_none(tmp_path: Path) -> None:
    # Tool mode inlines the playbook and attaches no skill by default.
    assert run_openeqa._resolve_skill(None, tools=True) is None


def test_resolve_skill_tools_with_explicit_path(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# skill", encoding="utf-8")
    skill = run_openeqa._resolve_skill(skill_path, tools=True)
    assert skill is not None
    assert skill.name == "openeqa-codex-tools"


def test_as_reasoning_effort_invalid_raises() -> None:
    with pytest.raises(ValueError):
        run_openeqa._as_reasoning_effort("turbo")


def test_as_reasoning_summary_invalid_raises() -> None:
    with pytest.raises(ValueError):
        run_openeqa._as_reasoning_summary("verbose")


def test_build_config_with_reasoning_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AGENT_REASONING_SUMMARY", raising=False)
    config = _build_config(reasoning_summary="concise")
    assert config.reasoning_summary == "concise"


def test_build_judge_disabled_returns_none() -> None:
    assert (
        run_openeqa._build_judge(no_judge=True, llm_config=None, judge_model=None)
        is None
    )


def test_main_runs_selected_questions(
    openeqa_fixture: OpenEqaFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_questions(**kwargs: Any) -> OpenEqaRunSummary:
        captured.update(kwargs)
        return OpenEqaRunSummary(n=1, n_judged=0, mnas=0.0)

    monkeypatch.setattr(run_openeqa, "run_questions", _fake_run_questions)

    exit_code = run_openeqa.main(
        [
            "--questions",
            str(openeqa_fixture.questions_path),
            "--data-root",
            str(openeqa_fixture.data_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--limit",
            "1",
            "--no-judge",
        ]
    )
    assert exit_code == 0
    assert captured["judge"] is None
    assert captured["tools_enabled"] is False
    assert [q.question_id for q in captured["questions"]] == ["q-scannet-1"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["n"] == 1


def test_main_tools_flag_threads_tools_enabled(
    openeqa_fixture: OpenEqaFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    captured: dict[str, Any] = {}

    def _fake_run_questions(**kwargs: Any) -> OpenEqaRunSummary:
        captured.update(kwargs)
        return OpenEqaRunSummary(n=1, n_judged=0, mnas=0.0)

    monkeypatch.setattr(run_openeqa, "run_questions", _fake_run_questions)

    exit_code = run_openeqa.main(
        [
            "--questions",
            str(openeqa_fixture.questions_path),
            "--data-root",
            str(openeqa_fixture.data_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--limit",
            "1",
            "--no-judge",
            "--tools",
        ]
    )
    assert exit_code == 0
    assert captured["tools_enabled"] is True
    capsys.readouterr()


def test_main_errors_when_no_local_scenes(
    tmp_path: Path,
) -> None:
    # A questions file whose clip has no local scene under an empty data root.
    questions_path = tmp_path / "q.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "question": "q?",
                    "answer": "a",
                    "question_id": "x",
                    "episode_history": "scannet-v0/000-scannet-sceneXXXX_00",
                }
            ]
        ),
        encoding="utf-8",
    )
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(SystemExit):
        run_openeqa.main(
            [
                "--questions",
                str(questions_path),
                "--data-root",
                str(empty_root),
                "--output-dir",
                str(tmp_path / "out"),
                "--no-judge",
            ]
        )
