"""Tests keeping the inlined OpenEQA tools playbook in sync with the dispatcher."""

from __future__ import annotations

from codex_agent.openeqa.playbook import OPENEQA_TOOL_NAMES, OPENEQA_TOOLS_PLAYBOOK
from codex_agent.openeqa.tools.dispatch import TOOL_NAMES


def test_playbook_tool_names_match_dispatcher() -> None:
    assert set(OPENEQA_TOOL_NAMES) == set(TOOL_NAMES)


def test_playbook_documents_every_tool() -> None:
    for name in OPENEQA_TOOL_NAMES:
        assert name in OPENEQA_TOOLS_PLAYBOOK, f"playbook missing tool {name!r}"


def test_playbook_requires_view_image_before_trust() -> None:
    assert "view_image" in OPENEQA_TOOLS_PLAYBOOK


def test_playbook_warns_against_reading_skill_files() -> None:
    lowered = OPENEQA_TOOLS_PLAYBOOK.lower()
    assert "skill" in lowered and "do not" in lowered
