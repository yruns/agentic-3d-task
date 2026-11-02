"""Regression checks for the repository NR3D Codex skill."""

from __future__ import annotations

from pathlib import Path

from codex_agent.nr3d.playbook import NR3D_TOOL_NAMES


def test_nr3d_tools_skill_tracks_inline_playbook_contract() -> None:
    skill_path = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "nr3d-codex-tools"
        / "SKILL.md"
    )
    skill_text = skill_path.read_text(encoding="utf-8")

    for tool_name in NR3D_TOOL_NAMES:
        assert tool_name in skill_text

    for field_name in (
        "proposal_id",
        "confidence",
        "summary",
        "uncertainties",
        "cited_frame_indices",
    ):
        assert field_name in skill_text

    assert "tools_enabled=True" in skill_text
    assert "Do not re-open this skill" in skill_text
    assert "Never repeat a tool with identical arguments" in skill_text
    assert "Emit the final JSON immediately" in skill_text
