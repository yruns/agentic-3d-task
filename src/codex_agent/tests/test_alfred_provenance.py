"""Evaluation provenance rejects tracked and untracked executable drift."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from codex_agent.alfred import provenance


def test_rejects_dirty_tracked_tree_before_reading_executable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def git(command: list[str], *, text: bool) -> str:
        calls.append(command)
        return " M src/codex_agent/alfred/policy.py\n"

    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="clean tracked working tree"):
        provenance.code_provenance(tmp_path)
    assert len(calls) == 1
    assert calls[0][3:] == ["status", "--porcelain", "--untracked-files=no"]


@pytest.mark.parametrize(
    "filename", ["src/new.py", "scripts/run.py", "configs/new.toml"]
)
def test_rejects_untracked_executable_source_or_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    def git(command: list[str], *, text: bool) -> str:
        return "" if "status" in command else filename + "\n"

    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="untracked files"):
        provenance.code_provenance(tmp_path)


def test_clean_code_records_branch_head_and_actual_executed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = {
        "src/actor.py": b"answer = 42\n",
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
    }
    for relative, content in contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def git(command: list[str], *, text: bool) -> str:
        if "status" in command or "--others" in command:
            return ""
        if "ls-files" in command:
            return "\n".join(contents)
        if "branch" in command:
            return "feat/alfred-benchmark\n"
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    expected = hashlib.sha256()
    for relative, content in sorted(contents.items()):
        expected.update(relative.encode() + b"\0" + content + b"\0")
    assert provenance.code_provenance(tmp_path) == {
        "branch": "feat/alfred-benchmark",
        "head_sha": "a" * 40,
        "executed_code_sha256": expected.hexdigest(),
        "tracked_worktree_drift": "",
    }
