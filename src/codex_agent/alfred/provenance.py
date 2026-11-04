"""Fail-closed code provenance for ALFRED evaluation invocations."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash large checkpoints without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance(project_root: Path) -> dict[str, object]:
    """Require a clean tracked tree and committed executable source/config.

    An untracked source file is executable drift even when ``git diff`` is empty.
    Evaluation cannot be enabled by merely leaving new implementation untracked.
    """

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments], text=True
        ).strip()

    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise ValueError(
            "evaluation requires a clean tracked working tree; commit the reviewed change first"
        )
    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "src",
        "scripts",
        "configs",
        "pyproject.toml",
    )
    if untracked:
        raise ValueError(
            "evaluation source/config contains untracked files; commit the reviewed change first"
        )
    paths = git("ls-files", "src", "scripts", "configs", "pyproject.toml").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode() + b"\0")
        digest.update((project_root / relative).read_bytes())
        digest.update(b"\0")
    return {
        "branch": git("branch", "--show-current"),
        "head_sha": git("rev-parse", "HEAD"),
        "executed_code_sha256": digest.hexdigest(),
        "tracked_worktree_drift": tracked,
    }
