"""Evaluation provenance rejects tracked and untracked executable drift."""

from __future__ import annotations

import hashlib
import json
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


def test_runtime_capture_records_calling_interpreter_without_parent_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = {
        name: {
            "version": "worker-package-version",
            "distribution_record_sha256": "a" * 64,
        }
        for name in provenance.PERCEPTION_PACKAGES
    }
    monkeypatch.setattr(provenance, "model_source_provenance", lambda: packages)
    monkeypatch.setattr(provenance.sys, "executable", "/worker-environment/bin/python")
    monkeypatch.setattr(provenance.sys, "prefix", "/worker-environment")
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "GPU-01234567-89ab-cdef-0123-456789abcdef"
    )
    captured = provenance.capture_runtime_provenance(model_worker=True)
    assert captured.executable == "/worker-environment/bin/python"
    assert captured.prefix == "/worker-environment"
    assert captured.packages["torch"].version == "worker-package-version"
    assert captured.visible_devices == "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert captured.package_source_sha256 == provenance.alfred_source_sha256()


def test_package_metadata_excludes_private_distribution_download_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Distribution:
        version = "test-version"

        def read_text(self, name: str) -> str | None:
            if name == "RECORD":
                return "installed.py,sha256=fixture,1\n"
            return json.dumps(
                {
                    "url": "https://private.invalid/source?credential=SYNTHETIC_PRIVATE_MARKER"
                }
            )

    monkeypatch.setattr(
        provenance.distribution_metadata, "distribution", lambda name: Distribution()
    )
    captured = provenance.package_source_provenance(("fixture-package",))
    assert captured["fixture-package"]["version"] == "test-version"
    assert "distribution_record_sha256" in captured["fixture-package"]
    assert "SYNTHETIC_PRIVATE_MARKER" not in json.dumps(captured)


def test_missing_package_and_unfingerprinted_package_are_explicit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise provenance.distribution_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.distribution_metadata, "distribution", unavailable)
    with pytest.raises(RuntimeError, match="not installed"):
        provenance.package_source_provenance(("missing-package",))
    with pytest.raises(ValueError, match="requires.*digest"):
        provenance.PackageProvenance(version="unverified")
