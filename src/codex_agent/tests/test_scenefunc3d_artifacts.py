"""Tests for SceneFunc3D artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.tools.mask_artifacts import (
    ArtifactStatus,
    SceneFunc3dRunSummary,
    artifact_paths_for,
    write_run_summary,
)


def test_artifact_paths_for(tmp_path: Path) -> None:
    paths = artifact_paths_for(tmp_path, "421254", "desc-a")
    assert paths.root == tmp_path / "421254" / "desc-a"
    assert paths.summary_json == paths.root / "summary.json"
    assert paths.events_jsonl == paths.root / "events.jsonl"
    assert paths.overlays_dir == paths.root / "overlays"
    assert paths.fragments_dir == paths.root / "fragments"
    assert paths.fused_dir == paths.root / "fused"


@pytest.mark.parametrize(
    ("visit_id", "desc_id"),
    [
        ("", "desc-a"),
        (" ", "desc-a"),
        (".", "desc-a"),
        ("..", "desc-a"),
        ("/escape", "desc-a"),
        ("421254/escape", "desc-a"),
        ("421254\\escape", "desc-a"),
        ("421254", ""),
        ("421254", " "),
        ("421254", "."),
        ("421254", ".."),
        ("421254", "../escape"),
        ("421254", "a/b"),
        ("421254", "a\\b"),
        ("421254", "/escape"),
    ],
)
def test_artifact_paths_for_rejects_unsafe_path_components(
    tmp_path: Path, visit_id: str, desc_id: str
) -> None:
    with pytest.raises(SceneFunc3dDataError, match="invalid artifact path component"):
        artifact_paths_for(tmp_path, visit_id, desc_id)


def test_write_run_summary(tmp_path: Path) -> None:
    paths = artifact_paths_for(tmp_path, "421254", "desc-a")
    summary = SceneFunc3dRunSummary(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the drawer.",
        status=ArtifactStatus.IN_PROGRESS,
        selected_frame_ids=("000050",),
        accepted_fragment_ids=(),
        failure_type="",
        stop_reason="",
    )

    write_run_summary(paths, summary)

    assert paths.root.is_dir()
    assert paths.raw_outputs_dir.is_dir()
    assert paths.overlays_dir.is_dir()
    assert paths.fragments_dir.is_dir()
    assert paths.fused_dir.is_dir()

    summary_text = paths.summary_json.read_text(encoding="utf-8")
    assert summary_text.endswith("\n")
    assert summary_text.startswith("{\n  ")
    assert '\n  "sample_id":' in summary_text

    payload = json.loads(summary_text)
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["selected_frame_ids"] == ["000050"]
    assert payload["status"] == "in_progress"
