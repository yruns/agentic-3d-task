"""Tests for SceneFunc3D artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

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

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["selected_frame_ids"] == ["000050"]
    assert payload["status"] == "in_progress"
