"""Tests for SceneFunc3D sample loading and hidden-GT separation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.sample import (
    SceneFunc3dSampleId,
    load_sample,
    safe_sample_id,
    scene_dir_for,
)


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "conceptgraph").mkdir()
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sample_id_parse() -> None:
    parsed = SceneFunc3dSampleId.parse("421254::desc-a")
    assert parsed.visit_id == "421254"
    assert parsed.desc_id == "desc-a"
    assert parsed.raw == "421254::desc-a"


def test_sample_id_rejects_bad_form() -> None:
    with pytest.raises(SceneFunc3dDataError, match="must have 2"):
        SceneFunc3dSampleId.parse("421254")


def test_scene_dir_for() -> None:
    assert scene_dir_for(Path("/data"), "421254") == Path("/data/421254")


def test_safe_sample_id() -> None:
    assert safe_sample_id("421254::desc-a") == "421254__desc-a"


def test_load_sample_hides_gt_indices(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    sample = load_sample(tmp_path, "421254::desc-a")

    assert sample.sample_id == "421254::desc-a"
    assert sample.visit_id == "421254"
    assert sample.desc_id == "desc-a"
    assert sample.task_description == "Open the lower drawer."
    assert sample.annotation_ids == ("annot-a",)
    assert sample.motion_hints[0].motion_type == "trans"
    assert sample.agent_context == {
        "visit_id": "421254",
        "desc_id": "desc-a",
        "task_description": "Open the lower drawer.",
        "annotation_ids": ["annot-a"],
        "motion_hints": [
            {
                "motion_id": "motion-a",
                "annotation_id": "annot-a",
                "motion_type": "trans",
                "motion_dir": [1.0, 0.0, 0.0],
            }
        ],
    }
    assert "indices" not in json.dumps(sample.agent_context)


def test_load_sample_requires_matching_visit(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    path = tmp_path / "421254" / "421254_descriptions.json"
    path.write_text(
        json.dumps({"visit_id": "wrong", "descriptions": []}), encoding="utf-8"
    )

    with pytest.raises(SceneFunc3dDataError, match="visit_id"):
        load_sample(tmp_path, "421254::desc-a")
