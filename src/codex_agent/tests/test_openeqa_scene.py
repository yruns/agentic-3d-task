"""Unit tests for OpenEQA scene loading and frame discovery.

The QA turn attaches no default frames; the agent fetches frames via the CLI
tools, so this module only covers scene discovery + the local-scene filter.
"""

from __future__ import annotations

import pytest

from codex_agent.errors import OpenEqaDataError
from codex_agent.openeqa.question import OpenEqaQuestion
from codex_agent.openeqa.scene import (
    OpenEqaScene,
    filter_questions_with_local_scenes,
    has_local_scene,
    scene_dir_for,
)
from codex_agent.tests.conftest import OpenEqaFixture


def _scene(fixture: OpenEqaFixture) -> OpenEqaScene:
    return OpenEqaScene.load(scene_dir_for(fixture.data_root, fixture.clip_id))


def test_load_discovers_rgb_frame_ids(openeqa_fixture: OpenEqaFixture) -> None:
    scene = _scene(openeqa_fixture)
    assert scene.clip_id == openeqa_fixture.clip_id
    assert scene.rgb_frame_ids == (0, 1, 2, 3, 4)
    assert scene.total_frames == 5


def test_load_missing_scene_dir_raises(openeqa_fixture: OpenEqaFixture) -> None:
    with pytest.raises(OpenEqaDataError):
        OpenEqaScene.load(openeqa_fixture.data_root / "does-not-exist")


def test_load_missing_raw_dir_raises(tmp_path) -> None:
    scene_dir = tmp_path / "clip"
    scene_dir.mkdir()
    with pytest.raises(OpenEqaDataError):
        OpenEqaScene.load(scene_dir)


def test_load_no_frames_raises(tmp_path) -> None:
    scene_dir = tmp_path / "clip"
    (scene_dir / "raw").mkdir(parents=True)
    with pytest.raises(OpenEqaDataError):
        OpenEqaScene.load(scene_dir)


def test_raw_rgb_path_uses_six_digit_frame_id(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    scene = _scene(openeqa_fixture)
    assert scene.raw_rgb_path(3) == scene.raw_dir / "000003-rgb.png"


def test_has_local_scene(openeqa_fixture: OpenEqaFixture) -> None:
    assert has_local_scene(openeqa_fixture.data_root, openeqa_fixture.clip_id) is True
    assert has_local_scene(openeqa_fixture.data_root, "missing-clip") is False


def test_filter_questions_with_local_scenes(openeqa_fixture: OpenEqaFixture) -> None:
    present = OpenEqaQuestion(
        question_id="a",
        question="q?",
        answer="x",
        category="object recognition",
        episode_history=f"scannet-v0/{openeqa_fixture.clip_id}",
        clip_id=openeqa_fixture.clip_id,
        scene_id=openeqa_fixture.scene_id,
    )
    absent = OpenEqaQuestion(
        question_id="b",
        question="q?",
        answer="x",
        category="object recognition",
        episode_history="scannet-v0/999-scannet-sceneZZZZ_00",
        clip_id="999-scannet-sceneZZZZ_00",
        scene_id="sceneZZZZ_00",
    )
    kept = filter_questions_with_local_scenes(
        [present, absent], openeqa_fixture.data_root
    )
    assert [q.question_id for q in kept] == ["a"]


def test_downsize_rgb_for_view_writes_jpeg(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    from PIL import Image

    from codex_agent.openeqa.scene import downsize_rgb_for_view

    scene = _scene(openeqa_fixture)
    destination = tmp_path / "out.jpg"
    result = downsize_rgb_for_view(scene.raw_rgb_path(0), destination, max_size=32)
    assert result == destination
    with Image.open(destination) as image:
        assert max(image.size) <= 32
        assert image.mode == "RGB"


def test_downsize_rgb_for_view_rejects_bad_max_size(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    from codex_agent.openeqa.scene import downsize_rgb_for_view

    scene = _scene(openeqa_fixture)
    with pytest.raises(OpenEqaDataError):
        downsize_rgb_for_view(scene.raw_rgb_path(0), tmp_path / "o.jpg", max_size=0)
