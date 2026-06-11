"""Unit tests for OpenEQA scene loading and first-person frame sampling."""

from __future__ import annotations

import pytest
from PIL import Image

from codex_agent.errors import OpenEqaDataError
from codex_agent.openeqa.question import OpenEqaQuestion
from codex_agent.openeqa.scene import (
    OpenEqaScene,
    filter_questions_with_local_scenes,
    has_local_scene,
    scene_dir_for,
    uniform_frame_ids,
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


def test_uniform_frame_ids_returns_all_when_fewer() -> None:
    assert uniform_frame_ids([0, 1, 2], 8) == (0, 1, 2)


def test_uniform_frame_ids_subsamples_evenly() -> None:
    ids = list(range(100))
    selected = uniform_frame_ids(ids, 5)
    assert selected == (0, 20, 40, 60, 80)
    assert len(selected) == 5


def test_uniform_frame_ids_rejects_non_positive() -> None:
    with pytest.raises(OpenEqaDataError):
        uniform_frame_ids([0, 1], 0)


def test_select_frame_ids_uses_uniform_sampling(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    scene = _scene(openeqa_fixture)
    assert scene.select_frame_ids(2) == (0, 2)
    assert scene.select_frame_ids(10) == (0, 1, 2, 3, 4)


def test_select_frame_ids_rejects_non_positive(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    scene = _scene(openeqa_fixture)
    with pytest.raises(OpenEqaDataError):
        scene.select_frame_ids(0)


def test_raw_rgb_path_uses_six_digit_frame_id(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    scene = _scene(openeqa_fixture)
    assert scene.raw_rgb_path(3) == scene.raw_dir / "000003-rgb.png"


def test_prepare_frames_missing_raw_frame_raises(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    scene = _scene(openeqa_fixture)
    # A scene that claims a frame id whose PNG was removed from disk.
    broken = OpenEqaScene(
        clip_id=scene.clip_id, scene_dir=scene.scene_dir, rgb_frame_ids=(999,)
    )
    with pytest.raises(OpenEqaDataError):
        broken.prepare_frames(cache_dir=tmp_path / "cache", num_frames=1)


def test_prepare_frames_downsizes_and_caches(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    scene = _scene(openeqa_fixture)
    cache_dir = tmp_path / "cache"
    frames = scene.prepare_frames(cache_dir=cache_dir, num_frames=3, max_image_size=32)
    assert len(frames) == 3
    for frame in frames:
        assert frame.image_path.exists()
        assert frame.image_path.suffix == ".jpg"
        with Image.open(frame.image_path) as image:
            assert max(image.size) <= 32
            assert image.mode == "RGB"


def test_prepare_frames_is_idempotent(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    scene = _scene(openeqa_fixture)
    cache_dir = tmp_path / "cache"
    first = scene.prepare_frames(cache_dir=cache_dir, num_frames=2, max_image_size=32)
    mtime = first[0].image_path.stat().st_mtime_ns
    second = scene.prepare_frames(cache_dir=cache_dir, num_frames=2, max_image_size=32)
    # Cached file is reused, not re-encoded.
    assert second[0].image_path == first[0].image_path
    assert second[0].image_path.stat().st_mtime_ns == mtime


def test_prepare_frames_rejects_bad_max_size(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    scene = _scene(openeqa_fixture)
    with pytest.raises(OpenEqaDataError):
        scene.prepare_frames(cache_dir=tmp_path, num_frames=1, max_image_size=0)


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
