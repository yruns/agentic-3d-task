"""Unit tests for NR3D sample-id parsing and sample/scene loading."""

from __future__ import annotations

import pytest

from codex_agent.errors import Nr3dDataError
from codex_agent.nr3d.sample import (
    Nr3dSampleId,
    Nr3dScene,
    load_sample,
    safe_sample_id,
    scene_dir_for,
)
from codex_agent.tests.conftest import Nr3dFixture


def test_parse_sample_id() -> None:
    parsed = Nr3dSampleId.parse("scannet/scene0011_00::15::17427")
    assert parsed.scan_id == "scannet/scene0011_00"
    assert parsed.scene_id == "scene0011_00"
    assert parsed.target_id == 15
    assert parsed.assignment_id == "17427"


@pytest.mark.parametrize(
    "bad",
    ["no-colons", "a::b::c::d", "scene::notanint::1", "::5::1"],
)
def test_parse_sample_id_rejects_bad(bad: str) -> None:
    with pytest.raises(Nr3dDataError):
        Nr3dSampleId.parse(bad)


def test_safe_sample_id() -> None:
    assert (
        safe_sample_id("scannet/scene0011_00::15::17427")
        == "scannet__scene0011_00__15__17427"
    )


def test_load_sample(nr3d_fixture: Nr3dFixture) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    assert sample.scene_id == nr3d_fixture.scene_id
    assert sample.target_id == 3
    assert sample.query == "the red chair"
    assert sample.gt_bbox_3d_9dof == nr3d_fixture.gt_bbox


def test_load_sample_missing_raises(nr3d_fixture: Nr3dFixture) -> None:
    with pytest.raises(Nr3dDataError):
        load_sample(nr3d_fixture.data_root, "scannet/scene0001_00::3::999")


def test_load_scene(nr3d_fixture: Nr3dFixture) -> None:
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    assert scene.scene_id == nr3d_fixture.scene_id
    assert scene.scene_category == "office"
    assert scene.total_frames == 12
    assert scene.frame_id_range == (0, 11)
    assert scene.bev_image_path.exists()
    assert scene.proposal_pool.ids() == {3, 7}


def test_load_scene_missing_dir_raises(nr3d_fixture: Nr3dFixture) -> None:
    with pytest.raises(Nr3dDataError):
        Nr3dScene.load(nr3d_fixture.data_root / "scene9999_00" / "missing_pack")
