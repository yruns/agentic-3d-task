"""Unit tests for the NR3D proposal-pool loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import Nr3dDataError
from codex_agent.nr3d.proposals import ProposalPool
from codex_agent.tests.conftest import Nr3dFixture


def test_loads_pool_with_inverted_visibility(nr3d_fixture: Nr3dFixture) -> None:
    pool = ProposalPool.from_files(
        nr3d_fixture.scene_dir / "proposals.jsonl",
        nr3d_fixture.scene_dir / "visibility.json",
    )
    assert pool.source == "gt"
    assert pool.ids() == {3, 7}

    chair = pool.require(3)
    assert chair.category == "chair"
    assert chair.enriched_category == "office chair"
    assert chair.visible_frame_ids == (10, 11)

    table = pool.require(7)
    assert table.visible_frame_ids == (10,)


def test_ids_by_category_sorted(nr3d_fixture: Nr3dFixture) -> None:
    pool = ProposalPool.from_files(
        nr3d_fixture.scene_dir / "proposals.jsonl",
        nr3d_fixture.scene_dir / "visibility.json",
    )
    assert pool.ids_by_category() == {"chair": [3], "table": [7]}


def test_require_missing_id_raises(nr3d_fixture: Nr3dFixture) -> None:
    pool = ProposalPool.from_files(
        nr3d_fixture.scene_dir / "proposals.jsonl",
        nr3d_fixture.scene_dir / "visibility.json",
    )
    assert pool.get(999) is None
    with pytest.raises(Nr3dDataError):
        pool.require(999)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Nr3dDataError):
        ProposalPool.from_files(tmp_path / "nope.jsonl", tmp_path / "vis.json")


def test_bad_source_raises(tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.jsonl"
    visibility = tmp_path / "visibility.json"
    proposals.write_text(json.dumps({"source": "bogus", "proposals": []}))
    visibility.write_text(json.dumps({}))
    with pytest.raises(Nr3dDataError):
        ProposalPool.from_files(proposals, visibility)


def test_bad_bbox_length_raises(tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.jsonl"
    visibility = tmp_path / "visibility.json"
    proposals.write_text(
        json.dumps(
            {
                "source": "gt",
                "proposals": [
                    {"id": 1, "bbox_3d": [0, 0, 0], "score": 1, "label": "x"}
                ],
            }
        )
    )
    visibility.write_text(json.dumps({}))
    with pytest.raises(Nr3dDataError):
        ProposalPool.from_files(proposals, visibility)
