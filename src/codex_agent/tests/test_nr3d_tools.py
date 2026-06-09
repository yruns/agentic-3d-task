"""Unit tests for the NR3D agent tool layer (catalog/frame/spatial/bev + dispatch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_agent.nr3d.sample import Nr3dScene
from codex_agent.nr3d.tools.bev_tools import ViewBevArgs, view_bev
from codex_agent.nr3d.tools.catalog_tools import (
    InspectProposalArgs,
    ListSceneProposalsArgs,
    inspect_proposal,
    list_scene_proposals,
)
from codex_agent.nr3d.tools.dispatch import TOOL_NAMES, run_tool
from codex_agent.nr3d.tools.frame_annotation import MarkFrameArgs, mark_frame_with_bbox
from codex_agent.nr3d.tools.frame_tools import (
    ListFrameProposalsArgs,
    SelectByProposalArgs,
    list_frame_proposals,
    select_by_proposal,
)
from codex_agent.nr3d.tools.models import ToolInputError
from codex_agent.nr3d.tools.spatial_tools import (
    CompareCandidatesToAnchorsArgs,
    CompareProposalsSpatialArgs,
    canonical_spatial_relation,
    compare_candidates_to_anchors,
    compare_proposals_spatial,
)
from codex_agent.tests.conftest import Nr3dToolsFixture


@pytest.fixture
def scene(nr3d_tools_fixture: Nr3dToolsFixture) -> Nr3dScene:
    return Nr3dScene.load(nr3d_tools_fixture.scene_dir)


# ----- proposal enrichment / frame views --------------------------------------


def test_proposal_parses_frame_views_and_enrichment(scene: Nr3dScene) -> None:
    chair = scene.proposal_pool.require(3)
    assert set(chair.frame_views) == {0, 1}
    assert chair.frame_views[0].bbox_2d == (10, 10, 40, 60)
    assert chair.enrichment is not None
    assert chair.enrichment.color == "red"
    assert chair.enrichment.nearby_objects == ("table",)
    assert chair.position_3d == (0.0, 0.0, 0.0)


def test_frame_to_proposal_ids_index(scene: Nr3dScene) -> None:
    index = scene.proposal_pool.frame_to_proposal_ids()
    assert index[0] == [3, 7]
    assert index[1] == [3, 9]


# ----- catalog tools ----------------------------------------------------------


def test_inspect_proposal_returns_enrichment(scene: Nr3dScene) -> None:
    result = inspect_proposal(scene, InspectProposalArgs(proposal_id=3))
    payload = result.to_payload()
    assert payload["category"] == "chair"
    assert payload["frames_appeared"] == [0, 1]
    assert payload["enrichment"]["color"] == "red"


def test_inspect_proposal_unknown_id_raises(scene: Nr3dScene) -> None:
    with pytest.raises(ToolInputError):
        inspect_proposal(scene, InspectProposalArgs(proposal_id=999))


def test_list_scene_proposals_category_filter(scene: Nr3dScene) -> None:
    result = list_scene_proposals(scene, ListSceneProposalsArgs(category="chair"))
    payload = result.to_payload()
    assert payload["count"] == 1
    assert payload["proposals"][0]["proposal_id"] == 3


def test_list_scene_proposals_region_and_limit(scene: Nr3dScene) -> None:
    region = list_scene_proposals(
        scene, ListSceneProposalsArgs(region_bev=[1.0, -1.0, 3.0, 1.0])
    )
    assert [p["proposal_id"] for p in region.to_payload()["proposals"]] == [7]
    limited = list_scene_proposals(scene, ListSceneProposalsArgs(limit=2))
    assert limited.to_payload()["count"] == 2


# ----- frame tools ------------------------------------------------------------


def test_select_by_proposal_union_vs_require_all(scene: Nr3dScene) -> None:
    union = select_by_proposal(
        scene, SelectByProposalArgs(proposal_ids=[3, 9], require_all=False)
    )
    assert {f.frame_id for f in union.frames} == {0, 1}

    require_all = select_by_proposal(
        scene, SelectByProposalArgs(proposal_ids=[3, 7], require_all=True)
    )
    assert [f.frame_id for f in require_all.frames] == [0]


def test_select_by_proposal_require_all_empty_note(scene: Nr3dScene) -> None:
    result = select_by_proposal(
        scene, SelectByProposalArgs(proposal_ids=[7, 9], require_all=True)
    )
    assert result.frames == ()
    assert "no single frame" in result.note


def test_select_by_proposal_missing_id_raises(scene: Nr3dScene) -> None:
    with pytest.raises(ToolInputError):
        select_by_proposal(scene, SelectByProposalArgs(proposal_ids=[3, 404]))


def test_list_frame_proposals_left_to_right(scene: Nr3dScene) -> None:
    result = list_frame_proposals(scene, ListFrameProposalsArgs(frame_id=0))
    payload = result.to_payload()
    # chair (center_x=25) is left of table (center_x=77.5)
    assert payload["visible_proposal_ids"] == [3, 7]
    assert payload["left_to_right"] == ["#3 chair", "#7 table"]


def test_list_frame_proposals_unknown_frame_raises(scene: Nr3dScene) -> None:
    with pytest.raises(ToolInputError):
        list_frame_proposals(scene, ListFrameProposalsArgs(frame_id=99))


# ----- spatial tools ----------------------------------------------------------


def test_relation_alias_normalization() -> None:
    assert canonical_spatial_relation("closer to") == "closest_to"
    assert canonical_spatial_relation("furthest_from") == "farthest_from"
    assert canonical_spatial_relation("to the left of") == "left_of"


def test_compare_closest_to_ranks_by_distance(scene: Nr3dScene) -> None:
    result = compare_proposals_spatial(
        scene,
        CompareProposalsSpatialArgs(
            candidate_ids=[3, 9], anchor_id=7, relation="closest_to"
        ),
    )
    payload = result.to_payload()
    # chair (dist 2.0 from table) is closer than lamp (dist ~2.83)
    assert payload["ranked_ids"] == [3, 9]


def test_compare_above_uses_vertical_offset(scene: Nr3dScene) -> None:
    result = compare_proposals_spatial(
        scene,
        CompareProposalsSpatialArgs(
            candidate_ids=[7, 9], anchor_id=3, relation="above"
        ),
    )
    # lamp (z=2) is above chair; table (z=0) is not
    assert result.to_payload()["ranked_ids"][0] == 9


def test_compare_left_of_uses_coview_votes(scene: Nr3dScene) -> None:
    result = compare_proposals_spatial(
        scene,
        CompareProposalsSpatialArgs(
            candidate_ids=[3, 9], anchor_id=7, relation="left_of"
        ),
    )
    payload = result.to_payload()
    # chair is co-visible left of table in frame 0; lamp shares no frame
    assert payload["ranked_ids"][0] == 3
    assert payload["shared_frame_counts"][0] == 1
    assert payload["supporting_frame_counts"][0] == 1


def test_compare_unsupported_relation_raises(scene: Nr3dScene) -> None:
    with pytest.raises(ToolInputError):
        compare_proposals_spatial(
            scene,
            CompareProposalsSpatialArgs(
                candidate_ids=[3], anchor_id=7, relation="orbiting"
            ),
        )


def test_compare_candidates_to_anchors_consistency(scene: Nr3dScene) -> None:
    result = compare_candidates_to_anchors(
        scene,
        CompareCandidatesToAnchorsArgs(
            candidate_ids=[3], anchor_ids=[7, 9], relation="closest_to"
        ),
    )
    payload = result.to_payload()
    assert payload["globally_consistent_top1"] == 3
    assert payload["anchor_disagreement"] is False


# ----- render tools -----------------------------------------------------------


def test_mark_frame_with_bbox_writes_image(scene: Nr3dScene, tmp_path: Path) -> None:
    out_dir = tmp_path / "scratch"
    result = mark_frame_with_bbox(
        scene, MarkFrameArgs(frame_id=0, ids=[3, 7]), out_dir=out_dir
    )
    payload = result.to_payload()
    assert Path(payload["image_path"]).exists()
    assert payload["left_to_right"] == ["#3 chair", "#7 table"]


def test_mark_frame_requires_ids_or_labels(scene: Nr3dScene, tmp_path: Path) -> None:
    with pytest.raises(ToolInputError):
        mark_frame_with_bbox(scene, MarkFrameArgs(frame_id=0), out_dir=tmp_path)


def test_mark_frame_no_match_raises(scene: Nr3dScene, tmp_path: Path) -> None:
    with pytest.raises(ToolInputError):
        mark_frame_with_bbox(
            scene, MarkFrameArgs(frame_id=0, ids=[9]), out_dir=tmp_path
        )


def test_view_bev_default_returns_base(scene: Nr3dScene, tmp_path: Path) -> None:
    result = view_bev(scene, ViewBevArgs(), out_dir=tmp_path)
    payload = result.to_payload()
    assert payload["view"] == "default"
    assert Path(payload["image_path"]) == scene.bev_image_path


def test_view_bev_highlight_writes_overlay(scene: Nr3dScene, tmp_path: Path) -> None:
    out_dir = tmp_path / "bev"
    result = view_bev(scene, ViewBevArgs(highlight=[3]), out_dir=out_dir)
    payload = result.to_payload()
    assert payload["view"] == "highlighted"
    assert payload["highlight_ids"] == [3]
    assert Path(payload["image_path"]).exists()


def test_view_bev_reports_missing_category(scene: Nr3dScene, tmp_path: Path) -> None:
    result = view_bev(scene, ViewBevArgs(categories=["unicorn"]), out_dir=tmp_path)
    assert result.to_payload()["categories_with_no_matches"] == ["unicorn"]


# ----- dispatch ---------------------------------------------------------------


def test_run_tool_routes_and_validates(scene: Nr3dScene, tmp_path: Path) -> None:
    payload = run_tool(
        scene, "inspect_proposal", {"proposal_id": 3}, out_dir=tmp_path
    ).to_payload()
    assert payload["proposal_id"] == 3


def test_run_tool_unknown_name_raises(scene: Nr3dScene, tmp_path: Path) -> None:
    with pytest.raises(ToolInputError):
        run_tool(scene, "teleport", {}, out_dir=tmp_path)


def test_run_tool_invalid_args_raise_tool_input_error(
    scene: Nr3dScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError):
        run_tool(scene, "inspect_proposal", {"wrong_key": 1}, out_dir=tmp_path)


def test_tool_names_cover_all_nine() -> None:
    assert len(TOOL_NAMES) == 9
