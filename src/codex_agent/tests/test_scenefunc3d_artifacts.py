"""Tests for SceneFunc3D artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.final_mask_artifacts import FinalMaskMultiViewAction
from codex_agent.scenefunc3d.task import ApprovalAction
from codex_agent.scenefunc3d.tools.mask_artifacts import (
    ArtifactStatus,
    SceneFunc3dRunSummary,
    artifact_paths_for,
    write_run_summary,
)
from codex_agent.scenefunc3d.tools.mask_inspection import (
    AcceptedFragment,
    AcceptedFragmentInput,
    AcceptedFragmentReviewArtifacts,
    AcceptedFragmentReviewArtifactsInput,
    FuseAcceptedMasksArgs,
    FusedMaskPayload,
    FusedMaskResult,
    InspectMaskArtifactArgs,
    MaskInspectionPayload,
    MaskInspectionResult,
    MultiViewDecisionInput,
    MultiViewDecisionPayload,
    SeedLiftStatus,
    SuggestAdditionalViewsArgs,
    SuggestedView,
    SuggestedViewsPayload,
    SuggestedViewsResult,
    fuse_accepted_masks,
    inspect_mask_artifact,
    suggest_additional_views,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

_APPROVED_FRAGMENT_ACTIONS: tuple[ApprovalAction, ...] = (
    ApprovalAction.SELECT_EVIDENCE,
    ApprovalAction.PROPOSE_MOLMO_POINT,
    ApprovalAction.APPROVE_MOLMO_POINT,
    ApprovalAction.PROPOSE_SAM_CANDIDATES,
    ApprovalAction.APPROVE_SAM_CANDIDATE,
    ApprovalAction.CREATE_FIRST_LIFT,
    ApprovalAction.APPROVE_FIRST_LIFT,
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


def test_mask_inspection_payload() -> None:
    result = MaskInspectionResult(
        artifact_path=Path("/tmp/summary.json"),
        overlay_paths=(Path("/tmp/overlay.jpg"),),
        lifted_point_count=42,
        status="success",
    )

    payload = cast(MaskInspectionPayload, result.to_payload())

    assert payload["artifact_path"] == "/tmp/summary.json"
    assert payload["overlay_paths"] == ["/tmp/overlay.jpg"]


def test_inspect_mask_artifact_validates_npz_and_ply(tmp_path: Path) -> None:
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "fragment-a")
    overlay_path = tmp_path / "overlay.txt"
    overlay_path.write_text("reviewed\n", encoding="utf-8")
    args = InspectMaskArtifactArgs.model_validate(
        {
            "mask_npz_path": str(mask_npz_path),
            "mask_ply_path": str(mask_ply_path),
            "overlay_paths": [str(overlay_path)],
        }
    )

    result = inspect_mask_artifact(args)

    assert result.status == "valid"
    assert result.lifted_point_count == 2
    assert result.overlay_paths == (overlay_path,)


def test_inspect_mask_artifact_ignores_review_only_frame_fields(
    tmp_path: Path,
) -> None:
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "fragment-a")
    overlay_path = tmp_path / "overlay.txt"
    overlay_path.write_text("reviewed\n", encoding="utf-8")

    args = InspectMaskArtifactArgs.model_validate(
        {
            "frame_id": "000050",
            "candidate_id": "mask_00",
            "mask_npz_path": str(mask_npz_path),
            "mask_ply_path": str(mask_ply_path),
            "overlay_path": str(overlay_path),
        }
    )

    assert args.mask_npz_path == mask_npz_path
    assert args.mask_ply_path == mask_ply_path
    assert args.overlay_paths == (overlay_path,)


def test_suggested_views_payload() -> None:
    result = SuggestedViewsResult(
        seed_fragment_id="000050_mask_00",
        seed_lift_point_count=42,
        seed_lift_status=SeedLiftStatus.USABLE,
        expansion_recommendation=FinalMaskMultiViewAction.STOP,
        expansion_reason="seed has enough lifted points",
        views=(
            SuggestedView(
                frame_id="000060",
                reason="different view angle",
                rank=1,
                has_depth=True,
                has_intrinsics=True,
                has_pose=True,
                view_diversity_score=0.5,
            ),
        ),
    )

    payload = cast(SuggestedViewsPayload, result.to_payload())

    assert payload["views"][0]["reason"] == "different view angle"
    assert payload["expansion_recommendation"] == "stop"
    assert payload["views"][0]["has_pose"] is True


def test_suggest_additional_views_ranks_geometry_ready_views(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    for frame_id in ("000010", "000020", "000040"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    _write_pose(raw_dir / "pose" / "000010.txt", x_translation=0.0)
    _write_geometry_assets(raw_dir, "000040", x_translation=2.0)
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
        candidate_frame_ids=("000020", "000040"),
        min_seed_point_count=4,
        k=2,
    )

    result = suggest_additional_views(tool_scene, args)

    assert result.seed_fragment_id == "000010_mask_00"
    assert result.seed_lift_point_count == 2
    assert result.seed_lift_status == "sparse"
    assert result.expansion_recommendation is FinalMaskMultiViewAction.EXPAND
    assert tuple(view.frame_id for view in result.views) == ("000040", "000020")
    assert result.views[0].reason == (
        "seed_geometry_sparse: point_count=2 below min_seed_point_count=4; "
        "candidate has depth, intrinsics, and pose; "
        "view_diversity_score=2.000; "
        "temporal_distance=30"
    )
    assert result.views[0].has_depth is True
    assert result.views[0].has_intrinsics is True
    assert result.views[0].has_pose is True
    assert result.views[0].view_diversity_score == 2.0
    assert result.views[1].has_depth is False
    payload = cast(SuggestedViewsPayload, result.to_payload())
    assert payload["seed_lift_point_count"] == 2
    assert payload["seed_lift_status"] == "sparse"
    assert payload["expansion_recommendation"] == "expand"


def test_suggest_additional_views_expands_usable_seed_when_views_exist(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    for frame_id in ("000010", "000020"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
        min_seed_point_count=1,
        k=1,
    )

    result = suggest_additional_views(tool_scene, args)

    assert result.seed_lift_status is SeedLiftStatus.USABLE
    assert result.expansion_recommendation is FinalMaskMultiViewAction.EXPAND
    assert tuple(view.frame_id for view in result.views) == ("000020",)
    assert "meets min_seed_point_count=1" in result.expansion_reason
    assert "additional candidate views are available" in result.expansion_reason


def test_suggest_additional_views_prioritizes_query_visible_object_frames(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "421393"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.parent.mkdir(parents=True)
    for frame_id in ("000010", "000020", "000040"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    _write_geometry_assets(raw_dir, "000010", x_translation=0.0)
    _write_geometry_assets(raw_dir, "000020", x_translation=1.0)
    _write_geometry_assets(raw_dir, "000040", x_translation=3.0)
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000020-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 3,
                                "class_name": "heater",
                                "score": 0.8,
                                "bbox_xyxy": [120, 700, 560, 1500],
                            }
                        ],
                    },
                    "1": {
                        "view_id": 1,
                        "frame_name": "000040-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 4,
                                "class_name": "wall",
                                "score": 0.9,
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
        candidate_frame_ids=("000020", "000040"),
        task_description="Adjust room temperature using the radiator dial",
        min_seed_point_count=1,
        k=2,
    )

    result = suggest_additional_views(tool_scene, args)
    payload = cast(SuggestedViewsPayload, result.to_payload())

    assert tuple(view.frame_id for view in result.views) == ("000020", "000040")
    assert "visible_object_query_match" in result.views[0].reason
    assert payload["views"][0]["matched_objects"] == [
        {
            "object_id": "3",
            "label": "heater",
            "score": 0.8,
            "bbox_xyxy": [120.0, 700.0, 560.0, 1500.0],
            "bbox_format": "pixel_xyxy",
            "source": "object_frame_map",
        }
    ]


def test_suggest_additional_views_recommends_crops_from_matched_object_bboxes(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "421393"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.parent.mkdir(parents=True)
    for frame_id in ("000010", "000073"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    _write_geometry_assets(raw_dir, "000010", x_translation=0.0)
    _write_geometry_assets(raw_dir, "000073", x_translation=2.0)
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000073-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 8,
                                "class_name": "heater",
                                "score": 0.86,
                                "bbox_xyxy": [100, 200, 500, 1000],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
        task_description="Adjust room temperature using the radiator dial",
        min_seed_point_count=1,
        k=1,
    )

    result = suggest_additional_views(tool_scene, args)
    payload = cast(SuggestedViewsPayload, result.to_payload())

    assert payload["views"][0]["recommended_crops"] == [
        {
            "bbox_xyxy": [100.0, 200.0, 500.0, 1000.0],
            "bbox_format": "pixel_xyxy",
            "reason": "matched_object_full_bbox",
            "source_object_id": "8",
            "source_object_label": "heater",
        },
        {
            "bbox_xyxy": [360.0, 760.0, 540.0, 1040.0],
            "bbox_format": "pixel_xyxy",
            "reason": "right_lower_affordance_crop_from_matched_object_bbox",
            "source_object_id": "8",
            "source_object_label": "heater",
        },
    ]


def test_suggest_additional_views_stops_when_only_accepted_frame_exists(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
        min_seed_point_count=1,
        k=1,
    )

    result = suggest_additional_views(tool_scene, args)

    assert result.seed_lift_status is SeedLiftStatus.USABLE
    assert result.expansion_recommendation is FinalMaskMultiViewAction.STOP
    assert result.views == ()
    assert "no additional candidate views are available" in result.expansion_reason


def test_suggest_additional_views_rejects_seed_artifact_mismatch(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, _mask_ply_path = _write_points_artifact(seed_dir)
    mismatched_ply_path = _write_one_point_ply(seed_dir / "bad.ply")
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mismatched_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
    )

    with pytest.raises(ToolInputError, match="mask npz point count"):
        suggest_additional_views(tool_scene, args)


def test_suggest_additional_views_rejects_lift_overlay_frame_mismatch(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000020",
        candidate_id="mask_00",
    )
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=mask_npz_path,
        seed_mask_ply_path=mask_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
    )

    with pytest.raises(ToolInputError, match="seed lift overlay"):
        suggest_additional_views(tool_scene, args)


def test_suggest_additional_views_rejects_seed_npz_from_different_fragment(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000010_mask_00"
    other_dir = tmp_path / "fragments" / "000020_mask_01"
    _seed_npz_path, _seed_ply_path = _write_points_artifact(seed_dir)
    other_npz_path, other_ply_path = _write_points_artifact(other_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000010",
        candidate_id="mask_00",
    )
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="000010_mask_00",
        accepted_frame_id="000010",
        seed_mask_npz_path=other_npz_path,
        seed_mask_ply_path=other_ply_path,
        seed_lift_overlay_path=lift_overlay_path,
    )

    with pytest.raises(ToolInputError, match="same seed fragment directory"):
        suggest_additional_views(tool_scene, args)


def test_suggest_additional_views_requires_seed_artifact() -> None:
    with pytest.raises(ValueError, match="seed_mask_npz_path"):
        SuggestAdditionalViewsArgs.model_validate(
            {
                "seed_fragment_id": "frag-a",
                "accepted_frame_id": "000010",
                "k": 2,
            }
        )


def test_suggest_additional_views_args_accept_seed_frame_id_alias(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000050_mask_02"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000050",
        candidate_id="mask_02",
    )

    args = SuggestAdditionalViewsArgs.model_validate(
        {
            "seed_fragment_id": "000050_mask_02",
            "seed_frame_id": "000050",
            "seed_mask_npz_path": str(mask_npz_path),
            "seed_mask_ply_path": str(mask_ply_path),
            "seed_lift_overlay_path": str(lift_overlay_path),
            "task_description": "Open the lower drawer.",
        }
    )

    assert args.accepted_frame_id == "000050"


def test_suggest_additional_views_args_accept_seed_candidate_id_context(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000050_mask_02"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000050",
        candidate_id="mask_02",
    )

    args = SuggestAdditionalViewsArgs.model_validate(
        {
            "seed_fragment_id": "000050_mask_02",
            "seed_frame_id": "000050",
            "seed_candidate_id": "mask_02",
            "seed_mask_npz_path": str(mask_npz_path),
            "seed_mask_ply_path": str(mask_ply_path),
            "seed_lift_overlay_path": str(lift_overlay_path),
            "task_description": "Adjust room temperature using the radiator dial.",
        }
    )

    assert args.accepted_frame_id == "000050"
    assert args.task_description == "Adjust room temperature using the radiator dial."


def test_suggest_additional_views_args_infers_accepted_frame_from_seed_fragment(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / "000050_mask_02"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000050",
        candidate_id="mask_02",
    )

    args = SuggestAdditionalViewsArgs.model_validate(
        {
            "seed_fragment_id": "000050_mask_02",
            "seed_mask_npz_path": str(mask_npz_path),
            "seed_mask_ply_path": str(mask_ply_path),
            "seed_lift_overlay_path": str(lift_overlay_path),
        }
    )

    assert args.accepted_frame_id == "000050"


@pytest.mark.parametrize("seed_fragment_id", ["frag-a", "000050_"])
def test_suggest_additional_views_args_keeps_frame_required_for_unparseable_seed(
    seed_fragment_id: str,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "fragments" / seed_fragment_id
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000050",
        candidate_id="mask_02",
    )

    with pytest.raises(ValueError, match="accepted_frame_id"):
        SuggestAdditionalViewsArgs.model_validate(
            {
                "seed_fragment_id": seed_fragment_id,
                "seed_mask_npz_path": str(mask_npz_path),
                "seed_mask_ply_path": str(mask_ply_path),
                "seed_lift_overlay_path": str(lift_overlay_path),
            }
        )


def test_fused_mask_payload(tmp_path: Path) -> None:
    review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    result = FusedMaskResult(
        accepted_fragments=(
            AcceptedFragment(
                fragment_id="000050_mask_00",
                frame_id="000050",
                point_count=1119,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifacts(
                    molmo_raw_text_path=Path(review_artifacts["molmo_raw_text_path"]),
                    molmo_overlay_path=Path(review_artifacts["molmo_overlay_path"]),
                    sam_contact_sheet_path=Path(
                        review_artifacts["sam_contact_sheet_path"]
                    ),
                    sam_candidate_overlay_path=Path(
                        review_artifacts["sam_candidate_overlay_path"]
                    ),
                    lift_overlay_path=Path(review_artifacts["lift_overlay_path"]),
                ),
            ),
        ),
        multi_view_decision=_single_view_decision_input("000050_mask_00").to_domain(),
        mask_artifact_path=Path("/tmp/fused/mask_artifact.json"),
        mask_npz_path=Path("/tmp/fused/mask_data.npz"),
        mask_ply_path=Path("/tmp/fused/lifted_points.ply"),
    )

    payload = cast(FusedMaskPayload, result.to_payload())

    assert payload["accepted_fragments"][0]["fragment_id"] == "000050_mask_00"
    assert payload["accepted_fragments"][0]["frame_id"] == "000050"
    assert payload["accepted_fragments"][0]["approval_actions"] == _action_values(
        _APPROVED_FRAGMENT_ACTIONS
    )
    assert payload["accepted_fragments"][0]["review_artifacts"] == review_artifacts
    assert payload["accepted_frame_ids"] == ["000050"]
    assert payload["multi_view_decision"]["action"] == "stop"
    assert payload["mask_artifact_path"] == "/tmp/fused/mask_artifact.json"


def test_accepted_fragment_rejects_invalid_approval_actions() -> None:
    with pytest.raises(ValueError, match="approval_actions"):
        AcceptedFragment(
            fragment_id="000050_mask_00",
            frame_id="000050",
            point_count=1119,
            approval_actions=(
                ApprovalAction.SELECT_EVIDENCE,
                ApprovalAction.PROPOSE_SAM_CANDIDATES,
            ),
            review_artifacts=AcceptedFragmentReviewArtifacts(
                molmo_raw_text_path=Path("/tmp/molmo_raw.txt"),
                molmo_overlay_path=Path("/tmp/molmo_overlay.jpg"),
                sam_contact_sheet_path=Path("/tmp/sam_contact.jpg"),
                sam_candidate_overlay_path=Path("/tmp/sam_candidate.jpg"),
                lift_overlay_path=Path("/tmp/lift_overlay.txt"),
            ),
        )


def test_fuse_accepted_masks_writes_valid_final_artifact(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    first_npz_path, first_ply_path = _write_points_artifact(
        tmp_path / "frag-a", point_indices=(10, 12)
    )
    second_npz_path, second_ply_path = _write_points_artifact(
        tmp_path / "frag-b", point_indices=(20, 22)
    )
    first_review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    second_review_artifacts = _write_review_artifacts(tmp_path / "review-b")
    args = FuseAcceptedMasksArgs(
        fragments=(
            AcceptedFragmentInput(
                fragment_id="frag-a",
                frame_id="000010",
                mask_npz_path=first_npz_path,
                mask_ply_path=first_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    first_review_artifacts
                ),
            ),
            AcceptedFragmentInput(
                fragment_id="frag-b",
                frame_id="000020",
                mask_npz_path=second_npz_path,
                mask_ply_path=second_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    second_review_artifacts
                ),
            ),
        ),
        multi_view_decision=MultiViewDecisionInput.model_validate(
            _expand_decision_payload()
        ),
    )

    result = fuse_accepted_masks(args, out_dir=tmp_path / "out")

    payload = json.loads(result.mask_artifact_path.read_text(encoding="utf-8"))
    assert payload["accepted_frame_ids"] == ["000010", "000020"]
    assert payload["accepted_fragments"] == [
        {
            "fragment_id": "frag-a",
            "frame_id": "000010",
            "point_count": 2,
            "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
            "review_artifacts": first_review_artifacts,
        },
        {
            "fragment_id": "frag-b",
            "frame_id": "000020",
            "point_count": 2,
            "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
            "review_artifacts": second_review_artifacts,
        },
    ]
    assert result.to_payload()["accepted_frame_ids"] == ["000010", "000020"]
    assert Path(payload["mask_npz_path"]) == result.mask_npz_path
    assert Path(payload["mask_ply_path"]) == result.mask_ply_path
    with np.load(result.mask_npz_path) as archive:
        np.testing.assert_array_equal(
            archive["point_indices"], np.array([10, 12, 20, 22], dtype=np.int64)
        )
    assert result.mask_npz_path.exists()
    assert result.mask_ply_path.read_text(encoding="ascii").startswith("ply\n")


def test_fuse_accepted_masks_records_multi_view_decision(tmp_path: Path) -> None:
    first_npz_path, first_ply_path = _write_points_artifact(
        tmp_path / "frag-a", point_indices=(10, 12)
    )
    second_npz_path, second_ply_path = _write_points_artifact(
        tmp_path / "frag-b", point_indices=(20, 22)
    )
    first_review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    second_review_artifacts = _write_review_artifacts(tmp_path / "review-b")
    multi_view_decision = {
        "seed_fragment_id": "frag-a",
        "action": "expand",
        "reason": "first lift is sparse and the handle side may be occluded",
        "suggested_frame_ids": ["000020"],
    }
    args = FuseAcceptedMasksArgs.model_validate(
        {
            "fragments": [
                {
                    "fragment_id": "frag-a",
                    "frame_id": "000010",
                    "mask_npz_path": str(first_npz_path),
                    "mask_ply_path": str(first_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": first_review_artifacts,
                },
                {
                    "fragment_id": "frag-b",
                    "frame_id": "000020",
                    "mask_npz_path": str(second_npz_path),
                    "mask_ply_path": str(second_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": second_review_artifacts,
                },
            ],
            "multi_view_decision": multi_view_decision,
        }
    )

    result = fuse_accepted_masks(args, out_dir=tmp_path / "out")

    payload = json.loads(result.mask_artifact_path.read_text(encoding="utf-8"))
    assert payload["multi_view_decision"] == multi_view_decision
    assert result.to_payload()["multi_view_decision"] == multi_view_decision


def test_fuse_accepted_masks_args_accept_accepted_fragments_alias(
    tmp_path: Path,
) -> None:
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "000050_mask_02")
    review_artifacts = _write_review_artifacts(tmp_path / "review")

    args = FuseAcceptedMasksArgs.model_validate(
        {
            "accepted_fragments": [
                {
                    "fragment_id": "000050_mask_02",
                    "frame_id": "000050",
                    "candidate_id": "mask_02",
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": review_artifacts,
                }
            ],
            "multi_view_decision": {
                "seed_fragment_id": "000050_mask_02",
                "action": "stop",
                "reason": "first lift is enough",
                "suggested_frame_ids": ["000112", "000116"],
            },
            "task_description": "Open the lower drawer by pulling the handle.",
        }
    )

    assert tuple(fragment.fragment_id for fragment in args.fragments) == (
        "000050_mask_02",
    )


def test_fuse_accepted_masks_ignores_redundant_candidate_id_on_fragments(
    tmp_path: Path,
) -> None:
    mask_npz_path, mask_ply_path = _write_points_artifact(
        tmp_path / "frag-a", point_indices=(10, 12)
    )
    review_artifacts = _write_review_artifacts(tmp_path / "review-a")

    args = FuseAcceptedMasksArgs.model_validate(
        {
            "fragments": [
                {
                    "fragment_id": "frag-a",
                    "frame_id": "000010",
                    "candidate_id": "mask_00",
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": review_artifacts,
                }
            ],
            "multi_view_decision": {
                "seed_fragment_id": "frag-a",
                "action": "stop",
                "reason": "first lift covers the accepted target part",
                "suggested_frame_ids": [],
            },
        }
    )

    assert args.fragments[0].fragment_id == "frag-a"
    assert args.fragments[0].frame_id == "000010"


def test_fuse_accepted_masks_rejects_multi_view_seed_mismatch(
    tmp_path: Path,
) -> None:
    args = _two_fragment_fuse_args(
        tmp_path,
        multi_view_decision=_expand_decision_payload(seed_fragment_id="frag-b"),
    )

    with pytest.raises(ToolInputError, match="seed_fragment_id"):
        fuse_accepted_masks(args, out_dir=tmp_path / "out")


def test_fuse_accepted_masks_rejects_stop_decision_for_multiple_frames(
    tmp_path: Path,
) -> None:
    args = _two_fragment_fuse_args(
        tmp_path,
        multi_view_decision={
            "seed_fragment_id": "frag-a",
            "action": FinalMaskMultiViewAction.STOP.value,
            "reason": "incorrectly stops despite accepting a follow-up frame",
            "suggested_frame_ids": [],
        },
    )

    with pytest.raises(ToolInputError, match="action must be 'expand'"):
        fuse_accepted_masks(args, out_dir=tmp_path / "out")


def test_fuse_accepted_masks_rejects_expand_without_accepted_suggested_frame(
    tmp_path: Path,
) -> None:
    args = _two_fragment_fuse_args(
        tmp_path,
        multi_view_decision=_expand_decision_payload(suggested_frame_ids=("000030",)),
    )

    with pytest.raises(ToolInputError, match="accepted follow-up frame"):
        fuse_accepted_masks(args, out_dir=tmp_path / "out")


def test_fuse_accepted_masks_rejects_fragment_without_point_indices(
    tmp_path: Path,
) -> None:
    mask_npz_path, mask_ply_path = _write_points_world_only_artifact(
        tmp_path / "frag-a"
    )
    review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    args = FuseAcceptedMasksArgs(
        fragments=(
            AcceptedFragmentInput(
                fragment_id="frag-a",
                frame_id="000010",
                mask_npz_path=mask_npz_path,
                mask_ply_path=mask_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    review_artifacts
                ),
            ),
        ),
        multi_view_decision=_single_view_decision_input(),
    )

    with pytest.raises(ToolInputError, match="point_indices"):
        fuse_accepted_masks(args, out_dir=tmp_path / "out")


def test_fuse_accepted_masks_deduplicates_accepted_frame_ids(
    tmp_path: Path,
) -> None:
    first_npz_path, first_ply_path = _write_points_artifact(tmp_path / "frag-a")
    second_npz_path, second_ply_path = _write_points_artifact(tmp_path / "frag-b")
    first_review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    second_review_artifacts = _write_review_artifacts(tmp_path / "review-b")
    args = FuseAcceptedMasksArgs(
        fragments=(
            AcceptedFragmentInput(
                fragment_id="frag-a",
                frame_id="000010",
                mask_npz_path=first_npz_path,
                mask_ply_path=first_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    first_review_artifacts
                ),
            ),
            AcceptedFragmentInput(
                fragment_id="frag-b",
                frame_id="000010",
                mask_npz_path=second_npz_path,
                mask_ply_path=second_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    second_review_artifacts
                ),
            ),
        ),
        multi_view_decision=_single_view_decision_input(),
    )

    result = fuse_accepted_masks(args, out_dir=tmp_path / "out")

    payload = json.loads(result.mask_artifact_path.read_text(encoding="utf-8"))
    assert payload["accepted_frame_ids"] == ["000010"]
    assert result.to_payload()["accepted_frame_ids"] == ["000010"]
    assert [fragment["frame_id"] for fragment in payload["accepted_fragments"]] == [
        "000010",
        "000010",
    ]
    assert [
        fragment["approval_actions"] for fragment in payload["accepted_fragments"]
    ] == [
        _action_values(_APPROVED_FRAGMENT_ACTIONS),
        _action_values(_APPROVED_FRAGMENT_ACTIONS),
    ]
    assert [
        fragment["review_artifacts"] for fragment in payload["accepted_fragments"]
    ] == [
        first_review_artifacts,
        second_review_artifacts,
    ]


def _action_values(actions: tuple[ApprovalAction, ...]) -> list[str]:
    return [action.value for action in actions]


def _single_view_decision_input(
    seed_fragment_id: str = "frag-a",
) -> MultiViewDecisionInput:
    return MultiViewDecisionInput(
        seed_fragment_id=seed_fragment_id,
        action=FinalMaskMultiViewAction.STOP,
        reason="first lift covers the target part in the accepted frame",
    )


def _expand_decision_payload(
    *,
    seed_fragment_id: str = "frag-a",
    suggested_frame_ids: tuple[str, ...] = ("000020",),
) -> MultiViewDecisionPayload:
    return {
        "seed_fragment_id": seed_fragment_id,
        "action": FinalMaskMultiViewAction.EXPAND.value,
        "reason": "first lift is sparse and the target part needs another view",
        "suggested_frame_ids": list(suggested_frame_ids),
    }


def _two_fragment_fuse_args(
    tmp_path: Path,
    *,
    multi_view_decision: MultiViewDecisionPayload,
) -> FuseAcceptedMasksArgs:
    first_npz_path, first_ply_path = _write_points_artifact(
        tmp_path / "frag-a", point_indices=(10, 12)
    )
    second_npz_path, second_ply_path = _write_points_artifact(
        tmp_path / "frag-b", point_indices=(20, 22)
    )
    first_review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    second_review_artifacts = _write_review_artifacts(tmp_path / "review-b")
    return FuseAcceptedMasksArgs.model_validate(
        {
            "fragments": [
                {
                    "fragment_id": "frag-a",
                    "frame_id": "000010",
                    "mask_npz_path": str(first_npz_path),
                    "mask_ply_path": str(first_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": first_review_artifacts,
                },
                {
                    "fragment_id": "frag-b",
                    "frame_id": "000020",
                    "mask_npz_path": str(second_npz_path),
                    "mask_ply_path": str(second_ply_path),
                    "approval_actions": _action_values(_APPROVED_FRAGMENT_ACTIONS),
                    "review_artifacts": second_review_artifacts,
                },
            ],
            "multi_view_decision": multi_view_decision,
        }
    )


def _write_review_artifacts(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "molmo_raw_text_path": root / "molmo_raw.txt",
        "molmo_overlay_path": root / "molmo_overlay.jpg",
        "sam_contact_sheet_path": root / "sam_contact_sheet.jpg",
        "sam_candidate_overlay_path": root / "sam_candidate_overlay.jpg",
        "lift_overlay_path": root / "lift_overlay.txt",
    }
    for path in paths.values():
        path.write_text("reviewed\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _write_points_artifact(
    root: Path, *, point_indices: tuple[int, ...] = (10, 12)
) -> tuple[Path, Path]:
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    points_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    point_indices_array = np.array(point_indices, dtype=np.int64)
    mask_npz_path = write_lift_npz(
        root / "mask_data.npz", points_world, point_indices=point_indices_array
    )
    mask_ply_path = write_lift_ply(root / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path


def _write_one_point_ply(path: Path) -> Path:
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_ply

    return write_lift_ply(
        path,
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    )


def _write_lift_overlay(path: Path, *, frame_id: str, candidate_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"frame_id={frame_id}\n"
        f"candidate_id={candidate_id}\n"
        "lifted_point_count=2\n",
        encoding="utf-8",
    )
    return path


def _write_geometry_assets(
    raw_dir: Path, frame_id: str, *, x_translation: float
) -> None:
    (raw_dir / "depth").mkdir(parents=True, exist_ok=True)
    (raw_dir / "depth" / f"{frame_id}.png").write_bytes(b"depth")
    (raw_dir / f"{frame_id}-intrinsics.txt").write_text(
        "1 0 0\n0 1 0\n0 0 1\n",
        encoding="utf-8",
    )
    _write_pose(raw_dir / "pose" / f"{frame_id}.txt", x_translation=x_translation)


def _write_pose(path: Path, *, x_translation: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1 0 0 {x_translation}\n" "0 1 0 0\n" "0 0 1 0\n" "0 0 0 1\n",
        encoding="utf-8",
    )


def _write_points_world_only_artifact(root: Path) -> tuple[Path, Path]:
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    points_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    mask_npz_path = write_lift_npz(root / "mask_data.npz", points_world)
    mask_ply_path = write_lift_ply(root / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path
