"""Tests for SceneFunc3D artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from codex_agent.errors import SceneFunc3dDataError
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
    SuggestAdditionalViewsArgs,
    SuggestedView,
    SuggestedViewsPayload,
    SuggestedViewsResult,
    fuse_accepted_masks,
    inspect_mask_artifact,
    suggest_additional_views,
)
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


def test_suggested_views_payload() -> None:
    result = SuggestedViewsResult(
        seed_fragment_id="000050_mask_00",
        views=(
            SuggestedView(frame_id="000060", reason="different view angle", rank=1),
        ),
    )

    payload = cast(SuggestedViewsPayload, result.to_payload())

    assert payload["views"][0]["reason"] == "different view angle"


def test_suggest_additional_views_ranks_temporal_neighbors(tmp_path: Path) -> None:
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    for frame_id in ("000000", "000010", "000020", "000040"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    tool_scene = SceneFunc3dToolScene.load(scene_dir)
    args = SuggestAdditionalViewsArgs(
        seed_fragment_id="frag-a",
        accepted_frame_id="000010",
        k=2,
    )

    result = suggest_additional_views(tool_scene, args)

    assert result.seed_fragment_id == "frag-a"
    assert tuple(view.frame_id for view in result.views) == ("000000", "000020")


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
                frame_id="000020",
                mask_npz_path=second_npz_path,
                mask_ply_path=second_ply_path,
                approval_actions=_APPROVED_FRAGMENT_ACTIONS,
                review_artifacts=AcceptedFragmentReviewArtifactsInput.model_validate(
                    second_review_artifacts
                ),
            ),
        )
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
    assert result.mask_npz_path.exists()
    assert result.mask_ply_path.read_text(encoding="ascii").startswith("ply\n")


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
        )
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


def _write_points_artifact(root: Path) -> tuple[Path, Path]:
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    points_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    mask_npz_path = write_lift_npz(root / "mask_data.npz", points_world)
    mask_ply_path = write_lift_ply(root / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path
