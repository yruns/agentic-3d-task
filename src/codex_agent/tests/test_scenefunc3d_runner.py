"""Tests for the SceneFunc3D single-case runtime runner."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pytest

import codex_agent.scenefunc3d.runner as runner
from codex_agent.errors import CodexResponseError
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.scenefunc3d.runner import (
    SCENEFUNC3D_ALLOWED_TOOL_NAMES,
    SceneFunc3dMaskOutcome,
    SceneFunc3dMaskTask,
    SceneFunc3dRunnerConfig,
    build_arg_parser,
    check_sidecar_health,
    main,
    run_single_sample,
)
from codex_agent.scenefunc3d.sample import SceneFunc3dSample, SceneFuncMotionHint
from codex_agent.scenefunc3d.servers.http_json import (
    JsonObject,
    make_json_handler,
)
from codex_agent.scenefunc3d.task import ApprovalAction
from codex_agent.tasks.base import CodexExecutor, CodexTask

ResultT = TypeVar("ResultT")

_APPROVED_FRAGMENT_ACTIONS = (
    ApprovalAction.SELECT_EVIDENCE.value,
    ApprovalAction.PROPOSE_MOLMO_POINT.value,
    ApprovalAction.APPROVE_MOLMO_POINT.value,
    ApprovalAction.PROPOSE_SAM_CANDIDATES.value,
    ApprovalAction.APPROVE_SAM_CANDIDATE.value,
    ApprovalAction.CREATE_FIRST_LIFT.value,
    ApprovalAction.APPROVE_FIRST_LIFT.value,
)


def test_build_arg_parser_accepts_single_case_runtime_options(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.dataset_root == tmp_path / "data"
    assert args.sample_id == "421254::desc-a"
    assert args.backend_config == tmp_path / "backends.toml"
    assert args.output_dir == tmp_path / "out"


def test_mask_task_prompt_inlines_tools_without_attachments(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "421254"
    backend_config_path = tmp_path / "backends.toml"
    output_dir = tmp_path / "out"
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=backend_config_path,
    )

    request = task.build_turn_request()

    assert request.skills == ()
    assert request.image_paths == ()
    assert "Molmo point" in request.prompt
    assert "SAM candidates" in request.prompt
    assert "--backend-config" in request.prompt
    assert (
        f"python -m codex_agent.scenefunc3d.tools <tool> "
        f"--scene-root {scene_root} "
        f"--backend-config {backend_config_path} "
        f"--out-dir {output_dir} --args '<json>'"
    ) in request.prompt
    hard_limits = request.prompt.split("Hard limits:\n", maxsplit=1)[1].split(
        "\nFinal JSON schema:", maxsplit=1
    )[0]
    assert "Use ONLY these SceneFunc3D tools plus view_image" in hard_limits
    assert ", ".join(SCENEFUNC3D_ALLOWED_TOOL_NAMES) in hard_limits
    for tool_name in SCENEFUNC3D_ALLOWED_TOOL_NAMES:
        assert tool_name in hard_limits
    assert "Do NOT read, cat, sed, head, grep, rg, or open" in hard_limits
    assert "Never re-run a tool with identical arguments" in hard_limits


def test_mask_task_parses_strict_final_json(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    payload = _outcome_payload(output_dir)

    outcome = task.parse_response(json.dumps(payload))

    assert task.is_valid_response(json.dumps(payload)) is True
    assert isinstance(outcome, SceneFunc3dMaskOutcome)
    assert outcome.mask_artifact_path == output_dir / "mask_artifact.json"
    assert outcome.mask_npz_path == output_dir / "mask.npz"
    assert outcome.mask_ply_path == output_dir / "mask.ply"
    assert outcome.selected_frame_ids == ("000010",)
    assert outcome.accepted_fragment_ids == ("frag-a",)
    assert outcome.confidence == 0.87
    assert outcome.uncertainties == ("partial occlusion",)
    assert outcome.to_payload() == payload


def test_mask_task_rejects_artifact_without_multi_view_decision(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload.pop("multi_view_decision", None)
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )

    with pytest.raises(CodexResponseError, match="multi_view_decision"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_artifact_without_fragment_lift_geometry(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir, include_fragment_lift_geometry=False)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="lift_geometry"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_fragment_lift_geometry_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir, include_fragment_lift_geometry=True)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["lift_geometry"][
        "max_extent_meters"
    ] = 0.1
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="lift_geometry.*must match"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_nonexistent_final_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    response_text = json.dumps(_outcome_payload(output_dir))

    assert task.is_valid_response(response_text) is False
    with pytest.raises(CodexResponseError, match="does not exist"):
        task.parse_response(response_text)


def test_mask_task_rejects_invalid_mask_npz_contents(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    (output_dir / "mask.npz").write_bytes(b"npz")
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="mask_npz_path"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_corrupt_zip_mask_npz_contents(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    (output_dir / "mask.npz").write_bytes(b"PK\x03\x04bad")
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    assert task.is_valid_response(json.dumps(_outcome_payload(output_dir))) is False
    with pytest.raises(CodexResponseError, match="mask_npz_path"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_mask_npz_without_point_indices(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    points_world = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    np.savez_compressed(output_dir / "mask.npz", points_world=points_world)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="point_indices"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_invalid_mask_ply_contents(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    (output_dir / "mask.ply").write_text("ply\n", encoding="ascii")
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="mask_ply_path"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_final_artifact_path_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["mask_npz_path"] = str(output_dir / "other.npz")
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="mask_npz_path"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_accepted_fragment_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir, accepted_fragment_ids=("frag-b",))
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="accepted_fragment_ids"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_fragment_point_count_sum_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"] = [
        {
            "fragment_id": "frag-a",
            "frame_id": "000010",
            "point_count": 1,
            "lift_geometry": _test_lift_geometry_payload(),
            "approval_actions": list(_APPROVED_FRAGMENT_ACTIONS),
            "review_artifacts": _write_review_artifacts(
                output_dir / "review_artifacts" / "frag-a"
            ),
        }
    ]
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="point_count"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_selected_frame_missing_from_scene(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254", frame_ids=("000010",))
    _write_outcome_artifacts(output_dir)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    payload = _outcome_payload(output_dir, selected_frame_ids=("000020",))
    with pytest.raises(CodexResponseError, match="selected_frame_ids"):
        task.parse_response(json.dumps(payload))


def test_mask_task_rejects_selected_frame_mismatch_with_final_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir, accepted_frame_ids=("000010",))
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    payload = _outcome_payload(output_dir, selected_frame_ids=("000020",))

    with pytest.raises(CodexResponseError, match="selected_frame_ids"):
        task.parse_response(json.dumps(payload))


def test_mask_task_rejects_artifact_accepted_frame_mismatch_with_fragments(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_frame_ids"] = ["000020"]
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    payload = _outcome_payload(output_dir, selected_frame_ids=("000020",))

    with pytest.raises(CodexResponseError, match="accepted_frame_ids"):
        task.parse_response(json.dumps(payload))


def test_mask_task_rejects_fragment_without_approval_actions(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0].pop("approval_actions")
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="approval_actions"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_fragment_approval_actions_out_of_order(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["approval_actions"] = [
        ApprovalAction.SELECT_EVIDENCE.value,
        ApprovalAction.PROPOSE_SAM_CANDIDATES.value,
    ]
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="approval_actions"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_missing_fragment_review_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"][
        "molmo_overlay_path"
    ] = str(output_dir / "missing_overlay.jpg")
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="review_artifacts"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_review_artifact_from_wrong_fragment(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    wrong_review_artifacts = _write_review_artifacts(
        output_dir / "review_artifacts" / "wrong-fragment"
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"][
        "lift_overlay_path"
    ] = wrong_review_artifacts["lift_overlay_path"]
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="review_artifacts"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_lift_overlay_with_nonstandard_filename(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(output_dir)
    nonstandard_overlay_path = output_dir / "review_artifacts" / "frag-a" / "other.txt"
    nonstandard_overlay_path.write_text("reviewed\n", encoding="utf-8")
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"][
        "lift_overlay_path"
    ] = str(nonstandard_overlay_path)
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="lift_overlay.txt"):
        task.parse_response(json.dumps(_outcome_payload(output_dir)))


def test_mask_task_rejects_molmo_raw_text_from_wrong_frame(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    wrong_raw_text_path = output_dir / "molmo" / "000011_raw.txt"
    wrong_raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_raw_text_path.write_text("wrong frame\n", encoding="utf-8")
    review_artifacts["molmo_raw_text_path"] = str(wrong_raw_text_path)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="molmo_raw_text_path"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_sam_contact_sheet_from_wrong_frame(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    wrong_contact_sheet_path = output_dir / "sam" / "000011" / "contact_sheet.jpg"
    wrong_contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_contact_sheet_path.write_text("wrong frame\n", encoding="utf-8")
    review_artifacts["sam_contact_sheet_path"] = str(wrong_contact_sheet_path)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="sam_contact_sheet_path"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_sam_candidate_overlay_from_wrong_candidate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    wrong_candidate_overlay_path = output_dir / "sam" / "000010" / "mask_01_overlay.jpg"
    wrong_candidate_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_candidate_overlay_path.write_text("wrong candidate\n", encoding="utf-8")
    review_artifacts["sam_candidate_overlay_path"] = str(wrong_candidate_overlay_path)
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="sam_candidate_overlay_path"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_standard_review_artifacts_from_other_run(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    other_output_dir = tmp_path / "other-out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        other_output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="expected_path"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_lift_overlay_candidate_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    _write_lift_overlay_summary(
        Path(review_artifacts["lift_overlay_path"]),
        frame_id="000010",
        candidate_id="mask_01",
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="candidate_id"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_lift_overlay_frame_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    _write_lift_overlay_summary(
        Path(review_artifacts["lift_overlay_path"]),
        frame_id="000011",
        candidate_id="mask_00",
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="frame_id"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_mask_task_rejects_standard_fragment_point_count_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    scene_root = _write_scene_root(tmp_path / "421254")
    _write_outcome_artifacts(
        output_dir,
        accepted_fragment_ids=("000010_mask_00",),
        accepted_frame_ids=("000010",),
    )
    review_artifacts = _write_standard_review_artifacts(
        output_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    _write_fragment_points_artifact(
        output_dir / "fragments" / "000010_mask_00",
        point_count=1,
    )
    artifact_payload = json.loads(
        (output_dir / "mask_artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["accepted_fragments"][0]["review_artifacts"] = review_artifacts
    (output_dir / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="point_count"):
        task.parse_response(
            json.dumps(
                _outcome_payload(
                    output_dir,
                    accepted_fragment_ids=("000010_mask_00",),
                )
            )
        )


def test_check_sidecar_health_passes_for_healthy_fake_servers(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecar_health_rejects_unloaded_sidecar(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server_with_state(
        model_name="fake-sam",
        model_loaded=False,
    )
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        with pytest.raises(RuntimeError, match="model_loaded=false"):
            check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecars_script_uses_backend_config(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )
    script_path = Path("scripts/scenefunc3d/check_sidecars.sh").resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    try:
        result = subprocess.run(
            [str(script_path), str(backend_config_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        molmo_server.close()
        sam_server.close()

    assert "sidecars ok" in result.stdout


def test_run_single_sample_writes_result_json_with_outcome_payload(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    _write_outcome_artifacts(
        sample_output_dir,
        rejected_suggested_frame_ids=("000020", "000030"),
    )
    outcome = SceneFunc3dMaskOutcome(
        mask_artifact_path=sample_output_dir / "mask_artifact.json",
        mask_npz_path=sample_output_dir / "mask.npz",
        mask_ply_path=sample_output_dir / "mask.ply",
        selected_frame_ids=("000010",),
        accepted_fragment_ids=("frag-a",),
        confidence=0.87,
        uncertainties=("partial occlusion",),
    )
    executor = _FakeExecutor(
        outcome,
        on_execute=lambda: _write_suggest_additional_views_event(
            sample_output_dir,
            expansion_recommendation="expand",
            frame_ids=("000020", "000030"),
        ),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    result_path = run_single_sample(
        config,
        sample_id="421254::desc-a",
        executor=executor,
        check_sidecars=False,
    )

    assert result_path == tmp_path / "out" / "421254" / "desc-a" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "scenefunc3d_mask_generation"
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["outcome"] == _outcome_payload(sample_output_dir)
    assert payload["artifact"] == {
        "accepted_frame_ids": ["000010"],
        "accepted_fragment_ids": ["frag-a"],
        "final_point_count": 2,
        "multi_view_decision": {
            "seed_fragment_id": "frag-a",
            "action": "stop",
            "reason": "test artifact records the first-lift multi-view decision",
            "suggested_frame_ids": [],
            "rejected_suggested_frame_ids": ["000020", "000030"],
        },
    }
    assert payload["turn"] == {
        "turn_id": "fake-turn",
        "status": "completed",
        "duration_ms": None,
        "usage": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "reasoning_summary": None,
        "run_home": None,
        "attempts": [],
    }
    assert executor.task_name == "scenefunc3d_mask_generation"
    assert str(tmp_path / "data" / "421254") in executor.prompt
    summary_path = sample_output_dir / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["sample_id"] == "421254::desc-a"
    assert summary_payload["visit_id"] == "421254"
    assert summary_payload["desc_id"] == "desc-a"
    assert summary_payload["task_description"] == "Open the lower drawer."
    assert summary_payload["status"] == "success"
    assert summary_payload["selected_frame_ids"] == ["000010"]
    assert summary_payload["accepted_fragment_ids"] == ["frag-a"]
    assert summary_payload["failure_type"] == ""
    assert summary_payload["stop_reason"] == "single_view_complete"
    assert summary_payload["mask_artifact_path"] == str(
        sample_output_dir / "mask_artifact.json"
    )
    assert summary_payload["mask_npz_path"] == str(sample_output_dir / "mask.npz")
    assert summary_payload["mask_ply_path"] == str(sample_output_dir / "mask.ply")
    assert summary_payload["confidence"] == 0.87
    assert summary_payload["uncertainties"] == ["partial occlusion"]
    assert summary_payload["final_point_count"] == 2
    assert summary_payload["multi_view_decision"] == {
        "seed_fragment_id": "frag-a",
        "action": "stop",
        "reason": "test artifact records the first-lift multi-view decision",
        "suggested_frame_ids": [],
        "rejected_suggested_frame_ids": ["000020", "000030"],
    }
    event_lines = (
        (sample_output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(event_lines) == 2
    completion_event = json.loads(event_lines[-1])
    assert completion_event == {
        "event_type": "run_completed",
        "sample_id": "421254::desc-a",
        "status": "success",
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "mask_artifact_path": str(sample_output_dir / "mask_artifact.json"),
    }


def test_run_single_sample_rejects_stop_missing_expand_rejected_frames(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    _write_outcome_artifacts(sample_output_dir)
    outcome = SceneFunc3dMaskOutcome(
        mask_artifact_path=sample_output_dir / "mask_artifact.json",
        mask_npz_path=sample_output_dir / "mask.npz",
        mask_ply_path=sample_output_dir / "mask.ply",
        selected_frame_ids=("000010",),
        accepted_fragment_ids=("frag-a",),
        confidence=0.87,
        uncertainties=("partial occlusion",),
    )
    executor = _FakeExecutor(
        outcome,
        on_execute=lambda: _write_suggest_additional_views_event(
            sample_output_dir,
            expansion_recommendation="expand",
            frame_ids=("000020", "000030"),
        ),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="missing_rejected_suggested"):
        run_single_sample(
            config,
            sample_id="421254::desc-a",
            executor=executor,
            check_sidecars=False,
        )


def test_run_single_sample_ignores_stale_expand_events_from_previous_run(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    _write_outcome_artifacts(sample_output_dir)
    _write_suggest_additional_views_event(
        sample_output_dir,
        expansion_recommendation="expand",
        frame_ids=("000020", "000030"),
    )
    outcome = SceneFunc3dMaskOutcome(
        mask_artifact_path=sample_output_dir / "mask_artifact.json",
        mask_npz_path=sample_output_dir / "mask.npz",
        mask_ply_path=sample_output_dir / "mask.ply",
        selected_frame_ids=("000010",),
        accepted_fragment_ids=("frag-a",),
        confidence=0.87,
        uncertainties=("partial occlusion",),
    )
    executor = _FakeExecutor(outcome)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    result_path = run_single_sample(
        config,
        sample_id="421254::desc-a",
        executor=executor,
        check_sidecars=False,
    )

    event_lines = (
        (sample_output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(event_lines) == 1
    assert json.loads(event_lines[0])["event_type"] == "run_completed"
    assert result_path == sample_output_dir / "result.json"


def test_main_with_score_prints_result_path_and_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_scene(tmp_path / "data")

    def fake_run_single_sample(
        config: SceneFunc3dRunnerConfig,
        *,
        sample_id: str,
        executor: object,
        check_sidecars: bool = True,
    ) -> Path:
        assert sample_id == "421254::desc-a"
        assert check_sidecars is False
        _ = executor
        sample_output_dir = config.output_dir / "421254" / "desc-a"
        _write_outcome_artifacts(sample_output_dir)
        result_path = sample_output_dir / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "task_name": "scenefunc3d_mask_generation",
                    "sample_id": sample_id,
                    "outcome": _outcome_payload(sample_output_dir),
                    "turn": {},
                }
            ),
            encoding="utf-8",
        )
        return result_path

    def fake_build_executor() -> CodexExecutor:
        return _UnusedExecutor()

    monkeypatch.setattr(runner, "run_single_sample", fake_run_single_sample)
    monkeypatch.setattr(runner, "_build_executor", fake_build_executor)

    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-sidecar-health-check",
            "--score",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result_path"] == str(
        tmp_path / "out" / "421254" / "desc-a" / "result.json"
    )
    score = payload["score"]
    assert score["sample_id"] == "421254::desc-a"
    assert score["metrics"]["iou"] == 0.0
    assert score["metrics"]["predicted_count"] == 2
    assert score["metrics"]["gt_count"] == 3


def test_build_executor_uses_tool_writable_runtime_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.delenv("CODEX_AGENT_SANDBOX", raising=False)
    monkeypatch.delenv("CODEX_AGENT_SANDBOX_NETWORK", raising=False)

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert executor.config.sandbox == "workspace_write"
    assert executor.config.sandbox_network_access is True


def test_build_executor_preserves_full_access_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "full_access")
    monkeypatch.delenv("CODEX_AGENT_SANDBOX_NETWORK", raising=False)

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert executor.config.sandbox == "full_access"
    assert executor.config.sandbox_network_access is False


def test_build_executor_upgrades_read_only_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.runtime import CodexAgentRuntime

    config_path = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(config_path))
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "read_only")
    monkeypatch.setenv("CODEX_AGENT_SANDBOX_NETWORK", "false")

    executor = runner._build_executor()

    assert isinstance(executor, CodexAgentRuntime)
    assert executor.config.sandbox == "workspace_write"
    assert executor.config.sandbox_network_access is True


def test_run_single_sample_revalidates_executor_outcome(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    _write_outcome_artifacts(sample_output_dir)
    (sample_output_dir / "mask.npz").write_bytes(b"npz")
    outcome = SceneFunc3dMaskOutcome(
        mask_artifact_path=sample_output_dir / "mask_artifact.json",
        mask_npz_path=sample_output_dir / "mask.npz",
        mask_ply_path=sample_output_dir / "mask.ply",
        selected_frame_ids=("000010",),
        accepted_fragment_ids=("frag-a",),
        confidence=0.87,
        uncertainties=("partial occlusion",),
    )
    executor = _FakeExecutor(outcome)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(CodexResponseError, match="mask_npz_path"):
        run_single_sample(
            config,
            sample_id="421254::desc-a",
            executor=executor,
            check_sidecars=False,
        )


def _do_nothing() -> None:
    return None


class _FakeExecutor:
    def __init__(
        self,
        outcome: SceneFunc3dMaskOutcome,
        *,
        on_execute: Callable[[], None] = _do_nothing,
    ) -> None:
        self._outcome = outcome
        self._on_execute = on_execute
        self.task_name = ""
        self.prompt = ""

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        request = task.build_turn_request()
        self.task_name = task.task_name
        self.prompt = request.prompt
        self._on_execute()
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(ResultT, self._outcome),
            turn=CodexTurnResult(
                final_response=json.dumps(self._outcome.to_payload()),
                metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
            ),
        )


class _UnusedExecutor:
    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        _ = task
        raise AssertionError("test patched run_single_sample must not execute tasks")


@dataclass(frozen=True)
class _HealthServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _start_health_server(*, model_name: str) -> _HealthServer:
    return _start_health_server_with_state(model_name=model_name, model_loaded=True)


def _start_health_server_with_state(
    *,
    model_name: str,
    model_loaded: bool,
) -> _HealthServer:
    def health_route(_payload: JsonObject) -> JsonObject:
        return {
            "status": "ok",
            "model_name": model_name,
            "model_loaded": model_loaded,
        }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_json_handler({"/health": health_route})
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return _HealthServer(
        base_url=f"http://127.0.0.1:{port}",
        server=server,
        thread=thread,
    )


def _sample() -> SceneFunc3dSample:
    return SceneFunc3dSample(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the lower drawer.",
        annotation_ids=("annot-a",),
        motion_hints=(
            SceneFuncMotionHint(
                motion_id="motion-a",
                annotation_id="annot-a",
                motion_type="trans",
                motion_dir=(1.0, 0.0, 0.0),
            ),
        ),
    )


def _outcome_payload(
    root: Path,
    *,
    selected_frame_ids: tuple[str, ...] = ("000010",),
    accepted_fragment_ids: tuple[str, ...] = ("frag-a",),
) -> dict[str, object]:
    return {
        "mask_artifact_path": str(root / "mask_artifact.json"),
        "mask_npz_path": str(root / "mask.npz"),
        "mask_ply_path": str(root / "mask.ply"),
        "selected_frame_ids": list(selected_frame_ids),
        "accepted_fragment_ids": list(accepted_fragment_ids),
        "confidence": 0.87,
        "uncertainties": ["partial occlusion"],
    }


def _write_outcome_artifacts(
    root: Path,
    *,
    accepted_fragment_ids: tuple[str, ...] = ("frag-a",),
    accepted_frame_ids: tuple[str, ...] = ("000010",),
    rejected_suggested_frame_ids: tuple[str, ...] = (),
    include_fragment_lift_geometry: bool = True,
) -> None:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    root.mkdir(parents=True, exist_ok=True)
    points_world = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    point_indices = np.array([10, 12], dtype=np.int64)
    mask_npz_path = write_lift_npz(
        root / "mask.npz", points_world, point_indices=point_indices
    )
    mask_ply_path = write_lift_ply(root / "mask.ply", points_world)
    accepted_fragments: list[dict[str, object]] = []
    for fragment_id, frame_id in zip(
        accepted_fragment_ids, accepted_frame_ids, strict=True
    ):
        fragment_payload: dict[str, object] = {
            "fragment_id": fragment_id,
            "frame_id": frame_id,
            "point_count": int(points_world.shape[0]),
            "approval_actions": list(_APPROVED_FRAGMENT_ACTIONS),
            "review_artifacts": _write_review_artifacts(
                root / "review_artifacts" / fragment_id
            ),
        }
        if include_fragment_lift_geometry:
            fragment_payload["lift_geometry"] = _test_lift_geometry_payload()
        accepted_fragments.append(fragment_payload)
    unique_frame_ids = _unique_frame_ids(accepted_frame_ids)
    multi_view_action = "expand" if len(unique_frame_ids) > 1 else "stop"
    artifact_payload: dict[str, object] = {
        "accepted_frame_ids": list(accepted_frame_ids),
        "accepted_fragments": accepted_fragments,
        "multi_view_decision": {
            "seed_fragment_id": accepted_fragment_ids[0],
            "action": multi_view_action,
            "reason": "test artifact records the first-lift multi-view decision",
            "suggested_frame_ids": (
                list(unique_frame_ids[1:]) if multi_view_action == "expand" else []
            ),
            "rejected_suggested_frame_ids": list(rejected_suggested_frame_ids),
        },
        "mask_npz_path": str(mask_npz_path),
        "mask_ply_path": str(mask_ply_path),
    }
    (root / "mask_artifact.json").write_text(
        json.dumps(artifact_payload), encoding="utf-8"
    )


def _test_lift_geometry_payload() -> dict[str, object]:
    return {
        "bbox_min_xyz": [1.0, 2.0, 3.0],
        "bbox_max_xyz": [4.0, 5.0, 6.0],
        "bbox_extent_xyz": [3.0, 3.0, 3.0],
        "max_extent_meters": 3.0,
    }


def _write_suggest_additional_views_event(
    root: Path,
    *,
    expansion_recommendation: str,
    frame_ids: tuple[str, ...],
) -> None:
    event_payload = {
        "event_type": "tool_completed",
        "tool_name": "suggest_additional_views",
        "status": "success",
        "args": {
            "seed_fragment_id": "frag-a",
            "accepted_frame_id": "000010",
        },
        "result": {
            "seed_fragment_id": "frag-a",
            "seed_lift_point_count": 2,
            "seed_lift_status": "usable",
            "expansion_recommendation": expansion_recommendation,
            "expansion_reason": "test suggested views",
            "views": [
                {
                    "frame_id": frame_id,
                    "reason": "test follow-up view",
                    "rank": rank,
                    "has_depth": True,
                    "has_intrinsics": True,
                    "has_pose": True,
                    "view_diversity_score": 1.0,
                }
                for rank, frame_id in enumerate(frame_ids, start=1)
            ],
        },
        "error": "",
    }
    events_path = root / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


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


def _write_standard_review_artifacts(
    root: Path, *, frame_id: str, candidate_id: str
) -> dict[str, str]:
    fragment_id = f"{frame_id}_{candidate_id}"
    paths = {
        "molmo_raw_text_path": root / "molmo" / f"{frame_id}_raw.txt",
        "molmo_overlay_path": root / "molmo" / f"{frame_id}_points.jpg",
        "sam_contact_sheet_path": root / "sam" / frame_id / "contact_sheet.jpg",
        "sam_candidate_overlay_path": (
            root / "sam" / frame_id / f"{candidate_id}_overlay.jpg"
        ),
        "lift_overlay_path": root / "fragments" / fragment_id / "lift_overlay.txt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed\n", encoding="utf-8")
    _write_lift_overlay_summary(
        paths["lift_overlay_path"],
        frame_id=frame_id,
        candidate_id=candidate_id,
    )
    _write_fragment_points_artifact(root / "fragments" / fragment_id, point_count=2)
    return {key: str(path) for key, path in paths.items()}


def _write_fragment_points_artifact(root: Path, *, point_count: int) -> None:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    root.mkdir(parents=True, exist_ok=True)
    if point_count == 2:
        points_world = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    else:
        points_world = np.array(
            [
                [float(index), float(index + 1), float(index + 2)]
                for index in range(point_count)
            ],
            dtype=np.float64,
        )
    point_indices = np.arange(point_count, dtype=np.int64)
    write_lift_npz(root / "mask_data.npz", points_world, point_indices=point_indices)
    write_lift_ply(root / "lifted_points.ply", points_world)


def _write_lift_overlay_summary(
    path: Path, *, frame_id: str, candidate_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"frame_id={frame_id}\n"
        f"candidate_id={candidate_id}\n"
        "lifted_point_count=2\n",
        encoding="utf-8",
    )


def _unique_frame_ids(frame_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen_frame_ids: set[str] = set()
    unique_frame_ids: list[str] = []
    for frame_id in frame_ids:
        if frame_id not in seen_frame_ids:
            unique_frame_ids.append(frame_id)
            seen_frame_ids.add(frame_id)
    return tuple(unique_frame_ids)


def _write_backend_config(
    path: Path,
    *,
    molmo_url: str,
    sam_url: str,
    root: Path,
) -> None:
    path.write_text(
        "\n".join(
            (
                f"molmo_url = {json.dumps(molmo_url)}",
                f"sam_url = {json.dumps(sam_url)}",
                "request_timeout_seconds = 2.0",
                f"artifact_staging_root = {json.dumps(str(root / 'stage'))}",
                f"allowed_image_roots = [{json.dumps(str(root))}]",
                f"allowed_output_roots = [{json.dumps(str(root / 'out'))}]",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    _write_scene_root(scene_dir)
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


def _write_scene_root(
    scene_dir: Path, *, frame_ids: tuple[str, ...] = ("000010", "000020")
) -> Path:
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"fake-png")
    return scene_dir
