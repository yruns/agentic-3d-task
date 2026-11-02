"""Tests for the SceneFunc3D sidecar smoke driver contract."""

from __future__ import annotations

from pathlib import Path


def test_sidecar_smoke_runs_multiview_suggestion_before_fusion() -> None:
    script_text = _sidecar_smoke_script_path().read_text(encoding="utf-8")

    suggest_index = script_text.index('"suggest_additional_views"')
    fuse_index = script_text.index('"fuse_accepted_masks"')

    assert suggest_index < fuse_index
    assert 'run_root / "suggest_additional_views.json"' in script_text
    assert '"seed_mask_npz_path": lift_payload["mask_npz_path"]' in script_text
    assert '"seed_mask_ply_path": lift_payload["mask_ply_path"]' in script_text
    assert '"seed_lift_overlay_path": lift_payload["overlay_path"]' in script_text
    assert '"action": suggest_payload["expansion_recommendation"]' in script_text
    assert '"reason": suggest_payload["expansion_reason"]' in script_text


def test_sidecar_smoke_uses_lift_fragment_id_for_seed_provenance() -> None:
    script_text = _sidecar_smoke_script_path().read_text(encoding="utf-8")

    assert "fragment_id = f\"000050_{candidate['candidate_id']}\"" in script_text
    assert '"fragment_id": fragment_id' in script_text
    assert '"seed_fragment_id": fragment_id' in script_text


def test_sidecar_smoke_allows_generated_crop_artifacts_as_inputs() -> None:
    script_text = _sidecar_smoke_script_path().read_text(encoding="utf-8")

    assert (
        'f\'allowed_image_roots = ["{dataset_root.parent}", "{run_root}"]\''
        in script_text
    )


def test_sidecar_smoke_defaults_outputs_to_repo_tmp() -> None:
    script_text = _sidecar_smoke_script_path().read_text(encoding="utf-8")

    assert (
        'RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/scenefunc3d/sidecar_tool_smoke_20260629}"'
        in script_text
    )
    assert (
        'RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/sidecar_tool_smoke_20260629}"'
        not in (script_text)
    )


def _sidecar_smoke_script_path() -> Path:
    return (
        Path.cwd()
        / "docs"
        / "benchmark"
        / "scenefunc_molmo_sam3d"
        / "assets"
        / "run_sidecar_tool_smoke_20260629.sh"
    )
