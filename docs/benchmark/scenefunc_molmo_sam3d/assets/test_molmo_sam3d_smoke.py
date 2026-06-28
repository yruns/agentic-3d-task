"""Tests for the SceneFunc Molmo/SAM/3D smoke asset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from molmo_sam3d_smoke import (
    MaskSummary,
    SmokeConfig,
    assess_smoke_success,
    extract_single_molmo_point,
    parse_manual_point,
    patch_molmo_remote_code,
    select_sam_candidate,
)


def test_extract_single_molmo_point_rejects_multiple_points() -> None:
    generated_text = '<point x="10" y="20">a</point><point x="30" y="40">b</point>'

    with pytest.raises(ValueError, match="exactly one valid point"):
        extract_single_molmo_point(
            generated_text,
            image_width=1000,
            image_height=500,
        )


def test_parse_manual_point_rejects_out_of_bounds_point() -> None:
    with pytest.raises(ValueError, match="outside image bounds"):
        parse_manual_point("640,480", image_width=640, image_height=480)


def test_smoke_config_rejects_non_positive_depth_scale() -> None:
    with pytest.raises(ValueError, match="depth_scale must be > 0"):
        SmokeConfig(
            dataset_root=Path("/tmp/dataset"),
            scene_id="421254",
            frame_id="000050",
            prompt="point to the target",
            output_dir=Path("/tmp/output"),
            molmo_model_id="allenai/Molmo-7B-D-0924",
            molmo_cache_dir=Path("/tmp/cache"),
            sam_checkpoint=Path("/tmp/sam.pth"),
            desc_id="case",
            depth_scale=0.0,
            max_new_tokens=120,
            max_lift_points=50000,
            manual_point="",
            sam_selection="smallest",
            max_success_coverage_percent=5.0,
        )


def test_select_sam_candidate_smallest_uses_smallest_non_empty_mask() -> None:
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, :, :] = True
    masks[1, 0, 0] = True
    scores = np.asarray([0.99, 0.1, 0.2], dtype=np.float32)

    selected = select_sam_candidate(masks, scores, selection="smallest")

    assert selected.score == pytest.approx(0.1)
    assert int(np.count_nonzero(selected.mask)) == 1


def test_assess_smoke_success_rejects_large_mask() -> None:
    summary = MaskSummary(
        mask_index=0,
        sam_score=0.99,
        pixel_count=10,
        image_coverage_percent=31.0,
        lifted_point_count=10,
        saved_point_count=10,
        overlay_path="overlay.jpg",
        mask_npz_path="mask.npz",
        ply_path="points.ply",
    )

    assessment = assess_smoke_success((summary,), max_success_coverage_percent=5.0)

    assert not assessment.success
    assert assessment.status == "mask_too_large"


def test_patch_molmo_remote_code_applies_idempotent_patches(tmp_path: Path) -> None:
    (tmp_path / "image_preprocessing_molmo.py").write_text(
        "\n".join(
            [
                '"""fake image processor"""',
                "import tensorflow as tf",
                "",
                "def resize_and_pad(resize_method):",
                '    if resize_method == "tensorflow":',
                "        return tf.image.resize",
                "    return None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "modeling_molmo.py").write_text(
        "\n".join(
            [
                "class MolmoForCausalLM:",
                '    _no_split_modules = ["MolmoBlock"]',
                "    def _update_model_kwargs_for_generation(self, model_kwargs, num_new_tokens):",
                '        if "cache_position" in model_kwargs:',
                '            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens',
                "        return model_kwargs",
                "    def tie_weights(self):",
                "        return None",
                "",
            ]
        ),
        encoding="utf-8",
    )

    first_reports = patch_molmo_remote_code(tmp_path)
    second_reports = patch_molmo_remote_code(tmp_path)

    assert any(report.changed for report in first_reports)
    assert not any(report.changed for report in second_reports)
    modeling_text = (tmp_path / "modeling_molmo.py").read_text(encoding="utf-8")
    image_text = (tmp_path / "image_preprocessing_molmo.py").read_text(encoding="utf-8")
    assert "all_tied_weights_keys = {}" in modeling_text
    assert "def tie_weights(self, *args, **kwargs):" in modeling_text
    assert 'model_kwargs.get("cache_position")' in modeling_text
    assert "import tensorflow as tf" not in image_text
    assert 'tf = import_module("tensorflow")' in image_text
