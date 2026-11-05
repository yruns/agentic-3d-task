"""Strict scene TOML boundaries with local placeholder assets, never model loads."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.alfred.perception import ScenePerceptionConfig
from codex_agent.alfred.settings import load_scene_settings


def local_scene_config(root: Path) -> ScenePerceptionConfig:
    """Build local model-file fixtures for external model-boundary tests."""
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("dino_config", "dino_checkpoint", "sam_checkpoint", "clip_checkpoint"):
        path = root / name
        path.write_bytes(("fixture-" + name).encode())
        files[name] = path
    encoder = root / "text_encoder"
    encoder.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
    ):
        (encoder / name).write_bytes(b"local-encoder-fixture")
    depth = root / "depth"
    depth.mkdir()
    (depth / "config.json").write_text(json.dumps({"model_name": "da3metric-large"}))
    (depth / "model.safetensors").write_bytes(b"local-depth-fixture")
    return ScenePerceptionConfig(
        python_executable=Path(sys.executable),
        dino_config=files["dino_config"],
        dino_checkpoint=files["dino_checkpoint"],
        dino_text_encoder_dir=encoder,
        sam_checkpoint=files["sam_checkpoint"],
        clip_checkpoint=files["clip_checkpoint"],
        depth_model_dir=depth,
        gpu_lock_path=root / "gpu.lock",
        device="cpu",
    )


def write_scene_settings(
    path: Path,
    config: ScenePerceptionConfig,
    *,
    extra: str = "",
    replacements: dict[str, object] | None = None,
) -> Path:
    """Serialize fixture primitives independently of the production parser."""
    values: dict[str, object] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    values.update(replacements or {})
    path.write_text(
        "\n".join(
            [
                "[perception]",
                *(f"{key} = {json.dumps(value)}" for key, value in values.items()),
                extra,
            ]
        )
    )
    return path


def test_scene_settings_load_typed_nested_configuration(tmp_path: Path) -> None:
    config = local_scene_config(tmp_path / "assets")
    path = write_scene_settings(
        tmp_path / "scene.toml",
        config,
        extra="""
[memory]
max_depth_m = 6.0
[memory.mapping]
association_threshold = 1.4
confirmation_observations = 2
[spatial]
region_size_m = 1.0
robot_radius_m = 0.22
""",
    )
    settings = load_scene_settings(path)
    assert settings.perception == config
    assert settings.memory.max_depth_m == 6.0
    assert settings.memory.mapping.association_threshold == 1.4
    assert settings.memory.mapping.confirmation_observations == 2
    assert settings.spatial.region_size_m == 1.0
    assert settings.spatial.robot_radius_m == 0.22


@pytest.mark.parametrize(
    "extra",
    [
        "[unexpected]\nvalue = 1",
        "[memory]\nmisspelled = 1",
        "[memory.mapping]\nunknown_threshold = 0.5",
        "[spatial]\nlegacy_cell_size = 0.25",
    ],
)
def test_scene_settings_reject_unknown_sections_or_fields(
    tmp_path: Path, extra: str
) -> None:
    path = write_scene_settings(
        tmp_path / "scene.toml", local_scene_config(tmp_path / "assets"), extra=extra
    )
    with pytest.raises((ValueError, ValidationError), match="unknown|unexpected"):
        load_scene_settings(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("caption_batch_size", True),
        ("max_detections", "100"),
        ("timeout_seconds", False),
        ("box_threshold", "0.2"),
        ("physical_gpu_index", True),
        ("physical_gpu_index", "4"),
    ],
)
def test_perception_numbers_are_strict(
    tmp_path: Path, field: str, value: object
) -> None:
    path = write_scene_settings(
        tmp_path / "scene.toml",
        local_scene_config(tmp_path / "assets"),
        replacements={field: value},
    )
    with pytest.raises(ValidationError):
        load_scene_settings(path)


@pytest.mark.parametrize(
    "extra",
    [
        "[memory.mapping]\nconfirmation_observations = true",
        "[spatial]\nrobot_radius_m = false",
        "[memory]\nmax_object_views = 3.5",
    ],
)
def test_mapping_and_spatial_numbers_reject_invalid_types(
    tmp_path: Path, extra: str
) -> None:
    path = write_scene_settings(
        tmp_path / "scene.toml", local_scene_config(tmp_path / "assets"), extra=extra
    )
    with pytest.raises(ValidationError):
        load_scene_settings(path)


@pytest.mark.parametrize(
    "path_value", ["weights/model.pth", "$MODEL_ROOT/model.pth", "~/models/checkpoint"]
)
def test_asset_paths_must_be_explicit_absolute_paths(
    tmp_path: Path, path_value: str
) -> None:
    path = write_scene_settings(
        tmp_path / "scene.toml",
        local_scene_config(tmp_path / "assets"),
        replacements={"dino_checkpoint": path_value},
    )
    with pytest.raises(ValueError, match="absolute local path"):
        load_scene_settings(path)


def test_missing_model_file_is_not_replaced_or_downloaded(tmp_path: Path) -> None:
    config = local_scene_config(tmp_path / "assets")
    path = write_scene_settings(tmp_path / "scene.toml", config)
    config.sam_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="sam_checkpoint"):
        load_scene_settings(path)


def test_settings_require_perception_and_table_sections(tmp_path: Path) -> None:
    path = tmp_path / "scene.toml"
    path.write_text("[memory]\nmax_depth_m = 8.0\n")
    with pytest.raises(ValueError, match="require.*perception"):
        load_scene_settings(path)
    path.write_text("perception = 'not a table'\n")
    with pytest.raises(ValueError, match="must be a table"):
        load_scene_settings(path)
