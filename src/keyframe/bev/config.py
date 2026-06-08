"""Static configuration for scene BEV rendering."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

RGB = tuple[int, int, int]

#: Environment variable that adds an extra root to search for ScanNet meshes.
SCANNET_DATA_ROOT_ENV_VAR = "SCANNET_DATA_ROOT"


def scannet_data_root_override() -> Path | None:
    """Return the ScanNet mesh root from ``$SCANNET_DATA_ROOT``, if set.

    Centralizes the only environment-variable read in the BEV package so mesh
    resolution code does not read ``os.environ`` directly.
    """
    override = os.environ.get(SCANNET_DATA_ROOT_ENV_VAR)
    return Path(override) if override else None


class SceneBEVConfig(BaseModel):
    """Immutable rendering parameters for a top-down scene BEV.

    The defaults reproduce the ScanNet perspective-from-above look used for
    keyframe-parser visual context: a wide-FOV virtual camera placed above the
    scene centroid, ceiling triangles removed, and the camera trajectory drawn
    on top. Object markers / labels are overlaid afterwards.
    """

    model_config = ConfigDict(frozen=True)

    image_size: int = 1500
    fov_deg: float = 80.0
    camera_height_factor: float = 0.7

    # Per-frame frustum visibility filtering (drops geometry never seen).
    frustum_stride: int = 10
    frustum_margin_px: int = 50
    frustum_max_depth: float = 10.0
    source_image_width: int = 1296
    source_image_height: int = 968

    # Ceiling removal: triangles whose (aligned) normal points down past this.
    ceiling_normal_threshold: float = -0.5

    background_color: RGB = (245, 245, 245)
    trajectory_color: RGB = (59, 130, 246)
    trajectory_thickness: int = 3

    # Crop the rendered mesh tightly to its non-background bounding box.
    crop_margin: int = 3
    crop_white_threshold: int = 245

    # Object overlay.
    label_objects: bool = True
    marker_radius: int = 4
    marker_radius_highlight: int = 7
    label_color_default: RGB = (32, 32, 32)
    label_color_highlight: RGB = (255, 64, 64)
    label_bg_default: RGB = (255, 255, 255)
    label_bg_highlight: RGB = (255, 255, 0)
    label_font_scale: float = 1.0
    label_font_thickness: int = 3
    label_outline_extra_thickness: int = 3
    label_declutter_gap_px: int = 6


#: Shared immutable default; safe to use as a function-argument default.
DEFAULT_SCENE_BEV_CONFIG = SceneBEVConfig()
