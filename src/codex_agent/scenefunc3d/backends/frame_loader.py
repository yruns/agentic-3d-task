"""Lazy per-frame camera/depth/RGB access for a prepared SceneFunc3D scene.

Depth is read one frame at a time (never all frames at once): a single depth
array is ~22 MB at 1440x1920, so materialising all 170 frames would need
~3.7 GB. Callers stream frames and discard depth after use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.camera_io import (
    load_camera_geometry,
    read_depth_meters,
)
from codex_agent.scenefunc3d.backends.frame_assets import (
    resolve_available_frame_geometry_assets,
    resolve_frame_geometry_assets,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

_RGB_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def iter_frame_geometry(
    scene: SceneFunc3dToolScene,
) -> Iterator[tuple[str, CameraGeometry]]:
    """Yield ``(frame_id, CameraGeometry)`` for every frame with full geometry.

    Frames missing any of depth/intrinsics/pose are skipped. Fails closed when
    no frame in the scene has complete geometry, so downstream stages never
    silently see an empty frame set.
    """
    yielded = 0
    for frame_id in scene.rgb_frame_ids:
        assets = resolve_available_frame_geometry_assets(scene, frame_id)
        if (
            assets.depth_path is None
            or assets.intrinsics_path is None
            or assets.pose_path is None
        ):
            continue
        geometry = load_camera_geometry(
            intrinsics_path=assets.intrinsics_path,
            pose_path=assets.pose_path,
        )
        yielded += 1
        yield frame_id, geometry
    if yielded == 0:
        raise SceneFunc3dDataError(
            "no SceneFunc3D frame has complete depth/intrinsics/pose geometry: "
            f"scene_root={scene.scene_root}"
        )


def load_frame_geometry(scene: SceneFunc3dToolScene, frame_id: str) -> CameraGeometry:
    """Load one frame's validated camera geometry."""
    assets = resolve_frame_geometry_assets(scene, frame_id)
    return load_camera_geometry(
        intrinsics_path=assets.intrinsics_path,
        pose_path=assets.pose_path,
    )


def read_frame_depth(scene: SceneFunc3dToolScene, frame_id: str) -> FloatArray:
    """Read one frame's depth image as float64 metres."""
    assets = resolve_frame_geometry_assets(scene, frame_id)
    return read_depth_meters(assets.depth_path)


def frame_rgb_path(scene: SceneFunc3dToolScene, frame_id: str) -> Path:
    """Resolve one frame's RGB image path, failing closed when absent."""
    indexed = scene.source_frame_raw_rgb_path(frame_id)
    if indexed is not None and indexed.is_file():
        return indexed
    for suffix in _RGB_SUFFIXES:
        candidate = scene.raw_dir / f"{frame_id}-rgb{suffix}"
        if candidate.is_file():
            return candidate
    raise ToolInputError(
        "frame RGB image is missing: " f"frame_id={frame_id!r}; raw_dir={scene.raw_dir}"
    )


__all__ = [
    "frame_rgb_path",
    "iter_frame_geometry",
    "load_frame_geometry",
    "read_frame_depth",
]
