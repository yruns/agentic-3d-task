"""Bird's-eye-view rendering of prepared 3D scenes for visual parsing context.

``render_scene_bev`` rasterizes a colored scene mesh from above and overlays the
camera trajectory and object markers. ``SceneBEVBuilder`` / ``Nr3dSceneBEVBuilder``
resolve per-benchmark assets, handle the raw->aligned mesh transform, and cache
the rendered PNG.
"""

from __future__ import annotations

from keyframe.bev.builder import (
    BEVScenePaths,
    Nr3dSceneBEVBuilder,
    SceneBEVBuilder,
)
from keyframe.bev.config import SceneBEVConfig
from keyframe.bev.mesh import TriangleMesh, load_ply_mesh
from keyframe.bev.render import (
    BEVMarker,
    CameraView,
    RenderedBEV,
    render_scene_bev,
)

__all__ = [
    "SceneBEVConfig",
    "TriangleMesh",
    "load_ply_mesh",
    "BEVMarker",
    "CameraView",
    "RenderedBEV",
    "render_scene_bev",
    "BEVScenePaths",
    "SceneBEVBuilder",
    "Nr3dSceneBEVBuilder",
]
