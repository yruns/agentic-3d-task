"""Perspective-from-above BEV rendering for a colored scene mesh.

The pipeline mirrors the production ScanNet BEV: filter mesh triangles to those
seen along the camera trajectory, drop ceiling faces, rasterize from a virtual
camera placed above the scene centroid (painter's algorithm), draw the camera
trajectory, crop to content, and overlay object markers / ``#id category``
labels projected through the *same* camera.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from keyframe.bev.config import DEFAULT_SCENE_BEV_CONFIG, RGB, SceneBEVConfig
from keyframe.bev.mesh import TriangleMesh


class BEVMarker(BaseModel):
    """A scene object to mark on the BEV, positioned in the render frame."""

    model_config = ConfigDict(frozen=True)

    obj_id: int
    category: str
    position: tuple[float, float, float]


class CameraView(BaseModel):
    """Virtual look-down camera shared by the mesh raster and label overlay."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    R: NDArray[np.float64]
    t: NDArray[np.float64]
    f: float
    c: float
    image_size: int
    crop_offset: tuple[int, int] = (0, 0)


class RenderedBEV(BaseModel):
    """A rendered RGB BEV image plus the camera used to draw it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: NDArray[np.uint8]
    view: CameraView


def render_scene_bev(
    mesh: TriangleMesh,
    camera_poses: NDArray[np.float64],
    intrinsic: NDArray[np.float64],
    markers: Sequence[BEVMarker] = (),
    *,
    highlight_ids: frozenset[int] = frozenset(),
    config: SceneBEVConfig = DEFAULT_SCENE_BEV_CONFIG,
) -> RenderedBEV:
    """Render the scene BEV. ``camera_poses`` are ``(N, 4, 4)`` cam-to-world."""
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
        raise ValueError(f"camera_poses must be (N, 4, 4), got {camera_poses.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be 3x3, got {intrinsic.shape}")

    visible = _frustum_visibility(mesh.vertices, camera_poses, intrinsic, config)
    kept = _filter_triangles(mesh, visible, config)

    view = _look_down_camera(mesh.vertices, kept, config)
    image = _rasterize(mesh, kept, view, config)
    image = _draw_trajectory(image, camera_poses[:, :3, 3], view, config)
    image, offset = _crop_to_content(image, config)
    view = view.model_copy(update={"crop_offset": offset})
    image = _overlay_markers(image, markers, view, highlight_ids, config)
    return RenderedBEV(image=image, view=view)


def _frustum_visibility(
    vertices: NDArray[np.float64],
    camera_poses: NDArray[np.float64],
    intrinsic: NDArray[np.float64],
    config: SceneBEVConfig,
) -> NDArray[np.bool_]:
    """Per-vertex visibility across sampled camera frustums."""
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    margin = config.frustum_margin_px
    width, height = config.source_image_width, config.source_image_height

    visible = np.zeros(vertices.shape[0], dtype=np.bool_)
    for i in range(0, camera_poses.shape[0], config.frustum_stride):
        pose = camera_poses[i]
        if not np.isfinite(pose).all():
            continue
        world_to_cam = np.linalg.inv(pose)
        cam = (world_to_cam[:3, :3] @ vertices.T).T + world_to_cam[:3, 3]
        z = cam[:, 2]
        safe_z = np.maximum(z, 0.01)
        u = fx * cam[:, 0] / safe_z + cx
        v = fy * cam[:, 1] / safe_z + cy
        visible |= (
            (z > 0.1)
            & (z < config.frustum_max_depth)
            & (u >= -margin)
            & (u < width + margin)
            & (v >= -margin)
            & (v < height + margin)
        )
    return visible


def _filter_triangles(
    mesh: TriangleMesh, visible: NDArray[np.bool_], config: SceneBEVConfig
) -> NDArray[np.int64]:
    """Keep frustum-visible, non-ceiling triangles (fallback: all visible)."""
    tris = mesh.triangles
    tri_visible = visible[tris[:, 0]] & visible[tris[:, 1]] & visible[tris[:, 2]]
    facing_down = mesh.triangle_normals()[:, 2] < config.ceiling_normal_threshold
    kept = tris[tri_visible & ~facing_down]
    if kept.shape[0] == 0:
        kept = tris[tri_visible]
    if kept.shape[0] == 0:
        kept = tris
    return kept


def _look_down_camera(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    config: SceneBEVConfig,
) -> CameraView:
    """Build a virtual camera above the (visible) scene centroid, looking down."""
    used = np.unique(triangles.ravel())
    visible_verts = vertices[used]
    bmin = visible_verts.min(axis=0)
    bmax = visible_verts.max(axis=0)
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin

    max_xy = float(max(extent[0], extent[1]))
    eye = np.array(
        [center[0], center[1], bmax[2] + max_xy * config.camera_height_factor],
        dtype=np.float64,
    )
    R = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )
    t = -R @ eye
    size = config.image_size
    f = size / (2.0 * np.tan(np.radians(config.fov_deg) / 2.0))
    return CameraView(R=R, t=t, f=float(f), c=size / 2.0, image_size=size)


def _rasterize(
    mesh: TriangleMesh,
    triangles: NDArray[np.int64],
    view: CameraView,
    config: SceneBEVConfig,
) -> NDArray[np.uint8]:
    """Painter's-algorithm rasterization of triangles into an RGB image."""
    size = config.image_size
    cam = (view.R @ mesh.vertices.T).T + view.t
    z = np.clip(cam[:, 2], 0.01, None)
    x_img = view.f * cam[:, 0] / z + view.c
    y_img = view.f * cam[:, 1] / z + view.c

    depths = cam[triangles][:, :, 2].mean(axis=1)
    order = np.argsort(depths)

    tri_x = x_img[triangles].astype(np.int32)
    tri_y = y_img[triangles].astype(np.int32)
    tri_pts = np.stack([tri_x, tri_y], axis=-1)
    tri_bgr = (mesh.colors[triangles].mean(axis=1) * 255.0).astype(np.uint8)[:, ::-1]

    canvas = np.full((size, size, 3), 0, dtype=np.uint8)
    canvas[:] = config.background_color[::-1]
    for j in order:
        pts = tri_pts[j]
        if np.any(pts < -500) or np.any(pts > size + 500):
            continue
        cv2.fillPoly(
            canvas, [pts], (int(tri_bgr[j][0]), int(tri_bgr[j][1]), int(tri_bgr[j][2]))
        )
    rgb: NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )
    return rgb


def _draw_trajectory(
    image: NDArray[np.uint8],
    positions: NDArray[np.float64],
    view: CameraView,
    config: SceneBEVConfig,
) -> NDArray[np.uint8]:
    """Draw the camera path as a connected polyline on a copy of ``image``."""
    out: NDArray[np.uint8] = image.copy()
    color: RGB = config.trajectory_color
    for i in range(positions.shape[0] - 1):
        p1 = _project_world(positions[i], view, apply_crop=False)
        p2 = _project_world(positions[i + 1], view, apply_crop=False)
        if p1 is None or p2 is None:
            continue
        cv2.line(out, p1, p2, color, config.trajectory_thickness, cv2.LINE_AA)
    return out


def _crop_to_content(
    image: NDArray[np.uint8], config: SceneBEVConfig
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    """Crop to the non-background bounding box plus a margin."""
    threshold = config.crop_white_threshold
    non_bg = np.any(image < threshold, axis=2)
    if not non_bg.any():
        return image, (0, 0)
    rows = np.where(non_bg.any(axis=1))[0]
    cols = np.where(non_bg.any(axis=0))[0]
    margin = config.crop_margin
    y0 = max(0, int(rows.min()) - margin)
    y1 = min(image.shape[0], int(rows.max()) + 1 + margin)
    x0 = max(0, int(cols.min()) - margin)
    x1 = min(image.shape[1], int(cols.max()) + 1 + margin)
    return image[y0:y1, x0:x1], (x0, y0)


def _project_world(
    position: NDArray[np.float64], view: CameraView, *, apply_crop: bool
) -> tuple[int, int] | None:
    """Project a 3D world point to (u, v); ``None`` if behind the camera."""
    cam = view.R @ position + view.t
    if cam[2] < 0.01:
        return None
    u = view.f * cam[0] / cam[2] + view.c
    v = view.f * cam[1] / cam[2] + view.c
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    bound = view.image_size + 10
    if u < -10 or u > bound or v < -10 or v > bound:
        return None
    if apply_crop:
        return int(u) - view.crop_offset[0], int(v) - view.crop_offset[1]
    return int(u), int(v)


def _overlay_markers(
    image: NDArray[np.uint8],
    markers: Sequence[BEVMarker],
    view: CameraView,
    highlight_ids: frozenset[int],
    config: SceneBEVConfig,
) -> NDArray[np.uint8]:
    """Draw per-object dots and (optionally) ``#id category`` text labels."""
    out: NDArray[np.uint8] = image.copy()
    placed: list[tuple[BEVMarker, int, int]] = []
    for marker in markers:
        position = np.asarray(marker.position, dtype=np.float64)
        uv = _project_world(position, view, apply_crop=True)
        if uv is None:
            continue
        highlighted = marker.obj_id in highlight_ids
        radius = config.marker_radius_highlight if highlighted else config.marker_radius
        dot: RGB = (
            config.label_color_highlight if highlighted else config.label_color_default
        )
        cv2.circle(out, uv, radius, dot, -1)
        placed.append((marker, uv[0], uv[1]))

    if not config.label_objects or not placed:
        return out

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = config.label_font_scale
    thickness = config.label_font_thickness
    longest = max((m.category for m, _, _ in placed), key=len)
    (ref_w, ref_h), _ = cv2.getTextSize(longest, font, scale, thickness)
    anchors = _stack_overlapping_anchors(
        [(u, v) for _, u, v in placed],
        ref_w=ref_w,
        ref_h=ref_h,
        gap_px=config.label_declutter_gap_px,
    )
    for (marker, _u, _v), (ux, vy) in zip(placed, anchors):
        highlighted = marker.obj_id in highlight_ids
        _draw_label(out, marker, (ux, vy), highlighted, font, config)
    return out


def _draw_label(
    image: NDArray[np.uint8],
    marker: BEVMarker,
    anchor: tuple[int, int],
    highlighted: bool,
    font: int,
    config: SceneBEVConfig,
) -> None:
    """Draw one ``#id category`` label, kept inside the image bounds.

    The label sits to the right of its dot, flipping to the left when it would
    overflow the right edge, and is clamped vertically so it stays readable.
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    scale = config.label_font_scale
    thickness = config.label_font_thickness
    outline = thickness + config.label_outline_extra_thickness
    label = f"#{marker.obj_id} {marker.category}"
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    anchor_x, anchor_y = anchor
    if anchor_x + 6 + tw + 4 > width:
        org_x = anchor_x - 6 - tw
    else:
        org_x = anchor_x + 6
    org_x = max(4, min(org_x, width - tw - 4))
    org_y = max(th + 4, min(anchor_y - 6, height - baseline - 4))
    org = (org_x, org_y)
    bg: RGB = config.label_bg_highlight if highlighted else config.label_bg_default
    cv2.rectangle(
        image,
        (org[0] - 4, org[1] - th - 4),
        (org[0] + tw + 4, org[1] + baseline + 4),
        bg,
        -1,
    )
    cv2.putText(image, label, org, font, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image, label, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _stack_overlapping_anchors(
    anchors: list[tuple[int, int]], *, ref_w: int, ref_h: int, gap_px: int
) -> list[tuple[int, int]]:
    """Nudge overlapping label anchors downward until separable."""
    out: list[tuple[int, int]] = []
    for u, v in anchors:
        offset = 0
        while any(
            abs(u - ou) <= ref_w and abs(v + offset - ov) <= ref_h for ou, ov in out
        ):
            offset += ref_h + gap_px
        out.append((u, v + offset))
    return out
