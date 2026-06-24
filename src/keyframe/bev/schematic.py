"""Mesh-free, top-down *schematic* BEV for scenes without a textured mesh.

The perspective mesh renderer (``keyframe.bev.render.render_scene_bev``) needs a
colored ``mesh.ply``. OpenEQA ScanNet clips ship a prepared ConceptGraph pack
(objects + camera trajectory) but **no** mesh, so this module draws a top-down
*floor-plan* instead: each object becomes an axis-aligned footprint rectangle
with a ``#id category`` label, the camera path is drawn on top, and the whole
thing is laid out with a simple orthographic (metric) transform.

Inputs come straight from the already-loaded lightweight ConceptGraph objects
(centroid + 3D bbox extent) and the camera trajectory, so nothing heavier than
NumPy + OpenCV is required. The transform is returned as an
:class:`OrthoBevView` so callers can project further world points onto the same
image later (parity with the mesh renderer's ``CameraView``).

Requires the ``vision`` extra (OpenCV + NumPy); imported lazily by callers.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from keyframe.bev.config import RGB
from keyframe.bev.render import BEVMarker

#: Smallest footprint half-extent (meters) drawn for an object, so a marker with
#: a near-zero or missing extent is still visible as a small square.
_MIN_HALF_EXTENT_M = 0.12


class SchematicBevConfig(BaseModel):
    """Immutable rendering parameters for the mesh-free schematic BEV."""

    model_config = ConfigDict(frozen=True)

    #: Longest side budget (pixels) for the rendered map; chosen to match the
    #: agent ``view_image`` budget so the BEV needs no extra downscale.
    image_size: int = 768
    margin_px: int = 32
    #: Metric scale is clamped into this band (pixels per meter) so tiny scenes
    #: are not blown up and large ones stay within ``image_size``.
    min_pixels_per_meter: float = 14.0
    max_pixels_per_meter: float = 110.0

    background_color: RGB = (245, 245, 245)
    grid: bool = True
    grid_color: RGB = (228, 228, 228)
    grid_step_m: float = 1.0

    footprint_fill: RGB = (203, 219, 240)
    footprint_outline: RGB = (60, 96, 150)
    footprint_fill_highlight: RGB = (255, 233, 120)
    footprint_outline_highlight: RGB = (220, 48, 48)
    marker_radius: int = 3

    trajectory_color: RGB = (59, 130, 246)
    trajectory_thickness: int = 2
    traj_start_color: RGB = (34, 197, 94)
    traj_end_color: RGB = (239, 68, 68)
    traj_endpoint_radius: int = 5

    label_objects: bool = True
    label_font_scale: float = 0.5
    label_font_thickness: int = 1
    label_declutter_gap_px: int = 4
    #: Cap on labeled objects (largest footprints win) to avoid an unreadable map.
    max_labeled_objects: int = 70


#: Shared immutable default; safe to use as a function-argument default.
DEFAULT_SCHEMATIC_BEV_CONFIG = SchematicBevConfig()


class OrthoBevView(BaseModel):
    """Orthographic metric transform from world ``(x, y)`` to BEV pixels.

    Pixels increase right (``x``) and *down* (``-y``), so the map reads like a
    floor plan with world ``+y`` pointing up. Serializable so a later tool call
    can project the same world points onto the cached render.
    """

    model_config = ConfigDict(frozen=True)

    min_x: float
    max_y: float
    scale: float
    width: int
    height: int

    def project(self, x: float, y: float) -> tuple[int, int]:
        """Project a world ``(x, y)`` point to integer ``(px, py)`` pixels."""
        px = int(round((x - self.min_x) * self.scale))
        py = int(round((self.max_y - y) * self.scale))
        return px, py

    def to_payload(self) -> dict[str, float | int]:
        """Return a JSON-ready mapping of the transform parameters."""
        return {
            "min_x": self.min_x,
            "max_y": self.max_y,
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
        }


def render_schematic_bev(
    markers: Sequence[BEVMarker],
    camera_xy: NDArray[np.float64],
    *,
    highlight_ids: frozenset[int] = frozenset(),
    config: SchematicBevConfig = DEFAULT_SCHEMATIC_BEV_CONFIG,
) -> tuple[NDArray[np.uint8], OrthoBevView]:
    """Render a top-down schematic BEV from object footprints + camera path.

    Args:
        markers: Objects to draw; ``extent`` (when present) sizes the footprint.
        camera_xy: ``(N, 2)`` world XY camera positions (may be empty).
        highlight_ids: Object ids drawn in the highlight color.
        config: Rendering parameters.

    Returns:
        ``(image_rgb, view)`` — the rendered ``H×W×3`` uint8 RGB image and the
        orthographic transform used to draw it.

    Raises:
        ValueError: If there is no geometry to render (no markers and no poses).
    """
    footprints = [_footprint(marker) for marker in markers]
    view = _build_view(footprints, camera_xy, config)
    canvas = np.empty((view.height, view.width, 3), dtype=np.uint8)
    canvas[:] = _bgr(config.background_color)
    if config.grid:
        _draw_grid(canvas, view, config)
    _draw_trajectory(canvas, camera_xy, view, config)
    placed = _draw_footprints(canvas, markers, footprints, view, highlight_ids, config)
    if config.label_objects:
        _draw_labels(canvas, placed, highlight_ids, config)
    rgb: NDArray[np.uint8] = np.asarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )
    return rgb, view


def _bgr(color: RGB) -> tuple[int, int, int]:
    r, g, b = color
    return (b, g, r)


def _footprint(marker: BEVMarker) -> tuple[float, float, float, float]:
    """Return the world-space ``(xmin, ymin, xmax, ymax)`` footprint of a marker."""
    cx, cy = float(marker.position[0]), float(marker.position[1])
    if marker.extent is not None:
        hx = max(_MIN_HALF_EXTENT_M, abs(float(marker.extent[0])) / 2.0)
        hy = max(_MIN_HALF_EXTENT_M, abs(float(marker.extent[1])) / 2.0)
    else:
        hx = hy = _MIN_HALF_EXTENT_M
    return cx - hx, cy - hy, cx + hx, cy + hy


def _build_view(
    footprints: Sequence[tuple[float, float, float, float]],
    camera_xy: NDArray[np.float64],
    config: SchematicBevConfig,
) -> OrthoBevView:
    xs: list[float] = []
    ys: list[float] = []
    for xmin, ymin, xmax, ymax in footprints:
        xs.extend((xmin, xmax))
        ys.extend((ymin, ymax))
    if camera_xy.size:
        xs.extend(float(v) for v in camera_xy[:, 0])
        ys.extend(float(v) for v in camera_xy[:, 1])
    if not xs or not ys:
        raise ValueError("schematic BEV needs at least one object or camera pose")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    usable = max(config.image_size - 2 * config.margin_px, 1)
    scale = usable / max(span_x, span_y)
    scale = min(max(scale, config.min_pixels_per_meter), config.max_pixels_per_meter)

    margin = config.margin_px
    width = int(round(span_x * scale)) + 2 * margin
    height = int(round(span_y * scale)) + 2 * margin
    return OrthoBevView(
        min_x=min_x - margin / scale,
        max_y=max_y + margin / scale,
        scale=scale,
        width=max(width, 2 * margin + 1),
        height=max(height, 2 * margin + 1),
    )


def _draw_grid(
    canvas: NDArray[np.uint8], view: OrthoBevView, config: SchematicBevConfig
) -> None:
    color = _bgr(config.grid_color)
    step = max(config.grid_step_m, 1e-3)
    x_start = np.floor(view.min_x / step) * step
    x_world = x_start
    while True:
        px, _ = view.project(x_world, view.max_y)
        if px > view.width:
            break
        if 0 <= px < view.width:
            cv2.line(canvas, (px, 0), (px, view.height - 1), color, 1, cv2.LINE_AA)
        x_world += step
    min_y = view.max_y - view.height / view.scale
    y_start = np.floor(min_y / step) * step
    y_world = y_start
    while y_world <= view.max_y:
        _, py = view.project(view.min_x, y_world)
        if 0 <= py < view.height:
            cv2.line(canvas, (0, py), (view.width - 1, py), color, 1, cv2.LINE_AA)
        y_world += step


def _draw_trajectory(
    canvas: NDArray[np.uint8],
    camera_xy: NDArray[np.float64],
    view: OrthoBevView,
    config: SchematicBevConfig,
) -> None:
    if not camera_xy.size:
        return
    pixels = [view.project(float(x), float(y)) for x, y in camera_xy]
    traj_color = _bgr(config.trajectory_color)
    for start, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(
            canvas, start, end, traj_color, config.trajectory_thickness, cv2.LINE_AA
        )
    cv2.circle(
        canvas,
        pixels[0],
        config.traj_endpoint_radius,
        _bgr(config.traj_start_color),
        -1,
    )
    cv2.circle(
        canvas, pixels[-1], config.traj_endpoint_radius, _bgr(config.traj_end_color), -1
    )


def _draw_footprints(
    canvas: NDArray[np.uint8],
    markers: Sequence[BEVMarker],
    footprints: Sequence[tuple[float, float, float, float]],
    view: OrthoBevView,
    highlight_ids: frozenset[int],
    config: SchematicBevConfig,
) -> list[tuple[BEVMarker, int, int]]:
    """Draw object footprints; return ``(marker, anchor_x, anchor_y)`` for labels."""
    placed: list[tuple[BEVMarker, int, int]] = []
    for marker, (xmin, ymin, xmax, ymax) in zip(markers, footprints):
        highlighted = marker.obj_id in highlight_ids
        fill = _bgr(
            config.footprint_fill_highlight if highlighted else config.footprint_fill
        )
        outline = _bgr(
            config.footprint_outline_highlight
            if highlighted
            else config.footprint_outline
        )
        top_left = view.project(xmin, ymax)
        bottom_right = view.project(xmax, ymin)
        cv2.rectangle(canvas, top_left, bottom_right, fill, -1)
        cv2.rectangle(canvas, top_left, bottom_right, outline, 1, cv2.LINE_AA)
        center = view.project(float(marker.position[0]), float(marker.position[1]))
        cv2.circle(canvas, center, config.marker_radius, outline, -1)
        placed.append((marker, top_left[0], top_left[1]))
    return placed


def _draw_labels(
    canvas: NDArray[np.uint8],
    placed: Sequence[tuple[BEVMarker, int, int]],
    highlight_ids: frozenset[int],
    config: SchematicBevConfig,
) -> None:
    if not placed:
        return
    ranked = sorted(placed, key=_label_priority, reverse=True)
    labeled = ranked[: config.max_labeled_objects]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = config.label_font_scale
    thickness = config.label_font_thickness
    longest = max((m.category for m, _, _ in labeled), key=len, default="")
    (ref_w, ref_h), _ = cv2.getTextSize(longest or "x", font, scale, thickness)
    anchors = _stack_overlapping_anchors(
        [(x, y) for _, x, y in labeled],
        ref_w=ref_w,
        ref_h=ref_h,
        gap_px=config.label_declutter_gap_px,
    )
    for (marker, _x, _y), (ax, ay) in zip(labeled, anchors):
        highlighted = marker.obj_id in highlight_ids
        _draw_label(canvas, marker, (ax, ay), highlighted, font, config)


def _label_priority(item: tuple[BEVMarker, int, int]) -> float:
    marker = item[0]
    if marker.extent is None:
        return 0.0
    return abs(float(marker.extent[0])) * abs(float(marker.extent[1]))


def _draw_label(
    canvas: NDArray[np.uint8],
    marker: BEVMarker,
    anchor: tuple[int, int],
    highlighted: bool,
    font: int,
    config: SchematicBevConfig,
) -> None:
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    scale = config.label_font_scale
    thickness = config.label_font_thickness
    label = f"#{marker.obj_id} {marker.category}"
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    anchor_x, anchor_y = anchor
    if anchor_x + 4 + tw + 4 > width:
        org_x = anchor_x - 4 - tw
    else:
        org_x = anchor_x + 4
    org_x = max(3, min(org_x, width - tw - 3))
    org_y = max(th + 3, min(anchor_y - 3, height - baseline - 3))
    bg = _bgr(config.footprint_fill_highlight if highlighted else (255, 255, 255))
    cv2.rectangle(
        canvas,
        (org_x - 3, org_y - th - 3),
        (org_x + tw + 3, org_y + baseline),
        bg,
        -1,
    )
    cv2.putText(
        canvas, label, (org_x, org_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )


def _stack_overlapping_anchors(
    anchors: list[tuple[int, int]], *, ref_w: int, ref_h: int, gap_px: int
) -> list[tuple[int, int]]:
    """Nudge overlapping label anchors downward until separable."""
    out: list[tuple[int, int]] = []
    for x, y in anchors:
        offset = 0
        while any(
            abs(x - ox) <= ref_w and abs(y + offset - oy) <= ref_h for ox, oy in out
        ):
            offset += ref_h + gap_px
        out.append((x, y + offset))
    return out


__all__ = [
    "SchematicBevConfig",
    "DEFAULT_SCHEMATIC_BEV_CONFIG",
    "OrthoBevView",
    "render_schematic_bev",
]
