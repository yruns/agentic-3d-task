"""Global-alignment (image) tool: ``view_bev``.

Returns the prepared top-down BEV render. With ``highlight``/``categories`` it
projects the chosen proposals' 3D centers onto the BEV using the same look-down
camera that produced the base render (read from the ``.view.json`` sidecar),
draws labeled markers, writes the overlay to a scratch dir, and returns its
path so the agent can ``view_image`` it and tie first-person frames to the map.

Requires the ``vision`` extra (OpenCV + NumPy); imported lazily by the
dispatcher only when the tool runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..proposals import Proposal
from ..sample import Nr3dScene
from ..scene_assets import BevViewParams
from .image_io import downscale_for_view
from .models import ToolInputError

_HIGHLIGHT_RENDERER_VERSION = "v1"
_MIN_PROJECTION_DEPTH = 0.01
_MARKER_RADIUS = 9
_MARKER_COLOR = (255, 64, 64)
_LABEL_BG = (255, 255, 0)


class ViewBevArgs(BaseModel):
    """Arguments for :func:`view_bev`."""

    model_config = ConfigDict(extra="forbid")

    highlight: list[int] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ViewBevResult:
    """A BEV image path plus what was highlighted on it."""

    image_path: Path
    highlight_ids: tuple[int, ...]
    missing_categories: tuple[str, ...]
    is_default_view: bool

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(self.image_path),
            "highlight_ids": list(self.highlight_ids),
            "view": "default" if self.is_default_view else "highlighted",
        }
        if self.missing_categories:
            payload["categories_with_no_matches"] = list(self.missing_categories)
        return payload


def view_bev(scene: Nr3dScene, args: ViewBevArgs, *, out_dir: Path) -> ViewBevResult:
    """Return the BEV render, optionally highlighting proposals by id/category."""
    highlight_ids, missing = _resolve_highlight_ids(scene, args)
    if not highlight_ids:
        return ViewBevResult(
            image_path=scene.bev_image_path,
            highlight_ids=(),
            missing_categories=tuple(missing),
            is_default_view=True,
        )
    if scene.bev_view_params is None:
        raise ToolInputError(
            "cannot highlight the BEV: this scene is missing its "
            "bev/scene_bev_nr3d.view.json sidecar"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_token = "_".join(str(i) for i in highlight_ids)
    out_path = (
        out_dir
        / f"{scene.scene_id}_bev_h_{ids_token}_{_HIGHLIGHT_RENDERER_VERSION}.png"
    )
    _render_highlighted_bev(
        base_png=scene.bev_image_path,
        view=scene.bev_view_params,
        proposals=[scene.proposal_pool.require(pid) for pid in highlight_ids],
        out_path=out_path,
    )
    return ViewBevResult(
        image_path=out_path,
        highlight_ids=tuple(highlight_ids),
        missing_categories=tuple(missing),
        is_default_view=False,
    )


def _resolve_highlight_ids(
    scene: Nr3dScene, args: ViewBevArgs
) -> tuple[list[int], list[str]]:
    pool = scene.proposal_pool
    ids: set[int] = set()
    missing_ids = sorted({pid for pid in args.highlight if pool.get(pid) is None})
    if missing_ids:
        raise ToolInputError(f"highlight ids not in pool: {missing_ids}")
    ids.update(args.highlight)

    by_category: dict[str, list[int]] = {}
    for proposal in pool.proposals:
        by_category.setdefault(proposal.category.strip().lower(), []).append(
            proposal.proposal_id
        )
    missing_categories: list[str] = []
    for raw in args.categories:
        key = raw.strip().lower()
        if not key:
            continue
        hits = by_category.get(key, [])
        if not hits:
            missing_categories.append(raw)
        ids.update(hits)
    return sorted(ids), missing_categories


def _render_highlighted_bev(
    *,
    base_png: Path,
    view: BevViewParams,
    proposals: list[Proposal],
    out_path: Path,
) -> None:
    loaded = cv2.imread(str(base_png))
    if loaded is None:
        raise ToolInputError(f"BEV image is not readable: {base_png}")
    image: NDArray[np.uint8] = loaded.astype(np.uint8, copy=False)
    rotation = np.asarray(view.rotation, dtype=np.float64)
    translation = np.asarray(view.translation, dtype=np.float64)
    for proposal in proposals:
        pixel = _project_to_bev(proposal.position_3d, rotation, translation, view)
        if pixel is None:
            continue
        _draw_marker(image, pixel, f"#{proposal.proposal_id} {proposal.category}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the written overlay within the view_image token budget.
    rendered, _ = downscale_for_view(image)
    cv2.imwrite(str(out_path), rendered)


def _project_to_bev(
    position_3d: tuple[float, float, float],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    view: BevViewParams,
) -> tuple[int, int] | None:
    point = np.asarray(position_3d, dtype=np.float64)
    cam = rotation @ point + translation
    if cam[2] < _MIN_PROJECTION_DEPTH:
        return None
    u = view.focal * cam[0] / cam[2] + view.center
    v = view.focal * cam[1] / cam[2] + view.center
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)) - view.crop_offset[0], int(round(v)) - view.crop_offset[1]


def _draw_marker(image: NDArray[np.uint8], pixel: tuple[int, int], label: str) -> None:
    height, width = int(image.shape[0]), int(image.shape[1])
    px, py = pixel
    if not (-_MARKER_RADIUS <= px < width + _MARKER_RADIUS):
        return
    if not (-_MARKER_RADIUS <= py < height + _MARKER_RADIUS):
        return
    cv2.circle(image, (px, py), _MARKER_RADIUS, (0, 0, 0), -1)
    cv2.circle(image, (px, py), _MARKER_RADIUS - 2, _MARKER_COLOR[::-1], -1)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    org_x = max(4, min(px + 8, width - text_w - 4))
    org_y = max(text_h + 6, min(py - 6, height - baseline - 4))
    cv2.rectangle(
        image,
        (org_x - 3, org_y - text_h - 3),
        (org_x + text_w + 3, org_y + baseline + 3),
        _LABEL_BG[::-1],
        -1,
    )
    cv2.putText(
        image, label, (org_x, org_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )


__all__ = [
    "ViewBevArgs",
    "ViewBevResult",
    "view_bev",
]
