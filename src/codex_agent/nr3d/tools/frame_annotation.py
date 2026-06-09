"""Frame annotation (image) tool: ``mark_frame_with_bbox``.

Draws high-contrast, labeled 2D boxes for the named proposal ids (and/or
categories) onto a first-person RGB frame, writes the result into a writable
scratch directory, and returns its path plus the left→right 2D layout. The
agent then calls the built-in ``view_image`` on that path to read the pixels.

Requires the ``vision`` extra (OpenCV + Pillow + NumPy); this module is imported
lazily by the dispatcher only when the tool is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ..proposals import Proposal
from ..sample import Nr3dScene
from .models import ToolInputError

#: High-contrast stroke colors cycled across the marked boxes (RGB).
_BBOX_PALETTE: tuple[tuple[int, int, int], ...] = (
    (34, 197, 94),
    (239, 68, 68),
    (59, 130, 246),
    (234, 179, 8),
    (168, 85, 247),
    (20, 184, 166),
    (249, 115, 22),
    (236, 72, 153),
)
_BLACK_OUTLINE_PAD = 2
_LABEL_PADDING_PX = 5
_LABEL_FONT_SCALE_FLOOR = 0.84
_LABEL_FONT_SCALE_REF_WIDTH = 1500
_LABEL_FONT_THICKNESS_DIVISOR = 500
_LABEL_LUMINANCE_BLACK_TEXT_THRESHOLD = 140.0


class MarkFrameArgs(BaseModel):
    """Arguments for :func:`mark_frame_with_bbox`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int
    ids: list[int] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class MarkFrameResult:
    """A written annotated frame plus its left→right 2D layout."""

    frame_id: int
    image_path: Path
    visible_proposal_ids: tuple[int, ...]
    left_to_right: tuple[str, ...]
    boxes_2d: dict[int, list[int]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
            "visible_proposal_ids": list(self.visible_proposal_ids),
            "left_to_right": list(self.left_to_right),
            "boxes_2d": self.boxes_2d,
        }


def mark_frame_with_bbox(
    scene: Nr3dScene, args: MarkFrameArgs, *, out_dir: Path
) -> MarkFrameResult:
    """Render labeled boxes for the named ids/categories on one frame."""
    if not args.ids and not args.labels:
        raise ToolInputError(
            "mark_frame_with_bbox requires at least one of {ids, labels}"
        )
    visible = _matching_proposals(scene, args.frame_id, args.ids, args.labels)
    if not visible:
        present = sorted(
            scene.proposal_pool.frame_to_proposal_ids().get(args.frame_id, [])
        )
        raise ToolInputError(
            f"no proposals matched ids={args.ids} labels={args.labels} on "
            f"frame_id={args.frame_id}; visible proposal ids here={present}"
        )

    ordered = sorted(
        visible, key=lambda p: (p.frame_views[args.frame_id].center_x, p.proposal_id)
    )
    raw_path = scene.resolve_raw_rgb(ordered[0].frame_views[args.frame_id])
    if not raw_path.exists():
        raise ToolInputError(f"raw RGB frame is missing on disk: {raw_path}")

    image = np.asarray(Image.open(raw_path).convert("RGB")).copy()
    for index, proposal in enumerate(ordered):
        color = _BBOX_PALETTE[index % len(_BBOX_PALETTE)]
        box = proposal.frame_views[args.frame_id].bbox_2d
        _draw_palette_bbox(image, box, color)
        _draw_label(image, f"#{proposal.proposal_id} {proposal.category}", box, color)

    out_dir.mkdir(parents=True, exist_ok=True)
    ids_token = "_".join(str(p.proposal_id) for p in ordered)
    out_path = out_dir / f"{scene.scene_id}_frame_{args.frame_id}_ids_{ids_token}.png"
    Image.fromarray(image).save(out_path, format="PNG")

    return MarkFrameResult(
        frame_id=args.frame_id,
        image_path=out_path,
        visible_proposal_ids=tuple(p.proposal_id for p in ordered),
        left_to_right=tuple(f"#{p.proposal_id} {p.category}" for p in ordered),
        boxes_2d={
            p.proposal_id: list(p.frame_views[args.frame_id].bbox_2d) for p in ordered
        },
    )


def _matching_proposals(
    scene: Nr3dScene, frame_id: int, ids: list[int], labels: list[str]
) -> list[Proposal]:
    wanted_ids = {int(i) for i in ids}
    wanted_labels = {label.strip().lower() for label in labels if label.strip()}
    matched: list[Proposal] = []
    for proposal in scene.proposal_pool.proposals:
        if frame_id not in proposal.frame_views:
            continue
        if proposal.proposal_id in wanted_ids or (
            wanted_labels and proposal.category.strip().lower() in wanted_labels
        ):
            matched.append(proposal)
    return matched


def _draw_palette_bbox(
    image: NDArray[np.uint8],
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    stroke = max(5, image.shape[1] // 200)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), stroke + 2 * _BLACK_OUTLINE_PAD)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, stroke)


def _draw_label(
    image: NDArray[np.uint8],
    text: str,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(_LABEL_FONT_SCALE_FLOOR, image.shape[1] / _LABEL_FONT_SCALE_REF_WIDTH)
    thickness = max(2, image.shape[1] // _LABEL_FONT_THICKNESS_DIVISOR)
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    bg_x1, bg_y1, bg_x2, bg_y2 = _label_box(box, text_w, text_h, image.shape[0])
    cv2.rectangle(
        image,
        (bg_x1 - _BLACK_OUTLINE_PAD, bg_y1 - _BLACK_OUTLINE_PAD),
        (bg_x2 + _BLACK_OUTLINE_PAD, bg_y2 + _BLACK_OUTLINE_PAD),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    text_color = _text_color_for_bg(color)
    cv2.putText(
        image,
        text,
        (bg_x1 + _LABEL_PADDING_PX, bg_y1 + text_h + _LABEL_PADDING_PX),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _label_box(
    box: tuple[int, int, int, int],
    text_w: int,
    text_h: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    label_w = text_w + 2 * _LABEL_PADDING_PX
    label_h = text_h + 2 * _LABEL_PADDING_PX
    if label_w <= max(1, x2 - x1) and label_h <= max(1, y2 - y1):
        return x1, y1, x1 + label_w, y1 + label_h
    above_top = max(0, y1 - label_h)
    return x1, above_top, x1 + label_w, above_top + label_h


def _text_color_for_bg(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > _LABEL_LUMINANCE_BLACK_TEXT_THRESHOLD:
        return (0, 0, 0)
    return (255, 255, 255)


__all__ = [
    "MarkFrameArgs",
    "MarkFrameResult",
    "mark_frame_with_bbox",
]
