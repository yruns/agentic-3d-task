"""Frame annotation & high-res zoom (image) tool: ``mark_frame_with_bbox``.

Draws high-contrast, labeled 2D boxes for the named proposal ids (and/or
categories) onto one or more first-person RGB frames, writes the result into a
writable scratch directory, and returns its path plus the left→right 2D layout.

Also supports:
* **Multi-frame batching (``frame_ids``)**: annotate up to 4 frames in one call
  and compose a combined ``contact_sheet`` so the agent can compare multiple
  viewpoints in a single ``view_image`` call.
* **High-resolution unoccluded crop sheet (``crops_sheet_path``)**: cuts each
  matched proposal's 2D box (with 20% context margin) directly from the
  full-resolution raw RGB *without* thick bbox strokes covering small objects,
  and tiles them into a labelled comparison sheet so the agent can inspect fine
  appearance, color, and texture.

Requires the ``vision`` extra (OpenCV + Pillow + NumPy); this module is imported
lazily by the dispatcher only when the tool is invoked.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ..proposals import Proposal
from ..sample import Nr3dScene
from .image_io import downscale_for_view, scale_box
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
_MAX_BATCH_FRAMES = 4
_CROP_MARGIN_RATIO = 0.22
_CROP_TILE_PX = 360


class MarkFrameArgs(BaseModel):
    """Arguments for :func:`mark_frame_with_bbox`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int | None = None
    frame_ids: list[int] = Field(default_factory=list)
    ids: list[int] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    include_crops: bool = True


@dataclass(frozen=True)
class MarkFrameResult:
    """A written annotated frame (or multi-frame sheet) plus 2D layout & crops."""

    frame_id: int
    image_path: Path
    visible_proposal_ids: tuple[int, ...]
    left_to_right: tuple[str, ...]
    boxes_2d: dict[int, list[int]]
    crops_sheet_path: Path | None = None
    contact_sheet_path: Path | None = None
    per_frame: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
            "visible_proposal_ids": list(self.visible_proposal_ids),
            "left_to_right": list(self.left_to_right),
            "boxes_2d": self.boxes_2d,
        }
        if self.crops_sheet_path is not None:
            payload["crops_sheet_path"] = str(self.crops_sheet_path)
        if self.contact_sheet_path is not None:
            payload["contact_sheet"] = str(self.contact_sheet_path)
        if len(self.per_frame) > 1:
            payload["frames"] = list(self.per_frame)
        return payload


def mark_frame_with_bbox(
    scene: Nr3dScene, args: MarkFrameArgs, *, out_dir: Path
) -> MarkFrameResult:
    """Render labeled boxes (and high-res crops) for ids/categories on frame(s)."""
    if not args.ids and not args.labels:
        raise ToolInputError(
            "mark_frame_with_bbox requires at least one of {ids, labels}"
        )
    requested_fids = _resolve_frame_ids(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered_frames: list[dict[str, Any]] = []
    crop_tiles: list[tuple[str, NDArray[np.uint8]]] = []

    for fid in requested_fids[:_MAX_BATCH_FRAMES]:
        visible = _matching_proposals(scene, fid, args.ids, args.labels)
        if not visible:
            if len(requested_fids) == 1:
                present = sorted(
                    scene.proposal_pool.frame_to_proposal_ids().get(fid, [])
                )
                raise ToolInputError(
                    f"no proposals matched ids={args.ids} labels={args.labels} on "
                    f"frame_id={fid}; visible proposal ids here={present}"
                )
            continue

        ordered = sorted(
            visible, key=lambda p: (p.frame_views[fid].center_x, p.proposal_id)
        )
        raw_path = scene.resolve_raw_rgb(ordered[0].frame_views[fid])
        if not raw_path.exists():
            raise ToolInputError(f"raw RGB frame is missing on disk: {raw_path}")

        raw_rgb = np.asarray(Image.open(raw_path).convert("RGB"))
        image = raw_rgb.copy()
        for index, proposal in enumerate(ordered):
            color = _BBOX_PALETTE[index % len(_BBOX_PALETTE)]
            box = proposal.frame_views[fid].bbox_2d
            _draw_palette_bbox(image, box, color)
            _draw_label(
                image, f"#{proposal.proposal_id} {proposal.category}", box, color
            )
            if args.include_crops:
                crop_arr = _extract_raw_crop(raw_rgb, box)
                if crop_arr is not None:
                    crop_tiles.append(
                        (
                            f"#{proposal.proposal_id} {proposal.category} (f{fid})",
                            crop_arr,
                        )
                    )

        rendered, scale = downscale_for_view(image)
        ids_token = "_".join(str(p.proposal_id) for p in ordered)
        out_path = out_dir / f"{scene.scene_id}_frame_{fid}_ids_{ids_token}.png"
        _atomic_write_png(rendered, out_path)

        rendered_frames.append(
            {
                "frame_id": fid,
                "image_path": out_path,
                "rendered_rgb": rendered,
                "visible_proposal_ids": tuple(p.proposal_id for p in ordered),
                "left_to_right": tuple(
                    f"#{p.proposal_id} {p.category}" for p in ordered
                ),
                "boxes_2d": {
                    p.proposal_id: scale_box(p.frame_views[fid].bbox_2d, scale)
                    for p in ordered
                },
            }
        )

    if not rendered_frames:
        present_any = {
            fid: sorted(scene.proposal_pool.frame_to_proposal_ids().get(fid, []))
            for fid in requested_fids[:_MAX_BATCH_FRAMES]
        }
        raise ToolInputError(
            f"no proposals matched ids={args.ids} labels={args.labels} on "
            f"frame_ids={requested_fids}; visible proposal ids={present_any}"
        )

    crops_sheet: Path | None = None
    if args.include_crops and crop_tiles:
        f_tok = "_".join(str(r["frame_id"]) for r in rendered_frames)
        i_tok = ("_".join(str(i) for i in args.ids) if args.ids else "lbl")[:32]
        crops_path = out_dir / f"{scene.scene_id}_crops_f{f_tok}_ids_{i_tok}.png"
        crops_sheet = _compose_tile_sheet(crop_tiles, crops_path, tile_px=_CROP_TILE_PX)

    contact_sheet: Path | None = None
    if len(rendered_frames) > 1:
        frame_tiles = [
            (
                f"frame {r['frame_id']}: {', '.join(r['left_to_right'])}",
                r["rendered_rgb"],
            )
            for r in rendered_frames
        ]
        f_tok = "_".join(str(r["frame_id"]) for r in rendered_frames)
        sheet_path = out_dir / f"{scene.scene_id}_marked_sheet_{f_tok}.png"
        contact_sheet = _compose_tile_sheet(frame_tiles, sheet_path, tile_px=384)

    first = rendered_frames[0]
    per_frame_payloads = tuple(
        {
            "frame_id": r["frame_id"],
            "image_path": str(r["image_path"]),
            "visible_proposal_ids": list(r["visible_proposal_ids"]),
            "left_to_right": list(r["left_to_right"]),
            "boxes_2d": r["boxes_2d"],
        }
        for r in rendered_frames
    )
    return MarkFrameResult(
        frame_id=int(first["frame_id"]),
        image_path=contact_sheet if contact_sheet is not None else first["image_path"],
        visible_proposal_ids=first["visible_proposal_ids"],
        left_to_right=first["left_to_right"],
        boxes_2d=first["boxes_2d"],
        crops_sheet_path=crops_sheet,
        contact_sheet_path=contact_sheet,
        per_frame=per_frame_payloads,
    )


def _resolve_frame_ids(args: MarkFrameArgs) -> list[int]:
    ids: list[int] = []
    if args.frame_id is not None:
        ids.append(int(args.frame_id))
    ids.extend(int(f) for f in args.frame_ids)
    deduped = list(dict.fromkeys(ids))
    if not deduped:
        raise ToolInputError("mark_frame_with_bbox requires frame_id or frame_ids")
    return deduped


def _extract_raw_crop(
    raw_rgb: NDArray[np.uint8], box: tuple[int, int, int, int]
) -> NDArray[np.uint8] | None:
    h, w = int(raw_rgb.shape[0]), int(raw_rgb.shape[1])
    if h <= 0 or w <= 0:
        return None
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(12, int(round(bw * _CROP_MARGIN_RATIO)))
    pad_y = max(12, int(round(bh * _CROP_MARGIN_RATIO)))
    cx1 = max(0, min(w - 1, x1 - pad_x))
    cy1 = max(0, min(h - 1, y1 - pad_y))
    cx2 = max(cx1 + 1, min(w, x2 + pad_x))
    cy2 = max(cy1 + 1, min(h, y2 + pad_y))
    return raw_rgb[cy1:cy2, cx1:cx2].copy()


def _compose_tile_sheet(
    tiles: Sequence[tuple[str, NDArray[np.uint8]]],
    out_path: Path,
    *,
    tile_px: int = 360,
) -> Path | None:
    if not tiles:
        return None
    count = len(tiles)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    label_h = 26
    cell_w = tile_px
    cell_h = tile_px + label_h
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)

    for idx, (label, arr) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x0 = c * cell_w
        y0 = r * cell_h
        th, tw = int(arr.shape[0]), int(arr.shape[1])
        longest = max(th, tw, 1)
        scale = tile_px / float(longest)
        nw = max(1, min(tile_px, int(round(tw * scale))))
        nh = max(1, min(tile_px, int(round(th * scale))))
        resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        ox = x0 + (tile_px - nw) // 2
        oy = y0 + (tile_px - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
        cv2.rectangle(
            canvas,
            (x0, y0 + tile_px),
            (x0 + cell_w, y0 + cell_h),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            canvas,
            label[:42],
            (x0 + 6, y0 + tile_px + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    downscaled, _ = downscale_for_view(canvas)
    _atomic_write_png(downscaled, out_path)
    return out_path


def _atomic_write_png(rgb_array: NDArray[np.uint8], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f".{out_path.stem}.{os.getpid()}.{threading.get_ident()}{out_path.suffix}"
    )
    try:
        Image.fromarray(rgb_array).save(tmp_path, format="PNG")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
