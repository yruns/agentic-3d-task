"""Image helpers for the OpenEQA agent tools: high-res crop + contact sheet.

Two operations the trace analysis showed were missing:

* :func:`crop_rgb_for_view` cuts a region out of the **full-resolution** raw RGB
  (not the ≤768 px view the agent first sees) so fine detail — a label, a small
  object — becomes legible. This directly attacks the resolution ceiling that
  forced confident-but-wrong fine-grained answers.
* :func:`compose_contact_sheet` tiles several frames into one labelled image so
  the agent can inspect a whole retrieval batch with a *single* ``view_image``
  action instead of one per frame, cutting the tool-budget tax that exhausted
  the call cap.

Pillow is an optional (``vision`` extra) dependency, imported lazily so the
package keeps importing without it (mirrors
:func:`codex_agent.openeqa.scene.downsize_rgb_for_view`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...errors import OpenEqaDataError
from ..scene import DEFAULT_MAX_IMAGE_SIZE

if TYPE_CHECKING:
    from PIL import Image as PILImage

#: JPEG quality for crops / contact sheets (matches ``scene.downsize_rgb_for_view``).
_JPEG_QUALITY = 85
#: Minimum crop side (px) in the SOURCE image; tiny boxes are padded out to this
#: so a crop always carries usable context rather than a few pixels.
_MIN_CROP_SIDE = 32
#: Per-tile pixel size on a contact sheet (longest side of each cell).
_DEFAULT_TILE_PX = 384
#: Contact-sheet background / label colors.
_SHEET_BG = (245, 245, 245)
_LABEL_BG = (0, 0, 0)
_LABEL_FG = (255, 255, 255)


def _load_pillow() -> tuple[object, object, object]:
    """Return ``(Image, ImageDraw, ImageFont)`` or raise a clear error."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # optional vision dependency
        raise OpenEqaDataError(
            "Pillow is required for OpenEQA image tools; install the 'vision' "
            "extra (pip install -e '.[vision]')"
        ) from exc
    return Image, ImageDraw, ImageFont


def clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a pixel box to the image and grow it to a minimum size.

    Args:
        box: ``(left, top, right, bottom)`` in source-image pixels (any order of
            the two corners is accepted).
        width: Source image width in pixels.
        height: Source image height in pixels.

    Returns:
        An integer ``(left, top, right, bottom)`` box inside ``[0, width] ×
        [0, height]`` with both sides at least ``_MIN_CROP_SIDE`` where the image
        allows it.

    Raises:
        OpenEqaDataError: If the image has no area.
    """
    if width <= 0 or height <= 0:
        raise OpenEqaDataError(f"source image has no area: {width}x{height}")
    left, right = sorted((box[0], box[2]))
    top, bottom = sorted((box[1], box[3]))
    left_i = max(0, min(int(math.floor(left)), width - 1))
    top_i = max(0, min(int(math.floor(top)), height - 1))
    right_i = max(left_i + 1, min(int(math.ceil(right)), width))
    bottom_i = max(top_i + 1, min(int(math.ceil(bottom)), height))

    if right_i - left_i < _MIN_CROP_SIDE:
        center = (left_i + right_i) // 2
        half = min(_MIN_CROP_SIDE, width) // 2
        left_i = max(0, center - half)
        right_i = min(width, left_i + min(_MIN_CROP_SIDE, width))
        left_i = max(0, right_i - min(_MIN_CROP_SIDE, width))
    if bottom_i - top_i < _MIN_CROP_SIDE:
        center = (top_i + bottom_i) // 2
        half = min(_MIN_CROP_SIDE, height) // 2
        top_i = max(0, center - half)
        bottom_i = min(height, top_i + min(_MIN_CROP_SIDE, height))
        top_i = max(0, bottom_i - min(_MIN_CROP_SIDE, height))
    return left_i, top_i, right_i, bottom_i


def crop_rgb_for_view(
    source: Path,
    destination: Path,
    *,
    box: tuple[float, float, float, float],
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
    jpeg_quality: int = _JPEG_QUALITY,
) -> tuple[Path, tuple[int, int, int, int]]:
    """Crop ``source`` to ``box`` (source pixels) and save a JPEG view.

    The crop is taken from the full-resolution image, then downsized only if its
    longest side still exceeds ``max_size`` — so a small region ends up far more
    legible than the same region inside a whole-frame ≤768 px view.

    Returns:
        ``(destination, applied_box)`` where ``applied_box`` is the clamped
        integer pixel box actually used.

    Raises:
        OpenEqaDataError: If Pillow is missing or ``max_size`` is non-positive.
    """
    if max_size <= 0:
        raise OpenEqaDataError(f"max_size must be positive, got {max_size}")
    image_mod, _, _ = _load_pillow()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with image_mod.open(source) as image:  # type: ignore[attr-defined]
        rgb = image.convert("RGB")
        applied = clamp_box(box, rgb.width, rgb.height)
        crop = rgb.crop(applied)
        longest = max(crop.width, crop.height)
        if longest > max_size:
            scale = max_size / longest
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
            )
        crop.save(destination, format="JPEG", quality=jpeg_quality)
    return destination, applied


def normalized_box_to_pixels(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Scale a normalized ``[0, 1]`` box to pixel coordinates."""
    return (box[0] * width, box[1] * height, box[2] * width, box[3] * height)


def box_looks_normalized(box: tuple[float, float, float, float]) -> bool:
    """Whether every coordinate is within ``[0, 1]`` (a normalized box)."""
    return all(0.0 <= v <= 1.0 for v in box)


def compose_contact_sheet(
    tiles: Sequence[tuple[str, Path]],
    destination: Path,
    *,
    tile_px: int = _DEFAULT_TILE_PX,
    columns: int = 0,
    jpeg_quality: int = _JPEG_QUALITY,
) -> Path:
    """Tile labelled images into one contact sheet and save it as JPEG.

    Args:
        tiles: ``(label, image_path)`` pairs, drawn left-to-right, top-to-bottom.
        destination: Output JPEG path.
        tile_px: Longest-side pixel size of each cell.
        columns: Grid width; ``0`` picks a near-square layout.
        jpeg_quality: Output JPEG quality.

    Returns:
        ``destination``.

    Raises:
        OpenEqaDataError: If Pillow is missing or ``tiles`` is empty.
    """
    if not tiles:
        raise OpenEqaDataError("compose_contact_sheet needs at least one tile")
    image_mod, draw_mod, font_mod = _load_pillow()
    count = len(tiles)
    cols = columns if columns > 0 else max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    label_h = max(14, tile_px // 12)
    cell_w = tile_px
    cell_h = tile_px + label_h

    font = font_mod.load_default()  # type: ignore[attr-defined]
    sheet = image_mod.new(  # type: ignore[attr-defined]
        "RGB", (cols * cell_w, rows * cell_h), color=_SHEET_BG
    )
    drawer = draw_mod.Draw(sheet)  # type: ignore[attr-defined]

    for index, (label, path) in enumerate(tiles):
        row, col = divmod(index, cols)
        x0 = col * cell_w
        y0 = row * cell_h
        _paste_tile(image_mod, sheet, path, x0, y0, tile_px)
        drawer.rectangle([x0, y0 + tile_px, x0 + cell_w, y0 + cell_h], fill=_LABEL_BG)
        drawer.text((x0 + 4, y0 + tile_px + 1), label, fill=_LABEL_FG, font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=jpeg_quality)
    return destination


def _paste_tile(
    image_mod: object,
    sheet: PILImage.Image,
    path: Path,
    x0: int,
    y0: int,
    tile_px: int,
) -> None:
    """Fit one image into a ``tile_px`` square cell, centered, preserving aspect."""
    with image_mod.open(path) as raw:  # type: ignore[attr-defined]
        tile = raw.convert("RGB")
        longest = max(tile.width, tile.height)
        if longest > tile_px:
            scale = tile_px / longest
            tile = tile.resize(
                (max(1, round(tile.width * scale)), max(1, round(tile.height * scale)))
            )
        offset_x = x0 + (tile_px - tile.width) // 2
        offset_y = y0 + (tile_px - tile.height) // 2
        sheet.paste(tile, (offset_x, offset_y))


__all__ = [
    "clamp_box",
    "crop_rgb_for_view",
    "normalized_box_to_pixels",
    "box_looks_normalized",
    "compose_contact_sheet",
]
