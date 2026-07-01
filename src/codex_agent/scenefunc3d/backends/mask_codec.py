"""Value codecs for compact SceneFunc3D mask payloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MaskRlePayload:
    """Row-major run-length encoding for a 2D boolean mask."""

    height: int
    width: int
    counts: tuple[int, ...]


def encode_bool_mask_rle(mask: npt.NDArray[np.bool_]) -> MaskRlePayload:
    """Encode a 2D boolean mask as row-major false/true run lengths."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D; got ndim={mask.ndim}")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask dtype must be bool; got dtype={mask.dtype}")

    flat_mask = mask.reshape(-1, order="C")
    counts: list[int] = []
    current_value = False
    run_length = 0
    for raw_pixel in flat_mask:
        pixel_value = bool(raw_pixel)
        if pixel_value == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        current_value = pixel_value
        run_length = 1
    counts.append(run_length)

    return MaskRlePayload(
        height=int(mask.shape[0]),
        width=int(mask.shape[1]),
        counts=tuple(counts),
    )


def decode_bool_mask_rle(payload: MaskRlePayload) -> npt.NDArray[np.bool_]:
    """Decode row-major false/true run lengths into a 2D boolean mask."""
    _validate_rle_payload(payload)
    total_pixels = payload.height * payload.width
    flat_mask = np.empty(total_pixels, dtype=np.bool_)
    offset = 0
    current_value = False
    for count in payload.counts:
        next_offset = offset + count
        flat_mask[offset:next_offset] = current_value
        offset = next_offset
        current_value = not current_value
    return flat_mask.reshape((payload.height, payload.width), order="C")


def _validate_rle_payload(payload: MaskRlePayload) -> None:
    if (
        not isinstance(payload.height, int)
        or isinstance(payload.height, bool)
        or not isinstance(payload.width, int)
        or isinstance(payload.width, bool)
    ):
        raise ValueError("RLE height and width must be integers")
    if payload.height <= 0 or payload.width <= 0:
        raise ValueError(
            "RLE height and width must describe a positive 2D mask: "
            f"height={payload.height}; width={payload.width}"
        )
    if not payload.counts:
        raise ValueError("RLE counts must not be empty")

    total_count = 0
    for count in payload.counts:
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"RLE count must be an integer: count={count!r}")
        if count < 0:
            raise ValueError(f"RLE count must be non-negative: count={count!r}")
        total_count += count

    expected_total = payload.height * payload.width
    if total_count != expected_total:
        raise ValueError(
            "RLE counts must sum to height * width: "
            f"sum={total_count}; expected={expected_total}"
        )


__all__ = [
    "MaskRlePayload",
    "decode_bool_mask_rle",
    "encode_bool_mask_rle",
]
