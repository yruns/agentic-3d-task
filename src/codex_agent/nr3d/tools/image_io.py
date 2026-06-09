"""Shared image sizing for tools whose output the agent later ``view_image``s.

Large frames/BEV renders are the heaviest single contribution to a turn's
context, and once a big image enters the context the Codex app-server starts
windowing older history — which is what tips the agent into the degenerate
``SKILL.md`` re-read loop (see
``docs/codex_agent/skill_loop_and_reasoning_dropped_20260609.md``). Downscaling
every viewed image to a fixed budget keeps the picture legible (labeled boxes
stay sharp at this size) while cutting its token cost and the context churn that
follows it.

Requires the ``vision`` extra (OpenCV + NumPy); imported lazily by the image
tools that use it.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

#: Longest-side budget (pixels) for any image written for ``view_image``. 768 px
#: keeps labeled boxes/markers readable while bounding image token cost; raising
#: it re-introduces the post-image context churn that triggers the re-read loop.
MAX_VIEW_IMAGE_DIM = 768


def downscale_for_view(
    image: NDArray[np.uint8], *, max_dim: int = MAX_VIEW_IMAGE_DIM
) -> tuple[NDArray[np.uint8], float]:
    """Shrink ``image`` so its longest side is at most ``max_dim``.

    Args:
        image: An ``H×W`` or ``H×W×C`` array (channel order is irrelevant).
        max_dim: Longest-side budget in pixels. ``0`` disables downscaling.

    Returns:
        ``(scaled_image, scale)`` where ``scale`` is the multiplicative factor
        applied to pixel coordinates (``1.0`` when no resize happened). Callers
        that report pixel coordinates alongside the image must multiply them by
        ``scale`` to stay consistent with the written image.
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    longest = max(height, width)
    if max_dim <= 0 or longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False), scale


def scale_box(box: tuple[int, int, int, int], scale: float) -> list[int]:
    """Scale a 2D ``(x1, y1, x2, y2)`` box by ``scale`` (rounded to ints)."""
    return [int(round(coord * scale)) for coord in box]


__all__ = ["MAX_VIEW_IMAGE_DIM", "downscale_for_view", "scale_box"]
