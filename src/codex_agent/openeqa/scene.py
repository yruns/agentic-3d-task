"""Prepared-scene assets for OpenEQA (first-person frame discovery + downsizing).

Each ScanNet clip is prepared under ``<data_root>/<clip_id>/`` with first-person
RGB frames at ``raw/<frame_id:06d>-rgb.png``.

The QA turn attaches **no** default frames: the agent fetches every frame it needs
on demand through the OpenEQA CLI tools (``keyframe_selector`` / ``view_frame`` /
``view_bev``), which is why this module only exposes frame *discovery*
(:class:`OpenEqaScene`) and the shared :func:`downsize_rgb_for_view` helper the
tools use to keep fetched images within the image/token budget — not any
uniform-sampling / attachment logic.

Frame paths are resolved from the canonical ``<data_root>/<clip_id>/raw``
directory rather than any absolute path embedded in scene metadata (those point
at the machine that generated the assets).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import OpenEqaDataError
from .question import OpenEqaQuestion

#: Default longest-side pixel size for a downsized frame the agent views.
DEFAULT_MAX_IMAGE_SIZE = 768
#: Default JPEG quality for cached, downsized frames.
_JPEG_QUALITY = 85

_RAW_DIRNAME = "raw"
_RGB_FRAME_RE = re.compile(r"^(\d+)-rgb\.png$")


@dataclass(frozen=True)
class OpenEqaScene:
    """Prepared first-person assets for one OpenEQA ScanNet clip."""

    clip_id: str
    scene_dir: Path
    rgb_frame_ids: tuple[int, ...]

    @property
    def raw_dir(self) -> Path:
        """Directory holding the clip's raw first-person RGB/depth/pose files."""
        return self.scene_dir / _RAW_DIRNAME

    def raw_rgb_path(self, frame_id: int) -> Path:
        """Absolute path to a frame's raw RGB PNG."""
        return self.raw_dir / f"{frame_id:06d}-rgb.png"

    @property
    def total_frames(self) -> int:
        """Number of raw RGB frames discovered for the clip."""
        return len(self.rgb_frame_ids)

    @classmethod
    def load(cls, scene_dir: Path) -> OpenEqaScene:
        """Discover the raw RGB frames for a prepared clip directory.

        Raises:
            OpenEqaDataError: If the clip or its ``raw`` directory is missing,
                or no ``*-rgb.png`` frames are present.
        """
        if not scene_dir.is_dir():
            raise OpenEqaDataError(f"scene directory is missing: {scene_dir}")
        raw_dir = scene_dir / _RAW_DIRNAME
        if not raw_dir.is_dir():
            raise OpenEqaDataError(f"scene raw frames directory is missing: {raw_dir}")
        frame_ids = sorted(
            int(match.group(1))
            for match in (_RGB_FRAME_RE.match(p.name) for p in raw_dir.iterdir())
            if match is not None
        )
        if not frame_ids:
            raise OpenEqaDataError(f"no '*-rgb.png' frames found under {raw_dir}")
        return cls(
            clip_id=scene_dir.name,
            scene_dir=scene_dir,
            rgb_frame_ids=tuple(frame_ids),
        )


def scene_dir_for(data_root: Path, clip_id: str) -> Path:
    """Return the canonical ``<data_root>/<clip_id>`` scene directory."""
    return data_root / clip_id


def has_local_scene(data_root: Path, clip_id: str) -> bool:
    """Whether a clip's prepared ``raw`` frame directory exists locally."""
    return (scene_dir_for(data_root, clip_id) / _RAW_DIRNAME).is_dir()


def filter_questions_with_local_scenes(
    questions: Sequence[OpenEqaQuestion], data_root: Path
) -> tuple[OpenEqaQuestion, ...]:
    """Keep only questions whose clip has prepared frames under ``data_root``."""
    return tuple(q for q in questions if has_local_scene(data_root, q.clip_id))


def downsize_rgb_for_view(
    source: Path,
    destination: Path,
    *,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
    jpeg_quality: int = _JPEG_QUALITY,
) -> Path:
    """Downsize ``source`` so its longest side is ``max_size`` and save as JPEG.

    Used by the agent frame/keyframe/BEV tools so every image the agent views
    stays within the same token budget. Pillow is an optional (``vision`` extra)
    dependency, imported lazily so the package keeps importing without it.

    Returns:
        ``destination`` (for convenient chaining).

    Raises:
        OpenEqaDataError: If Pillow is not installed or arguments are invalid.
    """
    if max_size <= 0:
        raise OpenEqaDataError(f"max_size must be positive, got {max_size}")
    try:
        from PIL import Image
    except ImportError as exc:  # optional vision dependency
        raise OpenEqaDataError(
            "Pillow is required to prepare OpenEQA frames; install the 'vision' "
            "extra (pip install -e '.[vision]')"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        longest = max(width, height)
        if longest > max_size:
            scale = max_size / longest
            rgb = rgb.resize(
                (max(1, round(width * scale)), max(1, round(height * scale)))
            )
        rgb.save(destination, format="JPEG", quality=jpeg_quality)
    return destination


__all__ = [
    "DEFAULT_MAX_IMAGE_SIZE",
    "OpenEqaScene",
    "scene_dir_for",
    "has_local_scene",
    "filter_questions_with_local_scenes",
    "downsize_rgb_for_view",
]
