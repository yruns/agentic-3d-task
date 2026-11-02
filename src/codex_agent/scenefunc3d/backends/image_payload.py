"""Inline image payload helpers for SceneFunc3D sidecar backends."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from codex_agent.scenefunc3d.servers.schemas import (
    InlineImageMimeType,
    InlineImagePayload,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError

from .config import ensure_path_under_roots

_ALLOWED_INLINE_IMAGE_MIME_TYPES: frozenset[InlineImageMimeType] = frozenset(
    ("image/jpeg", "image/png", "image/webp")
)


def build_inline_image_payload(
    image_path: Path,
    *,
    allowed_roots: tuple[Path, ...],
) -> InlineImagePayload:
    """Build a validated inline image payload from an allowed local image path."""
    normalized_path = ensure_path_under_roots(
        image_path,
        roots=allowed_roots,
        field_name="image_path",
    )
    mime_type = _guess_allowed_mime_type(normalized_path)
    try:
        image_bytes = normalized_path.read_bytes()
    except OSError as exc:
        raise ToolInputError(
            "could not read image for inline sidecar payload: "
            f"path={normalized_path}; error_type={exc.__class__.__name__}"
        ) from exc

    return InlineImagePayload(
        filename=normalized_path.name,
        mime_type=mime_type,
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        data_base64=base64.b64encode(image_bytes).decode("ascii"),
    )


def materialize_inline_image(
    payload: InlineImagePayload,
    *,
    parent_dir: Path,
) -> Path:
    """Write a validated inline image payload under ``parent_dir``."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    image_path = parent_dir / payload.filename
    image_path.write_bytes(payload.decoded_bytes())
    return image_path


def is_remote_backend_url(url: str) -> bool:
    """Return true when a backend URL targets a remote HTTPS sidecar."""
    return urlparse(url).scheme.lower() == "https"


def _guess_allowed_mime_type(image_path: Path) -> InlineImageMimeType:
    guessed_mime_type = mimetypes.guess_type(image_path.name)[0]
    if guessed_mime_type in _ALLOWED_INLINE_IMAGE_MIME_TYPES:
        return cast(InlineImageMimeType, guessed_mime_type)
    raise ToolInputError(
        "image_path must use a supported inline sidecar image extension: "
        f"path={image_path}; mime_type={guessed_mime_type!r}; "
        "allowed_mime_types=image/jpeg,image/png,image/webp"
    )


__all__ = [
    "build_inline_image_payload",
    "is_remote_backend_url",
    "materialize_inline_image",
]
