"""Crop-to-full-frame metadata for SceneFunc3D evidence images."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, FilePath, StringConstraints

from codex_agent.scenefunc3d.backends.config import ensure_path_under_roots

from .models import ToolInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictCoordinateInt = Annotated[int, Field(strict=True)]


class CropMetadataPayload(TypedDict):
    """JSON-ready crop metadata persisted next to a rendered crop image."""

    frame_id: str
    crop_image_path: str
    source_image_path: str
    source_image_width: int
    source_image_height: int
    crop_image_width: int
    crop_image_height: int
    crop_bbox_xyxy: list[int]


class CropMetadata(BaseModel):
    """Validated mapping from a crop image back to its full source frame."""

    model_config = ConfigDict(extra="forbid")

    frame_id: NonEmptyText
    crop_image_path: FilePath
    source_image_path: FilePath
    source_image_width: int = Field(gt=0, strict=True)
    source_image_height: int = Field(gt=0, strict=True)
    crop_image_width: int = Field(gt=0, strict=True)
    crop_image_height: int = Field(gt=0, strict=True)
    crop_bbox_xyxy: tuple[
        StrictCoordinateInt,
        StrictCoordinateInt,
        StrictCoordinateInt,
        StrictCoordinateInt,
    ]

    def model_post_init(self, __context: object) -> None:
        """Validate crop bounds against the source frame dimensions."""
        left, top, right, bottom = self.crop_bbox_xyxy
        if left < 0 or top < 0:
            raise ValueError(
                f"crop bbox origin must be non-negative: {self.crop_bbox_xyxy}"
            )
        if left >= right or top >= bottom:
            raise ValueError(
                f"crop bbox must satisfy left < right and top < bottom: {self.crop_bbox_xyxy}"
            )
        if right > self.source_image_width or bottom > self.source_image_height:
            raise ValueError(
                "crop bbox must fit inside source image: "
                f"bbox={self.crop_bbox_xyxy}; "
                f"source_size={(self.source_image_width, self.source_image_height)}"
            )
        bbox_size = (right - left, bottom - top)
        crop_size = (self.crop_image_width, self.crop_image_height)
        if bbox_size != crop_size:
            raise ValueError(
                "crop bbox size must match crop image size: "
                f"bbox_size={bbox_size}; crop_size={crop_size}"
            )

    def to_payload(self) -> CropMetadataPayload:
        """Return a JSON-serializable payload."""
        return {
            "frame_id": self.frame_id,
            "crop_image_path": str(self.crop_image_path),
            "source_image_path": str(self.source_image_path),
            "source_image_width": self.source_image_width,
            "source_image_height": self.source_image_height,
            "crop_image_width": self.crop_image_width,
            "crop_image_height": self.crop_image_height,
            "crop_bbox_xyxy": list(self.crop_bbox_xyxy),
        }


def crop_metadata_path_for_image(image_path: Path) -> Path:
    """Return the sidecar metadata path for a rendered crop image."""
    return image_path.with_suffix(".crop.json")


def write_crop_metadata(metadata: CropMetadata) -> Path:
    """Persist crop metadata next to the crop image and return the metadata path."""
    metadata_path = crop_metadata_path_for_image(metadata.crop_image_path)
    try:
        metadata_path.write_text(
            json.dumps(metadata.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ToolInputError(
            "could not write crop metadata: "
            f"path={metadata_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return metadata_path


def load_crop_metadata_for_image(
    image_path: Path,
    *,
    expected_frame_id: str,
    allowed_image_roots: tuple[Path, ...],
) -> CropMetadata | None:
    """Load crop metadata for ``image_path`` when a crop sidecar exists."""
    validated_image_path = ensure_path_under_roots(
        image_path,
        roots=allowed_image_roots,
        field_name="image_path",
    )
    metadata_path = crop_metadata_path_for_image(validated_image_path)
    if not metadata_path.is_file():
        return None
    try:
        payload: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolInputError(
            "could not read crop metadata: "
            f"path={metadata_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except JSONDecodeError as exc:
        raise ToolInputError(
            f"crop metadata is not valid JSON: path={metadata_path}"
        ) from exc
    try:
        metadata = CropMetadata.model_validate(payload)
    except ValueError as exc:
        raise ToolInputError(
            f"crop metadata failed validation: path={metadata_path}; error={exc}"
        ) from exc
    _validate_metadata_matches_images(
        metadata,
        image_path=validated_image_path,
        expected_frame_id=expected_frame_id,
        allowed_image_roots=allowed_image_roots,
        metadata_path=metadata_path,
    )
    return metadata


def _validate_metadata_matches_images(
    metadata: CropMetadata,
    *,
    image_path: Path,
    expected_frame_id: str,
    allowed_image_roots: tuple[Path, ...],
    metadata_path: Path,
) -> None:
    if metadata.frame_id != expected_frame_id:
        raise ToolInputError(
            "crop metadata frame_id mismatch: "
            f"metadata_path={metadata_path}; expected={expected_frame_id!r}; "
            f"actual={metadata.frame_id!r}"
        )

    validated_crop_image_path = ensure_path_under_roots(
        metadata.crop_image_path,
        roots=allowed_image_roots,
        field_name="crop_image_path",
    )
    validated_source_image_path = ensure_path_under_roots(
        metadata.source_image_path,
        roots=allowed_image_roots,
        field_name="source_image_path",
    )

    if validated_crop_image_path != image_path:
        raise ToolInputError(
            "crop metadata does not describe image_path: "
            f"metadata_path={metadata_path}; image_path={image_path}; "
            f"metadata_crop_image_path={metadata.crop_image_path}"
        )

    crop_image_size = _read_image_size(
        validated_crop_image_path,
        image_role="crop",
        metadata_path=metadata_path,
    )
    expected_crop_size = (metadata.crop_image_width, metadata.crop_image_height)
    if crop_image_size != expected_crop_size:
        raise ToolInputError(
            "crop metadata crop image size mismatch: "
            f"metadata_path={metadata_path}; actual={crop_image_size}; "
            f"expected={expected_crop_size}"
        )

    source_image_size = _read_image_size(
        validated_source_image_path,
        image_role="source",
        metadata_path=metadata_path,
    )
    expected_source_size = (metadata.source_image_width, metadata.source_image_height)
    if source_image_size != expected_source_size:
        raise ToolInputError(
            "crop metadata source image size mismatch: "
            f"metadata_path={metadata_path}; actual={source_image_size}; "
            f"expected={expected_source_size}"
        )


def _read_image_size(
    image_path: Path, *, image_role: str, metadata_path: Path
) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to validate crop metadata images; install the "
            "'vision' extra"
        ) from exc

    try:
        with Image.open(image_path) as image:
            image_size = (image.width, image.height)
    except OSError as exc:
        raise ToolInputError(
            "could not read crop metadata image: "
            f"metadata_path={metadata_path}; image_role={image_role}; "
            f"image_path={image_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return image_size


def expand_crop_mask_to_source_frame(
    crop_mask: npt.NDArray[np.bool_], metadata: CropMetadata
) -> npt.NDArray[np.bool_]:
    """Embed a crop-sized mask into the full source frame mask coordinates."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolInputError(
            "numpy is required to expand crop masks to source-frame masks; install "
            "the 'vision' extra"
        ) from exc

    left, top, right, bottom = metadata.crop_bbox_xyxy
    expected_shape = (bottom - top, right - left)
    if crop_mask.shape != expected_shape:
        raise ToolInputError(
            "crop mask shape does not match crop metadata bbox: "
            f"mask_shape={crop_mask.shape}; expected_shape={expected_shape}; "
            f"metadata_crop_image_path={metadata.crop_image_path}"
        )
    full_frame_mask = np.zeros(
        (metadata.source_image_height, metadata.source_image_width),
        dtype=np.bool_,
    )
    full_frame_mask[top:bottom, left:right] = crop_mask.astype(np.bool_, copy=False)
    return full_frame_mask


__all__ = [
    "CropMetadata",
    "CropMetadataPayload",
    "crop_metadata_path_for_image",
    "expand_crop_mask_to_source_frame",
    "load_crop_metadata_for_image",
    "write_crop_metadata",
]
