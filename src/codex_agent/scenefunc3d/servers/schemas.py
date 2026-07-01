"""Shared Pydantic schemas for SceneFunc3D sidecar HTTP services."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_validator,
    model_validator,
)
from pydantic.types import StringConstraints
from typing_extensions import Self

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
InlineImageMimeType: TypeAlias = Literal["image/jpeg", "image/png", "image/webp"]
ImageSourceKind: TypeAlias = Literal["path", "inline"]
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictSchema(BaseModel):
    """Base model for closed SceneFunc3D sidecar request/response schemas."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_StrictSchema):
    """Health check response shared by local sidecar services."""

    status: Literal["ok"]
    model_name: NonEmptyString
    model_loaded: bool


class InlineImagePayload(_StrictSchema):
    """Base64-encoded image payload for remote SceneFunc3D sidecar requests."""

    filename: NonEmptyString
    mime_type: InlineImageMimeType
    sha256: str
    data_base64: NonEmptyString

    @field_validator("filename")
    @classmethod
    def _require_filename_not_path(cls, value: str) -> str:
        if value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError("filename must be a file name, not a path")
        return value

    @field_validator("sha256")
    @classmethod
    def _require_lower_hex_sha256(cls, value: str) -> str:
        if not _SHA256_HEX_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _require_matching_sha256(self) -> Self:
        decoded_bytes = self.decoded_bytes()
        actual_sha256 = hashlib.sha256(decoded_bytes).hexdigest()
        if actual_sha256 != self.sha256:
            raise ValueError("sha256 must match decoded data_base64 bytes")
        return self

    def decoded_bytes(self) -> bytes:
        """Decode and validate the base64 image bytes."""
        try:
            return base64.b64decode(
                self.data_base64.encode("ascii"),
                validate=True,
            )
        except UnicodeEncodeError as exc:
            raise ValueError("data_base64 must contain ASCII base64 text") from exc
        except binascii.Error as exc:
            raise ValueError("data_base64 must be valid base64") from exc


class MolmoPointRequest(_StrictSchema):
    """Request body for the Molmo point sidecar endpoint."""

    request_id: NonEmptyString
    image_path: FilePath | None = None
    image: InlineImagePayload | None = None
    prompt: NonEmptyString
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def _require_exactly_one_image_source(self) -> Self:
        _require_exactly_one_image_source(
            image_path=self.image_path,
            image=self.image,
        )
        return self

    @property
    def image_source(self) -> ImageSourceKind:
        """Return whether this request uses a path-backed or inline image."""
        if self.image_path is not None:
            return "path"
        return "inline"

    def require_image_path(self) -> Path:
        """Return the path-backed image or raise for an inline request."""
        if self.image_path is None:
            raise ValueError("MolmoPointRequest does not contain image_path")
        return self.image_path

    def require_inline_image(self) -> InlineImagePayload:
        """Return the inline image payload or raise for a path request."""
        if self.image is None:
            raise ValueError("MolmoPointRequest does not contain image")
        return self.image

    def with_image_path(self, image_path: Path) -> MolmoPointRequest:
        """Return a path-backed equivalent request for existing runners."""
        return MolmoPointRequest(
            request_id=self.request_id,
            image_path=image_path,
            prompt=self.prompt,
            image_width=self.image_width,
            image_height=self.image_height,
        )


class MolmoImagePoint(_StrictSchema):
    """One MolmoPoint sidecar point in image pixel coordinates."""

    x_px: float = Field(ge=0.0, strict=True)
    y_px: float = Field(ge=0.0, strict=True)
    source: str = ""
    label: str = ""

    @field_validator("x_px", "y_px", mode="before")
    @classmethod
    def _require_float_coordinate(cls, value: object) -> object:
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("Molmo point coordinates must be finite floats")
        return value


class MolmoPointResponse(_StrictSchema):
    """Response body for the Molmo point sidecar endpoint."""

    request_id: NonEmptyString
    model_name: NonEmptyString
    raw_text: str
    image_points: tuple[MolmoImagePoint, ...] = ()
    latency_ms: float = Field(ge=0)

    @field_validator("latency_ms")
    @classmethod
    def _require_finite_latency(cls, value: float) -> float:
        return _require_finite_float(value, field_name="latency_ms")


class SamPointPrompt(_StrictSchema):
    """One approved 2D point prompt for SAM mask generation."""

    x_px: float = Field(ge=0.0, strict=True)
    y_px: float = Field(ge=0.0, strict=True)
    label: str = ""
    source: str = ""

    @field_validator("x_px", "y_px", mode="before")
    @classmethod
    def _require_float_coordinate(cls, value: object) -> object:
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("SAM point coordinates must be finite floats")
        return value


class SamMaskRequest(_StrictSchema):
    """Request body for the SAM mask sidecar endpoint."""

    request_id: NonEmptyString
    image_path: FilePath | None = None
    image: InlineImagePayload | None = None
    points: tuple[SamPointPrompt, ...] = Field(min_length=1)
    staging_dir: Path | None = None

    @model_validator(mode="after")
    def _require_exactly_one_image_source(self) -> Self:
        _require_exactly_one_image_source(
            image_path=self.image_path,
            image=self.image,
        )
        return self

    @property
    def image_source(self) -> ImageSourceKind:
        """Return whether this request uses a path-backed or inline image."""
        if self.image_path is not None:
            return "path"
        return "inline"

    def require_image_path(self) -> Path:
        """Return the path-backed image or raise for an inline request."""
        if self.image_path is None:
            raise ValueError("SamMaskRequest does not contain image_path")
        return self.image_path

    def require_inline_image(self) -> InlineImagePayload:
        """Return the inline image payload or raise for a path request."""
        if self.image is None:
            raise ValueError("SamMaskRequest does not contain image")
        return self.image

    def require_staging_dir(self) -> Path:
        """Return the staging directory or raise when server staging is needed."""
        if self.staging_dir is None:
            raise ValueError("SamMaskRequest does not contain staging_dir")
        return self.staging_dir

    def with_image_path(
        self,
        image_path: Path,
        *,
        staging_dir: Path | None = None,
    ) -> SamMaskRequest:
        """Return a path-backed equivalent request for existing runners."""
        return SamMaskRequest(
            request_id=self.request_id,
            image_path=image_path,
            points=self.points,
            staging_dir=self.staging_dir if staging_dir is None else staging_dir,
        )

    def with_staging_dir(self, staging_dir: Path) -> SamMaskRequest:
        """Return an equivalent request with an explicit staging directory."""
        return SamMaskRequest(
            request_id=self.request_id,
            image_path=self.image_path,
            image=self.image,
            points=self.points,
            staging_dir=staging_dir,
        )


class SamMaskCandidateResponse(_StrictSchema):
    """Metadata for one SAM candidate mask written to disk."""

    candidate_id: NonEmptyString
    score: float = Field(ge=0.0, le=1.0)
    mask_npz_path: Path
    mask_npz_base64: str = ""
    mask_npz_sha256: str = ""
    pixel_count: int = Field(ge=0, strict=True)
    coverage_percent: float = Field(ge=0.0, le=100.0)


class SamMaskResponse(_StrictSchema):
    """Response body for the SAM mask sidecar endpoint."""

    request_id: NonEmptyString
    model_name: NonEmptyString
    candidates: tuple[SamMaskCandidateResponse, ...]
    latency_ms: float = Field(ge=0)

    @field_validator("latency_ms")
    @classmethod
    def _require_finite_latency(cls, value: float) -> float:
        return _require_finite_float(value, field_name="latency_ms")


def _require_finite_float(value: float, *, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _require_exactly_one_image_source(
    *,
    image_path: Path | None,
    image: InlineImagePayload | None,
) -> None:
    if (image_path is None) == (image is None):
        raise ValueError("request must include exactly one of image_path or image")


__all__ = [
    "HealthResponse",
    "ImageSourceKind",
    "InlineImageMimeType",
    "InlineImagePayload",
    "MolmoImagePoint",
    "MolmoPointRequest",
    "MolmoPointResponse",
    "SamMaskCandidateResponse",
    "SamMaskRequest",
    "SamMaskResponse",
    "SamPointPrompt",
]
