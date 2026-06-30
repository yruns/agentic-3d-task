"""Shared Pydantic schemas for SceneFunc3D sidecar HTTP services."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
from pydantic.types import StringConstraints

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class _StrictSchema(BaseModel):
    """Base model for closed SceneFunc3D sidecar request/response schemas."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_StrictSchema):
    """Health check response shared by local sidecar services."""

    status: Literal["ok"]
    model_name: NonEmptyString
    model_loaded: bool


class MolmoPointRequest(_StrictSchema):
    """Request body for the Molmo point sidecar endpoint."""

    request_id: NonEmptyString
    image_path: FilePath
    prompt: NonEmptyString
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)


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
    image_path: FilePath
    points: tuple[SamPointPrompt, ...] = Field(min_length=1)
    staging_dir: Path


class SamMaskCandidateResponse(_StrictSchema):
    """Metadata for one SAM candidate mask written to disk."""

    candidate_id: NonEmptyString
    score: float = Field(ge=0.0, le=1.0)
    mask_npz_path: FilePath
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


__all__ = [
    "HealthResponse",
    "MolmoImagePoint",
    "MolmoPointRequest",
    "MolmoPointResponse",
    "SamMaskCandidateResponse",
    "SamMaskRequest",
    "SamMaskResponse",
    "SamPointPrompt",
]
