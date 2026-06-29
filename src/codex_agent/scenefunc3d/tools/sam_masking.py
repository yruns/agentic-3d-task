"""Data contracts and sidecar integration for SceneFunc3D SAM masks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    StrictStr,
    StringConstraints,
    model_validator,
)

from ...errors import SceneFunc3dDataError
from .crop_metadata import (
    expand_crop_mask_to_source_frame,
    load_crop_metadata_for_image,
)
from .models import ToolInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from PIL.Image import Image as PillowImage

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SafePathComponentText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]
StrictPixelCoordinate = Annotated[float, Field(ge=0.0, strict=True)]
_SAFE_PATH_COMPONENT_RE: Pattern[str] = re.compile(r"^[A-Za-z0-9_-]+$")
_MASK_OVERLAY_ALPHA = 96
_CONTACT_LABEL_HEIGHT_PX = 24
_CONTACT_LABEL_PADDING_PX = 4
_JPEG_QUALITY = 95


class SamCandidatePayload(TypedDict):
    """JSON-ready payload for one SAM mask candidate."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    mask_npz_path: str
    overlay_path: str


class SamMaskResultPayload(TypedDict):
    """JSON-ready payload for SAM mask candidate results."""

    frame_id: str
    candidates: list[SamCandidatePayload]
    contact_sheet_path: str


class SamMaskArgsPayload(TypedDict):
    """JSON-ready payload for a future SAM mask request."""

    frame_id: str
    image_path: str
    points: list[SamPointInputPayload]


class SamPointInputPayload(TypedDict):
    """JSON-ready payload for one SAM point prompt."""

    x_px: float
    y_px: float
    source: str
    label: str


class SamPointInput(BaseModel):
    """Strict JSON-boundary point input for future SAM mask generation."""

    model_config = ConfigDict(extra="forbid")

    x_px: StrictPixelCoordinate
    y_px: StrictPixelCoordinate
    source: StrictStr = ""
    label: StrictStr = ""

    def to_payload(self) -> SamPointInputPayload:
        """Return this point prompt as a JSON-ready mapping."""
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "source": self.source,
            "label": self.label,
        }


class SamMaskArgs(BaseModel):
    """Arguments for a SAM mask candidate tool."""

    model_config = ConfigDict(extra="forbid")

    frame_id: SafePathComponentText
    image_path: FilePath
    points: tuple[SamPointInput, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_agent_aliases(cls, payload: object) -> object:
        """Accept common single-point payloads emitted by the agent."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        if "points" not in values and "point" in values:
            values["points"] = (values.pop("point"),)
        values.pop("image_width", None)
        values.pop("image_height", None)
        values.pop("point_label", None)
        values.pop("point_source", None)
        values.pop("crop_metadata_path", None)
        return values

    def to_payload(self) -> SamMaskArgsPayload:
        """Return this request as a JSON-ready mapping."""
        return {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
            "points": [point.to_payload() for point in self.points],
        }


@dataclass(frozen=True)
class SamBackendConfig:
    """Explicit SAM backend selection."""

    model_name: str
    checkpoint_path: Path

    def __post_init__(self) -> None:
        """Validate backend identity before any heavy loading attempt."""
        if not self.model_name.strip():
            raise ToolInputError("SAM backend model_name must not be empty")


def run_sam_backend(config: SamBackendConfig) -> None:
    """Validate configured SAM backend before heavy model loading."""
    if not config.checkpoint_path.exists():
        raise ToolInputError(
            f"SAM backend unavailable: {config.model_name} at {config.checkpoint_path}"
        )
    raise ToolInputError(
        "SAM backend execution is not configured: "
        f"{config.model_name} at {config.checkpoint_path}"
    )


@dataclass(frozen=True)
class SamCandidate:
    """One SAM mask candidate proposed from a Molmo point prompt."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    mask_npz_path: Path
    overlay_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed candidate contracts."""
        if not self.candidate_id.strip():
            raise SceneFunc3dDataError("candidate_id must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise SceneFunc3dDataError(
                f"candidate score must be in [0, 1]; got {self.score!r}"
            )
        if self.pixel_count < 0:
            raise SceneFunc3dDataError(
                f"candidate pixel_count must be non-negative; got {self.pixel_count!r}"
            )
        if not 0.0 <= self.coverage_percent <= 100.0:
            raise SceneFunc3dDataError(
                "candidate coverage_percent must be in [0, 100]; "
                f"got {self.coverage_percent!r}"
            )

    def to_payload(self) -> SamCandidatePayload:
        """Return this candidate as a JSON-ready mapping."""
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "pixel_count": self.pixel_count,
            "coverage_percent": self.coverage_percent,
            "mask_npz_path": str(self.mask_npz_path),
            "overlay_path": str(self.overlay_path),
        }


@dataclass(frozen=True)
class SamMaskResult:
    """SAM mask candidates and rendered contact-sheet artifact for one frame."""

    frame_id: str
    candidates: tuple[SamCandidate, ...]
    contact_sheet_path: Path

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        return {
            "frame_id": self.frame_id,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "contact_sheet_path": str(self.contact_sheet_path),
        }


@dataclass(frozen=True)
class _ContactSheetEntry:
    """One labeled overlay tile for SAM candidate review."""

    candidate_id: str
    overlay_path: Path


def sam_mask(
    args: SamMaskArgs,
    *,
    out_dir: Path,
    backend_config_path: Path | None,
) -> SamMaskResult:
    """Call SAM sidecar, validate masks, and write overlay artifacts."""
    if backend_config_path is None:
        raise ToolInputError("sam_mask backend config is required")

    from codex_agent.scenefunc3d.backends.config import (
        ensure_path_under_roots,
        load_backend_settings,
    )
    from codex_agent.scenefunc3d.backends.sam_rpc import request_sam_masks

    settings = load_backend_settings(backend_config_path)
    artifact_out_dir = ensure_path_under_roots(
        out_dir,
        roots=settings.allowed_output_roots,
        field_name="out_dir",
    )
    frame_artifact_dir = artifact_out_dir / "sam" / args.frame_id
    crop_metadata = load_crop_metadata_for_image(
        args.image_path,
        expected_frame_id=args.frame_id,
        allowed_image_roots=settings.allowed_image_roots,
    )
    staging_dir = frame_artifact_dir / "candidates"
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolInputError(
            "could not create SAM staging directory: "
            f"staging_dir={staging_dir}; error_type={exc.__class__.__name__}"
        ) from exc
    request_id = f"{args.frame_id}_sam"
    response = request_sam_masks(
        settings,
        request_id=request_id,
        args=args,
        staging_dir=staging_dir,
    )

    candidates: list[SamCandidate] = []
    contact_sheet_entries: list[_ContactSheetEntry] = []
    for candidate_response in response.candidates:
        safe_candidate_id = _safe_path_component(candidate_response.candidate_id)
        mask_npz_path = ensure_path_under_roots(
            candidate_response.mask_npz_path,
            roots=settings.allowed_output_roots,
            field_name="mask_npz_path",
        )
        boolean_mask = _load_boolean_mask(mask_npz_path)
        candidate_mask_npz_path = mask_npz_path
        candidate_pixel_count = candidate_response.pixel_count
        candidate_coverage_percent = candidate_response.coverage_percent
        if crop_metadata is not None:
            full_frame_mask = expand_crop_mask_to_source_frame(
                boolean_mask, crop_metadata
            )
            candidate_mask_npz_path = _write_boolean_mask_npz(
                staging_dir / f"{safe_candidate_id}_full_frame.npz",
                full_frame_mask,
            )
            candidate_pixel_count = _mask_pixel_count(full_frame_mask)
            candidate_coverage_percent = _mask_coverage_percent(full_frame_mask)
        overlay_path = _write_candidate_overlay(
            image_path=args.image_path,
            mask=boolean_mask,
            overlay_path=(frame_artifact_dir / f"{safe_candidate_id}_overlay.jpg"),
        )
        contact_sheet_entries.append(
            _ContactSheetEntry(
                candidate_id=candidate_response.candidate_id,
                overlay_path=overlay_path,
            )
        )
        candidates.append(
            SamCandidate(
                candidate_id=candidate_response.candidate_id,
                score=candidate_response.score,
                pixel_count=candidate_pixel_count,
                coverage_percent=candidate_coverage_percent,
                mask_npz_path=candidate_mask_npz_path,
                overlay_path=overlay_path,
            )
        )

    contact_sheet_path = _write_contact_sheet(
        contact_sheet_entries,
        contact_sheet_path=frame_artifact_dir / "contact_sheet.jpg",
    )
    return SamMaskResult(
        frame_id=args.frame_id,
        candidates=tuple(candidates),
        contact_sheet_path=contact_sheet_path,
    )


def _load_boolean_mask(mask_npz_path: Path) -> npt.NDArray[np.bool_]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolInputError(
            "numpy is required to load SAM mask npz artifacts; install the "
            "'vision' extra"
        ) from exc

    try:
        with np.load(mask_npz_path) as archive:
            if "mask" not in archive.files:
                raise ToolInputError(
                    "SAM mask npz is missing required key 'mask': "
                    f"path={mask_npz_path}"
                )
            mask_array = archive["mask"]
    except ToolInputError:
        raise
    except (OSError, ValueError) as exc:
        raise ToolInputError(
            "could not load SAM mask npz artifact: "
            f"path={mask_npz_path}; error_type={exc.__class__.__name__}"
        ) from exc

    if mask_array.ndim != 2:
        raise ToolInputError(
            "SAM mask array must be 2D: "
            f"path={mask_npz_path}; ndim={mask_array.ndim}"
        )
    try:
        boolean_mask: npt.NDArray[np.bool_] = mask_array.astype(np.bool_, copy=False)
        return boolean_mask
    except (TypeError, ValueError) as exc:
        raise ToolInputError(
            "SAM mask array could not be converted to boolean: "
            f"path={mask_npz_path}; error_type={exc.__class__.__name__}"
        ) from exc


def _write_boolean_mask_npz(mask_npz_path: Path, mask: npt.NDArray[np.bool_]) -> Path:
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolInputError(
            "numpy is required to write SAM mask npz artifacts; install the "
            "'vision' extra"
        ) from exc

    try:
        mask_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_npz_path, mask=mask.astype(np.bool_, copy=False))
    except OSError as exc:
        raise ToolInputError(
            "could not write SAM full-frame mask npz artifact: "
            f"path={mask_npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return mask_npz_path


def _mask_pixel_count(mask: npt.NDArray[np.bool_]) -> int:
    return int(mask.sum())


def _mask_coverage_percent(mask: npt.NDArray[np.bool_]) -> float:
    if mask.size <= 0:
        raise ToolInputError("SAM mask coverage is undefined for an empty mask array")
    return float(_mask_pixel_count(mask) * 100.0 / mask.size)


def _write_candidate_overlay(
    *,
    image_path: Path,
    mask: npt.NDArray[np.bool_],
    overlay_path: Path,
) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render SAM mask overlays; install the "
            "'vision' extra"
        ) from exc

    try:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            base_image = image.convert("RGBA")
        if mask.shape != (base_image.height, base_image.width):
            raise ToolInputError(
                "SAM mask shape must match image dimensions: "
                f"image_path={image_path}; mask_shape={mask.shape}; "
                f"image_size={(base_image.width, base_image.height)}"
            )
        mask_image = Image.fromarray(mask.astype("uint8") * 255)
        color_layer = Image.new("RGBA", base_image.size, (255, 0, 0, 0))
        color_layer.putalpha(
            mask_image.point(
                lambda pixel_value: _MASK_OVERLAY_ALPHA if pixel_value else 0
            )
        )
        base_image.alpha_composite(color_layer)
        base_image.convert("RGB").save(
            overlay_path, format="JPEG", quality=_JPEG_QUALITY
        )
    except OSError as exc:
        raise ToolInputError(
            "could not render SAM mask overlay: "
            f"image_path={image_path}; overlay_path={overlay_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return overlay_path


def _write_contact_sheet(
    entries: list[_ContactSheetEntry],
    *,
    contact_sheet_path: Path,
) -> Path:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render SAM contact sheets; install the "
            "'vision' extra"
        ) from exc

    try:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        if not entries:
            raise ToolInputError(
                "cannot render SAM contact sheet without mask candidates: "
                f"contact_sheet_path={contact_sheet_path}"
            )

        images: list[PillowImage] = []
        try:
            for entry in entries:
                with Image.open(entry.overlay_path) as image:
                    images.append(image.convert("RGB"))
            width = max(image.width for image in images)
            overlay_height = max(image.height for image in images)
            sheet = Image.new(
                "RGB",
                (width * len(images), overlay_height + _CONTACT_LABEL_HEIGHT_PX),
                color=(255, 255, 255),
            )
            draw = ImageDraw.Draw(sheet)
            for index, image in enumerate(images):
                left = index * width
                sheet.paste(image, (left, 0))
                label_top = overlay_height
                draw.rectangle(
                    (
                        left,
                        label_top,
                        left + width,
                        label_top + _CONTACT_LABEL_HEIGHT_PX,
                    ),
                    fill=(245, 245, 245),
                    outline=(180, 180, 180),
                )
                draw.text(
                    (
                        left + _CONTACT_LABEL_PADDING_PX,
                        label_top + _CONTACT_LABEL_PADDING_PX,
                    ),
                    entries[index].candidate_id,
                    fill=(0, 0, 0),
                )
            sheet.save(contact_sheet_path, format="JPEG", quality=_JPEG_QUALITY)
        finally:
            for image in images:
                image.close()
    except OSError as exc:
        raise ToolInputError(
            "could not render SAM contact sheet: "
            f"contact_sheet_path={contact_sheet_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return contact_sheet_path


def _safe_path_component(value: str) -> str:
    if not _SAFE_PATH_COMPONENT_RE.fullmatch(value):
        raise ToolInputError(
            "SAM candidate_id must be a safe path component: " f"candidate_id={value!r}"
        )
    return value


__all__ = [
    "SamBackendConfig",
    "SamCandidate",
    "SamCandidatePayload",
    "SamMaskArgs",
    "SamMaskArgsPayload",
    "SamMaskResult",
    "SamMaskResultPayload",
    "SamPointInput",
    "SamPointInputPayload",
    "run_sam_backend",
    "sam_mask",
]
