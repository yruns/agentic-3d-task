"""Lightweight SceneFunc3D evidence-view tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

_RAW_RGB_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
_ERROR_FRAME_ID_PREVIEW = 8
_JPEG_QUALITY = 90


class SceneSummaryArgs(BaseModel):
    """Arguments for ``scene_summary``."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class SceneSummaryResult:
    """Summary payload for one prepared SceneFunc3D scene."""

    visit_id: str
    total_rgb_frames: int
    rgb_frame_ids: tuple[str, ...]
    has_conceptgraph: bool

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        return {
            "visit_id": self.visit_id,
            "total_rgb_frames": self.total_rgb_frames,
            "rgb_frame_ids": list(self.rgb_frame_ids),
            "has_conceptgraph": self.has_conceptgraph,
        }


def scene_summary(
    tool_scene: SceneFunc3dToolScene, args: SceneSummaryArgs
) -> SceneSummaryResult:
    """Return lightweight filesystem metadata for one SceneFunc3D scene."""
    _ = args
    return SceneSummaryResult(
        visit_id=tool_scene.visit_id,
        total_rgb_frames=len(tool_scene.rgb_frame_ids),
        rgb_frame_ids=tool_scene.rgb_frame_ids,
        has_conceptgraph=tool_scene.conceptgraph_dir.is_dir(),
    )


class ViewFrameArgs(BaseModel):
    """Arguments for ``view_frame``."""

    model_config = ConfigDict(extra="forbid")

    frame_ids: tuple[str, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ViewFrame:
    """One rendered SceneFunc3D RGB evidence frame."""

    frame_id: str
    image_path: Path

    def to_payload(self) -> dict[str, object]:
        """Return this frame as a JSON-ready mapping."""
        return {"frame_id": self.frame_id, "image_path": str(self.image_path)}


@dataclass(frozen=True)
class ViewFrameResult:
    """Rendered RGB evidence frames."""

    frames: tuple[ViewFrame, ...]

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        frame_payloads: list[dict[str, object]] = [
            frame.to_payload() for frame in self.frames
        ]
        return {"frames": frame_payloads}


def view_frame(
    tool_scene: SceneFunc3dToolScene, args: ViewFrameArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Copy requested RGB frames to the writable evidence-image directory."""
    _validate_frame_ids(tool_scene, args.frame_ids)
    frames = tuple(
        ViewFrame(
            frame_id=frame_id,
            image_path=_write_jpeg_copy(
                _resolve_rgb_source(tool_scene, frame_id),
                out_dir / tool_scene.visit_id / f"{frame_id}.jpg",
            ),
        )
        for frame_id in args.frame_ids
    )
    return ViewFrameResult(frames=frames)


def _validate_frame_ids(
    tool_scene: SceneFunc3dToolScene, requested_frame_ids: tuple[str, ...]
) -> None:
    available_frame_ids = set(tool_scene.rgb_frame_ids)
    missing_frame_ids = [
        frame_id
        for frame_id in requested_frame_ids
        if frame_id not in available_frame_ids
    ]
    if missing_frame_ids:
        raise ToolInputError(
            f"frame ids not in this SceneFunc3D scene: {missing_frame_ids}; "
            f"available frame ids include: "
            f"{list(tool_scene.rgb_frame_ids[:_ERROR_FRAME_ID_PREVIEW])} "
            f"({len(tool_scene.rgb_frame_ids)} total)"
        )


def _resolve_rgb_source(tool_scene: SceneFunc3dToolScene, frame_id: str) -> Path:
    conceptgraph_rgb_path = tool_scene.conceptgraph_rgb_vis_dir / f"{frame_id}-rgb.jpg"
    if conceptgraph_rgb_path.is_file():
        return conceptgraph_rgb_path
    for suffix in _RAW_RGB_SUFFIXES:
        raw_rgb_path = tool_scene.raw_dir / f"{frame_id}-rgb{suffix}"
        if raw_rgb_path.is_file():
            return raw_rgb_path
    raise ToolInputError(
        f"no RGB image found for frame id {frame_id!r}; checked "
        f"{conceptgraph_rgb_path} and raw RGB files under {tool_scene.raw_dir}"
    )


def _write_jpeg_copy(source_path: Path, destination_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render SceneFunc3D evidence frames; install the "
            "'vision' extra"
        ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source_path) as image:
            image.convert("RGB").save(
                destination_path, format="JPEG", quality=_JPEG_QUALITY
            )
    except OSError as exc:
        raise ToolInputError(
            f"could not render RGB image {source_path} as JPEG: {exc}"
        ) from exc
    return destination_path


__all__ = [
    "SceneSummaryArgs",
    "SceneSummaryResult",
    "ViewFrame",
    "ViewFrameArgs",
    "ViewFrameResult",
    "scene_summary",
    "view_frame",
]
