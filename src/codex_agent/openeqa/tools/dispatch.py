"""Tool dispatcher: validate arguments and route to the right handler.

Heavy / optional-dependency tool modules (Pillow for frame downsizing, the
ConceptGraph selector + OpenCV for ``list_objects`` / ``view_bev`` /
``keyframe_selector``) are imported lazily inside each branch so invoking a tool
never pays for imports it does not use.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolInputError, ToolPayload
from .scene_context import OpenEqaToolScene

#: Frame-switching/keyframe/BEV tools write images; ``list_objects`` is text-only.
TOOL_NAMES: tuple[str, ...] = (
    "list_objects",
    "keyframe_selector",
    "view_frame",
    "view_bev",
)

_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def run_tool(
    tool_scene: OpenEqaToolScene,
    name: str,
    raw_args: Mapping[str, Any],
    *,
    out_dir: Path,
) -> ToolPayload:
    """Validate ``raw_args`` for ``name`` and return the typed tool result.

    Raises:
        ToolInputError: Unknown tool name or invalid/recoverable arguments.
    """
    if name == "list_objects":
        from .object_tools import ListObjectsArgs, list_objects

        return list_objects(tool_scene, _parse(ListObjectsArgs, raw_args))
    if name == "view_frame":
        from .frame_tools import ViewFrameArgs, view_frame

        return view_frame(tool_scene, _parse(ViewFrameArgs, raw_args), out_dir=out_dir)
    if name == "keyframe_selector":
        from .keyframe_retrieval import KeyframeSelectorArgs, keyframe_selector

        return keyframe_selector(
            tool_scene, _parse(KeyframeSelectorArgs, raw_args), out_dir=out_dir
        )
    if name == "view_bev":
        from .bev_tools import ViewBevArgs, view_bev

        return view_bev(tool_scene, _parse(ViewBevArgs, raw_args), out_dir=out_dir)
    raise ToolInputError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")


def _parse(model: type[_ArgsT], raw_args: Mapping[str, Any]) -> _ArgsT:
    try:
        return model.model_validate(dict(raw_args))
    except ValidationError as exc:
        raise ToolInputError(
            f"invalid arguments for {model.__name__}: {_format_validation_error(exc)}"
        ) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


__all__ = ["TOOL_NAMES", "run_tool"]
