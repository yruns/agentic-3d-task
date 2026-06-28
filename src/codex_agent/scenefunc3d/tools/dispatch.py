"""SceneFunc3D tool dispatcher with typed argument validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolInputError, ToolPayload
from .scene_context import SceneFunc3dToolScene

TOOL_NAMES: tuple[str, ...] = ("scene_summary",)

_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def run_tool(
    tool_scene: SceneFunc3dToolScene,
    name: str,
    raw_args: Mapping[str, object],
    *,
    out_dir: Path,
) -> ToolPayload:
    """Validate ``raw_args`` for ``name`` and return the typed tool result."""
    _ = out_dir
    if name == "scene_summary":
        from .frame_views import SceneSummaryArgs, scene_summary

        return scene_summary(tool_scene, _parse(SceneSummaryArgs, raw_args))
    raise ToolInputError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")


def _parse(model: type[_ArgsT], raw_args: Mapping[str, object]) -> _ArgsT:
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
