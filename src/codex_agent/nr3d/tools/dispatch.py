"""Tool dispatcher: validate arguments and route to the right handler.

Heavy / optional-dependency tool modules (OpenCV for the frame and BEV tools,
the Stage-1 selector for ``keyframe_selector``) are imported lazily inside each
branch so that invoking a text tool never pays for vision or model imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..sample import Nr3dScene
from .models import ToolInputError, ToolPayload

TOOL_NAMES: tuple[str, ...] = (
    "inspect_proposal",
    "list_scene_proposals",
    "select_by_proposal",
    "list_frame_proposals",
    "keyframe_selector",
    "mark_frame_with_bbox",
    "compare_proposals_spatial",
    "compare_candidates_to_anchors",
    "view_bev",
)

_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def run_tool(
    scene: Nr3dScene,
    name: str,
    raw_args: Mapping[str, Any],
    *,
    out_dir: Path,
) -> ToolPayload:
    """Validate ``raw_args`` for ``name`` and return the typed tool result.

    Raises:
        ToolInputError: Unknown tool name or invalid/recoverable arguments.
    """
    if name == "inspect_proposal":
        from .catalog_tools import InspectProposalArgs, inspect_proposal

        return inspect_proposal(scene, _parse(InspectProposalArgs, raw_args))
    if name == "list_scene_proposals":
        from .catalog_tools import ListSceneProposalsArgs, list_scene_proposals

        return list_scene_proposals(scene, _parse(ListSceneProposalsArgs, raw_args))
    if name == "select_by_proposal":
        from .frame_tools import SelectByProposalArgs, select_by_proposal

        return select_by_proposal(scene, _parse(SelectByProposalArgs, raw_args))
    if name == "list_frame_proposals":
        from .frame_tools import ListFrameProposalsArgs, list_frame_proposals

        return list_frame_proposals(scene, _parse(ListFrameProposalsArgs, raw_args))
    if name == "keyframe_selector":
        from .keyframe_retrieval import KeyframeSelectorArgs, keyframe_selector

        return keyframe_selector(scene, _parse(KeyframeSelectorArgs, raw_args))
    if name == "mark_frame_with_bbox":
        from .frame_annotation import MarkFrameArgs, mark_frame_with_bbox

        return mark_frame_with_bbox(
            scene, _parse(MarkFrameArgs, raw_args), out_dir=out_dir
        )
    if name == "compare_proposals_spatial":
        from .spatial_tools import (
            CompareProposalsSpatialArgs,
            compare_proposals_spatial,
        )

        return compare_proposals_spatial(
            scene, _parse(CompareProposalsSpatialArgs, raw_args)
        )
    if name == "compare_candidates_to_anchors":
        from .spatial_tools import (
            CompareCandidatesToAnchorsArgs,
            compare_candidates_to_anchors,
        )

        return compare_candidates_to_anchors(
            scene, _parse(CompareCandidatesToAnchorsArgs, raw_args)
        )
    if name == "view_bev":
        from .bev_tools import ViewBevArgs, view_bev

        return view_bev(scene, _parse(ViewBevArgs, raw_args), out_dir=out_dir)
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
