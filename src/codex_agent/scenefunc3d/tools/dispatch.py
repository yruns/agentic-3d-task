"""SceneFunc3D tool dispatcher with typed argument validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolInputError, ToolPayload
from .scene_context import SceneFunc3dToolScene

TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "keyframe_selector",
    "view_frame",
    "view_crop",
    "view_bev",
    "frame_objects",
    "molmo_point",
    "sam_mask",
    "lift_mask_to_3d",
    "inspect_mask_artifact",
    "suggest_additional_views",
    "fuse_accepted_masks",
)

_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def run_tool(
    tool_scene: SceneFunc3dToolScene,
    name: str,
    raw_args: Mapping[str, object],
    *,
    out_dir: Path,
    backend_config_path: Path | None = None,
) -> ToolPayload:
    """Validate ``raw_args`` for ``name`` and return the typed tool result."""
    if name == "scene_summary":
        from .frame_views import SceneSummaryArgs, scene_summary

        return scene_summary(tool_scene, _parse(SceneSummaryArgs, raw_args))
    if name == "view_frame":
        from .frame_views import ViewFrameArgs, view_frame

        return view_frame(tool_scene, _parse(ViewFrameArgs, raw_args), out_dir=out_dir)
    if name == "view_crop":
        from .frame_views import ViewCropArgs, view_crop

        return view_crop(tool_scene, _parse(ViewCropArgs, raw_args), out_dir=out_dir)
    if name == "view_bev":
        from .frame_views import ViewBevArgs, view_bev

        return view_bev(tool_scene, _parse(ViewBevArgs, raw_args), out_dir=out_dir)
    if name == "frame_objects":
        from .frame_views import FrameObjectsArgs, frame_objects

        return frame_objects(tool_scene, _parse(FrameObjectsArgs, raw_args))
    if name == "keyframe_selector":
        from .keyframe_retrieval import KeyframeSelectorArgs, keyframe_selector

        return keyframe_selector(
            tool_scene, _parse(KeyframeSelectorArgs, raw_args), out_dir=out_dir
        )
    if name == "molmo_point":
        from .molmo_pointing import MolmoPointArgs, molmo_point

        return molmo_point(
            _parse(MolmoPointArgs, raw_args),
            out_dir=out_dir,
            backend_config_path=backend_config_path,
        )
    if name == "sam_mask":
        from .sam_masking import SamMaskArgs, sam_mask

        return sam_mask(
            _parse(SamMaskArgs, raw_args),
            out_dir=out_dir,
            backend_config_path=backend_config_path,
        )
    if name == "lift_mask_to_3d":
        from .mask_lifting import LiftMaskArgs, lift_mask_to_3d

        return lift_mask_to_3d(
            _parse(LiftMaskArgs, _with_resolved_lift_geometry(tool_scene, raw_args)),
            out_dir=out_dir,
            raw_mesh_path=tool_scene.raw_mesh_path,
        )
    if name == "inspect_mask_artifact":
        from .mask_inspection import InspectMaskArtifactArgs, inspect_mask_artifact

        return inspect_mask_artifact(_parse(InspectMaskArtifactArgs, raw_args))
    if name == "suggest_additional_views":
        from .mask_inspection import (
            SuggestAdditionalViewsArgs,
            suggest_additional_views,
        )

        return suggest_additional_views(
            tool_scene, _parse(SuggestAdditionalViewsArgs, raw_args)
        )
    if name == "fuse_accepted_masks":
        from .mask_inspection import FuseAcceptedMasksArgs, fuse_accepted_masks

        return fuse_accepted_masks(
            _parse(FuseAcceptedMasksArgs, raw_args), out_dir=out_dir
        )
    raise ToolInputError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")


def _parse(model: type[_ArgsT], raw_args: Mapping[str, object]) -> _ArgsT:
    try:
        return model.model_validate(dict(raw_args))
    except ValidationError as exc:
        raise ToolInputError(
            f"invalid arguments for {model.__name__}: {_format_validation_error(exc)}"
        ) from exc


def _with_resolved_lift_geometry(
    tool_scene: SceneFunc3dToolScene, raw_args: Mapping[str, object]
) -> dict[str, object]:
    values = dict(raw_args)
    if _has_all_lift_geometry_fields(values):
        return values
    frame_id = values.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        return values
    normalized_frame_id = frame_id.strip()

    from ..backends.frame_assets import resolve_frame_geometry_assets

    assets = resolve_frame_geometry_assets(tool_scene, normalized_frame_id)
    values.setdefault("depth_path", str(assets.depth_path))
    values.setdefault("intrinsics_path", str(assets.intrinsics_path))
    values.setdefault("pose_path", str(assets.pose_path))
    return values


def _has_all_lift_geometry_fields(values: Mapping[str, object]) -> bool:
    return all(
        field_name in values
        for field_name in ("depth_path", "intrinsics_path", "pose_path")
    )


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


__all__ = ["TOOL_NAMES", "run_tool"]
