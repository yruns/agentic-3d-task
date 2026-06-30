"""CLI entry point for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ...errors import CodexAgentError
from .dispatch import TOOL_NAMES, run_tool
from .mask_artifacts import (
    SceneFunc3dToolInvocationEvent,
    ToolEventStatus,
    append_tool_invocation_event,
)
from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

_DEFAULT_OUT_DIR = Path("tmp") / "scenefunc3d_tool_scratch"
_CHECKPOINT_ALLOWED_NEXT_TOOLS: Mapping[str, frozenset[str]] = {
    "molmo_point": frozenset(("sam_mask", "molmo_point")),
    "sam_mask": frozenset(("lift_mask_to_3d", "sam_mask")),
    "lift_mask_to_3d": frozenset(("inspect_mask_artifact",)),
    "suggest_additional_views": frozenset(
        ("view_crop", "view_frame", "molmo_point", "fuse_accepted_masks")
    ),
}
_CHECKPOINT_TOOL_NAMES = frozenset(
    (*_CHECKPOINT_ALLOWED_NEXT_TOOLS.keys(), "inspect_mask_artifact")
)
_MULTIVIEW_FOLLOWUP_EVIDENCE_TOOL_NAMES = frozenset(("view_crop", "view_frame"))
_MULTIVIEW_PROGRESS_TOOL_NAMES = frozenset(("molmo_point", "fuse_accepted_masks"))
_MAX_MULTIVIEW_EVIDENCE_EVENTS_BEFORE_POINT = 4
_IDENTICAL_RETRY_GUARDED_TOOL_NAMES = frozenset(
    ("molmo_point", "sam_mask", "lift_mask_to_3d")
)
_MAX_SUCCESSFUL_MOLMO_CALLS_BEFORE_SAM = 3
_MAX_SUCCESSFUL_INSPECTIONS_BEFORE_FUSE = 2


@dataclass(frozen=True)
class _SuccessfulToolEvent:
    tool_name: str
    args: dict[str, object]


@dataclass(frozen=True)
class _FollowupEvidenceSignature:
    tool_name: str
    args_json: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SceneFunc3D tools CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="codex_agent.scenefunc3d.tools", description=__doc__
    )
    parser.add_argument("tool", choices=list(TOOL_NAMES), help="Tool to run.")
    parser.add_argument("--scene-root", required=True, type=Path, help="Scene root.")
    parser.add_argument(
        "--args",
        default="{}",
        help="Tool arguments as a single JSON object.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Writable scratch directory for rendered images.",
    )
    parser.add_argument(
        "--backend-config",
        type=Path,
        default=None,
        help="Optional SceneFunc3D sidecar backend TOML config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one SceneFunc3D CLI tool and print a compact JSON result."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else _DEFAULT_OUT_DIR
    backend_config_path = cast(Path | None, args.backend_config)
    raw_args: dict[str, object] = {}
    try:
        raw_args = _parse_args_json(args.args)
        tool_scene = SceneFunc3dToolScene.load(args.scene_root)
        if args.out_dir is not None:
            _validate_tool_sequence(args.tool, raw_args, out_dir)
        payload = run_tool(
            tool_scene,
            args.tool,
            raw_args,
            out_dir=out_dir,
            backend_config_path=backend_config_path,
        )
        payload_data = payload.to_payload()
        _append_tool_event_or_exit(
            parser,
            out_dir=out_dir,
            event=SceneFunc3dToolInvocationEvent(
                tool_name=args.tool,
                status=ToolEventStatus.SUCCESS,
                args=raw_args,
                result=payload_data,
                error="",
            ),
        )
    except ToolInputError as exc:
        _append_tool_event_or_exit(
            parser,
            out_dir=out_dir,
            event=SceneFunc3dToolInvocationEvent(
                tool_name=args.tool,
                status=ToolEventStatus.FAILED,
                args=raw_args,
                result=None,
                error=str(exc),
            ),
        )
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    print(json.dumps(payload_data, ensure_ascii=False))
    return 0


def _append_tool_event_or_exit(
    parser: argparse.ArgumentParser,
    *,
    out_dir: Path,
    event: SceneFunc3dToolInvocationEvent,
) -> None:
    try:
        append_tool_invocation_event(out_dir, event)
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


def _parse_args_json(raw: str) -> dict[str, object]:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolInputError("--args must be a JSON object")
    return _copy_json_object(cast(Mapping[object, object], parsed))


def _validate_tool_sequence(
    tool_name: str, raw_args: Mapping[str, object], out_dir: Path
) -> None:
    events_path = out_dir / "events.jsonl"
    _validate_identical_checkpoint_retry(tool_name, raw_args, events_path)
    _validate_molmo_retry_budget(tool_name, events_path)
    _validate_repeated_inspection(tool_name, raw_args, events_path)
    _validate_inspection_budget(tool_name, events_path)
    _validate_repeated_multiview_followup_evidence(tool_name, raw_args, events_path)
    _validate_multiview_followup_sequence(tool_name, events_path)
    checkpoint_tool_name = _last_successful_checkpoint_tool_name(events_path)
    if checkpoint_tool_name is None:
        return
    allowed_tool_names = _CHECKPOINT_ALLOWED_NEXT_TOOLS.get(checkpoint_tool_name)
    if allowed_tool_names is None or tool_name in allowed_tool_names:
        _validate_checkpoint_tool_args(checkpoint_tool_name, tool_name, raw_args)
        return
    allowed = ", ".join(sorted(allowed_tool_names))
    progress_hint = _checkpoint_progress_hint(checkpoint_tool_name, tool_name)
    message = (
        f"after a successful {checkpoint_tool_name} call, the next SceneFunc3D "
        f"CLI tool must be one of: {allowed}; got {tool_name}. If rejecting the "
        "checkpoint result, retry the same checkpoint tool with changed evidence "
        "or arguments; otherwise continue to the required next approval gate."
    )
    if progress_hint:
        message = f"{message} {progress_hint}"
    raise ToolInputError(message)


def _checkpoint_progress_hint(checkpoint_tool_name: str, tool_name: str) -> str:
    if checkpoint_tool_name == "sam_mask" and tool_name == "inspect_mask_artifact":
        return (
            "After sam_mask, call lift_mask_to_3d first by copying candidate_id "
            "and mask_npz_path from the selected sam_mask candidate; "
            "inspect_mask_artifact only accepts mask_npz_path, mask_ply_path, "
            "and lift_overlay_path returned by lift_mask_to_3d."
        )
    return ""


def _validate_multiview_followup_sequence(tool_name: str, events_path: Path) -> None:
    successful_tool_names = _successful_tool_names(events_path)
    latest_suggest_index = _latest_tool_index(
        successful_tool_names, "suggest_additional_views"
    )
    if latest_suggest_index is None:
        return
    tool_names_after_suggest = successful_tool_names[latest_suggest_index + 1 :]
    if _has_multiview_progress(tool_names_after_suggest) or (
        tool_name in _MULTIVIEW_PROGRESS_TOOL_NAMES
    ):
        return
    evidence_count = sum(
        1
        for successful_tool_name in tool_names_after_suggest
        if successful_tool_name in _MULTIVIEW_FOLLOWUP_EVIDENCE_TOOL_NAMES
    )
    if (
        evidence_count < _MAX_MULTIVIEW_EVIDENCE_EVENTS_BEFORE_POINT
        or tool_name not in _MULTIVIEW_FOLLOWUP_EVIDENCE_TOOL_NAMES
    ):
        return
    raise ToolInputError(
        "after successful suggest_additional_views follow-up evidence, call "
        "molmo_point on a selected follow-up crop or frame, or call "
        "fuse_accepted_masks with rejected_suggested_frame_ids; do not keep "
        "opening more follow-up views before Molmo or fusion"
    )


def _validate_repeated_multiview_followup_evidence(
    tool_name: str, raw_args: Mapping[str, object], events_path: Path
) -> None:
    current_signature = _followup_evidence_signature(tool_name, raw_args)
    if current_signature is None:
        return
    successful_tool_events = _successful_tool_events(events_path)
    successful_tool_names = tuple(event.tool_name for event in successful_tool_events)
    latest_suggest_index = _latest_tool_index(
        successful_tool_names, "suggest_additional_views"
    )
    if latest_suggest_index is None:
        return
    tool_names_after_suggest = successful_tool_names[latest_suggest_index + 1 :]
    if _has_multiview_progress(tool_names_after_suggest):
        return
    for event in successful_tool_events[latest_suggest_index + 1 :]:
        prior_signature = _followup_evidence_signature(event.tool_name, event.args)
        if prior_signature == current_signature:
            raise ToolInputError(
                f"do not repeat follow-up {tool_name} with identical arguments "
                "after successful suggest_additional_views; change frame or crop, "
                "or call molmo_point on the selected evidence, or call "
                "fuse_accepted_masks with rejected_suggested_frame_ids"
            )


def _followup_evidence_signature(
    tool_name: str, raw_args: Mapping[str, object]
) -> _FollowupEvidenceSignature | None:
    if tool_name not in _MULTIVIEW_FOLLOWUP_EVIDENCE_TOOL_NAMES:
        return None
    args_json = json.dumps(
        dict(raw_args),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _FollowupEvidenceSignature(tool_name=tool_name, args_json=args_json)


def _validate_checkpoint_tool_args(
    checkpoint_tool_name: str, tool_name: str, raw_args: Mapping[str, object]
) -> None:
    if checkpoint_tool_name != "lift_mask_to_3d":
        return
    if tool_name != "inspect_mask_artifact":
        return
    if "lift_overlay_path" in raw_args:
        return
    raise ToolInputError(
        "after a successful lift_mask_to_3d call, inspect_mask_artifact args "
        "must include lift_overlay_path returned by lift_mask_to_3d"
    )


def _validate_repeated_inspection(
    tool_name: str, raw_args: Mapping[str, object], events_path: Path
) -> None:
    if tool_name != "inspect_mask_artifact":
        return
    if _last_successful_checkpoint_tool_name(events_path) != "inspect_mask_artifact":
        return
    prior_inspection_args = _last_successful_tool_args(
        events_path, "inspect_mask_artifact"
    )
    if prior_inspection_args == dict(raw_args):
        return
    raise ToolInputError(
        "do not call inspect_mask_artifact again without a new "
        "lift_mask_to_3d result; call lift_mask_to_3d for a different "
        "candidate if rejecting the inspected lift, or call "
        "suggest_additional_views or fuse_accepted_masks if accepting it"
    )


def _validate_inspection_budget(tool_name: str, events_path: Path) -> None:
    if tool_name == "fuse_accepted_masks":
        return
    inspection_count = sum(
        1
        for successful_tool_name in _successful_tool_names(events_path)
        if successful_tool_name == "inspect_mask_artifact"
    )
    if inspection_count < _MAX_SUCCESSFUL_INSPECTIONS_BEFORE_FUSE:
        return
    raise ToolInputError(
        "after 2 successful inspect_mask_artifact calls, call "
        "fuse_accepted_masks with the accepted fragments instead of searching "
        "more views or masks"
    )


def _validate_identical_checkpoint_retry(
    tool_name: str, raw_args: Mapping[str, object], events_path: Path
) -> None:
    if tool_name not in _IDENTICAL_RETRY_GUARDED_TOOL_NAMES:
        return
    prior_tool_args = _last_successful_tool_args(events_path, tool_name)
    if prior_tool_args is None or dict(raw_args) != prior_tool_args:
        return
    next_tool_names = _CHECKPOINT_ALLOWED_NEXT_TOOLS.get(tool_name, frozenset())
    progress_tool_names = sorted(next_tool_names - frozenset((tool_name,)))
    progress_hint = _identical_retry_progress_hint(
        tool_name, events_path, progress_tool_names
    )
    raise ToolInputError(
        f"do not repeat {tool_name} with identical arguments after a successful "
        f"{tool_name} call; change evidence or arguments, or call {progress_hint}"
    )


def _identical_retry_progress_hint(
    tool_name: str, events_path: Path, progress_tool_names: list[str]
) -> str:
    if tool_name == "lift_mask_to_3d" and _has_successful_tool_after_latest_tool(
        events_path,
        target_tool_name="lift_mask_to_3d",
        later_tool_name="inspect_mask_artifact",
    ):
        return (
            "suggest_additional_views or fuse_accepted_masks, or change "
            "candidate_id/mask_npz_path if rejecting the inspected lift"
        )
    return ", ".join(progress_tool_names)


def _has_successful_tool_after_latest_tool(
    events_path: Path, *, target_tool_name: str, later_tool_name: str
) -> bool:
    successful_tool_names = _successful_tool_names(events_path)
    target_index = _latest_tool_index(successful_tool_names, target_tool_name)
    later_index = _latest_tool_index(successful_tool_names, later_tool_name)
    if target_index is None or later_index is None:
        return False
    return later_index > target_index


def _validate_molmo_retry_budget(tool_name: str, events_path: Path) -> None:
    if tool_name != "molmo_point":
        return
    molmo_success_count = _trailing_successful_tool_count(
        _successful_tool_names(events_path), "molmo_point"
    )
    if molmo_success_count < _MAX_SUCCESSFUL_MOLMO_CALLS_BEFORE_SAM:
        return
    raise ToolInputError(
        "after 3 successful molmo_point calls before SAM, call sam_mask with "
        "the best approved point_xy instead of trying another Molmo point"
    )


def _successful_tool_names(events_path: Path) -> tuple[str, ...]:
    return tuple(event.tool_name for event in _successful_tool_events(events_path))


def _successful_tool_events(events_path: Path) -> tuple[_SuccessfulToolEvent, ...]:
    if not events_path.is_file():
        return ()
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ToolInputError(
            f"could not read SceneFunc3D tool sequence state: {events_path}"
        ) from exc
    successful_tool_events: list[_SuccessfulToolEvent] = []
    for line_number, raw_line in enumerate(event_lines, start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        try:
            event: object = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            raise ToolInputError(
                "could not read SceneFunc3D tool sequence state: invalid JSON "
                f"in {events_path} line {line_number}"
            ) from exc
        if not isinstance(event, dict):
            raise ToolInputError(
                "could not read SceneFunc3D tool sequence state: event line "
                f"{line_number} is not a JSON object"
            )
        tool_value = event.get("tool_name")
        if (
            event.get("event_type") == "tool_completed"
            and event.get("status") == "success"
            and isinstance(tool_value, str)
        ):
            args_value = event.get("args")
            successful_tool_events.append(
                _SuccessfulToolEvent(
                    tool_name=tool_value,
                    args=_copy_event_args(args_value),
                )
            )
    return tuple(successful_tool_events)


def _latest_tool_index(
    tool_names: tuple[str, ...], target_tool_name: str
) -> int | None:
    for index in range(len(tool_names) - 1, -1, -1):
        if tool_names[index] == target_tool_name:
            return index
    return None


def _trailing_successful_tool_count(
    tool_names: tuple[str, ...], target_tool_name: str
) -> int:
    count = 0
    for tool_name in reversed(tool_names):
        if tool_name != target_tool_name:
            return count
        count += 1
    return count


def _has_multiview_progress(tool_names: tuple[str, ...]) -> bool:
    return any(tool_name in _MULTIVIEW_PROGRESS_TOOL_NAMES for tool_name in tool_names)


def _last_successful_checkpoint_tool_name(events_path: Path) -> str | None:
    for event in reversed(_successful_tool_events(events_path)):
        if event.tool_name in _CHECKPOINT_TOOL_NAMES:
            return event.tool_name
    return None


def _last_successful_tool_args(
    events_path: Path, target_tool_name: str
) -> dict[str, object] | None:
    for event in reversed(_successful_tool_events(events_path)):
        if event.tool_name == target_tool_name:
            return event.args
    return None


def _copy_event_args(raw_args: object) -> dict[str, object]:
    if raw_args is None:
        return {}
    if not isinstance(raw_args, dict):
        raise ToolInputError(
            "could not read SceneFunc3D tool sequence state: successful tool "
            "event args must be a JSON object"
        )
    return _copy_json_object(cast(Mapping[object, object], raw_args))


def _copy_json_object(raw_args: Mapping[object, object]) -> dict[str, object]:
    copied_args: dict[str, object] = {}
    for key, value in raw_args.items():
        if not isinstance(key, str):
            raise ToolInputError("--args object keys must be strings")
        copied_args[key] = value
    return copied_args


if __name__ == "__main__":
    raise SystemExit(main())
