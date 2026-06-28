"""CLI entry point for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ...errors import CodexAgentError
from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

_DEFAULT_OUT_DIR = Path("tmp") / "scenefunc3d_tool_scratch"


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
    try:
        raw_args = _parse_args_json(args.args)
        tool_scene = SceneFunc3dToolScene.load(args.scene_root)
        payload = run_tool(
            tool_scene,
            args.tool,
            raw_args,
            out_dir=out_dir,
            backend_config_path=backend_config_path,
        )
    except ToolInputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    print(json.dumps(payload.to_payload(), ensure_ascii=False))
    return 0


def _parse_args_json(raw: str) -> dict[str, object]:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolInputError("--args must be a JSON object")
    return _copy_json_object(cast(Mapping[object, object], parsed))


def _copy_json_object(raw_args: Mapping[object, object]) -> dict[str, object]:
    copied_args: dict[str, object] = {}
    for key, value in raw_args.items():
        if not isinstance(key, str):
            raise ToolInputError("--args object keys must be strings")
        copied_args[key] = value
    return copied_args


if __name__ == "__main__":
    raise SystemExit(main())
