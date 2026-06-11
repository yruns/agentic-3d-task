"""CLI entry point for the OpenEQA agent tools.

Usage (run by the agent inside the sandbox)::

    python -m codex_agent.openeqa.tools <tool> \\
        --scene-dir <clip_dir> --args '<json>' [--out-dir <scratch>]

where ``<clip_dir>`` is ``<data_root>/<clip_id>`` (the directory holding both the
``raw/`` first-person frames and the ``conceptgraph/`` pack).

``list_objects`` prints a compact JSON object on stdout. The image tools
(``view_frame``, ``keyframe_selector``, ``view_bev``) additionally write
downsized PNG/JPEGs under ``--out-dir`` and include their ``image_path`` in the
JSON, which the agent then opens with the built-in ``view_image`` tool.

Recoverable problems (a bad frame id, an empty query, a missing object id) are
reported as ``{"error": ...}`` on stdout with exit code 0 so the agent can adjust
and continue; missing/corrupt scene assets fail loudly with exit code 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...errors import CodexAgentError
from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError
from .scene_context import OpenEqaToolScene

_DEFAULT_OUT_DIR = Path("tmp") / "openeqa_tool_scratch"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex_agent.openeqa.tools", description=__doc__
    )
    parser.add_argument("tool", choices=list(TOOL_NAMES), help="Tool to run.")
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Prepared clip directory (<data_root>/<clip_id>).",
    )
    parser.add_argument(
        "--args",
        default="{}",
        help="Tool arguments as a single JSON object (default: '{}').",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Writable scratch dir for images (default: ./tmp/openeqa_tool_scratch).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else _DEFAULT_OUT_DIR

    try:
        raw_args = _parse_args_json(args.args)
        tool_scene = OpenEqaToolScene.load(args.scene_dir)
        payload = run_tool(tool_scene, args.tool, raw_args, out_dir=out_dir)
    except ToolInputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")

    print(json.dumps(payload.to_payload(), ensure_ascii=False))
    return 0


def _parse_args_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolInputError("--args must be a JSON object, e.g. '{\"frame_id\": 120}'")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
