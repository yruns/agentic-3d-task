"""CLI entry point for the NR3D agent tools.

Usage (run by the agent inside the sandbox)::

    python -m codex_agent.nr3d.tools <tool> \\
        --scene-dir <pack_dir> --args '<json>' [--out-dir <scratch>]

Text tools print a compact JSON object on stdout. Frame/BEV tools additionally
write an annotated PNG under ``--out-dir`` and include its ``image_path`` in the
JSON, which the agent then opens with the built-in ``view_image`` tool.

Recoverable problems (bad ids, an unknown relation, an out-of-range frame) are
reported as ``{"error": ...}`` on stdout with exit code 0 so the agent can
adjust and continue; missing/corrupt scene assets fail loudly with exit code 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...errors import CodexAgentError
from ..sample import Nr3dScene
from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError

_DEFAULT_OUT_DIR = Path("tmp") / "nr3d_tool_scratch"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codex_agent.nr3d.tools", description=__doc__)
    parser.add_argument("tool", choices=list(TOOL_NAMES), help="Tool to run.")
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Prepared scene pack directory (…/<scene_id>/<pack_name>).",
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
        help="Writable scratch dir for annotated images (default: ./tmp/nr3d_tool_scratch).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else _DEFAULT_OUT_DIR

    try:
        raw_args = _parse_args_json(args.args)
        scene = Nr3dScene.load(args.scene_dir)
        payload = run_tool(scene, args.tool, raw_args, out_dir=out_dir)
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
        raise ToolInputError(
            "--args must be a JSON object, e.g. '{\"proposal_id\": 3}'"
        )
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
