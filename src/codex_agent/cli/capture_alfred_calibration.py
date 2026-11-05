"""Freeze and capture a deterministic 60-frame ALFRED train depth diagnostic.

This command never runs an actor, depth predictor, detector, or task scorer.
Ground-truth depth is confined to the offline calibration output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_agent.alfred.calibration_capture import (
    create_capture_plan,
    run_capture,
    write_capture_plan,
)
from codex_agent.alfred.provenance import code_provenance
from codex_agent.config import DEFAULT_PROJECT_ROOT


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose separate freeze and capture operations without loading a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Freeze seed-0 train selection and source hashes before capture."
    )
    for name in (
        "data-root",
        "splits",
        "alfred-root",
        "ai2thor-root",
        "unity-executable",
        "output-plan",
    ):
        plan.add_argument("--" + name, type=Path, required=True)
    capture = commands.add_parser(
        "capture",
        help="Capture only the frozen plan on matching clean committed source.",
    )
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--simulator-python", type=Path, required=True)
    capture.add_argument("--x-display", default="5")
    capture.add_argument("--worker-timeout", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate CLI arguments and write a frozen plan or offline capture manifest."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        code = code_provenance(DEFAULT_PROJECT_ROOT)
        plan = create_capture_plan(
            data_root=args.data_root,
            splits_path=args.splits,
            alfred_root=args.alfred_root,
            ai2thor_root=args.ai2thor_root,
            unity_executable=args.unity_executable,
            code=code,
        )
        write_capture_plan(plan, args.output_plan)
        print(
            json.dumps(
                {
                    "plan": str(args.output_plan.resolve()),
                    "split": "train",
                    "scenes": [item.scene_id for item in plan.trajectories],
                    "expected_frames": 60,
                    "source_sha256": plan.source_sha256,
                }
            )
        )
    else:
        manifest = run_capture(
            plan_path=args.plan,
            output_dir=args.output_dir,
            project_root=DEFAULT_PROJECT_ROOT,
            simulator_python=args.simulator_python,
            x_display=args.x_display,
            timeout_seconds=args.worker_timeout,
        )
        print(json.dumps({"manifest": str(manifest), "split": "train", "frames": 60}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
