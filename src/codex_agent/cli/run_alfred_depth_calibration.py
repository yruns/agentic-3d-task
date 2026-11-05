"""Compare DA3Metric and UniDepth on a frozen ALFRED training capture only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_agent.alfred.depth_calibration_runner import (
    CalibrationRunConfig,
    run_depth_calibration,
)
from codex_agent.config import DEFAULT_PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--da3-model-dir", required=True, type=Path)
    parser.add_argument("--unidepth-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-lock-path", required=True, type=Path)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=0,
        help="Physical nvidia-smi GPU index; logical model device remains cuda:0. GPU 1 is forbidden.",
    )
    parser.add_argument("--lock-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    config = CalibrationRunConfig(
        manifest_path=args.manifest,
        da3_model_dir=args.da3_model_dir,
        unidepth_model_dir=args.unidepth_model_dir,
        output_dir=args.output_dir,
        gpu_lock_path=args.gpu_lock_path,
        project_root=args.project_root,
        device=args.device,
        physical_gpu_index=args.physical_gpu_index,
        lock_timeout_seconds=args.lock_timeout_seconds,
    )
    report = run_depth_calibration(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selection": report["selection"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
