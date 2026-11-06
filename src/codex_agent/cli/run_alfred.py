"""Run a frozen ALFRED validation fold with RGB prediction and incremental memory.

This entry point validates arguments and wires dependencies. Episode execution,
official terminal scoring, provenance, checkpoints, and SQLite archival live in
``codex_agent.evaluation.alfred_runner``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

from codex_agent.alfred.execution import ExecutionConfig
from codex_agent.alfred.settings import load_scene_settings
from codex_agent.alfred.simulator import SimulatorConfig
from codex_agent.config import DEFAULT_PROJECT_ROOT, CodexAgentConfig, ReasoningEffort
from codex_agent.evaluation.alfred_runner import (
    AlfredRunConfig,
    preflight_run,
    run_validation,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"value={value!r} must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the public CLI without loading model or simulator dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Official json_2.1.0 root."
    )
    parser.add_argument(
        "--splits", required=True, type=Path, help="Official splits/oct21.json."
    )
    parser.add_argument(
        "--fold", required=True, type=Path, help="Frozen, hashed pilot manifest."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New run directory; existing paths are rejected.",
    )
    parser.add_argument(
        "--run-id", help="Unique archive ID; defaults to the output directory name."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "docs/benchmark/alfred/runs.sqlite",
    )
    parser.add_argument(
        "--simulator-python",
        required=True,
        type=Path,
        help="Existing simulator Python with pinned ai2thor==2.1.0.",
    )
    parser.add_argument(
        "--alfred-root", required=True, type=Path, help="Official ALFRED checkout."
    )
    parser.add_argument(
        "--scene-config",
        required=True,
        type=Path,
        help="TOML configuration for local perception, object mapping and spatial memory.",
    )
    parser.add_argument("--x-display", default="5")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=None,
        help="Optional physical GPU index override (e.g., 4, 5, 6; never 1).",
    )
    parser.add_argument(
        "--perception-server",
        default=None,
        help="Optional local HTTP perception server URL (e.g., http://127.0.0.1:8791).",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider-default", "minimal", "low", "medium", "high"),
        default=None,
    )
    parser.add_argument("--max-actions", type=_positive_int, default=1000)
    parser.add_argument("--max-failures", type=_positive_int, default=10)
    parser.add_argument("--max-decisions", type=_positive_int, default=40)
    parser.add_argument("--navigation-chunk", type=_positive_int, default=32)
    parser.add_argument("--retrieved-images", type=_positive_int, default=3)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--turn-timeout", type=float, default=900.0)
    parser.add_argument(
        "--use-step-instructions",
        action="store_true",
        help="Expose official low-level language instructions (H+L protocol).",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Run only this prefix of the selected frozen-fold shard; labeled diagnostic.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic shard index; requires 0 <= index < num-shards.",
    )
    parser.add_argument(
        "--num-shards",
        type=_positive_int,
        default=1,
        help="Select frozen manifest[index::num-shards] before applying --limit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Retry failed episodes once and continue the shard instead of aborting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing shard directory, skipping already-completed episodes.",
    )
    parser.add_argument(
        "--concurrency-state-file",
        type=Path,
        default=None,
        help="Optional shared JSON file for adaptive cross-shard concurrency control.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate fold/assets/provider readiness without simulation or evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI inputs and print a validated preflight or evaluation summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output_dir = args.output_dir.absolute()
    scene = load_scene_settings(args.scene_config)
    perception_cfg = scene.perception
    if args.physical_gpu is not None:
        perception_cfg = replace(
            perception_cfg,
            physical_gpu_index=args.physical_gpu,
            gpu_lock_path=Path(f"/tmp/agentic3d-alfred-gpu{args.physical_gpu}.lock"),
        )
    if args.perception_server is not None:
        perception_cfg = replace(
            perception_cfg,
            server_url=args.perception_server,
        )
    runtime = CodexAgentConfig.from_env(project_root=DEFAULT_PROJECT_ROOT)
    resolved_model = args.model or runtime.model
    if args.reasoning_effort is not None:
        resolved_effort = cast(ReasoningEffort, args.reasoning_effort)
    elif resolved_model.lower().startswith("gemini"):
        resolved_effort = "provider-default"
    else:
        resolved_effort = runtime.reasoning_effort
    runtime = replace(
        runtime,
        model=resolved_model,
        reasoning_effort=resolved_effort,
        turn_timeout_s=args.turn_timeout,
    )
    config = AlfredRunConfig(
        source_root=DEFAULT_PROJECT_ROOT,
        data_root=args.data_root,
        splits_path=args.splits,
        fold_path=args.fold,
        output_dir=output_dir,
        run_id=args.run_id or output_dir.name,
        database_path=args.database,
        simulator=SimulatorConfig(
            python_executable=args.simulator_python,
            alfred_root=args.alfred_root,
            output_dir=output_dir,
            x_display=args.x_display,
            timeout_seconds=args.worker_timeout,
        ),
        perception=perception_cfg,
        memory=scene.memory,
        spatial=scene.spatial,
        runtime=runtime,
        execution=ExecutionConfig(
            max_actions=args.max_actions,
            max_failures=args.max_failures,
            max_decisions=args.max_decisions,
            navigation_chunk=args.navigation_chunk,
            retrieved_images=args.retrieved_images,
            use_step_instructions=args.use_step_instructions,
        ),
        command=(
            sys.executable,
            "-m",
            "codex_agent.cli.run_alfred",
            *(sys.argv[1:] if argv is None else argv),
        ),
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        continue_on_error=args.continue_on_error,
        resume=args.resume,
        concurrency_state_path=args.concurrency_state_file,
    )
    if args.preflight_only:
        print(json.dumps(preflight_run(config), ensure_ascii=False, indent=2))
    else:
        print(run_validation(config).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
