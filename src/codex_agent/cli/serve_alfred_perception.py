"""Launch the persistent multi-GPU ALFRED scene perception HTTP server."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from codex_agent.alfred.perception_server import (
    ScenePerceptionServer,
    validate_gpu_pool_indices,
)
from codex_agent.alfred.settings import load_scene_settings


def _parse_gpus(raw: str) -> tuple[int, ...]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--gpus must specify at least one GPU index")
    try:
        indices = [int(item) for item in items]
        return validate_gpu_pool_indices(indices)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        required=True,
        type=Path,
        help="TOML configuration for local perception models.",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=None,
        help="Comma-separated physical GPU indices for the worker pool (e.g. 4,5,6; never 1).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        choices=("127.0.0.1", "localhost"),
        help="Loopback address to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8791,
        help="Loopback TCP port to listen on (default: 8791).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/agentic3d-alfred-perception-server"),
        help="Directory for per-GPU worker stderr logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    scene = load_scene_settings(args.scene_config)
    server = ScenePerceptionServer(
        scene.perception,
        log_dir=args.log_dir,
        host=args.host,
        port=args.port,
        gpu_indices=args.gpus,
    )
    logging.info(
        "ALFRED ScenePerceptionServer listening at %s with GPUs=%s (%s)",
        server.url,
        list(server.gpu_indices),
        json.dumps(server.health_payload()),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
