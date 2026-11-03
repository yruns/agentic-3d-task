"""Explicit operator commands; imports do not start clients or touch state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from codex_adapter.application.catalog import describe_model
from codex_adapter.config import load_settings
from codex_adapter.domain.errors import AdapterError


def build_parser() -> argparse.ArgumentParser:
    """Build commands sharing one explicit versioned registry configuration."""
    parser = argparse.ArgumentParser(
        description="State-preserving Codex Responses adapter"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "init-state", "serve", "models"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an operator command and report safe errors without payload tracebacks."""
    args = build_parser().parse_args(argv)
    try:
        return _execute(args.command, args.config)
    except AdapterError as error:
        print(json.dumps(error.payload()), file=sys.stderr)
        return 2


def _execute(command: str, config_path: Path) -> int:
    settings = load_settings(config_path)
    if command == "models":
        print(
            json.dumps(
                {
                    "models": [
                        describe_model(model) for model in settings.registry.models
                    ]
                },
                ensure_ascii=False,
            )
        )
        return 0
    if command == "check":
        print(
            json.dumps(
                {
                    "status": "configuration_valid",
                    "models": [model.id for model in settings.registry.models],
                    "upstream_probed": False,
                }
            )
        )
        return 0
    if command == "init-state":
        from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore

        store = SQLiteStateStore(settings.state)
        try:
            store.initialize()
        finally:
            store.close()
        print(json.dumps({"status": "state_initialized"}))
        return 0
    return _serve(config_path)


def _serve(config_path: Path) -> int:
    import uvicorn

    from codex_adapter.interfaces.app import create_app

    settings = load_settings(config_path)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
        access_log=False,
        log_level="warning",
    )
    return 0
