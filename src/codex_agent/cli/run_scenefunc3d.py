"""CLI: run SceneFunc3D mask-generation cases with the Codex Agent SDK.

This entry point keeps SceneFunc3D aligned with the other task-family CLIs under
``codex_agent.cli``. The runner implementation and argument semantics remain in
:mod:`codex_agent.scenefunc3d.runner`.

Example::

    python -m codex_agent.cli.run_scenefunc3d \\
        --dataset-root /path/to/SceneFuncVal-CG \\
        --sample-id 421254::desc-a \\
        --backend-config /path/to/scenefunc3d_backends.toml \\
        --output-dir tmp/scenefunc3d_run --score
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..scenefunc3d import runner


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SceneFunc3D CLI parser."""
    return runner.build_arg_parser(prog="codex_agent.cli.run_scenefunc3d")


def main(argv: Sequence[str] | None = None) -> int:
    """Run SceneFunc3D samples from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return runner.run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
