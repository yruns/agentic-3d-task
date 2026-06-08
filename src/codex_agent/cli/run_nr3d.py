"""CLI: run the NR3D grounding task over a fold with the Codex Agent SDK.

This entry point only parses arguments, wires dependencies, and prints the
headline metrics. All evaluation logic lives in
:mod:`codex_agent.evaluation.nr3d_runner`.

Example::

    python -m codex_agent.cli.run_nr3d \\
        --sample-ids fold.json \\
        --data-root /path/to/data/nr3d/scannet \\
        --output-dir tmp/nr3d_run --limit 10 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from loguru import logger

from ..config import DEFAULT_PROJECT_ROOT, CodexAgentConfig, SandboxMode
from ..evaluation.nr3d_runner import run_samples
from ..evaluation.sample_ids import load_sample_ids
from ..models import CodexSkill
from ..nr3d.sample import DEFAULT_PACK_NAME
from ..runtime import CodexAgentRuntime

DEFAULT_SKILL_PATH = (
    DEFAULT_PROJECT_ROOT / ".agents" / "skills" / "nr3d-codex-sdk" / "SKILL.md"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="NR3D ScanNet root; scenes live at <data_root>/<scene_id>/<pack_name>.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pack-name", default=DEFAULT_PACK_NAME)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N sample ids from the fold (e.g. 10 for a smoke run).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Codex model name (else CODEX_AGENT_MODEL / default).",
    )
    parser.add_argument(
        "--sandbox",
        default=None,
        choices=["read_only", "workspace_write", "full_access"],
        help="Override the Codex sandbox mode (default read_only).",
    )
    parser.add_argument(
        "--skill-path",
        type=Path,
        default=DEFAULT_SKILL_PATH,
        help="Path to the NR3D SKILL.md attached to each turn.",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Do not attach any skill (the prompt is self-contained).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    sample_ids = load_sample_ids(args.sample_ids)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        parser.error("no sample ids selected to run")

    runtime = CodexAgentRuntime(_build_config(model=args.model, sandbox=args.sandbox))
    skill = _resolve_skill(skill_path=args.skill_path, no_skill=args.no_skill)

    logger.info(
        "running {} NR3D samples (pack={}, skill={})",
        len(sample_ids),
        args.pack_name,
        skill.name if skill is not None else "none",
    )
    summary = run_samples(
        sample_ids=sample_ids,
        data_root=args.data_root,
        output_dir=args.output_dir,
        runtime=runtime,
        pack_name=args.pack_name,
        skill=skill,
        workers=args.workers,
        sample_retries=args.sample_retries,
    )
    print(
        json.dumps(
            summary.to_dict(include_per_sample=False), ensure_ascii=False, indent=2
        )
    )
    return 0


def _build_config(*, model: str | None, sandbox: str | None) -> CodexAgentConfig:
    config = CodexAgentConfig.from_env()
    if model:
        config = dataclasses.replace(config, model=model)
    if sandbox:
        config = dataclasses.replace(config, sandbox=_as_sandbox_mode(sandbox))
    return config


def _as_sandbox_mode(value: str) -> SandboxMode:
    # argparse 'choices' already restricts value to the valid set; map to the
    # Literal so the return type is precise.
    modes: dict[str, SandboxMode] = {
        "read_only": "read_only",
        "workspace_write": "workspace_write",
        "full_access": "full_access",
    }
    if value not in modes:
        raise ValueError(f"unexpected sandbox value: {value!r}")
    return modes[value]


def _resolve_skill(*, skill_path: Path, no_skill: bool) -> CodexSkill | None:
    if no_skill:
        return None
    if not skill_path.exists():
        logger.warning(
            "skill file not found at {}; running without a skill", skill_path
        )
        return None
    return CodexSkill(name="nr3d-codex-sdk", path=skill_path)


if __name__ == "__main__":
    raise SystemExit(main())
