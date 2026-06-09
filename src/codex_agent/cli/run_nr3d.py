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

from ..config import (
    DEFAULT_PROJECT_ROOT,
    CodexAgentConfig,
    ReasoningEffort,
    SandboxMode,
)
from ..evaluation.nr3d_runner import run_samples
from ..evaluation.sample_ids import load_sample_ids
from ..models import CodexSkill
from ..nr3d.sample import DEFAULT_PACK_NAME
from ..runtime import CodexAgentRuntime

DEFAULT_SKILL_PATH = (
    DEFAULT_PROJECT_ROOT / ".agents" / "skills" / "nr3d-codex-sdk" / "SKILL.md"
)
DEFAULT_TOOLS_SKILL_PATH = (
    DEFAULT_PROJECT_ROOT / ".agents" / "skills" / "nr3d-codex-tools" / "SKILL.md"
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
        default=None,
        help="Path to the NR3D SKILL.md attached to each turn (default depends "
        "on --tools).",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Do not attach any skill (the prompt is self-contained).",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable agent CLI tools (first-person frames, spatial ranking, BEV "
        "highlights). Implies workspace_write sandbox + network access and the "
        "nr3d-codex-tools skill unless overridden.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=None,
        help="Wall-clock budget (seconds) per agentic turn; a runaway tool loop "
        "is interrupted and finalized. Default 0 (off), or 900 when --tools.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Model reasoning effort. Lower is much faster per call; default "
        "'low' when --tools (the deterministic tools carry the heavy reasoning).",
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

    runtime = CodexAgentRuntime(
        _build_config(
            model=args.model,
            sandbox=args.sandbox,
            tools=args.tools,
            turn_timeout=args.turn_timeout,
            reasoning_effort=args.reasoning_effort,
        )
    )
    skill = _resolve_skill(
        skill_path=args.skill_path, no_skill=args.no_skill, tools=args.tools
    )

    logger.info(
        "running {} NR3D samples (pack={}, skill={}, tools={})",
        len(sample_ids),
        args.pack_name,
        skill.name if skill is not None else "none",
        args.tools,
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
        tools_enabled=args.tools,
    )
    print(
        json.dumps(
            summary.to_dict(include_per_sample=False), ensure_ascii=False, indent=2
        )
    )
    return 0


# A loose catastrophe guard, not a tight budget: per-turn wall-clock scales with
# adapter contention under concurrency, so a tight cap would interrupt
# legitimate slow turns. The loop fixes (no AGENTS.md, decisive skill) prevent
# the pathological runaway loops this guards against. ``0`` disables it.
_DEFAULT_TOOLS_TURN_TIMEOUT_S = 0.0


# The model default effort is used for tools (empty string). Low effort was
# observed to hurt spatial-grounding accuracy without a clear latency win.
_DEFAULT_TOOLS_REASONING_EFFORT: ReasoningEffort = ""


def _build_config(
    *,
    model: str | None,
    sandbox: str | None,
    tools: bool,
    turn_timeout: float | None,
    reasoning_effort: str | None,
) -> CodexAgentConfig:
    config = CodexAgentConfig.from_env()
    if model:
        config = dataclasses.replace(config, model=model)
    if tools and sandbox is None:
        # Tool turns shell out and write annotated images, and keyframe_selector
        # needs the network for query parsing.
        config = dataclasses.replace(
            config, sandbox="workspace_write", sandbox_network_access=True
        )
    if sandbox:
        resolved = _as_sandbox_mode(sandbox)
        config = dataclasses.replace(
            config,
            sandbox=resolved,
            sandbox_network_access=(
                config.sandbox_network_access
                or (tools and resolved == "workspace_write")
            ),
        )
    return dataclasses.replace(
        config,
        turn_timeout_s=_resolve_turn_timeout(turn_timeout, tools=tools, config=config),
        reasoning_effort=_resolve_reasoning_effort(
            reasoning_effort, tools=tools, config=config
        ),
    )


def _resolve_reasoning_effort(
    reasoning_effort: str | None, *, tools: bool, config: CodexAgentConfig
) -> ReasoningEffort:
    if reasoning_effort is not None:
        return _as_reasoning_effort(reasoning_effort)
    if config.reasoning_effort:
        return config.reasoning_effort
    return _DEFAULT_TOOLS_REASONING_EFFORT if tools else ""


def _as_reasoning_effort(value: str) -> ReasoningEffort:
    efforts: dict[str, ReasoningEffort] = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    if value not in efforts:
        raise ValueError(f"unexpected reasoning effort: {value!r}")
    return efforts[value]


def _resolve_turn_timeout(
    turn_timeout: float | None, *, tools: bool, config: CodexAgentConfig
) -> float:
    if turn_timeout is not None:
        return turn_timeout
    if config.turn_timeout_s > 0:
        return config.turn_timeout_s
    return _DEFAULT_TOOLS_TURN_TIMEOUT_S if tools else 0.0


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


def _resolve_skill(
    *, skill_path: Path | None, no_skill: bool, tools: bool
) -> CodexSkill | None:
    if no_skill:
        return None
    resolved_path = skill_path or (
        DEFAULT_TOOLS_SKILL_PATH if tools else DEFAULT_SKILL_PATH
    )
    if not resolved_path.exists():
        logger.warning(
            "skill file not found at {}; running without a skill", resolved_path
        )
        return None
    skill_name = "nr3d-codex-tools" if tools else "nr3d-codex-sdk"
    return CodexSkill(name=skill_name, path=resolved_path)


if __name__ == "__main__":
    raise SystemExit(main())
