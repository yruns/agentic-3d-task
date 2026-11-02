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

from ..config import CodexAgentConfig, ReasoningEffort, ReasoningSummary, SandboxMode
from ..evaluation.nr3d_runner import run_samples
from ..evaluation.sample_ids import load_sample_ids
from ..models import CodexSkill
from ..nr3d.sample import DEFAULT_PACK_NAME
from ..runtime import CodexAgentRuntime
from .runtime_preflight import preflight_codex_runtime


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
        help="Path to the NR3D SKILL.md attached to each turn. Requires --tools; "
        "default is no attached skill.",
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
        "highlights). Implies workspace_write sandbox + network access. Attach "
        "a skill explicitly with --skill-path when testing Codex Skill mode.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=None,
        help="Wall-clock budget (seconds) per agentic turn; a runaway tool loop "
        "is interrupted and finalized. Default 0 (off); the tool-call cap is the "
        "primary loop backstop.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Interrupt a turn after this many tool actions (the reliable loop "
        f"backstop). 0 disables. Default {_DEFAULT_TOOLS_MAX_TOOL_CALLS} when "
        "--tools, else 0.",
    )
    parser.add_argument(
        "--max-repeated-tool-calls",
        type=int,
        default=None,
        help="Interrupt a turn once the same tool action repeats this many "
        f"times. 0 disables. Default {_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS} "
        "when --tools, else 0.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Model reasoning effort. Default: medium (a real accuracy win for "
        "spatial grounding). Lower is faster per call but hurts accuracy; "
        "overridable here or via CODEX_AGENT_REASONING_EFFORT.",
    )
    parser.add_argument(
        "--reasoning-summary",
        default=None,
        choices=["auto", "concise", "detailed", "none"],
        help="Ask the model to emit a human-readable reasoning summary, captured "
        "into per-sample metadata for tracing/debugging. Default: off (no "
        "summary requested). 'none' explicitly disables it upstream.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        skill = _resolve_skill(
            skill_path=args.skill_path, no_skill=args.no_skill, tools=args.tools
        )
    except ValueError as exc:
        parser.error(str(exc))

    sample_ids = load_sample_ids(args.sample_ids)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        parser.error("no sample ids selected to run")

    config = _build_config(
        model=args.model,
        sandbox=args.sandbox,
        tools=args.tools,
        turn_timeout=args.turn_timeout,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        max_tool_calls=args.max_tool_calls,
        max_repeated_tool_calls=args.max_repeated_tool_calls,
    )
    preflight_codex_runtime(config)
    runtime = CodexAgentRuntime(config)

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


# Wall-clock budget is OFF by default: per-turn latency scales with adapter
# contention under concurrency, so a tight time cap would interrupt legitimate
# slow turns. The tool-call cap below is the primary, timing-independent loop
# backstop. ``0`` disables the time budget.
_DEFAULT_TOOLS_TURN_TIMEOUT_S = 0.0


# The reliable loop backstop. A legitimate tool run resolves in ~6-12 actions, so
# 30 gives generous headroom while bounding the degenerate re-read loop (observed
# at 80+ identical shell calls). Repeating one identical action 4x is already a
# rut with no recovery value, so trip there too.
_DEFAULT_TOOLS_MAX_TOOL_CALLS = 30
_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS = 4


def _build_config(
    *,
    model: str | None,
    sandbox: str | None,
    tools: bool,
    turn_timeout: float | None,
    reasoning_effort: str | None,
    reasoning_summary: str | None = None,
    max_tool_calls: int | None = None,
    max_repeated_tool_calls: int | None = None,
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
        reasoning_effort=_resolve_reasoning_effort(reasoning_effort, config=config),
        reasoning_summary=_resolve_reasoning_summary(reasoning_summary, config=config),
        max_tool_calls=_resolve_cap(
            max_tool_calls,
            tools=tools,
            current=config.max_tool_calls,
            tools_default=_DEFAULT_TOOLS_MAX_TOOL_CALLS,
        ),
        max_repeated_tool_calls=_resolve_cap(
            max_repeated_tool_calls,
            tools=tools,
            current=config.max_repeated_tool_calls,
            tools_default=_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS,
        ),
    )


def _resolve_cap(
    override: int | None, *, tools: bool, current: int, tools_default: int
) -> int:
    if override is not None:
        if override < 0:
            raise ValueError(f"cap must be non-negative, got {override}")
        return override
    if current > 0:
        return current
    return tools_default if tools else 0


def _resolve_reasoning_effort(
    reasoning_effort: str | None, *, config: CodexAgentConfig
) -> ReasoningEffort:
    # CLI flag wins; otherwise keep the env/config value (defaults to "medium",
    # see config.DEFAULT_REASONING_EFFORT).
    if reasoning_effort is not None:
        return _as_reasoning_effort(reasoning_effort)
    return config.reasoning_effort


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


def _resolve_reasoning_summary(
    reasoning_summary: str | None, *, config: CodexAgentConfig
) -> ReasoningSummary:
    # CLI flag wins; otherwise keep the env/config value (default "" = off).
    if reasoning_summary is not None:
        return _as_reasoning_summary(reasoning_summary)
    return config.reasoning_summary


def _as_reasoning_summary(value: str) -> ReasoningSummary:
    summaries: dict[str, ReasoningSummary] = {
        "auto": "auto",
        "concise": "concise",
        "detailed": "detailed",
        "none": "none",
    }
    if value not in summaries:
        raise ValueError(f"unexpected reasoning summary: {value!r}")
    return summaries[value]


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
    if skill_path is not None and not tools:
        raise ValueError("--skill-path requires --tools")
    # Tool mode inlines the playbook into the prompt and attaches no skill by
    # default: an advertised SKILL.md path is the bait for the unbounded re-read
    # loop. ``--skill-path`` still forces a skill for manual/legacy comparison.
    if skill_path is None:
        return None
    resolved_path = skill_path
    if not resolved_path.exists():
        logger.warning(
            "skill file not found at {}; running without a skill", resolved_path
        )
        return None
    return CodexSkill(name="nr3d-codex-tools", path=resolved_path)


if __name__ == "__main__":
    raise SystemExit(main())
