"""CLI: run the OpenEQA QA task over a question set with the Codex Agent SDK.

This entry point only parses arguments, wires dependencies, and prints the
headline metrics. All evaluation logic lives in
:mod:`codex_agent.evaluation.openeqa_runner`.

OpenEQA QA is **tool-based**: no frames are attached to a turn, and the agent
fetches all visual evidence through the OpenEQA CLI tools (``keyframe_selector``
/ ``view_frame`` / ``view_bev`` / ``list_objects``). The run therefore always
uses a ``workspace_write`` + network sandbox (the tools shell out and
``keyframe_selector`` calls the parsing LLM) with the tool-call loop caps.

Example::

    python -m codex_agent.cli.run_openeqa \\
        --questions data/open-eqa-v0.json \\
        --data-root data/OpenEQA/scannet \\
        --output-dir tmp/openeqa_run \\
        --limit 10 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from loguru import logger

from ..config import (
    CodexAgentConfig,
    ReasoningEffort,
    ReasoningSummary,
    SandboxMode,
)
from ..evaluation.openeqa_runner import run_questions
from ..evaluation.question_ids import load_question_ids
from ..openeqa.judge import JudgeScorer, LlmJudge
from ..openeqa.question import (
    OpenEqaQuestion,
    load_questions,
    select_questions,
)
from ..openeqa.scene import filter_questions_with_local_scenes
from ..runtime import CodexAgentRuntime
from .runtime_preflight import preflight_codex_runtime

# Wall-clock budget is OFF by default (per-turn latency scales with adapter
# contention under concurrency); the tool-call cap is the primary, timing-
# independent loop backstop. The v4 trace analysis found ~8% of questions hit the
# old cap of 24 because each retrieval batch spends one view_image per returned
# frame; 32 gives the diverse-retrieval + contact-sheet + view_crop tools room to
# gather enough evidence before the backstop fires. Repeating one identical
# action 4x is already a rut with no recovery value.
_DEFAULT_TOOLS_TURN_TIMEOUT_S = 0.0
_DEFAULT_TOOLS_MAX_TOOL_CALLS = 32
_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to open-eqa-v0.json (a JSON array of question records).",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="OpenEQA ScanNet root; clips live at <data_root>/<clip_id>/raw/.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--question-ids",
        type=Path,
        default=None,
        help="Optional fold file selecting a subset of question ids to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected questions (e.g. 10 for a smoke run).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Codex model name (else CODEX_AGENT_MODEL / default).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge model name (else the configs/llm.toml default).",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Path to the judge LLM config TOML (else configs/llm.toml).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring; only produce and checkpoint predictions.",
    )
    parser.add_argument(
        "--sandbox",
        default=None,
        choices=["read_only", "workspace_write", "full_access"],
        help="Override the Codex sandbox mode (default workspace_write + network, "
        "required for the agent CLI tools).",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=None,
        help="Wall-clock budget (seconds) per agentic turn; a runaway tool loop "
        "is interrupted and finalized. Default 0 (off).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Interrupt a turn after this many tool actions (the reliable loop "
        f"backstop). 0 disables. Default {_DEFAULT_TOOLS_MAX_TOOL_CALLS}.",
    )
    parser.add_argument(
        "--max-repeated-tool-calls",
        type=int,
        default=None,
        help="Interrupt a turn once the same tool action repeats this many times. "
        f"0 disables. Default {_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Model reasoning effort. Default: medium (config default).",
    )
    parser.add_argument(
        "--reasoning-summary",
        default=None,
        choices=["auto", "concise", "detailed", "none"],
        help="Ask the model to emit a reasoning summary, captured per question.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    available = filter_questions_with_local_scenes(
        load_questions(args.questions), args.data_root
    )
    if not available:
        parser.error(
            f"no questions with local scenes under {args.data_root}; check --data-root"
        )
    questions = _select_questions(
        available, question_ids_path=args.question_ids, limit=args.limit
    )
    if not questions:
        parser.error("no questions selected to run")

    config = _build_config(
        model=args.model,
        sandbox=args.sandbox,
        turn_timeout=args.turn_timeout,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        max_tool_calls=args.max_tool_calls,
        max_repeated_tool_calls=args.max_repeated_tool_calls,
    )
    preflight_codex_runtime(config)
    runtime = CodexAgentRuntime(config)
    judge = _build_judge(
        no_judge=args.no_judge, llm_config=args.llm_config, judge_model=args.judge_model
    )

    logger.info(
        "running {} OpenEQA questions (tools, judge={}, sandbox={})",
        len(questions),
        "off" if judge is None else (args.judge_model or "default"),
        config.sandbox,
    )
    summary = run_questions(
        questions=questions,
        data_root=args.data_root,
        output_dir=args.output_dir,
        runtime=runtime,
        judge=judge,
        workers=args.workers,
        sample_retries=args.sample_retries,
    )
    print(
        json.dumps(
            summary.to_dict(include_per_sample=False), ensure_ascii=False, indent=2
        )
    )
    return 0


def _select_questions(
    available: tuple[OpenEqaQuestion, ...],
    *,
    question_ids_path: Path | None,
    limit: int | None,
) -> tuple[OpenEqaQuestion, ...]:
    if question_ids_path is not None:
        selected = select_questions(available, load_question_ids(question_ids_path))
    else:
        selected = available
    if limit is not None:
        selected = selected[:limit]
    return selected


def _build_config(
    *,
    model: str | None,
    sandbox: str | None,
    turn_timeout: float | None,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
    max_tool_calls: int | None,
    max_repeated_tool_calls: int | None,
) -> CodexAgentConfig:
    # OpenEQA QA is always tool-based: the agent shells out to fetch frames and
    # keyframe_selector needs the network for query parsing, so the run defaults
    # to workspace_write + network unless the sandbox is explicitly overridden.
    config = CodexAgentConfig.from_env()
    if model:
        config = dataclasses.replace(config, model=model)
    resolved_sandbox: SandboxMode = (
        _as_sandbox_mode(sandbox) if sandbox else "workspace_write"
    )
    network = config.sandbox_network_access or resolved_sandbox in (
        "workspace_write",
        "full_access",
    )
    config = dataclasses.replace(
        config, sandbox=resolved_sandbox, sandbox_network_access=network
    )
    return dataclasses.replace(
        config,
        turn_timeout_s=_resolve_turn_timeout(turn_timeout, config=config),
        reasoning_effort=_resolve_reasoning_effort(reasoning_effort, config=config),
        reasoning_summary=_resolve_reasoning_summary(reasoning_summary, config=config),
        max_tool_calls=_resolve_cap(
            max_tool_calls,
            current=config.max_tool_calls,
            default=_DEFAULT_TOOLS_MAX_TOOL_CALLS,
        ),
        max_repeated_tool_calls=_resolve_cap(
            max_repeated_tool_calls,
            current=config.max_repeated_tool_calls,
            default=_DEFAULT_TOOLS_MAX_REPEATED_TOOL_CALLS,
        ),
    )


def _resolve_cap(override: int | None, *, current: int, default: int) -> int:
    if override is not None:
        if override < 0:
            raise ValueError(f"cap must be non-negative, got {override}")
        return override
    if current > 0:
        return current
    return default


def _resolve_turn_timeout(
    turn_timeout: float | None, *, config: CodexAgentConfig
) -> float:
    if turn_timeout is not None:
        return turn_timeout
    if config.turn_timeout_s > 0:
        return config.turn_timeout_s
    return _DEFAULT_TOOLS_TURN_TIMEOUT_S


def _as_sandbox_mode(value: str) -> SandboxMode:
    modes: dict[str, SandboxMode] = {
        "read_only": "read_only",
        "workspace_write": "workspace_write",
        "full_access": "full_access",
    }
    if value not in modes:
        raise ValueError(f"unexpected sandbox value: {value!r}")
    return modes[value]


def _build_judge(
    *, no_judge: bool, llm_config: Path | None, judge_model: str | None
) -> JudgeScorer | None:
    if no_judge:
        return None
    return LlmJudge.from_toml(llm_config, model=judge_model)


def _resolve_reasoning_effort(
    reasoning_effort: str | None, *, config: CodexAgentConfig
) -> ReasoningEffort:
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


if __name__ == "__main__":
    raise SystemExit(main())
