"""Run the NR3D grounding task over a fold and report IoU-based metrics.

Each sample is executed through :class:`~codex_agent.runtime.CodexAgentRuntime`,
scored with oriented 3D IoU against ground truth, and checkpointed to disk so a
run can resume without re-issuing completed turns.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ..models import CodexSkill
from ..nr3d.geometry import compute_oriented_iou_3d
from ..nr3d.grounding import Nr3dGroundingOutcome, Nr3dGroundingTask
from ..nr3d.sample import (
    DEFAULT_PACK_NAME,
    Nr3dScene,
    load_sample,
    safe_sample_id,
    scene_dir_for,
)
from ..tasks.base import CodexExecutor

_RETRYABLE_TOKENS: tuple[str, ...] = (
    "429",
    "500",
    "503",
    "timeout",
    "timed out",
    "connection reset",
    "internal error",
    "server busy",
    "overloaded",
)
_ACC_THRESHOLDS: tuple[float, ...] = (0.25, 0.50)


@dataclass(frozen=True)
class Nr3dSampleResult:
    """The scored outcome of one NR3D sample."""

    sample_id: str
    status: str
    iou: float
    predicted_bbox_9dof: tuple[float, ...] | None = None
    gt_bbox_9dof: tuple[float, ...] | None = None
    selected_object_id: int | None = None
    confidence: float | None = None
    query: str | None = None
    summary: str = ""
    reasoning_summary: str | None = None
    error: str | None = None
    turn_duration_ms: int | None = None
    turn_usage: dict[str, Any] | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint view of the result."""
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "iou": self.iou,
            "predicted_bbox_9dof": _list_or_none(self.predicted_bbox_9dof),
            "gt_bbox_9dof": _list_or_none(self.gt_bbox_9dof),
            "selected_object_id": self.selected_object_id,
            "confidence": self.confidence,
            "query": self.query,
            "summary": self.summary,
            "reasoning_summary": self.reasoning_summary,
            "error": self.error,
            "turn_duration_ms": self.turn_duration_ms,
            "turn_usage": self.turn_usage,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Nr3dSampleResult:
        """Rebuild a result from a checkpoint dict."""
        return cls(
            sample_id=str(payload["sample_id"]),
            status=str(payload["status"]),
            iou=float(payload["iou"]),
            predicted_bbox_9dof=_tuple_or_none(payload.get("predicted_bbox_9dof")),
            gt_bbox_9dof=_tuple_or_none(payload.get("gt_bbox_9dof")),
            selected_object_id=payload.get("selected_object_id"),
            confidence=payload.get("confidence"),
            query=payload.get("query"),
            summary=str(payload.get("summary", "")),
            reasoning_summary=payload.get("reasoning_summary"),
            error=payload.get("error"),
            turn_duration_ms=payload.get("turn_duration_ms"),
            turn_usage=payload.get("turn_usage"),
            input_tokens=payload.get("input_tokens"),
            cached_input_tokens=payload.get("cached_input_tokens"),
        )


@dataclass(frozen=True)
class Nr3dRunSummary:
    """Aggregate IoU + prompt-cache metrics over a fold."""

    n: int
    mean_iou: float
    acc_025: float
    acc_050: float
    cache_hit_rate: float = 0.0
    mean_cache_ratio: float = 0.0
    results: tuple[Nr3dSampleResult, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_per_sample: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n": self.n,
            "mean_iou": self.mean_iou,
            "Acc@0.25": self.acc_025,
            "Acc@0.50": self.acc_050,
            "cache_hit_rate": self.cache_hit_rate,
            "mean_cache_ratio": self.mean_cache_ratio,
        }
        if include_per_sample:
            payload["per_sample"] = [result.to_dict() for result in self.results]
        return payload


def run_one_sample(
    sample_id: str,
    *,
    data_root: Path,
    runtime: CodexExecutor,
    pack_name: str = DEFAULT_PACK_NAME,
    skill: CodexSkill | None = None,
    tools_enabled: bool = False,
) -> Nr3dSampleResult:
    """Execute and score a single NR3D sample."""
    sample = load_sample(data_root, sample_id, pack_name=pack_name)
    scene_dir = scene_dir_for(data_root, sample.scene_id, pack_name=pack_name)
    scene = Nr3dScene.load(scene_dir)
    task = Nr3dGroundingTask(
        sample=sample,
        scene=scene,
        skill=skill,
        tools_enabled=tools_enabled,
        scene_dir=scene_dir,
    )
    result = runtime.execute(task)
    outcome = result.outcome
    iou = _score_outcome(outcome, gt_bbox=sample.gt_bbox_3d_9dof)
    return Nr3dSampleResult(
        sample_id=sample_id,
        status=outcome.status,
        iou=iou,
        predicted_bbox_9dof=outcome.selected_bbox_9dof,
        gt_bbox_9dof=sample.gt_bbox_3d_9dof,
        selected_object_id=(outcome.proposal_id if outcome.target_present else None),
        confidence=outcome.confidence,
        query=sample.query,
        summary=outcome.summary,
        reasoning_summary=result.turn.metadata.reasoning_summary,
        turn_duration_ms=result.turn.metadata.duration_ms,
        turn_usage=(
            dict(result.turn.metadata.usage)
            if result.turn.metadata.usage is not None
            else None
        ),
        input_tokens=result.turn.metadata.input_tokens,
        cached_input_tokens=result.turn.metadata.cached_input_tokens,
    )


def run_samples(
    *,
    sample_ids: Sequence[str],
    data_root: Path,
    output_dir: Path,
    runtime: CodexExecutor,
    pack_name: str = DEFAULT_PACK_NAME,
    skill: CodexSkill | None = None,
    workers: int = 1,
    sample_retries: int = 2,
    tools_enabled: bool = False,
) -> Nr3dRunSummary:
    """Run a fold with per-sample checkpointing and return aggregate metrics."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if sample_retries < 0:
        raise ValueError("sample_retries must be non-negative")
    _validate_unique(sample_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        sample_id
        for sample_id in sample_ids
        if not _checkpoint_path(output_dir, sample_id, pack_name).exists()
    ]
    logger.info(
        "nr3d run: {} samples ({} pending, {} cached), workers={}",
        len(sample_ids),
        len(pending),
        len(sample_ids) - len(pending),
        workers,
    )

    def process(sample_id: str) -> None:
        result = _run_with_retries(
            sample_id,
            data_root=data_root,
            runtime=runtime,
            pack_name=pack_name,
            skill=skill,
            sample_retries=sample_retries,
            tools_enabled=tools_enabled,
        )
        _write_checkpoint(output_dir, result, pack_name)

    if workers == 1:
        for sample_id in pending:
            process(sample_id)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, sid) for sid in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    results = tuple(
        _read_checkpoint(output_dir, sample_id, pack_name) for sample_id in sample_ids
    )
    summary = _summarize(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_with_retries(
    sample_id: str,
    *,
    data_root: Path,
    runtime: CodexExecutor,
    pack_name: str,
    skill: CodexSkill | None,
    sample_retries: int,
    tools_enabled: bool,
) -> Nr3dSampleResult:
    last_error: Exception | None = None
    for attempt in range(sample_retries + 1):
        try:
            return run_one_sample(
                sample_id,
                data_root=data_root,
                runtime=runtime,
                pack_name=pack_name,
                skill=skill,
                tools_enabled=tools_enabled,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= sample_retries or not _is_retryable_error(exc):
                break
            logger.warning(
                "nr3d sample {} failed (attempt {}/{}): {}",
                sample_id,
                attempt + 1,
                sample_retries + 1,
                exc,
            )
            time.sleep(2.0 * (attempt + 1))
    logger.error("nr3d sample {} failed permanently: {}", sample_id, last_error)
    return Nr3dSampleResult(
        sample_id=sample_id,
        status="failed",
        iou=0.0,
        error=f"{type(last_error).__name__}: {str(last_error)[:480]}",
    )


def _score_outcome(
    outcome: Nr3dGroundingOutcome, *, gt_bbox: tuple[float, ...]
) -> float:
    if not outcome.target_present or outcome.selected_bbox_9dof is None:
        return 0.0
    return compute_oriented_iou_3d(outcome.selected_bbox_9dof, gt_bbox)


def _summarize(results: Sequence[Nr3dSampleResult]) -> Nr3dRunSummary:
    ious = [result.iou for result in results]
    n = len(ious)
    denom = max(n, 1)
    mean_iou = sum(ious) / denom if ious else 0.0
    acc_025 = sum(1 for v in ious if v >= _ACC_THRESHOLDS[0]) / denom
    acc_050 = sum(1 for v in ious if v >= _ACC_THRESHOLDS[1]) / denom

    cache_ratios = [
        result.cached_input_tokens / result.input_tokens
        for result in results
        if result.input_tokens and result.cached_input_tokens is not None
    ]
    measured = max(len(cache_ratios), 1)
    cache_hit_rate = sum(1 for r in cache_ratios if r > 0.0) / measured
    mean_cache_ratio = sum(cache_ratios) / measured if cache_ratios else 0.0
    return Nr3dRunSummary(
        n=n,
        mean_iou=mean_iou,
        acc_025=acc_025,
        acc_050=acc_050,
        cache_hit_rate=cache_hit_rate,
        mean_cache_ratio=mean_cache_ratio,
        results=tuple(results),
    )


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_TOKENS)


def _checkpoint_path(output_dir: Path, sample_id: str, pack_name: str) -> Path:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    safe = safe_sample_id(sample_id) or "sample"
    return output_dir / "per_sample" / pack_name / f"{safe}_{digest}.json"


def _write_checkpoint(
    output_dir: Path, result: Nr3dSampleResult, pack_name: str
) -> Path:
    path = _checkpoint_path(output_dir, result.sample_id, pack_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _read_checkpoint(
    output_dir: Path, sample_id: str, pack_name: str
) -> Nr3dSampleResult:
    path = _checkpoint_path(output_dir, sample_id, pack_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a JSON object: {path}")
    return Nr3dSampleResult.from_dict(payload)


def _validate_unique(sample_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        raise ValueError(
            f"sample_ids must be unique for checkpointing; duplicates={duplicates[:10]}"
        )


def _list_or_none(values: tuple[float, ...] | None) -> list[float] | None:
    return list(values) if values is not None else None


def _tuple_or_none(values: Any) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(v) for v in values)


__all__ = [
    "Nr3dSampleResult",
    "Nr3dRunSummary",
    "run_one_sample",
    "run_samples",
]
