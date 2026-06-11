"""Run the OpenEQA QA task over a set of questions and report MNAS metrics.

Each question is executed through a :class:`~codex_agent.tasks.base.CodexExecutor`,
the predicted answer is graded by an injected :class:`JudgeScorer` (the official
1-5 ``mmbench`` judge), and results are checkpointed to disk so a run can resume
without re-issuing completed turns or re-judging cached answers.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ..models import CodexSkill
from ..openeqa.judge import NO_PREDICTION_SCORE, JudgeScorer, score_to_mnas
from ..openeqa.qa import OpenEqaQuestionAnsweringTask
from ..openeqa.question import OpenEqaQuestion
from ..openeqa.scene import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NUM_FRAMES,
    OpenEqaScene,
    scene_dir_for,
)
from ..tasks.base import CodexExecutor

#: Substrings that mark a transient error worth retrying the whole sample for.
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
_FRAME_CACHE_DIRNAME = "frame_cache"


@dataclass(frozen=True)
class OpenEqaQuestionResult:
    """The scored outcome of one OpenEQA question."""

    question_id: str
    clip_id: str
    scene_id: str
    category: str
    status: str
    question: str = ""
    gt_answer: str = ""
    prediction: str = ""
    judge_score: int | None = None
    mnas: float | None = None
    confidence: float | None = None
    supporting_claims: tuple[str, ...] = field(default_factory=tuple)
    num_frames: int = 0
    reasoning_summary: str | None = None
    error: str | None = None
    turn_duration_ms: int | None = None
    turn_usage: dict[str, Any] | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint view of the result."""
        return {
            "question_id": self.question_id,
            "clip_id": self.clip_id,
            "scene_id": self.scene_id,
            "category": self.category,
            "status": self.status,
            "question": self.question,
            "gt_answer": self.gt_answer,
            "prediction": self.prediction,
            "judge_score": self.judge_score,
            "mnas": self.mnas,
            "confidence": self.confidence,
            "supporting_claims": list(self.supporting_claims),
            "num_frames": self.num_frames,
            "reasoning_summary": self.reasoning_summary,
            "error": self.error,
            "turn_duration_ms": self.turn_duration_ms,
            "turn_usage": self.turn_usage,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpenEqaQuestionResult:
        """Rebuild a result from a checkpoint dict."""
        return cls(
            question_id=str(payload["question_id"]),
            clip_id=str(payload.get("clip_id", "")),
            scene_id=str(payload.get("scene_id", "")),
            category=str(payload.get("category", "unknown")),
            status=str(payload["status"]),
            question=str(payload.get("question", "")),
            gt_answer=str(payload.get("gt_answer", "")),
            prediction=str(payload.get("prediction", "")),
            judge_score=_optional_int(payload.get("judge_score")),
            mnas=_optional_float(payload.get("mnas")),
            confidence=_optional_float(payload.get("confidence")),
            supporting_claims=tuple(payload.get("supporting_claims", []) or []),
            num_frames=int(payload.get("num_frames", 0)),
            reasoning_summary=payload.get("reasoning_summary"),
            error=payload.get("error"),
            turn_duration_ms=_optional_int(payload.get("turn_duration_ms")),
            turn_usage=payload.get("turn_usage"),
            input_tokens=_optional_int(payload.get("input_tokens")),
            cached_input_tokens=_optional_int(payload.get("cached_input_tokens")),
        )

    def prediction_record(self) -> dict[str, str]:
        """The OpenEQA prediction-file record for this question."""
        return {"question_id": self.question_id, "answer": self.prediction}


@dataclass(frozen=True)
class OpenEqaRunSummary:
    """Aggregate MNAS + completion + prompt-cache metrics over a question set."""

    n: int
    n_judged: int
    mnas: float
    per_category_mnas: Mapping[str, float] = field(default_factory=dict)
    completion_rate: float = 0.0
    cache_hit_rate: float = 0.0
    mean_cache_ratio: float = 0.0
    results: tuple[OpenEqaQuestionResult, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_per_sample: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n": self.n,
            "n_judged": self.n_judged,
            "MNAS": self.mnas,
            "per_category_MNAS": dict(self.per_category_mnas),
            "completion_rate": self.completion_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "mean_cache_ratio": self.mean_cache_ratio,
        }
        if include_per_sample:
            payload["per_sample"] = [result.to_dict() for result in self.results]
        return payload


def run_one_question(
    question: OpenEqaQuestion,
    *,
    data_root: Path,
    runtime: CodexExecutor,
    judge: JudgeScorer | None,
    frame_cache_dir: Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    skill: CodexSkill | None = None,
    tools_enabled: bool = False,
) -> OpenEqaQuestionResult:
    """Execute, answer, and (optionally) judge a single OpenEQA question."""
    scene_dir = scene_dir_for(data_root, question.clip_id)
    scene = OpenEqaScene.load(scene_dir)
    frames = scene.prepare_frames(
        cache_dir=frame_cache_dir,
        num_frames=num_frames,
        max_image_size=max_image_size,
    )
    task = OpenEqaQuestionAnsweringTask(
        question=question,
        frames=frames,
        skill=skill,
        tools_enabled=tools_enabled,
        scene_dir=scene_dir,
    )
    result = runtime.execute(task)
    outcome = result.outcome
    metadata = result.turn.metadata

    judge_score, judge_error = _judge_prediction(
        judge, question=question, prediction=outcome.answer, completed=outcome.status
    )
    return OpenEqaQuestionResult(
        question_id=question.question_id,
        clip_id=question.clip_id,
        scene_id=question.scene_id,
        category=question.category,
        status=outcome.status,
        question=question.question,
        gt_answer=question.answer,
        prediction=outcome.answer,
        judge_score=judge_score,
        mnas=score_to_mnas(judge_score) if judge_score is not None else None,
        confidence=outcome.confidence,
        supporting_claims=outcome.supporting_claims,
        num_frames=len(frames),
        reasoning_summary=metadata.reasoning_summary,
        error=judge_error,
        turn_duration_ms=metadata.duration_ms,
        turn_usage=dict(metadata.usage) if metadata.usage is not None else None,
        input_tokens=metadata.input_tokens,
        cached_input_tokens=metadata.cached_input_tokens,
    )


def run_questions(
    *,
    questions: Sequence[OpenEqaQuestion],
    data_root: Path,
    output_dir: Path,
    runtime: CodexExecutor,
    judge: JudgeScorer | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    skill: CodexSkill | None = None,
    workers: int = 1,
    sample_retries: int = 2,
    tools_enabled: bool = False,
) -> OpenEqaRunSummary:
    """Run a question set with per-question checkpointing and aggregate MNAS."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if sample_retries < 0:
        raise ValueError("sample_retries must be non-negative")
    _validate_unique(questions)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_dir = output_dir / _FRAME_CACHE_DIRNAME

    pending = [
        question
        for question in questions
        if not _checkpoint_path(output_dir, question.question_id).exists()
    ]
    logger.info(
        "openeqa run: {} questions ({} pending, {} cached), workers={}, judge={}",
        len(questions),
        len(pending),
        len(questions) - len(pending),
        workers,
        "on" if judge is not None else "off",
    )

    def process(question: OpenEqaQuestion) -> None:
        result = _run_with_retries(
            question,
            data_root=data_root,
            runtime=runtime,
            judge=judge,
            frame_cache_dir=frame_cache_dir,
            num_frames=num_frames,
            max_image_size=max_image_size,
            skill=skill,
            sample_retries=sample_retries,
            tools_enabled=tools_enabled,
        )
        _write_checkpoint(output_dir, result)

    if workers == 1:
        for question in pending:
            process(question)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, q) for q in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    results = tuple(
        _read_checkpoint(output_dir, question.question_id) for question in questions
    )
    summary = _summarize(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_predictions(output_dir, results)
    return summary


def _judge_prediction(
    judge: JudgeScorer | None,
    *,
    question: OpenEqaQuestion,
    prediction: str,
    completed: str,
) -> tuple[int | None, str | None]:
    """Return ``(judge_score, error)`` for one prediction.

    A turn that produced no answer scores the floor (``0``) so it still counts
    toward MNAS; a real prediction is graded by the judge. When scoring is
    disabled (``judge is None``) no score is produced. A judge failure on a real
    prediction is recorded as an error and excluded from MNAS rather than failing
    the question.
    """
    if judge is None:
        return None, None
    if completed != "completed" or not prediction.strip():
        return NO_PREDICTION_SCORE, None
    try:
        score = judge.score(
            question=question.question,
            gt_answer=question.answer,
            prediction=prediction,
            extra_answers=question.extra_answers,
        )
        return score, None
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed into success
        logger.error("openeqa judge failed for {}: {}", question.question_id, exc)
        return None, f"judge_error: {type(exc).__name__}: {str(exc)[:300]}"


def _run_with_retries(
    question: OpenEqaQuestion,
    *,
    data_root: Path,
    runtime: CodexExecutor,
    judge: JudgeScorer | None,
    frame_cache_dir: Path,
    num_frames: int,
    max_image_size: int,
    skill: CodexSkill | None,
    sample_retries: int,
    tools_enabled: bool,
) -> OpenEqaQuestionResult:
    last_error: Exception | None = None
    for attempt in range(sample_retries + 1):
        try:
            return run_one_question(
                question,
                data_root=data_root,
                runtime=runtime,
                judge=judge,
                frame_cache_dir=frame_cache_dir,
                num_frames=num_frames,
                max_image_size=max_image_size,
                skill=skill,
                tools_enabled=tools_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - sentinel result built below
            last_error = exc
            if attempt >= sample_retries or not _is_retryable_error(exc):
                break
            logger.warning(
                "openeqa question {} failed (attempt {}/{}): {}",
                question.question_id,
                attempt + 1,
                sample_retries + 1,
                exc,
            )
            time.sleep(2.0 * (attempt + 1))
    logger.error(
        "openeqa question {} failed permanently: {}", question.question_id, last_error
    )
    # A permanently-failed turn produced no answer; it scores the floor so it
    # still counts toward MNAS when scoring is enabled.
    failed_score = NO_PREDICTION_SCORE if judge is not None else None
    return OpenEqaQuestionResult(
        question_id=question.question_id,
        clip_id=question.clip_id,
        scene_id=question.scene_id,
        category=question.category,
        status="failed",
        question=question.question,
        gt_answer=question.answer,
        judge_score=failed_score,
        mnas=score_to_mnas(failed_score) if failed_score is not None else None,
        error=f"{type(last_error).__name__}: {str(last_error)[:480]}",
    )


def _summarize(results: Sequence[OpenEqaQuestionResult]) -> OpenEqaRunSummary:
    n = len(results)
    judged = [r for r in results if r.judge_score is not None]
    mnas_values = [
        score_to_mnas(r.judge_score) for r in judged if r.judge_score is not None
    ]
    mnas = sum(mnas_values) / len(mnas_values) if mnas_values else 0.0
    completed = sum(1 for r in results if r.status == "completed")
    completion_rate = completed / n if n else 0.0

    cache_ratios = [
        r.cached_input_tokens / r.input_tokens
        for r in results
        if r.input_tokens and r.cached_input_tokens is not None
    ]
    measured = max(len(cache_ratios), 1)
    cache_hit_rate = sum(1 for ratio in cache_ratios if ratio > 0.0) / measured
    mean_cache_ratio = sum(cache_ratios) / measured if cache_ratios else 0.0
    return OpenEqaRunSummary(
        n=n,
        n_judged=len(judged),
        mnas=mnas,
        per_category_mnas=_per_category_mnas(judged),
        completion_rate=completion_rate,
        cache_hit_rate=cache_hit_rate,
        mean_cache_ratio=mean_cache_ratio,
        results=tuple(results),
    )


def _per_category_mnas(
    judged: Sequence[OpenEqaQuestionResult],
) -> dict[str, float]:
    by_category: dict[str, list[float]] = {}
    for result in judged:
        if result.judge_score is None:
            continue
        by_category.setdefault(result.category, []).append(
            score_to_mnas(result.judge_score)
        )
    return {
        category: sum(values) / len(values)
        for category, values in sorted(by_category.items())
    }


def _write_predictions(
    output_dir: Path, results: Sequence[OpenEqaQuestionResult]
) -> Path:
    """Write the OpenEQA prediction file (``[{question_id, answer}, ...]``)."""
    path = output_dir / "predictions.json"
    path.write_text(
        json.dumps(
            [result.prediction_record() for result in results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_TOKENS)


def _checkpoint_path(output_dir: Path, question_id: str) -> Path:
    digest = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:12]
    safe = _safe_question_id(question_id)
    return output_dir / "per_sample" / f"{safe}_{digest}.json"


def _safe_question_id(question_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in question_id)[:64]


def _write_checkpoint(output_dir: Path, result: OpenEqaQuestionResult) -> Path:
    path = _checkpoint_path(output_dir, result.question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _read_checkpoint(output_dir: Path, question_id: str) -> OpenEqaQuestionResult:
    path = _checkpoint_path(output_dir, question_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a JSON object: {path}")
    return OpenEqaQuestionResult.from_dict(payload)


def _validate_unique(questions: Sequence[OpenEqaQuestion]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for question in questions:
        if question.question_id in seen:
            duplicates.append(question.question_id)
        seen.add(question.question_id)
    if duplicates:
        raise ValueError(
            f"question_ids must be unique for checkpointing; duplicates={duplicates[:10]}"
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "OpenEqaQuestionResult",
    "OpenEqaRunSummary",
    "run_one_question",
    "run_questions",
]
