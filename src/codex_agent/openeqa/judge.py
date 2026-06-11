"""OpenEQA LLM-as-judge scoring (the official 1-5 ``mmbench`` protocol).

The judge asks a separate LLM to grade a predicted answer against the reference
answer on an integer 1-5 scale (the upstream OpenEQA ``mmbench`` prompt), which
is then mapped to the headline metric, Mean Normalized Accuracy Score (MNAS):

    MNAS = 100 * (clip(score, 1, 5) - 1) / 4

The runner depends only on the :class:`JudgeScorer` protocol, so a fake judge
can be injected in tests while :class:`LlmJudge` (backed by the project's pooled
LLM client) is used in production.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger

from ..errors import OpenEqaJudgeError

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from keyframe.llm.client import LLMClient

#: A prediction below this score (i.e. a missing answer) earns the floor score.
NO_PREDICTION_SCORE = 0
_MIN_SCORE = 1
_MAX_SCORE = 5
_SCORE_TAG = "Your mark:"
_INTEGER_RE = re.compile(r"\d+")

#: Upstream OpenEQA judge prompt (``external/open-eqa/prompts/mmbench.txt``),
#: reproduced verbatim so scores match the official protocol.
_MMBENCH_PROMPT = """\
You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}
"""

_EXTRA_ANSWERS_LINE = "Extra Answers: {extra_answers}\n"

#: Judge sampling defaults (match the upstream ``get_llm_match_score`` defaults).
DEFAULT_JUDGE_TEMPERATURE = 0.2
DEFAULT_JUDGE_MAX_TOKENS = 32


@runtime_checkable
class JudgeScorer(Protocol):
    """Grades a predicted answer against a reference answer on a 1-5 scale."""

    def score(
        self,
        *,
        question: str,
        gt_answer: str,
        prediction: str,
        extra_answers: Sequence[str] = (),
    ) -> int:
        """Return an integer 1-5 (or :data:`NO_PREDICTION_SCORE` for no answer)."""
        ...


def build_judge_prompt(
    *,
    question: str,
    gt_answer: str,
    prediction: str,
    extra_answers: Sequence[str] = (),
) -> str:
    """Render the ``mmbench`` judge prompt for one (question, answer, prediction)."""
    prompt = _MMBENCH_PROMPT.format(
        question=question, answer=gt_answer, prediction=prediction
    )
    if extra_answers:
        prompt += _EXTRA_ANSWERS_LINE.format(extra_answers=list(extra_answers))
    return prompt


def parse_judge_score(output: str) -> int:
    """Parse the integer mark out of a judge response.

    Accepts a bare integer or text containing ``"Your mark: N"`` (the integer
    after the tag is used; otherwise the first integer in the text).

    Raises:
        OpenEqaJudgeError: If no integer can be found.
    """
    text = output.strip()
    if not text:
        raise OpenEqaJudgeError("judge returned an empty response")
    if text.isdigit():
        return int(text)
    tag_index = text.find(_SCORE_TAG)
    region = text[tag_index + len(_SCORE_TAG) :] if tag_index != -1 else text
    match = _INTEGER_RE.search(region)
    if match is None:
        raise OpenEqaJudgeError(
            f"could not parse a 1-5 score from judge output: {text[:200]!r}"
        )
    return int(match.group())


def score_to_mnas(score: int) -> float:
    """Map a raw 0-5 judge score to its normalized-accuracy contribution."""
    clamped = min(max(score, _MIN_SCORE), _MAX_SCORE)
    return 100.0 * (clamped - _MIN_SCORE) / (_MAX_SCORE - _MIN_SCORE)


def mean_mnas(scores: Sequence[int]) -> float:
    """Mean MNAS over a sequence of raw judge scores (``0.0`` when empty)."""
    if not scores:
        return 0.0
    return sum(score_to_mnas(score) for score in scores) / len(scores)


class LlmJudge:
    """A :class:`JudgeScorer` backed by the project's pooled chat LLM client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
        max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise OpenEqaJudgeError(
                f"max_retries must be non-negative, got {max_retries}"
            )
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    @classmethod
    def from_toml(
        cls,
        config_path: str | Path | None = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
    ) -> LlmJudge:
        """Build a judge from the project LLM config (``configs/llm.toml``)."""
        from keyframe.llm.client import LLMClient

        path = str(config_path) if config_path is not None else None
        return cls(LLMClient.from_toml(path), model=model, max_retries=max_retries)

    def score(
        self,
        *,
        question: str,
        gt_answer: str,
        prediction: str,
        extra_answers: Sequence[str] = (),
    ) -> int:
        """Grade ``prediction`` against ``gt_answer`` via the judge LLM."""
        if not prediction.strip():
            return NO_PREDICTION_SCORE
        prompt = build_judge_prompt(
            question=question,
            gt_answer=gt_answer,
            prediction=prediction,
            extra_answers=extra_answers,
        )
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                message = self._client.invoke(
                    prompt,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return parse_judge_score(_message_text(message))
            except Exception as exc:  # noqa: BLE001 - re-raised with context below
                last_error = exc
                logger.warning(
                    "openeqa judge attempt {}/{} failed: {}",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(1.0 * (attempt + 1))
        raise OpenEqaJudgeError(
            f"judge failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error


def _message_text(message: BaseMessage) -> str:
    """Flatten a LangChain message's content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


__all__ = [
    "NO_PREDICTION_SCORE",
    "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_JUDGE_MAX_TOKENS",
    "JudgeScorer",
    "LlmJudge",
    "build_judge_prompt",
    "parse_judge_score",
    "score_to_mnas",
    "mean_mnas",
]
