"""Unit tests for the OpenEQA LLM-as-judge: prompt, parsing, MNAS, client."""

from __future__ import annotations

import pytest

from codex_agent.errors import OpenEqaJudgeError
from codex_agent.openeqa.judge import (
    LlmJudge,
    build_judge_prompt,
    mean_mnas,
    parse_judge_score,
    score_to_mnas,
)


class _FakeMessage:
    """A minimal stand-in for a LangChain ``BaseMessage``."""

    def __init__(self, content: object) -> None:
        self.content = content


class _FakeClient:
    """A stand-in LLM client recording invocations and returning fixed text."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt, *, model=None, temperature=None, max_tokens=None):
        self.calls.append(prompt)
        value = self._responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return _FakeMessage(value)


def test_build_judge_prompt_contains_inputs() -> None:
    prompt = build_judge_prompt(
        question="What color is the rug?",
        gt_answer="tan",
        prediction="brown",
    )
    assert "Question: What color is the rug?" in prompt
    assert "Answer: tan" in prompt
    assert "Response: brown" in prompt
    assert "Your mark:" in prompt  # few-shot examples present


def test_build_judge_prompt_includes_extra_answers() -> None:
    prompt = build_judge_prompt(
        question="q",
        gt_answer="a",
        prediction="p",
        extra_answers=["b", "c"],
    )
    assert "Extra Answers:" in prompt
    assert "'b'" in prompt and "'c'" in prompt


def test_parse_judge_score_bare_integer() -> None:
    assert parse_judge_score("4") == 4


def test_parse_judge_score_with_tag() -> None:
    assert parse_judge_score("Reasoning...\nYour mark: 3") == 3


def test_parse_judge_score_with_trailing_text() -> None:
    assert parse_judge_score("Your mark: 5.") == 5


def test_parse_judge_score_no_tag_uses_first_integer() -> None:
    assert parse_judge_score("I would say 2 out of 5") == 2


def test_parse_judge_score_empty_raises() -> None:
    with pytest.raises(OpenEqaJudgeError):
        parse_judge_score("   ")


def test_parse_judge_score_no_integer_raises() -> None:
    with pytest.raises(OpenEqaJudgeError):
        parse_judge_score("no number here")


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, 0.0), (2, 25.0), (3, 50.0), (4, 75.0), (5, 100.0)],
)
def test_score_to_mnas_scale(score: int, expected: float) -> None:
    assert score_to_mnas(score) == pytest.approx(expected)


def test_score_to_mnas_clamps_out_of_range() -> None:
    assert score_to_mnas(0) == pytest.approx(0.0)  # no-prediction floor
    assert score_to_mnas(7) == pytest.approx(100.0)


def test_mean_mnas() -> None:
    assert mean_mnas([]) == 0.0
    assert mean_mnas([1, 5]) == pytest.approx(50.0)
    assert mean_mnas([5, 5, 3]) == pytest.approx((100 + 100 + 50) / 3)


def test_llm_judge_scores_prediction() -> None:
    client = _FakeClient(["Your mark: 4"])
    judge = LlmJudge(client)  # type: ignore[arg-type]
    score = judge.score(question="q", gt_answer="a", prediction="b")
    assert score == 4
    assert len(client.calls) == 1


def test_llm_judge_empty_prediction_scores_zero() -> None:
    client = _FakeClient([])
    judge = LlmJudge(client)  # type: ignore[arg-type]
    score = judge.score(question="q", gt_answer="a", prediction="   ")
    assert score == 0
    assert client.calls == []  # judge LLM not called for a missing prediction


def test_llm_judge_handles_list_content() -> None:
    client = _FakeClient([[{"type": "text", "text": "Your mark: 2"}]])
    judge = LlmJudge(client)  # type: ignore[arg-type]
    assert judge.score(question="q", gt_answer="a", prediction="b") == 2


def test_llm_judge_handles_scalar_content() -> None:
    # A message whose content is neither str nor list falls back to str().
    client = _FakeClient([5])
    judge = LlmJudge(client)  # type: ignore[arg-type]
    assert judge.score(question="q", gt_answer="a", prediction="b") == 5


def test_llm_judge_retries_then_succeeds() -> None:
    client = _FakeClient([RuntimeError("transient"), "Your mark: 5"])
    judge = LlmJudge(client, max_retries=1)  # type: ignore[arg-type]
    assert judge.score(question="q", gt_answer="a", prediction="b") == 5
    assert len(client.calls) == 2


def test_llm_judge_raises_after_exhausting_retries() -> None:
    client = _FakeClient([RuntimeError("a"), RuntimeError("b")])
    judge = LlmJudge(client, max_retries=1)  # type: ignore[arg-type]
    with pytest.raises(OpenEqaJudgeError):
        judge.score(question="q", gt_answer="a", prediction="b")


def test_llm_judge_rejects_negative_retries() -> None:
    client = _FakeClient([])
    with pytest.raises(OpenEqaJudgeError):
        LlmJudge(client, max_retries=-1)  # type: ignore[arg-type]
