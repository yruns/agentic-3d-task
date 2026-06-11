"""OpenEQA question-answering task built on the Codex Agent SDK runtime.

This task family answers open-ended OpenEQA questions over prepared ScanNet
clips: it samples first-person frames, asks Codex for a grounded answer, and
scores the answer with the official 1-5 ``mmbench`` LLM-as-judge mapped to MNAS.
"""

from __future__ import annotations

from .judge import (
    JudgeScorer,
    LlmJudge,
    build_judge_prompt,
    mean_mnas,
    parse_judge_score,
    score_to_mnas,
)
from .playbook import OPENEQA_TOOL_NAMES, OPENEQA_TOOLS_PLAYBOOK
from .qa import (
    DEFAULT_TOOL_CLI_MODULE,
    OpenEqaAnswerDecision,
    OpenEqaAnswerOutcome,
    OpenEqaQuestionAnsweringTask,
)
from .question import (
    OpenEqaQuestion,
    index_by_question_id,
    load_questions,
    select_questions,
)
from .scene import (
    OpenEqaFrame,
    OpenEqaScene,
    downsize_rgb_for_view,
    filter_questions_with_local_scenes,
    has_local_scene,
    scene_dir_for,
    uniform_frame_ids,
)

__all__ = [
    # Question domain
    "OpenEqaQuestion",
    "load_questions",
    "index_by_question_id",
    "select_questions",
    # Scene / frames
    "OpenEqaScene",
    "OpenEqaFrame",
    "scene_dir_for",
    "has_local_scene",
    "filter_questions_with_local_scenes",
    "uniform_frame_ids",
    "downsize_rgb_for_view",
    # Task
    "OpenEqaQuestionAnsweringTask",
    "OpenEqaAnswerDecision",
    "OpenEqaAnswerOutcome",
    "DEFAULT_TOOL_CLI_MODULE",
    # Tool playbook
    "OPENEQA_TOOLS_PLAYBOOK",
    "OPENEQA_TOOL_NAMES",
    # Judge / metrics
    "JudgeScorer",
    "LlmJudge",
    "build_judge_prompt",
    "parse_judge_score",
    "score_to_mnas",
    "mean_mnas",
]
