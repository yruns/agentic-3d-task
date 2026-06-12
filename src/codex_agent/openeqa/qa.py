"""OpenEQA question-answering task for the Codex Agent SDK runtime.

The task presents one open-ended question about a 3D indoor scene and attaches
**no images**: the agent has no visual information until it fetches it through the
OpenEQA CLI tools (``keyframe_selector`` / ``view_frame`` / ``view_bev`` /
``list_objects``) and views the returned images. The guide prompt tells the model
exactly how to call those tools, so retrieval is driven by the question instead of
a fixed uniform frame sample. The ground-truth answer is never placed in the
prompt — only the question text, its category, and the scene id — so the model
answers from the visual evidence it gathers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..errors import CodexResponseError
from ..json_extraction import extract_json_object
from ..models import CodexTurnRequest
from .playbook import OPENEQA_TOOLS_PLAYBOOK
from .question import OpenEqaQuestion

TASK_NAME = "openeqa_question_answering"
_DEFAULT_MAX_CLAIMS = 6
DEFAULT_TOOL_CLI_MODULE = "codex_agent.openeqa.tools"


class OpenEqaAnswerDecision(BaseModel):
    """The strict JSON contract requested from Codex for one question.

    All fields are required and extra keys are forbidden so the generated schema
    is compatible with the upstream's strict ``response_format`` (which requires
    ``additionalProperties: false``). Strict structured output keeps the model
    from ending the turn with a terse, unparseable reply.
    """

    model_config = {"extra": "forbid"}

    answer: str = Field(
        min_length=1,
        description="Concise factual answer (a short phrase), grounded in frames.",
    )
    supporting_claims: list[str] = Field(
        description="Brief evidence statements citing what the frames show."
    )
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class OpenEqaAnswerOutcome:
    """Parsed result of one OpenEQA answering turn."""

    answer: str
    confidence: float
    supporting_claims: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> Literal["completed", "failed"]:
        """``completed`` when a non-empty answer was produced, else ``failed``."""
        return "completed" if self.answer.strip() else "failed"


class OpenEqaQuestionAnsweringTask:
    """A :class:`codex_agent.tasks.base.CodexTask` for one OpenEQA question.

    The turn carries no attached images; the agent gathers visual evidence by
    running the OpenEQA CLI tools under ``scene_dir``. ``scene_dir`` is therefore
    required and the tool playbook is always inlined into the prompt (no on-disk
    skill is advertised, which avoids the context-window re-read loop).
    """

    def __init__(
        self,
        *,
        question: OpenEqaQuestion,
        scene_dir: Path,
        max_supporting_claims: int = _DEFAULT_MAX_CLAIMS,
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
    ) -> None:
        self.question = question
        self.scene_dir = scene_dir
        self.max_supporting_claims = max_supporting_claims
        self.tool_cli_module = tool_cli_module

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=OpenEqaAnswerDecision.model_json_schema(),
            skills=(),
            image_paths=(),
        )

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self._parse_decision(response_text)
        except CodexResponseError:
            return False
        return True

    def parse_response(self, response_text: str) -> OpenEqaAnswerOutcome:
        decision = self._parse_decision(response_text)
        return OpenEqaAnswerOutcome(
            answer=decision.answer.strip(),
            confidence=decision.confidence,
            supporting_claims=tuple(decision.supporting_claims),
        )

    def _parse_decision(self, response_text: str) -> OpenEqaAnswerDecision:
        payload = extract_json_object(response_text)
        try:
            return OpenEqaAnswerDecision.model_validate(payload)
        except ValidationError as exc:
            raise CodexResponseError(
                f"Codex response does not match the OpenEQA answer schema: {exc}"
            ) from exc

    def _build_prompt(self) -> str:
        schema = OpenEqaAnswerDecision.model_json_schema()
        return (
            "You are answering one OpenEQA question about a 3D indoor scene using "
            "the Codex Agent SDK.\n\n"
            "Rules:\n" + "\n".join(self._rules()) + "\n\nTask:\n"
            f"- question: {self.question.question}\n"
            f"- category: {self.question.category}\n"
            f"- scene_id: {self.question.scene_id}\n"
            + self._tools_section()
            + "\nOutput JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    def _rules(self) -> list[str]:
        return [
            "- NO images are attached to this message. You start with zero visual "
            "information about the scene.",
            "- You MUST gather the visual evidence yourself by running the OpenEQA "
            "CLI tools described below (start with keyframe_selector for a named "
            "object, or view_bev / list_objects for layout), then open each "
            "returned image_path with the view_image tool. An image you have not "
            "viewed is not evidence.",
            "- Answer concisely and factually as a short phrase, grounded ONLY in "
            "the tool images you fetched and viewed.",
            "- If the answer is not fully determinable, give your single best "
            "guess; do not refuse and do not answer with a question.",
            f"- Put at most {self.max_supporting_claims} brief evidence statements "
            "in supporting_claims.",
            "- Do not use benchmark ground-truth fields; none are provided.",
            "- When finished, your FINAL message must be exactly one JSON object "
            "matching the schema with all keys present: answer, supporting_claims, "
            "confidence. Do not reply with prose or a tool command.",
        ]

    def _tools_section(self) -> str:
        return (
            "\nHow to run a tool (in the shell):\n"
            f"- scene_dir: {self.scene_dir}\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--scene-dir {self.scene_dir} --args '<json>'\n"
            "- list_objects prints one JSON object; view_frame, keyframe_selector "
            "and view_bev also write an image and print its image_path — call "
            "view_image on that path before you rely on what it shows.\n"
            "\nYou are NOT exploring or editing a codebase — you are answering one "
            "question. Hard limits:\n"
            "- Everything you need is in THIS message. Do NOT read, cat, sed, "
            "head, grep, rg, or open any SKILL.md, AGENTS.md, README, docs, or "
            "source file, and do NOT list or search the repository. There is no "
            "separate skill file to consult.\n"
            "- Use ONLY the four OpenEQA tools plus view_image. Ignore any other "
            "skills, plugins, or playbooks.\n"
            "- Never re-run a tool with identical arguments and never re-view an "
            "image you have already seen; if a result is empty or errors, change "
            "approach instead of retrying.\n"
            "\n" + OPENEQA_TOOLS_PLAYBOOK + "\n"
        )


__all__ = [
    "TASK_NAME",
    "DEFAULT_TOOL_CLI_MODULE",
    "OpenEqaAnswerDecision",
    "OpenEqaAnswerOutcome",
    "OpenEqaQuestionAnsweringTask",
]
