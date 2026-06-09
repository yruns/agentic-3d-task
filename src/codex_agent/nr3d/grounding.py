"""NR3D visual-grounding task: catalog-first proposal selection via Codex.

The task presents the referring expression, a BEV render, and the scene's
proposal catalog, and asks Codex to pick exactly one ``proposal_id`` (or ``-1``
when the target is absent). For a ``source: gt`` pool the selected proposal's
box is the prediction scored against ground truth.

No ground-truth fields (the target id, category, or box) are ever placed in the
prompt — only the user's query and the candidate catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..errors import CodexResponseError
from ..json_extraction import extract_json_object
from ..models import CodexSkill, CodexTurnRequest
from .proposals import Proposal
from .sample import Nr3dSample, Nr3dScene

TASK_NAME = "nr3d_visual_grounding"
_TARGET_ABSENT_ID = -1
_DEFAULT_NOTE_CHARS = 220
_DEFAULT_VISIBLE_PREVIEW = 8
DEFAULT_TOOL_CLI_MODULE = "codex_agent.nr3d.tools"


class Nr3dGroundingDecision(BaseModel):
    """The strict JSON contract requested from Codex for one sample.

    All fields are required and extra keys are forbidden so the generated JSON
    schema is compatible with the upstream's strict ``response_format`` (it
    requires ``additionalProperties: false``). Strict structured output makes
    the model emit a complete, valid object as its final answer, so a tool-using
    agent cannot end the turn with a terse non-JSON reply.
    """

    model_config = {"extra": "forbid"}

    proposal_id: int = Field(
        description="Selected proposal id from the pool; -1 if the target is absent."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    uncertainties: list[str]
    cited_frame_indices: list[int]


@dataclass(frozen=True)
class Nr3dGroundingOutcome:
    """Parsed result of one grounding turn."""

    proposal_id: int
    selected_bbox_9dof: tuple[float, ...] | None
    confidence: float
    summary: str
    uncertainties: tuple[str, ...] = field(default_factory=tuple)
    cited_frame_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def target_present(self) -> bool:
        """Whether Codex claims the referred target exists in the pool."""
        return self.proposal_id != _TARGET_ABSENT_ID

    @property
    def status(self) -> Literal["completed", "failed"]:
        """``completed`` when a proposal was selected, else ``failed``."""
        return "completed" if self.target_present else "failed"


class Nr3dGroundingTask:
    """A :class:`codex_agent.tasks.base.CodexTask` for one NR3D sample."""

    def __init__(
        self,
        *,
        sample: Nr3dSample,
        scene: Nr3dScene,
        skill: CodexSkill | None = None,
        note_max_chars: int = _DEFAULT_NOTE_CHARS,
        visible_frames_preview: int = _DEFAULT_VISIBLE_PREVIEW,
        tools_enabled: bool = False,
        scene_dir: Path | None = None,
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
    ) -> None:
        self.sample = sample
        self.scene = scene
        self.skill = skill
        self.note_max_chars = note_max_chars
        self.visible_frames_preview = visible_frames_preview
        self.tools_enabled = tools_enabled and scene_dir is not None
        self.scene_dir = scene_dir
        self.tool_cli_module = tool_cli_module

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=Nr3dGroundingDecision.model_json_schema(),
            skills=(self.skill,) if self.skill is not None else (),
            image_paths=(self.scene.bev_image_path,),
        )

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self._parse_decision(response_text)
        except CodexResponseError:
            return False
        return True

    def parse_response(self, response_text: str) -> Nr3dGroundingOutcome:
        decision = self._parse_decision(response_text)
        valid_ids = self.scene.proposal_pool.ids()
        if (
            decision.proposal_id != _TARGET_ABSENT_ID
            and decision.proposal_id not in valid_ids
        ):
            raise CodexResponseError(
                f"Codex selected proposal_id={decision.proposal_id}, which is not "
                f"in the pool for scene {self.scene.scene_id}"
            )
        selected_bbox: tuple[float, ...] | None = None
        if decision.proposal_id != _TARGET_ABSENT_ID:
            selected_bbox = self.scene.proposal_pool.require(
                decision.proposal_id
            ).bbox_3d_9dof
        return Nr3dGroundingOutcome(
            proposal_id=decision.proposal_id,
            selected_bbox_9dof=selected_bbox,
            confidence=decision.confidence,
            summary=decision.summary,
            uncertainties=tuple(decision.uncertainties),
            cited_frame_indices=tuple(decision.cited_frame_indices),
        )

    def _parse_decision(self, response_text: str) -> Nr3dGroundingDecision:
        payload = extract_json_object(response_text)
        try:
            return Nr3dGroundingDecision.model_validate(payload)
        except ValidationError as exc:
            raise CodexResponseError(
                f"Codex response does not match the grounding schema: {exc}"
            ) from exc

    def _build_prompt(self) -> str:
        pool = self.scene.proposal_pool
        category_lines = [
            f"- {category}: {ids}" for category, ids in pool.ids_by_category().items()
        ]
        proposal_lines = [
            self._format_proposal(proposal)
            for proposal in sorted(pool.proposals, key=lambda p: p.proposal_id)
        ]
        schema = Nr3dGroundingDecision.model_json_schema()
        return (
            "You are solving one NR3D visual grounding sample using the Codex "
            "Agent SDK.\n\n"
            "Rules:\n" + "\n".join(self._rules()) + "\n\nTask:\n"
            f"- query: {self.sample.query}\n"
            f"- scene_id: {self.scene.scene_id}\n"
            f"- scene_category: {self.scene.scene_category or 'unknown'}\n"
            f"- total_frames: {self._fmt(self.scene.total_frames)}\n"
            f"- frame_id_range: {self._fmt_range(self.scene.frame_id_range)}\n"
            + self._tools_section()
            + "\nProposals by category:\n"
            + "\n".join(category_lines)
            + "\n\nProposal pool:\n"
            + "\n".join(proposal_lines)
            + "\n\nOutput JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    def _rules(self) -> list[str]:
        rules = [
            "- Pick exactly one proposal id from the provided proposal pool.",
            "- Use -1 only if the described target is absent from the pool.",
            "- A top-down BEV image of the scene is attached; use it together "
            "with the printed proposal metadata.",
            "- Do not use benchmark ground-truth fields; none are provided.",
        ]
        if self.tools_enabled:
            rules.append(
                "- You may run the NR3D CLI tools (below) to gather first-person "
                "frames, spatial rankings, and BEV highlights. Always open any "
                "returned image_path with the view_image tool before citing that "
                "frame as evidence."
            )
            rules.append(
                "- When finished, your FINAL message must be exactly one JSON "
                "object matching the schema with all keys present: proposal_id "
                "(an integer id from the pool, or -1 if absent), confidence, "
                "summary, uncertainties, cited_frame_indices. Example: "
                '{"proposal_id": 12, "confidence": 0.8, "summary": "...", '
                '"uncertainties": [], "cited_frame_indices": [52]}. Do not reply '
                "with a bare number or prose."
            )
        else:
            rules.append(
                "- Return only one JSON object matching the schema. Do not write "
                "files."
            )
        return rules

    def _tools_section(self) -> str:
        if not self.tools_enabled or self.scene_dir is None:
            return ""
        return (
            "\nEvidence tools (run in the shell; the attached skill explains the "
            "full loop):\n"
            f"- scene_dir: {self.scene_dir}\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--scene-dir {self.scene_dir} --args '<json>'\n"
            "- tools: inspect_proposal, list_scene_proposals, select_by_proposal, "
            "list_frame_proposals, keyframe_selector, mark_frame_with_bbox, "
            "compare_proposals_spatial, compare_candidates_to_anchors, view_bev\n"
            "- frame/BEV tools print an image_path; call view_image on it before "
            "you rely on what it shows.\n"
            "\nStay on task — hard limits:\n"
            "- Do NOT read, cat, sed, grep, or open any SKILL.md, AGENTS.md, "
            "README, docs, or source files. You already have every instruction "
            "you need in this prompt and the attached skill.\n"
            "- Use ONLY the nine NR3D tools above plus view_image. Ignore any "
            "other skills, plugins, or playbooks.\n"
            "- Never re-run a tool with the same arguments and never re-view an "
            "image you have already seen.\n"
            "- Decide within about 8 tool calls. Once you have inspected the top "
            "candidates and confirmed with one frame or one spatial comparison, "
            "output the final JSON immediately instead of gathering more.\n"
        )

    def _format_proposal(self, proposal: Proposal) -> str:
        cx, cy, cz, sx, sy, sz, rx, ry, rz = proposal.bbox_3d_9dof
        enriched = (
            f", enriched={proposal.enriched_category}"
            if proposal.enriched_category
            else ""
        )
        note = _truncate(
            proposal.compact_note or proposal.enriched_category or "",
            self.note_max_chars,
        )
        visible = proposal.visible_frame_ids
        preview = list(visible[: self.visible_frames_preview])
        return (
            f"- #{proposal.proposal_id}: category={proposal.category}{enriched}; "
            f"center=({cx:.3f},{cy:.3f},{cz:.3f}); "
            f"size=({sx:.3f},{sy:.3f},{sz:.3f}); "
            f"rot=({rx:.3f},{ry:.3f},{rz:.3f}); "
            f"visible_frames={len(visible)} {preview}; "
            f"note={note}"
        )

    @staticmethod
    def _fmt(value: int | None) -> str:
        return "unknown" if value is None else str(value)

    @staticmethod
    def _fmt_range(value: tuple[int, int] | None) -> str:
        return "unknown" if value is None else f"[{value[0]}, {value[1]}]"


def _truncate(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


__all__ = [
    "TASK_NAME",
    "Nr3dGroundingDecision",
    "Nr3dGroundingOutcome",
    "Nr3dGroundingTask",
]
