"""OpenEQA question-answering task for the Codex Agent SDK runtime.

The task uses a **Map-First + Evidence-Seeking** initialization strategy:
* **No first-person RGB frames are attached** to the initial turn — the agent
  must fetch and view first-person frames and high-resolution crops on demand
  via the OpenEQA CLI tools (``view_frame``, ``keyframe_selector``,
  ``view_crop``, ``list_objects``, ``view_bev``) and ``view_image``.
* When the prepared clip includes a ``conceptgraph/`` pack, the turn attaches
  the top-down **Schematic BEV map** (``1`` overview image) and inlines the
  **Scene Grid Layout** plus a compact **3D Object Catalog preview** (including
  each object's ``(row, col)`` grid cell and top ``best_frames``) directly in
  the prompt. This eliminates blind initial orientation calls while keeping all
  first-person visual evidence strictly on-demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..errors import CodexResponseError
from ..json_extraction import extract_json_object
from ..models import CodexTurnRequest
from .playbook import OPENEQA_TOOLS_PLAYBOOK
from .question import OpenEqaQuestion
from .scene import OpenEqaScene
from .tools.scene_context import build_selector

TASK_NAME = "openeqa_question_answering"
_DEFAULT_MAX_CLAIMS = 6
_MAX_CATALOG_PREVIEW_OBJECTS = 65
_PREVIEW_NOTE_CHARS = 110
_PREVIEW_LOC_CHARS = 90
DEFAULT_TOOL_CLI_MODULE = "codex_agent.openeqa.tools"

_CATEGORY_GUIDANCE: dict[str, str] = {
    "functional reasoning": (
        "CATEGORY STRATEGY (functional reasoning): Look for the MOST SPECIFIC, "
        "purpose-dedicated item in the room that satisfies the need (e.g. an "
        "hourglass on a desk rather than a generic phone; furry earmuffs or a "
        "jacket rather than bed covers; green boxes of paper on a cabinet; a "
        "wall-mounted phone). Do NOT stop at a generic furniture item, and NEVER "
        "say 'impossible / none' without inspecting first-person frames of desks, "
        "walls, shelves, and counters."
    ),
    "world knowledge": (
        "CATEGORY STRATEGY (world knowledge): Combine visual details from "
        "high-res crops (view_crop) with real-world knowledge. If the question "
        "asks about a portrait, mural, brand, sign, or room purpose/capacity, "
        "zoom in with view_crop on the relevant object/wall and count all "
        "workstations/fixtures across the full room."
    ),
    "object localization": (
        "CATEGORY STRATEGY (object localization): When answering 'Where is X?', "
        "always include BOTH its immediate support surface/container AND its "
        "spatial relation to the nearest salient landmark (e.g. 'on the work "
        "table to the right of the green garbage can', 'on the lower closet "
        "shelf to the right'). Verify the exact position in a first-person frame."
    ),
    "spatial understanding": (
        "CATEGORY STRATEGY (spatial understanding): Judge left / right / in front "
        "of / behind / between strictly from a co-visible FIRST-PERSON frame "
        "(never from world X/Y coordinates alone). Use view_crop to verify the "
        "exact identity of the neighboring object."
    ),
    "object recognition": (
        "CATEGORY STRATEGY (object recognition): Do NOT trust the 3D catalog "
        "label blindly (small items like erasers, carbonators, vacuums, or "
        "chargers are often mislabeled in 3D detection). Always fetch the "
        "closest first-person frame, call view_crop on the target region, and "
        "view_image the high-res crop before naming the object."
    ),
    "attribute recognition": (
        "CATEGORY STRATEGY (attribute recognition): Colors, materials, and "
        "surface details shift or blur at 768px. Always call view_crop on the "
        "target object and view_image the high-res crop before committing to a "
        "color or attribute."
    ),
    "object state recognition": (
        "CATEGORY STRATEGY (object state recognition): Inspect the object's "
        "exact physical state (open vs. closed, on vs. off, wet vs. dry, clean "
        "vs. cluttered) using a close-up frame and view_crop. Check fine cues "
        "(e.g. blind slat tilt, notebook spine/pages, indicator lights, doorway "
        "opening)."
    ),
}


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


@dataclass(frozen=True)
class _InitialSceneContext:
    """Pre-extracted scene metadata, grid layout, catalog preview, and BEV path."""

    total_frames: int | None = None
    frame_id_range: tuple[int, int] | None = None
    rgb_frame_ids: tuple[int, ...] = ()
    stride: int | None = None
    bev_image_path: Path | None = None
    grid_layout_text: str = ""
    catalog_preview_text: str = ""


class OpenEqaQuestionAnsweringTask:
    """A :class:`codex_agent.tasks.base.CodexTask` for one OpenEQA question.

    Attaches no first-person RGB frames; when ``conceptgraph/`` assets exist
    under ``scene_dir``, attaches the top-down schematic BEV map and inlines the
    scene grid layout + compact object catalog so the agent can immediately fetch
    targeted first-person frames and high-resolution crops.
    """

    def __init__(
        self,
        *,
        question: OpenEqaQuestion,
        scene_dir: Path,
        max_supporting_claims: int = _DEFAULT_MAX_CLAIMS,
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
        attach_initial_bev: bool = True,
    ) -> None:
        self.question = question
        self.scene_dir = scene_dir
        self.max_supporting_claims = max_supporting_claims
        self.tool_cli_module = tool_cli_module
        self.attach_initial_bev = attach_initial_bev

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        ctx = self._build_initial_context()
        image_paths: tuple[Path, ...] = (
            (ctx.bev_image_path,)
            if self.attach_initial_bev and ctx.bev_image_path is not None
            else ()
        )
        return CodexTurnRequest(
            prompt=self._build_prompt(ctx),
            output_schema=OpenEqaAnswerDecision.model_json_schema(),
            skills=(),
            image_paths=image_paths,
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

    def _build_initial_context(self) -> _InitialSceneContext:
        total_frames: int | None = None
        frame_id_range: tuple[int, int] | None = None
        rgb_frame_ids: tuple[int, ...] = ()
        try:
            scene = OpenEqaScene.load(self.scene_dir)
            total_frames = scene.total_frames
            rgb_frame_ids = scene.rgb_frame_ids
            if rgb_frame_ids:
                frame_id_range = (rgb_frame_ids[0], rgb_frame_ids[-1])
        except Exception:
            pass

        cg_dir = self.scene_dir / "conceptgraph"
        if not cg_dir.is_dir():
            return _InitialSceneContext(
                total_frames=total_frames,
                frame_id_range=frame_id_range,
                rgb_frame_ids=rgb_frame_ids,
            )

        try:
            selector = build_selector(cg_dir)
        except Exception:
            return _InitialSceneContext(
                total_frames=total_frames,
                frame_id_range=frame_id_range,
                rgb_frame_ids=rgb_frame_ids,
            )

        stride = max(1, int(getattr(selector, "stride", 1) or 1))
        bev_path: Path | None = None
        if self.attach_initial_bev:
            try:
                candidate = selector.generate_scene_bev(use_cache=True)
                if candidate.is_file():
                    bev_path = candidate
            except Exception:
                bev_path = None

        grid_text = ""
        try:
            grid_text = selector._build_grid_context()
        except Exception:
            grid_text = ""

        catalog_text = _format_catalog_preview(
            selector, rgb_frame_ids=rgb_frame_ids, stride=stride
        )
        return _InitialSceneContext(
            total_frames=total_frames,
            frame_id_range=frame_id_range,
            rgb_frame_ids=rgb_frame_ids,
            stride=stride,
            bev_image_path=bev_path,
            grid_layout_text=grid_text,
            catalog_preview_text=catalog_text,
        )

    def _build_prompt(self, ctx: _InitialSceneContext) -> str:
        schema = OpenEqaAnswerDecision.model_json_schema()
        cat_key = self.question.category.strip().lower()
        cat_guidance = _CATEGORY_GUIDANCE.get(cat_key, "")
        cat_block = f"\n{cat_guidance}\n" if cat_guidance else ""

        frame_meta_lines: list[str] = []
        if ctx.total_frames is not None and ctx.frame_id_range is not None:
            frame_meta_lines.append(
                f"- total_frames: {ctx.total_frames} "
                f"(valid frame_id_range: [{ctx.frame_id_range[0]}, {ctx.frame_id_range[1]}])"
            )
        if ctx.stride is not None:
            frame_meta_lines.append(f"- trajectory_stride: {ctx.stride}")
        frame_meta_str = "\n" + "\n".join(frame_meta_lines) if frame_meta_lines else ""

        scene_overview_parts: list[str] = []
        if ctx.grid_layout_text:
            scene_overview_parts.append(
                "Scene Grid Layout (matches attached schematic BEV map):\n"
                + ctx.grid_layout_text
            )
        if ctx.catalog_preview_text:
            scene_overview_parts.append(
                "Detected 3D Object Catalog Preview "
                "(routing hint ONLY — small/wall objects may be omitted or mislabeled!):\n"
                + ctx.catalog_preview_text
            )
        scene_overview_block = (
            "\n\n" + "\n\n".join(scene_overview_parts) + "\n"
            if scene_overview_parts
            else ""
        )

        return (
            "You are answering one OpenEQA question about a 3D indoor scene using "
            "the Codex Agent SDK.\n\n"
            "Rules:\n" + "\n".join(self._rules(ctx)) + "\n\nTask:\n"
            f"- question: {self.question.question}\n"
            f"- category: {self.question.category}\n"
            f"- scene_id: {self.question.scene_id}"
            + frame_meta_str
            + "\n"
            + cat_block
            + self._tools_section()
            + scene_overview_block
            + "\nOutput JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    def _rules(self, ctx: _InitialSceneContext) -> list[str]:
        bev_note = (
            "A top-down schematic BEV map and 3D object catalog preview are "
            "provided for global routing, but NO first-person RGB frames are "
            "attached."
            if ctx.bev_image_path is not None
            else "NO images are attached to this message. You start with zero "
            "visual information about the scene."
        )
        return [
            f"- {bev_note} NO first-person images are attached.",
            "- You MUST gather first-person visual evidence yourself by running "
            "the OpenEQA CLI tools described below (use view_frame / view_crop "
            "directly when best_frames are listed in the catalog, or "
            "keyframe_selector for relational/uncataloged queries), then open "
            "each returned image_path or contact_sheet with the view_image tool. "
            "An image you have not viewed is not evidence.",
            "- Do NOT rely on the 3D catalog or BEV labels alone: small tabletop "
            "items and wall fixtures are frequently missing or mislabeled in 3D "
            "detection. Always verify in first-person frames and use view_crop "
            "for small objects, colors, labels, or states.",
            "- Answer concisely and factually as a short phrase, grounded ONLY in "
            "the first-person tool images/crops you fetched and viewed.",
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
            "- list_objects prints one JSON object; view_frame, keyframe_selector, "
            "view_crop, and view_bev also write an image and print its image_path — "
            "call view_image on that path before you rely on what it shows.\n"
            "\nYou are NOT exploring or editing a codebase — you are answering one "
            "question. Hard limits:\n"
            "- Everything you need is in THIS message. Do NOT read, cat, sed, "
            "head, grep, rg, or open any SKILL.md, AGENTS.md, README, docs, or "
            "source file, and do NOT list or search the repository. There is no "
            "separate skill file to consult.\n"
            "- Use ONLY the five OpenEQA tools plus view_image. Ignore any other "
            "skills, plugins, or playbooks.\n"
            "- Never re-run a tool with identical arguments and never re-view an "
            "image you have already seen; if a result is empty or errors, change "
            "approach instead of retrying.\n"
            "\n" + OPENEQA_TOOLS_PLAYBOOK + "\n"
        )


def _format_catalog_preview(
    selector: Any,
    *,
    rgb_frame_ids: tuple[int, ...],
    stride: int,
) -> str:
    objects = getattr(selector, "objects", []) or []
    object_to_views: dict[int, list[tuple[int, float]]] = (
        getattr(selector, "object_to_views", {}) or {}
    )
    object_to_grids: dict[int, list[tuple[int, int]]] = (
        getattr(selector, "object_to_grids", {}) or {}
    )
    available_set = set(rgb_frame_ids)

    lines: list[str] = []
    for obj in objects[:_MAX_CATALOG_PREVIEW_OBJECTS]:
        if getattr(obj, "centroid", None) is None:
            continue
        oid = int(obj.obj_id)
        category = str(obj.object_tag or obj.category or "object")
        cells = object_to_grids.get(oid, [])
        grid_str = f"({cells[0][0]},{cells[0][1]})" if cells else "?"

        best_frames: list[int] = []
        for vid, _ in object_to_views.get(oid, []):
            target = int(vid) * stride
            if rgb_frame_ids:
                fid = (
                    target
                    if target in available_set
                    else min(rgb_frame_ids, key=lambda f: abs(f - target))
                )
            else:
                fid = target
            if fid not in best_frames:
                best_frames.append(fid)
            if len(best_frames) >= 3:
                break

        parts = [f"- #{oid} {category}: grid={grid_str}, best_frames={best_frames}"]
        colors = [
            str(c) for c in (getattr(obj, "color", []) or [])[:3] if str(c).strip()
        ]
        if colors:
            parts.append(f"color={'/' .join(colors)}")
        summary = _compact_str(getattr(obj, "summary", ""), _PREVIEW_NOTE_CHARS)
        if summary:
            parts.append(f"note={summary}")
        location = _compact_str(getattr(obj, "location", ""), _PREVIEW_LOC_CHARS)
        if location:
            parts.append(f"loc={location}")
        lines.append("; ".join(parts))

    if len(objects) > _MAX_CATALOG_PREVIEW_OBJECTS:
        lines.append(
            f"... ({len(objects) - _MAX_CATALOG_PREVIEW_OBJECTS} more objects; "
            "call list_objects to search all)"
        )
    return "\n".join(lines)


def _compact_str(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


__all__ = [
    "TASK_NAME",
    "DEFAULT_TOOL_CLI_MODULE",
    "OpenEqaAnswerDecision",
    "OpenEqaAnswerOutcome",
    "OpenEqaQuestionAnsweringTask",
]
