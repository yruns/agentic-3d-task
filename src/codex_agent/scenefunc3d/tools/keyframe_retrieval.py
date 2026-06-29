"""Lightweight SceneFunc3D keyframe retrieval tool."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import NotRequired

from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_QUERY_FRAME_ID_RE = re.compile(
    r"\b(?:frame|frame_id|frame id|keyframe|view)\s*[:#-]?\s*(?P<frame_id>\d{1,6})\b",
    flags=re.IGNORECASE,
)
_OBJECT_FRAME_MAP_PATH = Path("indices") / "object_frame_map.json"
_MIN_OBJECT_SCORE = 0.1
_CO_OCCURRENCE_BONUS = 2.0
_PRIMARY_OBJECT_BONUS = 2.0


@dataclass(frozen=True)
class _ObjectQueryTerm:
    """One object-index query term and its visible-label aliases."""

    query_triggers: tuple[str, ...]
    label_aliases: tuple[str, ...]
    weight: float
    is_primary_object: bool = False


_OBJECT_QUERY_TERMS: tuple[_ObjectQueryTerm, ...] = (
    _ObjectQueryTerm(
        query_triggers=("drawer", "drawers"),
        label_aliases=("drawer",),
        weight=3.0,
        is_primary_object=True,
    ),
    _ObjectQueryTerm(
        query_triggers=("cabinet", "cabinets"),
        label_aliases=("cabinet",),
        weight=2.0,
        is_primary_object=True,
    ),
    _ObjectQueryTerm(
        query_triggers=("tv", "television"),
        label_aliases=("television", "tv", "screen"),
        weight=2.0,
        is_primary_object=True,
    ),
    _ObjectQueryTerm(
        query_triggers=("handle", "knob", "pull"),
        label_aliases=("handle", "knob", "pull"),
        weight=4.0,
    ),
    _ObjectQueryTerm(
        query_triggers=("radiator", "radiators", "heater", "heaters", "temperature"),
        label_aliases=("radiator", "heater"),
        weight=3.0,
        is_primary_object=True,
    ),
    _ObjectQueryTerm(
        query_triggers=("button", "buttons", "switch", "switches"),
        label_aliases=("button", "switch"),
        weight=4.0,
    ),
    _ObjectQueryTerm(
        query_triggers=("dial", "dials", "valve", "valves", "control", "controls"),
        label_aliases=("dial", "valve", "knob", "control"),
        weight=4.0,
    ),
)


class _ObjectFrameMapObject(BaseModel):
    """One visible object entry from ``object_frame_map.json``."""

    model_config = ConfigDict(extra="ignore")

    object_id: int | str
    class_name: str = Field(min_length=1)
    score: float = Field(ge=0.0, allow_inf_nan=False)
    bbox_xyxy: tuple[float, float, float, float] | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox_xyxy(
        cls, bbox_xyxy: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        """Validate optional pixel-space object bounds."""
        if bbox_xyxy is None:
            return None
        _validate_pixel_bbox(bbox_xyxy)
        return bbox_xyxy


class _ObjectFrameMapFrame(BaseModel):
    """One frame entry from ``object_frame_map.json``."""

    model_config = ConfigDict(extra="ignore")

    view_id: int
    frame_name: str = Field(min_length=1)
    objects: tuple[_ObjectFrameMapObject, ...] = ()


class _ObjectFrameMapDocument(BaseModel):
    """Validated ``object_frame_map.json`` payload."""

    model_config = ConfigDict(extra="ignore")

    frame_to_objects: dict[str, _ObjectFrameMapFrame]


class KeyframeSelectorArgs(BaseModel):
    """Arguments for ``keyframe_selector``."""

    model_config = ConfigDict(extra="forbid")

    query: QueryText
    k: int = Field(default=4, ge=1, le=8, strict=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_agent_aliases(cls, payload: object) -> object:
        """Accept common task-context names emitted by the agent."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        task_description = values.pop("task_description", None)
        target = values.pop("target", None)
        object_hint = values.pop("object_hint", None)
        if "query" not in values:
            if task_description is not None:
                values["query"] = task_description
            elif target is not None:
                values["query"] = target
            elif object_hint is not None:
                values["query"] = object_hint
        if "k" not in values:
            if "max_frames" in values:
                values["k"] = values.pop("max_frames")
            elif "top_k" in values:
                values["k"] = values.pop("top_k")
        values.pop("max_frames", None)
        values.pop("top_k", None)
        values.pop("annotation_ids", None)
        values.pop("motion_type", None)
        values.pop("motion_hints", None)
        return values


class KeyframeMatchedObjectPayload(TypedDict):
    """JSON-ready object metadata explaining a visible-object frame match."""

    object_id: str
    label: str
    score: float
    source: str
    bbox_xyxy: NotRequired[list[float]]
    bbox_format: NotRequired[Literal["pixel_xyxy"]]


@dataclass(frozen=True)
class KeyframeSelection:
    """One ranked SceneFunc3D frame candidate."""

    frame_id: str
    rank: int
    score: float
    reason: str
    matched_objects: tuple[KeyframeMatchedObjectPayload, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return this selection as a JSON-ready mapping."""
        payload: dict[str, object] = {
            "frame_id": self.frame_id,
            "rank": self.rank,
            "score": self.score,
            "reason": self.reason,
        }
        if self.matched_objects:
            payload["matched_objects"] = list(self.matched_objects)
        return payload


@dataclass(frozen=True)
class VisibleObjectFrameScore:
    """Semantic score for one frame derived from ConceptGraph visible objects."""

    frame_id: str
    score: float
    matched_objects: tuple[KeyframeMatchedObjectPayload, ...]
    primary_object_match_count: int = 0


@dataclass(frozen=True)
class _ScoredObjectFrame:
    """Score and matched-object evidence for one visible-object frame."""

    score: float
    matched_objects: tuple[KeyframeMatchedObjectPayload, ...]
    primary_object_match_count: int


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Ranked frame candidates for a query."""

    query: str
    frames: tuple[KeyframeSelection, ...]
    strategy: str

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        frame_payloads: list[dict[str, object]] = [
            frame.to_payload() for frame in self.frames
        ]
        return {
            "query": self.query,
            "strategy": self.strategy,
            "frames": frame_payloads,
        }


def keyframe_selector(
    tool_scene: SceneFunc3dToolScene, args: KeyframeSelectorArgs, *, out_dir: Path
) -> KeyframeSelectorResult:
    """Return query-aware deterministic keyframe candidates without CLIP ranking."""
    _ = out_dir
    query_frame_ids = _query_frame_ids(args.query)
    matched_frame_ids = tuple(
        frame_id for frame_id in tool_scene.rgb_frame_ids if frame_id in query_frame_ids
    )
    if matched_frame_ids:
        selected_frame_ids = _fill_with_coverage_frames(
            tool_scene.rgb_frame_ids,
            selected_frame_ids=matched_frame_ids,
            k=args.k,
        )
        frames = tuple(
            _selection_for(
                frame_id,
                rank=rank,
                query_matched=frame_id in matched_frame_ids,
            )
            for rank, frame_id in enumerate(selected_frame_ids, start=1)
        )
        return KeyframeSelectorResult(
            query=args.query,
            frames=frames,
            strategy="frame_id_match",
        )

    visible_object_scores = visible_object_frame_scores(tool_scene, args.query)
    if visible_object_scores:
        matched_frame_ids = tuple(score.frame_id for score in visible_object_scores)
        selected_frame_ids = _fill_with_coverage_frames(
            tool_scene.rgb_frame_ids,
            selected_frame_ids=matched_frame_ids,
            k=args.k,
        )
        score_by_frame_id = {
            frame_score.frame_id: frame_score for frame_score in visible_object_scores
        }
        frames = tuple(
            _selection_for_visible_object_frame(
                frame_id,
                rank=rank,
                visible_object_match=score_by_frame_id.get(frame_id),
            )
            for rank, frame_id in enumerate(selected_frame_ids, start=1)
        )
        return KeyframeSelectorResult(
            query=args.query,
            frames=frames,
            strategy="visible_object_query_match",
        )

    selected_frame_ids = _coverage_frame_ids(tool_scene.rgb_frame_ids, args.k)
    frames = tuple(
        KeyframeSelection(
            frame_id=frame_id,
            rank=rank,
            score=0.0,
            reason="coverage_fallback_no_query_match",
        )
        for rank, frame_id in enumerate(selected_frame_ids, start=1)
    )
    return KeyframeSelectorResult(
        query=args.query,
        frames=frames,
        strategy="coverage_fallback_no_query_match",
    )


def _query_frame_ids(query: str) -> set[str]:
    return {
        match.group("frame_id").zfill(6) for match in _QUERY_FRAME_ID_RE.finditer(query)
    }


def _selection_for(
    frame_id: str,
    *,
    rank: int,
    query_matched: bool,
) -> KeyframeSelection:
    if query_matched:
        return KeyframeSelection(
            frame_id=frame_id,
            rank=rank,
            score=100.0,
            reason="query_mentions_frame_id",
        )
    return KeyframeSelection(
        frame_id=frame_id,
        rank=rank,
        score=0.0,
        reason="coverage_fallback_no_query_match",
    )


def _selection_for_visible_object_frame(
    frame_id: str,
    *,
    rank: int,
    visible_object_match: VisibleObjectFrameScore | None,
) -> KeyframeSelection:
    if visible_object_match is None:
        return KeyframeSelection(
            frame_id=frame_id,
            rank=rank,
            score=0.0,
            reason="coverage_fallback_no_query_match",
        )
    return KeyframeSelection(
        frame_id=frame_id,
        rank=rank,
        score=visible_object_match.score,
        reason="visible_object_query_match",
        matched_objects=visible_object_match.matched_objects,
    )


def _fill_with_coverage_frames(
    frame_ids: tuple[str, ...],
    *,
    selected_frame_ids: tuple[str, ...],
    k: int,
) -> tuple[str, ...]:
    if len(selected_frame_ids) >= k:
        return selected_frame_ids[:k]
    selected_set = set(selected_frame_ids)
    coverage_frame_ids = tuple(
        frame_id
        for frame_id in _coverage_frame_ids(frame_ids, k)
        if frame_id not in selected_set
    )
    return (selected_frame_ids + coverage_frame_ids)[:k]


def _coverage_frame_ids(frame_ids: tuple[str, ...], k: int) -> tuple[str, ...]:
    if k >= len(frame_ids):
        return frame_ids
    if k == 1:
        return (frame_ids[len(frame_ids) // 2],)
    last_index = len(frame_ids) - 1
    selected_indices = tuple(
        round(position * last_index / (k - 1)) for position in range(k)
    )
    return tuple(frame_ids[index] for index in selected_indices)


def visible_object_frame_scores(
    tool_scene: SceneFunc3dToolScene, query: str
) -> tuple[VisibleObjectFrameScore, ...]:
    """Return visible-object query scores for frames in one SceneFunc3D scene."""
    query_terms = _object_query_terms_for_query(query)
    if not query_terms:
        return ()
    object_frame_map_path = tool_scene.conceptgraph_dir / _OBJECT_FRAME_MAP_PATH
    if not object_frame_map_path.is_file():
        return ()
    object_frame_map = _load_object_frame_map(object_frame_map_path)
    available_frame_ids = set(tool_scene.rgb_frame_ids)
    seen_frame_ids: set[str] = set()
    scored_frames: list[VisibleObjectFrameScore] = []
    for record_key, frame_record in object_frame_map.frame_to_objects.items():
        frame_id = _frame_id_from_object_frame(record_key, frame_record.frame_name)
        if frame_id is None or frame_id not in available_frame_ids:
            continue
        if frame_id in seen_frame_ids:
            raise ToolInputError(
                "visible-object index contains duplicate frame id after "
                f"normalization: frame_id={frame_id!r}; "
                f"path={object_frame_map_path}"
            )
        seen_frame_ids.add(frame_id)
        scored_frame = _score_object_frame(frame_record, query_terms)
        if scored_frame.score > 0.0:
            scored_frames.append(
                VisibleObjectFrameScore(
                    frame_id=frame_id,
                    score=scored_frame.score,
                    matched_objects=scored_frame.matched_objects,
                    primary_object_match_count=(
                        scored_frame.primary_object_match_count
                    ),
                )
            )
    return tuple(
        sorted(
            scored_frames,
            key=lambda frame_score: (
                -frame_score.primary_object_match_count,
                -frame_score.score,
                _numeric_frame_sort_key(frame_score.frame_id),
                frame_score.frame_id,
            ),
        )
    )


def _object_query_terms_for_query(query: str) -> tuple[_ObjectQueryTerm, ...]:
    query_lower = query.lower()
    return tuple(
        term
        for term in _OBJECT_QUERY_TERMS
        if any(
            _query_contains_trigger(query_lower, trigger)
            for trigger in term.query_triggers
        )
    )


def _query_contains_trigger(query_lower: str, trigger: str) -> bool:
    return re.search(rf"\b{re.escape(trigger)}\b", query_lower) is not None


def _load_object_frame_map(object_frame_map_path: Path) -> _ObjectFrameMapDocument:
    try:
        payload: object = json.loads(object_frame_map_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolInputError(
            f"could not read visible-object index: {object_frame_map_path}"
        ) from exc
    except JSONDecodeError as exc:
        raise ToolInputError(
            f"visible-object index is not valid JSON: {object_frame_map_path}"
        ) from exc
    try:
        return _ObjectFrameMapDocument.model_validate(payload)
    except ValidationError as exc:
        raise ToolInputError(
            "visible-object index failed validation: "
            f"path={object_frame_map_path}; error={_format_validation_error(exc)}"
        ) from exc


def _frame_id_from_object_frame(record_key: str, frame_name: str) -> str | None:
    frame_id_from_name = _frame_id_from_frame_name(frame_name)
    if frame_id_from_name is not None:
        return frame_id_from_name
    if record_key.isdigit():
        return record_key.zfill(6)
    return None


def _frame_id_from_frame_name(frame_name: str) -> str | None:
    stem = Path(frame_name).stem
    if stem.endswith("-rgb"):
        stem = stem.removesuffix("-rgb")
    if stem.isdigit():
        return stem.zfill(6)
    return None


def _score_object_frame(
    frame_record: _ObjectFrameMapFrame, query_terms: tuple[_ObjectQueryTerm, ...]
) -> _ScoredObjectFrame:
    score = 0.0
    matched_term_count = 0
    matched_primary_object_count = 0
    matched_objects_by_id: dict[str, KeyframeMatchedObjectPayload] = {}
    for query_term in query_terms:
        term_score = 0.0
        for visible_object in frame_record.objects:
            if not _object_matches_query_term(visible_object, query_term):
                continue
            term_score += query_term.weight * max(
                visible_object.score, _MIN_OBJECT_SCORE
            )
            object_id = str(visible_object.object_id)
            if object_id not in matched_objects_by_id:
                matched_objects_by_id[object_id] = _matched_object_payload(
                    visible_object
                )
        if term_score > 0.0:
            matched_term_count += 1
            if query_term.is_primary_object:
                matched_primary_object_count += 1
            score += term_score
    if matched_term_count > 1:
        score += _CO_OCCURRENCE_BONUS * (matched_term_count - 1)
    if matched_primary_object_count > 0:
        score += _PRIMARY_OBJECT_BONUS * matched_primary_object_count
    return _ScoredObjectFrame(
        score=score,
        matched_objects=tuple(matched_objects_by_id.values()),
        primary_object_match_count=matched_primary_object_count,
    )


def _object_matches_query_term(
    visible_object: _ObjectFrameMapObject, query_term: _ObjectQueryTerm
) -> bool:
    object_label = visible_object.class_name.lower()
    return any(alias in object_label for alias in query_term.label_aliases)


def _matched_object_payload(
    visible_object: _ObjectFrameMapObject,
) -> KeyframeMatchedObjectPayload:
    payload: KeyframeMatchedObjectPayload = {
        "object_id": str(visible_object.object_id),
        "label": visible_object.class_name,
        "score": visible_object.score,
        "source": "object_frame_map",
    }
    if visible_object.bbox_xyxy is not None:
        payload["bbox_xyxy"] = [
            float(coordinate) for coordinate in visible_object.bbox_xyxy
        ]
        payload["bbox_format"] = "pixel_xyxy"
    return payload


def _validate_pixel_bbox(bbox: tuple[float, float, float, float]) -> None:
    left, top, right, bottom = bbox
    if any(not math.isfinite(value) for value in bbox):
        raise ValueError("pixel_xyxy bbox coordinates must be finite")
    if any(value < 0.0 for value in bbox):
        raise ValueError("pixel_xyxy bbox coordinates must be non-negative")
    if left >= right or top >= bottom:
        raise ValueError("pixel_xyxy bbox must satisfy left < right and top < bottom")


def _numeric_frame_sort_key(frame_id: str) -> int:
    if frame_id.isdigit():
        return int(frame_id)
    return 1_000_000_000


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


__all__ = [
    "KeyframeMatchedObjectPayload",
    "KeyframeSelection",
    "KeyframeSelectorArgs",
    "KeyframeSelectorResult",
    "VisibleObjectFrameScore",
    "keyframe_selector",
    "visible_object_frame_scores",
]
