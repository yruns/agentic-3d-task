"""Lightweight SceneFunc3D keyframe retrieval tool."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .scene_context import SceneFunc3dToolScene

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_QUERY_FRAME_ID_RE = re.compile(
    r"\b(?:frame|frame_id|frame id|keyframe|view)\s*[:#-]?\s*(?P<frame_id>\d{1,6})\b",
    flags=re.IGNORECASE,
)


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
        if "query" not in values and "task_description" in values:
            values["query"] = values.pop("task_description")
        values.pop("annotation_ids", None)
        return values


@dataclass(frozen=True)
class KeyframeSelection:
    """One ranked SceneFunc3D frame candidate."""

    frame_id: str
    rank: int
    score: float
    reason: str

    def to_payload(self) -> dict[str, object]:
        """Return this selection as a JSON-ready mapping."""
        return {
            "frame_id": self.frame_id,
            "rank": self.rank,
            "score": self.score,
            "reason": self.reason,
        }


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


__all__ = [
    "KeyframeSelection",
    "KeyframeSelectorArgs",
    "KeyframeSelectorResult",
    "keyframe_selector",
]
