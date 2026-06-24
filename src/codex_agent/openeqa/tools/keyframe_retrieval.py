"""Language->frame retrieval (image) tool: ``keyframe_selector``.

Wraps the production
:meth:`keyframe.keyframe_selector.KeyframeSelector.select_keyframes_v2`
(Stage-1 query parsing + geometric grounding + joint-coverage frame selection)
so the agent can ask, in free-form language, for the first-person frames most
likely to show the objects a question is about.

The trace analysis of the v4 run showed two failure modes this tool now defends
against (without touching the shared selector, so NR3D is unaffected):

* **Retrieval-pool collapse** — re-phrased queries kept returning the *same*
  overlapping frames, wasting tool rounds. We now over-fetch candidates and
  return a **spatially diverse** subset (camera position / viewing direction),
  and remember frames already returned in this turn so successive calls surface
  *new* viewpoints.
* **Grounding misses** — when the selector grounds nothing, instead of returning
  an empty result we **fall back** to category-visibility or a trajectory sweep
  so the agent still gets frames to look at.

Every call also writes a single labelled **contact sheet** of the returned
frames so the agent can inspect the whole batch with one ``view_image``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..scene import DEFAULT_MAX_IMAGE_SIZE, downsize_rgb_for_view
from .imaging import compose_contact_sheet
from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

#: Cap on frames returned to the agent (keeps the turn's image budget bounded).
_MAX_FRAMES = 4
#: Extra candidates over the requested ``k`` to give diversity filtering room.
_DIVERSITY_BUFFER = 4
#: Hard cap on how many candidate views we ask the selector to produce.
_MAX_INTERNAL_K = 10
#: Two views count as distinct if their cameras are ≥ this far apart (metres) ...
_MIN_POSITION_DIST = 0.5
#: ... OR their viewing directions differ by at least this angle (degrees).
_MIN_VIEW_ANGLE_DEG = 25.0
#: Bound on the per-turn "already returned" memory (keeps the state file small).
_MAX_SEEN = 60
_RGB_FRAME_RE = re.compile(r"(\d+)-rgb\.")
_FRAMEJPG_RE = re.compile(r"frame0*(\d+)\.")


class KeyframeSelectorArgs(BaseModel):
    """Arguments for :func:`keyframe_selector`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1)
    use_bev: bool = False
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)


@dataclass(frozen=True)
class KeyframeHit:
    """One language-retrieved frame, downsized for viewing."""

    frame_id: int
    image_path: Path

    def to_payload(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "image_path": str(self.image_path)}


@dataclass(frozen=True)
class _Candidate:
    """A frame the selector (or a fallback) proposed, with its source RGB."""

    frame_id: int
    view_id: int | None
    source_path: Path


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Language-retrieved frames plus a contact sheet and a grounding summary."""

    frames: tuple[KeyframeHit, ...]
    hypothesis_summary: str
    note: str
    contact_sheet_path: Path | None = None
    fallback: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hypothesis_summary": self.hypothesis_summary,
            "note": self.note,
            "frames": [frame.to_payload() for frame in self.frames],
        }
        if self.contact_sheet_path is not None:
            payload["contact_sheet"] = str(self.contact_sheet_path)
            payload["tip"] = (
                "view_image the contact_sheet to see all frames at once; "
                "view_crop a frame_id for a high-res zoom"
            )
        if self.fallback:
            payload["fallback"] = self.fallback
        return payload


def keyframe_selector(
    tool_scene: OpenEqaToolScene, args: KeyframeSelectorArgs, *, out_dir: Path
) -> KeyframeSelectorResult:
    """Retrieve up to ``k`` spatially-diverse first-person frames for a query."""
    conceptgraph_dir = tool_scene.require_conceptgraph()
    capped_k = min(args.k, _MAX_FRAMES)
    note = "" if args.k == capped_k else f"(k capped at {_MAX_FRAMES} from {args.k})"

    selector = build_selector(conceptgraph_dir)
    internal_k = min(_MAX_INTERNAL_K, capped_k + _DIVERSITY_BUFFER)
    result = _run_selector(selector, args.query, internal_k, use_bev=args.use_bev)

    candidates = _candidates_from_result(tool_scene, selector, result)
    fallback = ""
    if not candidates:
        candidates, fallback = _fallback_candidates(tool_scene, selector, args.query)

    out_dir.mkdir(parents=True, exist_ok=True)
    seen_path = _seen_state_path(out_dir, tool_scene.clip_id)
    seen = _load_seen(seen_path)
    selected = _select_diverse(selector, candidates, capped_k, seen)
    _record_seen(seen_path, seen, [c.frame_id for c in selected])

    frames = _downsize_selected(tool_scene, selected, out_dir, args.max_image_size)
    contact_sheet = _build_contact_sheet(tool_scene, frames, out_dir)
    return KeyframeSelectorResult(
        frames=tuple(frames),
        hypothesis_summary=_summarize(result),
        note=note or _empty_note(frames),
        contact_sheet_path=contact_sheet,
        fallback=fallback,
    )


def _run_selector(selector: Any, query: str, k: int, *, use_bev: bool) -> Any:
    try:
        return selector.select_keyframes_v2(
            query, k=k, viewpoint_aware=True, use_visual_context=use_bev
        )
    except Exception as exc:  # re-raised as a recoverable tool error for the agent
        raise ToolInputError(
            f"keyframe_selector failed ({type(exc).__name__}: {exc}); "
            "fall back to view_frame"
        ) from exc


def _candidates_from_result(
    tool_scene: OpenEqaToolScene, selector: Any, result: Any
) -> list[_Candidate]:
    """Build ordered candidates from the selector result (paths + view ids)."""
    view_by_frame = _view_by_frame(result)
    stride = _stride(selector)
    candidates: list[_Candidate] = []
    seen_ids: set[int] = set()
    for raw_path in getattr(result, "keyframe_paths", []) or []:
        path = Path(str(raw_path))
        frame_id = _frame_id_from_path(path)
        if frame_id is None or frame_id in seen_ids:
            continue
        seen_ids.add(frame_id)
        view_id = view_by_frame.get(frame_id)
        if view_id is None:
            view_id = frame_id // stride if stride > 0 else None
        candidates.append(
            _Candidate(frame_id=frame_id, view_id=view_id, source_path=path)
        )
    return candidates


def _fallback_candidates(
    tool_scene: OpenEqaToolScene, selector: Any, query: str
) -> tuple[list[_Candidate], str]:
    """When grounding is empty, surface frames by category visibility / a sweep."""
    stride = _stride(selector)
    category_views = _views_for_query_categories(selector, query)
    if category_views:
        candidates = _candidates_from_views(
            tool_scene, selector, category_views, stride
        )
        if candidates:
            return candidates, "category-visibility (grounding was empty)"
    sweep = _trajectory_sweep_views(selector, tool_scene)
    candidates = _candidates_from_views(tool_scene, selector, sweep, stride)
    return candidates, "trajectory sweep (grounding was empty)"


def _views_for_query_categories(selector: Any, query: str) -> list[int]:
    """Best views of any object whose category appears in the query text."""
    lowered = query.lower()
    object_to_views = getattr(selector, "object_to_views", {}) or {}
    scored: list[tuple[float, int]] = []
    for obj in getattr(selector, "objects", []) or []:
        category = (obj.object_tag or obj.category or "").strip().lower()
        if not category or category not in lowered:
            continue
        for view_id, score in object_to_views.get(obj.obj_id, []):
            scored.append((float(score), int(view_id)))
    scored.sort(reverse=True)
    ordered: list[int] = []
    for _, view_id in scored:
        if view_id not in ordered:
            ordered.append(view_id)
    return ordered


def _trajectory_sweep_views(selector: Any, tool_scene: OpenEqaToolScene) -> list[int]:
    """Evenly-spaced view ids across the camera trajectory (last resort)."""
    poses = getattr(selector, "camera_poses", []) or []
    count = len(poses)
    if count == 0:
        # No poses: spread across the raw frame ids directly (view==frame here).
        frame_ids = tool_scene.scene.rgb_frame_ids
        if not frame_ids:
            return []
        picks = _even_indices(len(frame_ids), _MAX_INTERNAL_K)
        stride = _stride(selector)
        return [frame_ids[i] // stride if stride > 0 else frame_ids[i] for i in picks]
    return _even_indices(count, _MAX_INTERNAL_K)


def _candidates_from_views(
    tool_scene: OpenEqaToolScene, selector: Any, view_ids: list[int], stride: int
) -> list[_Candidate]:
    available = set(tool_scene.scene.rgb_frame_ids)
    candidates: list[_Candidate] = []
    seen_ids: set[int] = set()
    for view_id in view_ids:
        frame_id = view_id * stride if stride > 0 else view_id
        if frame_id not in available:
            frame_id = _nearest(available, frame_id)
        if frame_id in seen_ids:
            continue
        seen_ids.add(frame_id)
        candidates.append(
            _Candidate(
                frame_id=frame_id,
                view_id=view_id,
                source_path=tool_scene.scene.raw_rgb_path(frame_id),
            )
        )
    return candidates


def _select_diverse(
    selector: Any, candidates: list[_Candidate], k: int, seen: list[int]
) -> list[_Candidate]:
    """Greedily pick ≤ k spatially-distinct candidates, avoiding ``seen`` frames.

    Relaxes progressively so a non-empty candidate set never yields zero frames:
    first try (fresh + diverse), then allow already-seen frames, then drop the
    spatial-diversity constraint entirely.
    """
    if not candidates:
        return []
    seen_set = set(seen)
    for allow_seen in (False, True):
        for enforce_spacing in (True, False):
            picked: list[_Candidate] = []
            for cand in candidates:
                if not allow_seen and cand.frame_id in seen_set:
                    continue
                if enforce_spacing and not _is_separated(selector, cand, picked):
                    continue
                picked.append(cand)
                if len(picked) >= k:
                    return picked
            if picked:
                return picked
    return candidates[:k]


def _is_separated(selector: Any, cand: _Candidate, picked: list[_Candidate]) -> bool:
    feat = _pose_feature(selector, cand.view_id)
    for other in picked:
        if other.frame_id == cand.frame_id:
            return False
        other_feat = _pose_feature(selector, other.view_id)
        if feat is None or other_feat is None:
            continue  # no geometry to compare; rely on frame-id distinctness
        position_a, forward_a = feat
        position_b, forward_b = other_feat
        import numpy as np

        dist = float(np.linalg.norm(position_a - position_b))
        cos = float(np.clip(np.dot(forward_a, forward_b), -1.0, 1.0))
        angle = math.degrees(math.acos(cos))
        if dist < _MIN_POSITION_DIST and angle < _MIN_VIEW_ANGLE_DEG:
            return False
    return True


def _pose_feature(selector: Any, view_id: int | None) -> tuple[Any, Any] | None:
    if view_id is None:
        return None
    poses = getattr(selector, "camera_poses", []) or []
    if view_id < 0 or view_id >= len(poses):
        return None
    import numpy as np

    pose = np.asarray(poses[view_id], dtype=np.float64)
    if pose.shape != (4, 4):
        return None
    position = pose[:3, 3]
    forward = -pose[:3, 2]
    norm = float(np.linalg.norm(forward))
    if norm > 1e-9:
        forward = forward / norm
    return position, forward


def _downsize_selected(
    tool_scene: OpenEqaToolScene,
    selected: list[_Candidate],
    out_dir: Path,
    max_size: int,
) -> list[KeyframeHit]:
    hits: list[KeyframeHit] = []
    for cand in selected:
        destination = out_dir / f"{tool_scene.clip_id}_kf_{cand.frame_id:06d}.jpg"
        hits.append(
            KeyframeHit(
                frame_id=cand.frame_id,
                image_path=downsize_rgb_for_view(
                    cand.source_path, destination, max_size=max_size
                ),
            )
        )
    return hits


def _build_contact_sheet(
    tool_scene: OpenEqaToolScene, frames: list[KeyframeHit], out_dir: Path
) -> Path | None:
    if len(frames) < 2:
        return None
    tiles = [(f"frame {hit.frame_id}", hit.image_path) for hit in frames]
    destination = out_dir / f"{tool_scene.clip_id}_kf_sheet.jpg"
    try:
        return compose_contact_sheet(tiles, destination)
    except Exception:  # a sheet is a convenience; never fail the tool over it
        return None


# ----- view/frame mapping + seen-state helpers --------------------------------


def _view_by_frame(result: Any) -> dict[int, int]:
    metadata = getattr(result, "metadata", None)
    mappings = getattr(metadata, "frame_mappings", []) if metadata is not None else []
    return {fm.resolved_frame_id: fm.resolved_view_id for fm in mappings}


def _stride(selector: Any) -> int:
    stride = int(getattr(selector, "stride", 1) or 1)
    return stride if stride > 0 else 1


def _seen_state_path(out_dir: Path, clip_id: str) -> Path | None:
    """Per-turn 'already returned' state file, or ``None`` to disable cross-call.

    Scoped by ``CODEX_HOME`` (unique per turn in production) so successive calls
    in the *same* turn dedupe against each other, without leaking across the
    many questions that share one clip and one scratch dir. When ``CODEX_HOME``
    is absent (tests / ad-hoc runs) cross-call dedup is disabled.
    """
    codex_home = os.environ.get("CODEX_HOME")
    if not codex_home:
        return None
    token = hashlib.sha1(codex_home.encode("utf-8")).hexdigest()[:12]
    return out_dir / f".kf_seen_{clip_id}_{token}.json"


def _load_seen(seen_path: Path | None) -> list[int]:
    if seen_path is None or not seen_path.exists():
        return []
    try:
        data = json.loads(seen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [int(v) for v in data if isinstance(v, int)]


def _record_seen(
    seen_path: Path | None, previous: list[int], new_ids: list[int]
) -> None:
    if seen_path is None:
        return
    merged = list(dict.fromkeys([*previous, *new_ids]))[-_MAX_SEEN:]
    try:
        seen_path.write_text(json.dumps(merged), encoding="utf-8")
    except OSError:
        pass  # the dedup memory is best-effort


def _even_indices(count: int, k: int) -> list[int]:
    if count <= 0 or k <= 0:
        return []
    if count <= k:
        return list(range(count))
    step = (count - 1) / (k - 1) if k > 1 else 0
    return sorted({int(round(i * step)) for i in range(k)})


def _nearest(available: set[int], target: int) -> int:
    return min(available, key=lambda fid: abs(fid - target))


def _frame_id_from_path(path: Path) -> int | None:
    match = _RGB_FRAME_RE.search(path.name) or _FRAMEJPG_RE.search(path.name)
    return int(match.group(1)) if match is not None else None


def _summarize(result: Any) -> str:
    target = getattr(result, "target_term", None)
    anchor = getattr(result, "anchor_term", None)
    parts = [f"target={target!r}"]
    if anchor:
        parts.append(f"anchor={anchor!r}")
    return " ".join(parts)


def _empty_note(frames: list[KeyframeHit]) -> str:
    if frames:
        return ""
    return "no frames grounded for this query; try view_frame or rephrase"


__all__ = [
    "KeyframeSelectorArgs",
    "KeyframeHit",
    "KeyframeSelectorResult",
    "keyframe_selector",
]
