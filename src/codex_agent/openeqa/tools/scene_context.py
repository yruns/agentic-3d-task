"""Per-clip scene context shared by the OpenEQA agent tools.

A *tool scene* is one prepared OpenEQA ScanNet clip directory
(``<data_root>/<clip_id>``) holding both the first-person ``raw/`` frames and the
ConceptGraph pack under ``conceptgraph/``. The lightweight frame view
(:class:`codex_agent.openeqa.scene.OpenEqaScene`) is loaded eagerly; the heavier
ConceptGraph scene representation (objects, trajectory, visibility) is built
lazily through :func:`build_selector` only by the tools that need it
(``list_objects``, ``view_bev``, ``keyframe_selector``), so ``view_frame`` never
pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...errors import OpenEqaDataError
from ..scene import OpenEqaScene
from .models import ToolInputError

if TYPE_CHECKING:
    from keyframe.keyframe_selector import KeyframeSelector

_CONCEPTGRAPH_DIRNAME = "conceptgraph"


@dataclass(frozen=True)
class OpenEqaToolScene:
    """A prepared OpenEQA clip directory for the agent tools."""

    scene_dir: Path
    scene: OpenEqaScene

    @property
    def clip_id(self) -> str:
        """On-disk clip directory name (also the BEV builder's ``scene_id``)."""
        return self.scene_dir.name

    @property
    def conceptgraph_dir(self) -> Path:
        """The ConceptGraph pack directory (objects, trajectory, visibility)."""
        return self.scene_dir / _CONCEPTGRAPH_DIRNAME

    @classmethod
    def load(cls, scene_dir: Path) -> OpenEqaToolScene:
        """Load the clip's frame view from ``<data_root>/<clip_id>``.

        Raises:
            OpenEqaDataError: If the clip directory or its ``raw`` frames are
                missing/malformed (raised by :meth:`OpenEqaScene.load`).
        """
        if not scene_dir.is_dir():
            raise OpenEqaDataError(f"scene directory is missing: {scene_dir}")
        return cls(scene_dir=scene_dir, scene=OpenEqaScene.load(scene_dir))

    def require_conceptgraph(self) -> Path:
        """Return the ConceptGraph dir or raise a recoverable tool error."""
        conceptgraph_dir = self.conceptgraph_dir
        if not conceptgraph_dir.is_dir():
            raise ToolInputError(
                "this scene has no ConceptGraph assets at "
                f"{conceptgraph_dir}; list_objects / view_bev / keyframe_selector "
                "are unavailable here — use view_frame instead"
            )
        return conceptgraph_dir


def build_selector(
    conceptgraph_dir: Path, *, model: str | None = None
) -> KeyframeSelector:
    """Build a ConceptGraph :class:`KeyframeSelector` for the OpenEQA dataset.

    Loads objects, camera trajectory, and the visibility index (no LLM call —
    query parsing only happens later inside ``select_keyframes_v2``), so the
    object ids it exposes match the ones the BEV highlights and keyframe results
    use.

    Raises:
        ToolInputError: If the scene cannot be loaded (reported recoverably so
            the agent can fall back to ``view_frame``).
    """
    try:
        from keyframe.keyframe_selector import KeyframeSelector

        return KeyframeSelector.from_scene_path(
            conceptgraph_dir, dataset="openeqa", model=model
        )
    except Exception as exc:  # re-raised as a recoverable tool error for the agent
        raise ToolInputError(
            f"could not load ConceptGraph scene at {conceptgraph_dir} "
            f"({type(exc).__name__}: {exc}); use view_frame instead"
        ) from exc


__all__ = ["OpenEqaToolScene", "build_selector"]
