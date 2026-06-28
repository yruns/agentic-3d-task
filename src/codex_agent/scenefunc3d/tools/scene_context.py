"""Prepared SceneFunc3D scene context for CLI tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...errors import SceneFunc3dDataError

_RAW_DIRNAME = "raw"
_CONCEPTGRAPH_DIRNAME = "conceptgraph"
_RGB_FRAME_RE = re.compile(r"^(\d{6})-rgb\.(png|jpg|jpeg)$")


@dataclass(frozen=True)
class SceneFunc3dToolScene:
    """Filesystem context for one prepared SceneFuncVal-CG scene."""

    visit_id: str
    scene_root: Path
    rgb_frame_ids: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        """Directory containing first-person RGB frames."""
        return self.scene_root / _RAW_DIRNAME

    @property
    def conceptgraph_dir(self) -> Path:
        """Directory containing the prepared ConceptGraph scene pack."""
        return self.scene_root / _CONCEPTGRAPH_DIRNAME

    @classmethod
    def load(cls, scene_root: Path) -> SceneFunc3dToolScene:
        """Load and validate one prepared SceneFunc3D scene root."""
        if not scene_root.is_dir():
            raise SceneFunc3dDataError(
                f"SceneFunc3D scene root is missing: {scene_root}"
            )
        raw_dir = scene_root / _RAW_DIRNAME
        if not raw_dir.is_dir():
            raise SceneFunc3dDataError(
                f"SceneFunc3D raw frame directory is missing: {raw_dir}"
            )
        frame_ids = sorted(
            match.group(1)
            for match in (_RGB_FRAME_RE.match(path.name) for path in raw_dir.iterdir())
            if match is not None
        )
        if not frame_ids:
            raise SceneFunc3dDataError(f"no RGB frames found under {raw_dir}")
        return cls(
            visit_id=scene_root.name,
            scene_root=scene_root,
            rgb_frame_ids=tuple(frame_ids),
        )


__all__ = ["SceneFunc3dToolScene"]
