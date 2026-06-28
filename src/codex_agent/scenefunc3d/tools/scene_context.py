"""Prepared SceneFunc3D scene context for CLI tools."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ...errors import SceneFunc3dDataError

_RAW_DIRNAME = "raw"
_CONCEPTGRAPH_DIRNAME = "conceptgraph"
_SOURCE_FRAMES_FILENAME = "source_frames.json"
_CONCEPTGRAPH_RGB_VIS_DIRNAME = "gsa_vis_ram_withbg_allclasses"
_RGB_FRAME_RE = re.compile(r"^(\d{6})-rgb\.(png|jpg|jpeg)$")
_FRAME_ID_FROM_NAME_RE = re.compile(
    r"^(?:.*[_-])?(\d{1,6})(?:[-_](?:rgb|depth|color))?(?:\.[^.]+)?$"
)
_SOURCE_FRAME_ID_FIELDS: tuple[str, ...] = (
    "frame_id",
    "frameId",
    "id",
    "name",
    "frame_name",
    "frameName",
    "file_name",
    "filename",
    "path",
    "rgb",
    "image",
    "image_path",
)


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

    @property
    def source_frames_json(self) -> Path:
        """Source-frame metadata index for real prepared SceneFuncVal-CG scenes."""
        return self.raw_dir / _SOURCE_FRAMES_FILENAME

    @property
    def conceptgraph_rgb_vis_dir(self) -> Path:
        """Likely ConceptGraph RGB visualization directory for frame tools."""
        return self.conceptgraph_dir / _CONCEPTGRAPH_RGB_VIS_DIRNAME

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
        frame_ids = _discover_frame_ids(raw_dir)
        if not frame_ids:
            source_frames_json = raw_dir / _SOURCE_FRAMES_FILENAME
            raise SceneFunc3dDataError(
                "no frame ids discovered from "
                f"{source_frames_json} or raw RGB frames under {raw_dir}"
            )
        return cls(
            visit_id=scene_root.name,
            scene_root=scene_root,
            rgb_frame_ids=frame_ids,
        )


def _discover_frame_ids(raw_dir: Path) -> tuple[str, ...]:
    source_frames_json = raw_dir / _SOURCE_FRAMES_FILENAME
    if source_frames_json.is_file():
        return _load_source_frame_ids(source_frames_json)
    return _discover_raw_rgb_frame_ids(raw_dir)


def _load_source_frame_ids(source_frames_json: Path) -> tuple[str, ...]:
    try:
        parsed: object = json.loads(source_frames_json.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SceneFunc3dDataError(
            f"could not read SceneFunc3D source frame index: {source_frames_json}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"invalid SceneFunc3D source frame index JSON: {source_frames_json}"
        ) from exc

    frame_ids = _frame_ids_from_source_payload(parsed)
    if not frame_ids:
        raise SceneFunc3dDataError(
            f"no frame ids discovered in SceneFunc3D source frame index: "
            f"{source_frames_json}"
        )
    return frame_ids


def _frame_ids_from_source_payload(payload: object) -> tuple[str, ...]:
    if isinstance(payload, list):
        return _frame_ids_from_source_sequence(cast(list[object], payload))
    if isinstance(payload, Mapping):
        return _frame_ids_from_source_mapping(cast(Mapping[object, object], payload))
    return ()


def _frame_ids_from_source_sequence(records: list[object]) -> tuple[str, ...]:
    frame_ids: list[str] = []
    for index, record in enumerate(records):
        frame_id = _source_frame_id_from_record(record)
        if frame_id is None:
            raise SceneFunc3dDataError(
                f"invalid source frame record {index}: could not derive frame id"
            )
        frame_ids.append(frame_id)
    return _deduplicate_frame_ids(frame_ids)


def _frame_ids_from_source_mapping(
    records_by_frame: Mapping[object, object],
) -> tuple[str, ...]:
    frame_ids: list[str] = []
    for raw_key, raw_record in records_by_frame.items():
        frame_id = _source_frame_id_from_record(raw_record)
        if frame_id is None:
            frame_id = _coerce_frame_id(raw_key)
        if frame_id is None:
            raise SceneFunc3dDataError(
                f"invalid source frame entry {raw_key!r}: could not derive frame id "
                "from record or key"
            )
        frame_ids.append(frame_id)
    return _deduplicate_frame_ids(frame_ids)


def _deduplicate_frame_ids(frame_ids: list[str]) -> tuple[str, ...]:
    seen_frame_ids: set[str] = set()
    for frame_id in frame_ids:
        if frame_id in seen_frame_ids:
            raise SceneFunc3dDataError(f"duplicate frame id {frame_id!r}")
        seen_frame_ids.add(frame_id)
    return tuple(frame_ids)


def _source_frame_id_from_record(record: object) -> str | None:
    if isinstance(record, Mapping):
        return _source_frame_id_from_mapping(cast(Mapping[object, object], record))
    return _coerce_frame_id(record)


def _source_frame_id_from_mapping(record: Mapping[object, object]) -> str | None:
    for field_name in _SOURCE_FRAME_ID_FIELDS:
        if field_name not in record:
            continue
        frame_id = _coerce_frame_id(record[field_name])
        if frame_id is not None:
            return frame_id
    return None


def _coerce_frame_id(raw_value: object) -> str | None:
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        if raw_value < 0:
            return None
        return f"{raw_value:06d}"
    if isinstance(raw_value, str):
        return _frame_id_from_text(raw_value)
    return None


def _frame_id_from_text(raw_value: str) -> str | None:
    stripped_value = raw_value.strip()
    if not stripped_value:
        return None
    name = Path(stripped_value.replace("\\", "/")).name
    match = _FRAME_ID_FROM_NAME_RE.match(name)
    if match is None:
        return None
    return match.group(1).zfill(6)


def _discover_raw_rgb_frame_ids(raw_dir: Path) -> tuple[str, ...]:
    frame_ids = sorted(
        match.group(1)
        for match in (_RGB_FRAME_RE.match(path.name) for path in raw_dir.iterdir())
        if match is not None
    )
    return tuple(frame_ids)


__all__ = ["SceneFunc3dToolScene"]
