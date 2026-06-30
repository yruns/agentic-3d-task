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
_SOURCE_FRAME_RGB_PATH_FIELDS: tuple[str, ...] = ("rgb", "image", "image_path", "path")
_SOURCE_FRAME_DEPTH_PATH_FIELDS: tuple[str, ...] = ("depth", "depth_path")
_SOURCE_FRAME_INTRINSICS_PATH_FIELDS: tuple[str, ...] = (
    "intrinsic",
    "intrinsics",
    "intrinsic_path",
    "intrinsics_path",
)
_SOURCE_FRAME_POSE_PATH_FIELDS: tuple[str, ...] = (
    "pose",
    "pose_path",
    "camera_pose",
    "extrinsic",
    "extrinsics",
    "extrinsic_path",
    "extrinsics_path",
)
_SOURCE_FRAME_RGB_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class SourceFrameRecord:
    """One source frame entry from ``raw/source_frames.json``."""

    frame_id: str
    raw_rgb_path: Path | None = None
    depth_path: Path | None = None
    intrinsics_path: Path | None = None
    pose_path: Path | None = None


@dataclass(frozen=True)
class SourceFrameIndex:
    """Typed index of source-frame metadata for one prepared scene."""

    records: tuple[SourceFrameRecord, ...]

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Return source frame ids in source index order."""
        return tuple(record.frame_id for record in self.records)

    def raw_rgb_path_for(self, frame_id: str) -> Path | None:
        """Return the indexed raw RGB path for ``frame_id``, if one exists."""
        record = self.record_for(frame_id)
        if record is None:
            return None
        return record.raw_rgb_path

    def depth_path_for(self, frame_id: str) -> Path | None:
        """Return the indexed depth path for ``frame_id``, if one exists."""
        record = self.record_for(frame_id)
        if record is None:
            return None
        return record.depth_path

    def intrinsics_path_for(self, frame_id: str) -> Path | None:
        """Return the indexed intrinsics path for ``frame_id``, if one exists."""
        record = self.record_for(frame_id)
        if record is None:
            return None
        return record.intrinsics_path

    def pose_path_for(self, frame_id: str) -> Path | None:
        """Return the indexed pose path for ``frame_id``, if one exists."""
        record = self.record_for(frame_id)
        if record is None:
            return None
        return record.pose_path

    def record_for(self, frame_id: str) -> SourceFrameRecord | None:
        """Return the indexed source-frame record for ``frame_id``, if present."""
        for record in self.records:
            if record.frame_id == frame_id:
                return record
        return None


@dataclass(frozen=True)
class SceneFunc3dToolScene:
    """Filesystem context for one prepared SceneFuncVal-CG scene."""

    visit_id: str
    scene_root: Path
    rgb_frame_ids: tuple[str, ...]
    source_frame_index: SourceFrameIndex

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

    def source_frame_raw_rgb_path(self, frame_id: str) -> Path | None:
        """Return the source-frame raw RGB path for ``frame_id``, if indexed."""
        return self.source_frame_index.raw_rgb_path_for(frame_id)

    def source_frame_depth_path(self, frame_id: str) -> Path | None:
        """Return the source-frame depth path for ``frame_id``, if indexed."""
        return self.source_frame_index.depth_path_for(frame_id)

    def source_frame_intrinsics_path(self, frame_id: str) -> Path | None:
        """Return the source-frame intrinsics path for ``frame_id``, if indexed."""
        return self.source_frame_index.intrinsics_path_for(frame_id)

    def source_frame_pose_path(self, frame_id: str) -> Path | None:
        """Return the source-frame pose path for ``frame_id``, if indexed."""
        return self.source_frame_index.pose_path_for(frame_id)

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
        source_frame_index = _load_source_frame_index_if_present(raw_dir)
        frame_ids = (
            source_frame_index.frame_ids
            if source_frame_index.records
            else _discover_raw_rgb_frame_ids(raw_dir)
        )
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
            source_frame_index=source_frame_index,
        )


def _load_source_frame_index_if_present(raw_dir: Path) -> SourceFrameIndex:
    source_frames_json = raw_dir / _SOURCE_FRAMES_FILENAME
    if source_frames_json.is_file():
        return _load_source_frame_index(source_frames_json)
    return SourceFrameIndex(records=())


def _load_source_frame_index(source_frames_json: Path) -> SourceFrameIndex:
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

    source_frame_index = _source_frame_index_from_payload(parsed, source_frames_json)
    if not source_frame_index.records:
        raise SceneFunc3dDataError(
            f"no frame ids discovered in SceneFunc3D source frame index: "
            f"{source_frames_json}"
        )
    return source_frame_index


def _source_frame_index_from_payload(
    payload: object, source_frames_json: Path
) -> SourceFrameIndex:
    if isinstance(payload, list):
        return _source_frame_index_from_sequence(
            cast(list[object], payload), source_frames_json.parent
        )
    if isinstance(payload, Mapping):
        return _source_frame_index_from_mapping(
            cast(Mapping[object, object], payload), source_frames_json.parent
        )
    return SourceFrameIndex(records=())


def _source_frame_index_from_sequence(
    records: list[object], source_frames_dir: Path
) -> SourceFrameIndex:
    source_records: list[SourceFrameRecord] = []
    for index, raw_record in enumerate(records):
        frame_id = _source_frame_id_from_record(raw_record)
        if frame_id is None:
            raise SceneFunc3dDataError(
                f"invalid source frame record {index}: could not derive frame id"
            )
        source_records.append(
            SourceFrameRecord(
                frame_id=frame_id,
                raw_rgb_path=_source_rgb_path_from_record(
                    raw_record, source_frames_dir
                ),
                depth_path=_source_depth_path_from_record(
                    raw_record, source_frames_dir
                ),
                intrinsics_path=_source_intrinsics_path_from_record(
                    raw_record, source_frames_dir
                ),
                pose_path=_source_pose_path_from_record(raw_record, source_frames_dir),
            )
        )
    return SourceFrameIndex(records=_deduplicate_source_frame_records(source_records))


def _source_frame_index_from_mapping(
    records_by_frame: Mapping[object, object], source_frames_dir: Path
) -> SourceFrameIndex:
    source_records: list[SourceFrameRecord] = []
    for raw_key, raw_record in records_by_frame.items():
        frame_id = _source_frame_id_from_record(raw_record)
        if frame_id is None:
            frame_id = _coerce_frame_id(raw_key)
        if frame_id is None:
            raise SceneFunc3dDataError(
                f"invalid source frame entry {raw_key!r}: could not derive frame id "
                "from record or key"
            )
        source_records.append(
            SourceFrameRecord(
                frame_id=frame_id,
                raw_rgb_path=_source_rgb_path_from_record(
                    raw_record, source_frames_dir
                ),
                depth_path=_source_depth_path_from_record(
                    raw_record, source_frames_dir
                ),
                intrinsics_path=_source_intrinsics_path_from_record(
                    raw_record, source_frames_dir
                ),
                pose_path=_source_pose_path_from_record(raw_record, source_frames_dir),
            )
        )
    return SourceFrameIndex(records=_deduplicate_source_frame_records(source_records))


def _deduplicate_source_frame_records(
    source_records: list[SourceFrameRecord],
) -> tuple[SourceFrameRecord, ...]:
    seen_frame_ids: set[str] = set()
    for record in source_records:
        if record.frame_id in seen_frame_ids:
            raise SceneFunc3dDataError(f"duplicate frame id {record.frame_id!r}")
        seen_frame_ids.add(record.frame_id)
    return tuple(source_records)


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


def _source_rgb_path_from_record(
    record: object, source_frames_dir: Path
) -> Path | None:
    if isinstance(record, Mapping):
        return _source_rgb_path_from_mapping(
            cast(Mapping[object, object], record), source_frames_dir
        )
    return _coerce_source_rgb_path(record, source_frames_dir)


def _source_rgb_path_from_mapping(
    record: Mapping[object, object], source_frames_dir: Path
) -> Path | None:
    for field_name in _SOURCE_FRAME_RGB_PATH_FIELDS:
        if field_name not in record:
            continue
        raw_rgb_path = _coerce_source_rgb_path(record[field_name], source_frames_dir)
        if raw_rgb_path is not None:
            return raw_rgb_path
    return None


def _source_depth_path_from_record(
    record: object, source_frames_dir: Path
) -> Path | None:
    if isinstance(record, Mapping):
        return _source_path_from_mapping(
            cast(Mapping[object, object], record),
            source_frames_dir,
            field_names=_SOURCE_FRAME_DEPTH_PATH_FIELDS,
        )
    return None


def _source_intrinsics_path_from_record(
    record: object, source_frames_dir: Path
) -> Path | None:
    if isinstance(record, Mapping):
        return _source_path_from_mapping(
            cast(Mapping[object, object], record),
            source_frames_dir,
            field_names=_SOURCE_FRAME_INTRINSICS_PATH_FIELDS,
        )
    return None


def _source_pose_path_from_record(
    record: object, source_frames_dir: Path
) -> Path | None:
    if isinstance(record, Mapping):
        return _source_path_from_mapping(
            cast(Mapping[object, object], record),
            source_frames_dir,
            field_names=_SOURCE_FRAME_POSE_PATH_FIELDS,
        )
    return None


def _source_path_from_mapping(
    record: Mapping[object, object],
    source_frames_dir: Path,
    *,
    field_names: tuple[str, ...],
) -> Path | None:
    for field_name in field_names:
        if field_name not in record:
            continue
        source_path = _coerce_source_path(record[field_name], source_frames_dir)
        if source_path is not None:
            return source_path
    return None


def _coerce_source_rgb_path(raw_value: object, source_frames_dir: Path) -> Path | None:
    if not isinstance(raw_value, str):
        return None
    stripped_value = raw_value.strip()
    if not stripped_value:
        return None
    source_path = Path(stripped_value)
    if source_path.suffix.lower() not in _SOURCE_FRAME_RGB_SUFFIXES:
        return None
    if source_path.is_absolute():
        return source_path
    return source_frames_dir / source_path


def _coerce_source_path(raw_value: object, source_frames_dir: Path) -> Path | None:
    if not isinstance(raw_value, str):
        return None
    stripped_value = raw_value.strip()
    if not stripped_value:
        return None
    source_path = Path(stripped_value)
    if source_path.is_absolute():
        return source_path
    return source_frames_dir / source_path


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


__all__ = ["SceneFunc3dToolScene", "SourceFrameIndex", "SourceFrameRecord"]
