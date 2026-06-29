"""Load SceneFunc3D samples from prepared SceneFuncVal-CG scenes."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import SceneFunc3dDataError

ModelT = TypeVar("ModelT", bound=BaseModel)
_VISIT_ID_PATTERN = re.compile(r"^[0-9]+$")
_MOTION_DIR_DOF = 3


class SceneFuncMotionHintPayload(TypedDict):
    """Agent-visible JSON payload for one motion hint."""

    motion_id: str
    annotation_id: str
    motion_type: str
    motion_dir: list[float]


class SceneFuncAgentContext(TypedDict):
    """Agent-visible SceneFunc3D task context without hidden GT fields."""

    visit_id: str
    desc_id: str
    task_description: str
    annotation_ids: list[str]
    motion_hints: list[SceneFuncMotionHintPayload]


class _RawDescription(BaseModel):
    """Validated description record from ``<visit_id>_descriptions.json``."""

    model_config = ConfigDict(extra="ignore")

    desc_id: str = Field(min_length=1)
    annot_id: str | list[str]
    description: str = Field(min_length=1)


class _RawDescriptionsFile(BaseModel):
    """Validated SceneFunc3D descriptions file."""

    model_config = ConfigDict(extra="ignore")

    visit_id: str
    descriptions: list[_RawDescription]


class _RawMotion(BaseModel):
    """Validated motion record from ``<visit_id>_motions.json``."""

    model_config = ConfigDict(extra="ignore")

    motion_id: str = Field(min_length=1)
    annot_id: str = Field(min_length=1)
    motion_type: str = Field(min_length=1)
    motion_dir: list[float]


class _RawMotionsFile(BaseModel):
    """Validated SceneFunc3D motions file."""

    model_config = ConfigDict(extra="ignore")

    visit_id: str
    motions: list[_RawMotion]


class _RawAnnotation(BaseModel):
    """Validated annotation record; hidden GT fields are intentionally unused."""

    model_config = ConfigDict(extra="ignore")

    annot_id: str = Field(min_length=1)


class _RawAnnotationsFile(BaseModel):
    """Validated SceneFunc3D annotations file."""

    model_config = ConfigDict(extra="ignore")

    visit_id: str
    annotations: list[_RawAnnotation]


@dataclass(frozen=True)
class SceneFunc3dSampleId:
    """A parsed SceneFunc3D ``<visit_id>::<desc_id>`` identifier."""

    raw: str
    visit_id: str
    desc_id: str

    @classmethod
    def parse(cls, sample_id: str) -> SceneFunc3dSampleId:
        """Parse the ``<visit_id>::<desc_id>`` sample id form."""
        if not isinstance(sample_id, str):
            raise SceneFunc3dDataError(f"sample_id must be a string, got {sample_id!r}")
        parts = sample_id.split("::")
        if len(parts) != 2:
            raise SceneFunc3dDataError(
                f"sample_id must have 2 '::'-separated parts, got {sample_id!r}"
            )
        visit_id, desc_id = parts
        if not _VISIT_ID_PATTERN.fullmatch(visit_id):
            raise SceneFunc3dDataError(
                f"visit_id must contain only digits, got {visit_id!r}"
            )
        if not desc_id:
            raise SceneFunc3dDataError(f"invalid SceneFunc3D sample_id: {sample_id!r}")
        return cls(raw=sample_id, visit_id=visit_id, desc_id=desc_id)


@dataclass(frozen=True)
class SceneFuncMotionHint:
    """Agent-visible motion metadata linked to one annotation id."""

    motion_id: str
    annotation_id: str
    motion_type: str
    motion_dir: tuple[float, ...]

    def to_agent_payload(self) -> SceneFuncMotionHintPayload:
        """Return the JSON-serializable motion hint shown to the agent."""
        return {
            "motion_id": self.motion_id,
            "annotation_id": self.annotation_id,
            "motion_type": self.motion_type,
            "motion_dir": list(self.motion_dir),
        }


@dataclass(frozen=True)
class SceneFunc3dSample:
    """One SceneFunc3D language-conditioned mask-generation sample."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    annotation_ids: tuple[str, ...]
    motion_hints: tuple[SceneFuncMotionHint, ...]

    @property
    def agent_context(self) -> SceneFuncAgentContext:
        """Return the agent-visible task context without hidden GT indices."""
        return {
            "visit_id": self.visit_id,
            "desc_id": self.desc_id,
            "task_description": self.task_description,
            "annotation_ids": list(self.annotation_ids),
            "motion_hints": [hint.to_agent_payload() for hint in self.motion_hints],
        }


def scene_dir_for(data_root: Path, visit_id: str) -> Path:
    """Return the canonical ``<data_root>/<visit_id>`` scene directory."""
    return data_root / visit_id


def safe_sample_id(sample_id: str) -> str:
    """Return a filesystem-safe sample id."""
    return sample_id.replace("::", "__").replace("/", "__")


def list_sample_ids(data_root: Path) -> tuple[str, ...]:
    """Return all SceneFunc3D sample ids under a prepared dataset root."""
    root = Path(data_root)
    if not root.is_dir():
        raise SceneFunc3dDataError(f"SceneFunc3D data root is missing: {root}")

    sample_ids: list[str] = []
    for scene_dir in _numeric_scene_dirs(root):
        visit_id = scene_dir.name
        descriptions = _load_scene_model(
            scene_dir / f"{visit_id}_descriptions.json", _RawDescriptionsFile
        )
        _validate_visit_id(descriptions.visit_id, visit_id, "descriptions")
        sample_ids.extend(
            f"{visit_id}::{description_id}"
            for description_id in _description_ids(descriptions)
        )
    return tuple(sample_ids)


def load_sample(data_root: Path, sample_id: str) -> SceneFunc3dSample:
    """Load one SceneFunc3D sample, excluding hidden annotation GT from context."""
    parsed = SceneFunc3dSampleId.parse(sample_id)
    scene_dir = scene_dir_for(data_root, parsed.visit_id)
    descriptions = _load_scene_model(
        scene_dir / f"{parsed.visit_id}_descriptions.json", _RawDescriptionsFile
    )
    motions = _load_scene_model(
        scene_dir / f"{parsed.visit_id}_motions.json", _RawMotionsFile
    )
    annotations = _load_scene_model(
        scene_dir / f"{parsed.visit_id}_annotations.json", _RawAnnotationsFile
    )
    _validate_visit_id(descriptions.visit_id, parsed.visit_id, "descriptions")
    _validate_visit_id(motions.visit_id, parsed.visit_id, "motions")
    _validate_visit_id(annotations.visit_id, parsed.visit_id, "annotations")

    description = _find_description(descriptions, parsed.desc_id)
    annotation_ids = _coerce_string_tuple(description.annot_id, "annot_id")
    _validate_selected_annotations(annotations, annotation_ids)
    return SceneFunc3dSample(
        sample_id=sample_id,
        visit_id=parsed.visit_id,
        desc_id=parsed.desc_id,
        task_description=_coerce_non_empty_string(
            description.description, "description"
        ),
        annotation_ids=annotation_ids,
        motion_hints=_motion_hints_for(motions, annotation_ids),
    )


def _load_scene_model(path: Path, model_type: type[ModelT]) -> ModelT:
    if not path.is_file():
        raise SceneFunc3dDataError(f"SceneFunc3D JSON is missing: {path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SceneFunc3dDataError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SceneFunc3dDataError(f"{path}: expected a JSON object")
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise SceneFunc3dDataError(f"{path}: failed validation: {exc}") from exc


def _numeric_scene_dirs(data_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                scene_dir
                for scene_dir in data_root.iterdir()
                if scene_dir.is_dir()
                and _VISIT_ID_PATTERN.fullmatch(scene_dir.name) is not None
            ),
            key=lambda path: path.name,
        )
    )


def _validate_visit_id(actual: str, visit_id: str, name: str) -> None:
    if actual != visit_id:
        raise SceneFunc3dDataError(f"{name} visit_id={actual!r}, expected {visit_id!r}")


def _find_description(payload: _RawDescriptionsFile, desc_id: str) -> _RawDescription:
    matches: list[_RawDescription] = []
    for item in payload.descriptions:
        if item.desc_id == desc_id:
            matches.append(item)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SceneFunc3dDataError(f"duplicate description id: {desc_id!r}")
    raise SceneFunc3dDataError(f"description id not found: {desc_id}")


def _description_ids(payload: _RawDescriptionsFile) -> tuple[str, ...]:
    seen_description_ids: set[str] = set()
    description_ids: list[str] = []
    for description in payload.descriptions:
        description_id = _coerce_non_empty_string(description.desc_id, "desc_id")
        if description_id in seen_description_ids:
            raise SceneFunc3dDataError(f"duplicate description id: {description_id!r}")
        seen_description_ids.add(description_id)
        description_ids.append(description_id)
    return tuple(description_ids)


def _validate_selected_annotations(
    payload: _RawAnnotationsFile, annotation_ids: tuple[str, ...]
) -> None:
    annotation_counts: dict[str, int] = {}
    for annotation in payload.annotations:
        annotation_counts[annotation.annot_id] = (
            annotation_counts.get(annotation.annot_id, 0) + 1
        )
    for annotation_id in annotation_ids:
        count = annotation_counts.get(annotation_id, 0)
        if count == 0:
            raise SceneFunc3dDataError(f"annotation id not found: {annotation_id!r}")
        if count > 1:
            raise SceneFunc3dDataError(
                f"duplicate annotation id in annotations: {annotation_id!r}"
            )


def _motion_hints_for(
    payload: _RawMotionsFile, annotation_ids: tuple[str, ...]
) -> tuple[SceneFuncMotionHint, ...]:
    annotation_set = set(annotation_ids)
    hints: list[SceneFuncMotionHint] = []
    for item in payload.motions:
        if item.annot_id not in annotation_set:
            continue
        hints.append(
            SceneFuncMotionHint(
                motion_id=_coerce_non_empty_string(item.motion_id, "motion_id"),
                annotation_id=item.annot_id,
                motion_type=_coerce_non_empty_string(item.motion_type, "motion_type"),
                motion_dir=_coerce_float_tuple(item.motion_dir, "motion_dir"),
            )
        )
    return tuple(hints)


def _coerce_string_tuple(raw: str | list[str], field_name: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        value = _coerce_non_empty_string(raw, field_name)
        return (value,)
    values = tuple(item for item in raw if item)
    if len(values) != len(raw):
        raise SceneFunc3dDataError(f"{field_name} contains empty values")
    if not values:
        raise SceneFunc3dDataError(f"{field_name} must not be empty")
    return values


def _coerce_non_empty_string(raw: str, field_name: str) -> str:
    if not raw.strip():
        raise SceneFunc3dDataError(f"{field_name} must be a non-empty string")
    return raw


def _coerce_float_tuple(raw: Sequence[float], field_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise SceneFunc3dDataError(f"{field_name} must contain floats") from exc
    if len(values) != _MOTION_DIR_DOF:
        raise SceneFunc3dDataError(
            f"{field_name} must contain exactly {_MOTION_DIR_DOF} floats, "
            f"got {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise SceneFunc3dDataError(f"{field_name} must contain only finite floats")
    return values


__all__ = [
    "SceneFuncMotionHintPayload",
    "SceneFuncAgentContext",
    "SceneFunc3dSampleId",
    "SceneFuncMotionHint",
    "SceneFunc3dSample",
    "scene_dir_for",
    "safe_sample_id",
    "list_sample_ids",
    "load_sample",
]
