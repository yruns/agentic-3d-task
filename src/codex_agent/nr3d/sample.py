"""Loaders for prepared NR3D samples and per-scene assets.

A *sample* is one referring expression with its ground-truth box; a *scene*
bundles the proposal pool plus the BEV render and catalog metadata shared by all
samples in that scene. Scene assets are located canonically from
``<data_root>/<scene_id>/<pack_name>/...`` rather than trusting any absolute
paths embedded in the artifact (those point at the machine that generated them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import Nr3dDataError
from .proposals import ProposalPool

DEFAULT_PACK_NAME = "pack_nr3d_v9_catalog_first"
_BBOX_DOF = 9
_BEV_FILENAME = "scene_bev_nr3d.png"


@dataclass(frozen=True)
class Nr3dSampleId:
    """A parsed NR3D ``<scan>::<target_id>::<assignment>`` identifier."""

    raw: str
    scan_id: str
    scene_id: str
    target_id: int
    assignment_id: str

    @classmethod
    def parse(cls, sample_id: str) -> Nr3dSampleId:
        """Parse the ``<scan>::<target_id>::<assignment>`` form.

        Raises:
            Nr3dDataError: If the identifier is malformed.
        """
        if not isinstance(sample_id, str):
            raise Nr3dDataError(
                "sample_id must be a string in "
                f"'<scan>::<target_id>::<assignment>' form, got {sample_id!r}"
            )
        parts = sample_id.split("::")
        if len(parts) != 3:
            raise Nr3dDataError(
                f"sample_id must have 3 '::'-separated parts, got {sample_id!r}"
            )
        scan_id, target_text, assignment_id = parts
        if not scan_id or not assignment_id:
            raise Nr3dDataError(f"invalid NR3D sample_id: {sample_id!r}")
        try:
            target_id = int(target_text)
        except ValueError as exc:
            raise Nr3dDataError(
                f"invalid target_id in sample_id {sample_id!r}"
            ) from exc
        return cls(
            raw=sample_id,
            scan_id=scan_id,
            scene_id=scan_id.split("/")[-1],
            target_id=target_id,
            assignment_id=assignment_id,
        )


@dataclass(frozen=True)
class Nr3dSample:
    """One prepared NR3D visual-grounding sample."""

    sample_id: str
    scene_id: str
    target_id: int
    category: str
    query: str
    gt_bbox_3d_9dof: tuple[float, ...]

    @classmethod
    def from_artifact(cls, payload: dict[str, Any], *, expected_id: str) -> Nr3dSample:
        """Build a sample from a prepared artifact dict, validating the id."""
        actual_id = payload.get("sample_id")
        if actual_id != expected_id:
            raise Nr3dDataError(
                f"sample artifact has sample_id={actual_id!r}, expected "
                f"{expected_id!r}"
            )
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise Nr3dDataError(f"{expected_id}: 'query' must be a non-empty string")
        return cls(
            sample_id=expected_id,
            scene_id=str(payload["scene_id"]),
            target_id=int(payload["target_id"]),
            category=str(payload.get("category", "")),
            query=query,
            gt_bbox_3d_9dof=_coerce_bbox_9dof(
                payload.get("gt_bbox_3d_9dof"), field_name=f"{expected_id}.gt_bbox"
            ),
        )


@dataclass(frozen=True)
class Nr3dScene:
    """Per-scene assets shared across all samples in a scene."""

    scene_id: str
    proposal_pool: ProposalPool
    bev_image_path: Path
    scene_category: str = ""
    total_frames: int | None = None
    frame_id_range: tuple[int, int] | None = None

    @classmethod
    def load(cls, scene_dir: Path) -> Nr3dScene:
        """Load proposal pool, BEV render, and catalog metadata for a scene.

        Raises:
            Nr3dDataError: If required assets are missing or malformed.
        """
        if not scene_dir.is_dir():
            raise Nr3dDataError(f"scene artifacts directory is missing: {scene_dir}")
        pool = ProposalPool.from_files(
            scene_dir / "proposals.jsonl", scene_dir / "visibility.json"
        )
        bev_image_path = scene_dir / "bev" / _BEV_FILENAME
        if not bev_image_path.exists():
            raise Nr3dDataError(f"BEV image is missing: {bev_image_path}")

        catalog = _load_optional_catalog(scene_dir / "scene_catalog.json")
        scene_category = str(catalog.get("scene_category") or "")
        total_frames = _optional_int(catalog.get("total_frames"))
        frame_id_range = _optional_int_pair(catalog.get("frame_id_range"))
        return cls(
            scene_id=str(catalog.get("scene_id") or pool.scene_id),
            proposal_pool=pool,
            bev_image_path=bev_image_path,
            scene_category=scene_category,
            total_frames=total_frames,
            frame_id_range=frame_id_range,
        )


def scene_dir_for(
    data_root: Path, scene_id: str, *, pack_name: str = DEFAULT_PACK_NAME
) -> Path:
    """Return the canonical ``<data_root>/<scene_id>/<pack_name>`` directory."""
    return data_root / scene_id / pack_name


def safe_sample_id(sample_id: str) -> str:
    """Return a filesystem-safe form of an NR3D sample id."""
    return sample_id.replace("/", "__").replace("::", "__")


def sample_artifact_path(
    data_root: Path, sample_id: str, *, pack_name: str = DEFAULT_PACK_NAME
) -> Path:
    """Return the expected prepared-sample artifact path."""
    parsed = Nr3dSampleId.parse(sample_id)
    return (
        scene_dir_for(data_root, parsed.scene_id, pack_name=pack_name)
        / "samples"
        / f"{safe_sample_id(sample_id)}.json"
    )


def load_sample(
    data_root: Path, sample_id: str, *, pack_name: str = DEFAULT_PACK_NAME
) -> Nr3dSample:
    """Load and validate one prepared NR3D sample artifact."""
    path = sample_artifact_path(data_root, sample_id, pack_name=pack_name)
    if not path.exists():
        raise Nr3dDataError(f"prepared sample artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Nr3dDataError(f"{path}: expected a JSON object")
    return Nr3dSample.from_artifact(payload, expected_id=sample_id)


def _load_optional_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Nr3dDataError(f"{path}: scene catalog must be a JSON object")
    return payload


def _coerce_bbox_9dof(raw: Any, *, field_name: str) -> tuple[float, ...]:
    if isinstance(raw, str):
        raw = json.loads(raw.strip())
    if not isinstance(raw, (list, tuple)):
        raise Nr3dDataError(
            f"{field_name} must be a list/tuple, got {type(raw).__name__}"
        )
    values = tuple(float(v) for v in raw)
    if len(values) != _BBOX_DOF:
        raise Nr3dDataError(
            f"{field_name} must contain exactly {_BBOX_DOF} floats, got {len(values)}"
        )
    return values


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_int_pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


__all__ = [
    "DEFAULT_PACK_NAME",
    "Nr3dSampleId",
    "Nr3dSample",
    "Nr3dScene",
    "scene_dir_for",
    "safe_sample_id",
    "sample_artifact_path",
    "load_sample",
]
