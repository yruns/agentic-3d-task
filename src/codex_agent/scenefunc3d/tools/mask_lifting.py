"""Data contracts and deterministic implementation for 2D-mask-to-3D lifting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Annotated, TypedDict

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    StringConstraints,
    model_validator,
)

from ...errors import SceneFunc3dDataError
from .models import ToolInputError

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.backends.lift_3d import FloatArray

SafePathComponentText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]
_SAFE_PATH_COMPONENT_RE: Pattern[str] = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_SCENE_POINT_ASSIGNMENT_DISTANCE_METERS = 0.05


class LiftMaskResultPayload(TypedDict):
    """JSON-ready payload for a lifted 3D mask result."""

    frame_id: str
    candidate_id: str
    lifted_point_count: int
    mask_npz_path: str
    mask_ply_path: str
    overlay_path: str


class LiftMaskArgs(BaseModel):
    """Arguments for deterministic 2D-mask-to-3D lifting."""

    model_config = ConfigDict(extra="forbid")

    frame_id: SafePathComponentText
    candidate_id: SafePathComponentText
    mask_path: FilePath = Field(
        validation_alias=AliasChoices("mask_path", "mask_npz_path")
    )
    depth_path: FilePath
    intrinsics_path: FilePath
    pose_path: FilePath

    @model_validator(mode="before")
    @classmethod
    def normalize_review_artifact_fields(cls, payload: object) -> object:
        """Ignore review-only fields that do not affect deterministic lifting."""
        if not isinstance(payload, Mapping):
            return payload
        values = dict(payload)
        values.pop("mask_overlay_path", None)
        return values


@dataclass(frozen=True)
class LiftMaskResult:
    """Lifted point-mask artifact paths for one 2D mask candidate."""

    frame_id: str
    candidate_id: str
    lifted_point_count: int
    mask_npz_path: Path
    mask_ply_path: Path
    overlay_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed lifting result contracts."""
        if not self.frame_id.strip():
            raise SceneFunc3dDataError("frame_id must not be empty")
        if not self.candidate_id.strip():
            raise SceneFunc3dDataError("candidate_id must not be empty")
        if self.lifted_point_count <= 0:
            raise SceneFunc3dDataError(
                "lifted_point_count must be positive; "
                f"got {self.lifted_point_count!r}"
            )

    def to_payload(self) -> dict[str, object]:
        """Return this lifting result as a JSON-ready mapping."""
        return {
            "frame_id": self.frame_id,
            "candidate_id": self.candidate_id,
            "lifted_point_count": self.lifted_point_count,
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "overlay_path": str(self.overlay_path),
        }


def lift_mask_to_3d(
    args: LiftMaskArgs, *, out_dir: Path, scene_mesh_path: Path
) -> LiftMaskResult:
    """Lift one 2D mask candidate into deterministic world-space point artifacts."""
    from codex_agent.scenefunc3d.backends.lift_3d import (
        CameraGeometry,
        assign_nearest_scene_point_indices,
        backproject_mask_to_world,
        load_mask_npz,
        load_scene_mesh_vertices,
        write_lift_npz,
        write_lift_ply,
    )

    mask = load_mask_npz(args.mask_path)
    depth_meters = _read_depth_meters(args.depth_path)
    intrinsics = _read_matrix(
        args.intrinsics_path,
        expected_shape=(3, 3),
        field_name="intrinsics",
    )
    camera_to_world = _read_matrix(
        args.pose_path,
        expected_shape=(4, 4),
        field_name="camera_to_world",
    )
    try:
        geometry = CameraGeometry(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
    except ToolInputError as exc:
        raise ToolInputError(
            "invalid lift camera geometry: "
            f"intrinsics_path={args.intrinsics_path}; pose_path={args.pose_path}; "
            f"error={exc}"
        ) from exc
    points_world = backproject_mask_to_world(mask, depth_meters, geometry)
    scene_points_world = load_scene_mesh_vertices(scene_mesh_path)
    point_indices = assign_nearest_scene_point_indices(
        points_world,
        scene_points_world,
        max_distance_meters=_MAX_SCENE_POINT_ASSIGNMENT_DISTANCE_METERS,
    )
    fragment_dir = _fragment_dir(
        out_dir, frame_id=args.frame_id, candidate_id=args.candidate_id
    )
    mask_npz_path = write_lift_npz(
        fragment_dir / "mask_data.npz",
        points_world,
        point_indices=point_indices,
    )
    mask_ply_path = write_lift_ply(fragment_dir / "lifted_points.ply", points_world)
    overlay_path = _write_lift_summary(
        fragment_dir / "lift_overlay.txt",
        args=args,
        scene_mesh_path=scene_mesh_path,
        lifted_point_count=int(points_world.shape[0]),
    )
    return LiftMaskResult(
        frame_id=args.frame_id,
        candidate_id=args.candidate_id,
        lifted_point_count=int(points_world.shape[0]),
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        overlay_path=overlay_path,
    )


def _read_depth_meters(depth_path: Path) -> FloatArray:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "numpy and Pillow are required to read lift depth images; install the "
            "'vision' extra"
        ) from exc

    try:
        with Image.open(depth_path) as image:
            depth_pixels = np.asarray(image, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read lift depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse lift depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc

    if depth_pixels.ndim != 2:
        raise ToolInputError(
            "lift depth image must be a single-channel 2D image: "
            f"path={depth_path}; shape={depth_pixels.shape}"
        )
    depth_meters: FloatArray = depth_pixels / 1000.0
    return depth_meters


def _read_matrix(
    matrix_path: Path,
    *,
    expected_shape: tuple[int, int],
    field_name: str,
) -> FloatArray:
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolInputError(
            "numpy is required to read lift camera matrices; install the "
            "'vision' extra"
        ) from exc

    expected_size = expected_shape[0] * expected_shape[1]
    try:
        raw_matrix = np.loadtxt(matrix_path, dtype=np.float64)
        matrix = np.asarray(raw_matrix, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read lift matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse lift matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if matrix.size != expected_size:
        raise ToolInputError(
            "lift matrix has wrong element count: "
            f"field={field_name}; path={matrix_path}; "
            f"expected={expected_size}; actual={matrix.size}"
        )
    reshaped_matrix: FloatArray = matrix.reshape(expected_shape)
    return reshaped_matrix


def _fragment_dir(out_dir: Path, *, frame_id: str, candidate_id: str) -> Path:
    if not _SAFE_PATH_COMPONENT_RE.fullmatch(frame_id):
        raise ToolInputError(f"frame_id must be a safe path component: {frame_id!r}")
    if not _SAFE_PATH_COMPONENT_RE.fullmatch(candidate_id):
        raise ToolInputError(
            f"candidate_id must be a safe path component: {candidate_id!r}"
        )
    return out_dir / "fragments" / f"{frame_id}_{candidate_id}"


def _write_lift_summary(
    overlay_path: Path,
    *,
    args: LiftMaskArgs,
    scene_mesh_path: Path,
    lifted_point_count: int,
) -> Path:
    summary = (
        f"frame_id={args.frame_id}\n"
        f"candidate_id={args.candidate_id}\n"
        f"lifted_point_count={lifted_point_count}\n"
        f"mask_path={args.mask_path}\n"
        f"depth_path={args.depth_path}\n"
        f"intrinsics_path={args.intrinsics_path}\n"
        f"pose_path={args.pose_path}\n"
        f"scene_mesh_path={scene_mesh_path}\n"
    )
    try:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(summary, encoding="utf-8")
    except OSError as exc:
        raise ToolInputError(
            "could not write lift overlay summary: "
            f"path={overlay_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return overlay_path


__all__ = [
    "LiftMaskArgs",
    "LiftMaskResult",
    "LiftMaskResultPayload",
    "lift_mask_to_3d",
]
