"""Deterministic 2D mask to 3D point lifting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt

from codex_agent.scenefunc3d.tools.models import ToolInputError

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
_POSE_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_POSE_LAST_ROW_ATOL = 1e-8
_MIN_ROTATION_DETERMINANT_ABS = 1e-12
_PLY_HEADER_MAX_LINES = 512
_FALLBACK_NEAREST_SCENE_CHUNK_SIZE = 4096
_PLY_LITTLE_ENDIAN_DTYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class _PlyVertexLayout:
    """Binary PLY vertex layout parsed from a SceneFuncVal-CG mesh header."""

    vertex_count: int
    dtype_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CameraGeometry:
    """Validated pinhole camera intrinsics and camera-to-world pose."""

    intrinsics: FloatArray
    camera_to_world: FloatArray

    def __post_init__(self) -> None:
        """Validate camera matrices before lifting any external frame data."""
        intrinsics = _coerce_float_array(
            self.intrinsics,
            field_name="intrinsics",
            expected_shape=(3, 3),
        )
        camera_to_world = _coerce_float_array(
            self.camera_to_world,
            field_name="camera_to_world",
            expected_shape=(4, 4),
        )
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ToolInputError(
                "invalid camera intrinsics: focal lengths must be positive"
            )
        if not bool(
            np.allclose(
                camera_to_world[3, :],
                _POSE_LAST_ROW,
                atol=_POSE_LAST_ROW_ATOL,
                rtol=0.0,
            )
        ):
            raise ToolInputError(
                "invalid camera_to_world pose: last row must be [0, 0, 0, 1]"
            )
        rotation_determinant = float(np.linalg.det(camera_to_world[:3, :3]))
        if abs(rotation_determinant) <= _MIN_ROTATION_DETERMINANT_ABS:
            raise ToolInputError(
                "invalid camera_to_world pose: rotation block must be non-degenerate"
            )
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)


def load_mask_npz(path: Path) -> BoolArray:
    """Load a 2D boolean mask from a SAM-style NPZ artifact."""
    try:
        with np.load(path) as archive:
            if "mask" not in archive.files:
                raise ToolInputError(
                    "mask npz is missing required key 'mask': " f"path={path}"
                )
            mask_array = np.asarray(archive["mask"])
    except ToolInputError:
        raise
    except (OSError, ValueError) as exc:
        raise ToolInputError(
            "could not load mask npz artifact: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc

    return _coerce_binary_mask(mask_array, field_name=f"mask array at {path}")


def backproject_mask_to_world(
    mask: BoolArray,
    depth_meters: FloatArray,
    geometry: CameraGeometry,
) -> FloatArray:
    """Back-project positive-depth mask pixels into world-space XYZ points."""
    boolean_mask = _coerce_bool_mask(mask, field_name="mask")
    depth = _coerce_depth(depth_meters)
    if depth.shape != boolean_mask.shape:
        raise ToolInputError(
            "depth_meters shape must match mask shape: "
            f"mask_shape={boolean_mask.shape}; depth_shape={depth.shape}"
        )

    valid_depth_mask = np.isfinite(depth) & (depth > 0.0)
    valid_mask = boolean_mask & valid_depth_mask
    if not bool(np.any(valid_mask)):
        raise ToolInputError(
            "lift_too_sparse: mask contains no pixels with finite positive depth"
        )

    rows, columns = np.nonzero(valid_mask)
    z_values = depth[rows, columns]
    fx = geometry.intrinsics[0, 0]
    fy = geometry.intrinsics[1, 1]
    cx = geometry.intrinsics[0, 2]
    cy = geometry.intrinsics[1, 2]
    x_values = (columns.astype(np.float64) - cx) * z_values / fx
    y_values = (rows.astype(np.float64) - cy) * z_values / fy
    camera_points = np.column_stack((x_values, y_values, z_values))
    homogeneous_camera_points = np.column_stack(
        (camera_points, np.ones(camera_points.shape[0], dtype=np.float64))
    )
    world_homogeneous = homogeneous_camera_points @ geometry.camera_to_world.T
    world_points = cast(FloatArray, world_homogeneous[:, :3].astype(np.float64))
    return _validate_points_world(world_points, field_name="points_world")


def load_scene_mesh_vertices(mesh_ply_path: Path) -> FloatArray:
    """Load ordered XYZ vertices from a SceneFuncVal-CG binary mesh PLY."""
    try:
        with mesh_ply_path.open("rb") as handle:
            layout = _read_binary_ply_vertex_layout(handle, mesh_ply_path)
            vertex_dtype = np.dtype(list(layout.dtype_fields))
            expected_bytes = vertex_dtype.itemsize * layout.vertex_count
            vertex_data = handle.read(expected_bytes)
    except OSError as exc:
        raise ToolInputError(
            "could not read SceneFunc3D mesh PLY: "
            f"path={mesh_ply_path}; error_type={exc.__class__.__name__}"
        ) from exc
    if len(vertex_data) != expected_bytes:
        raise ToolInputError(
            "SceneFunc3D mesh PLY ended before all vertex rows were read: "
            f"path={mesh_ply_path}; expected_bytes={expected_bytes}; "
            f"actual_bytes={len(vertex_data)}"
        )
    vertex_records = np.frombuffer(
        vertex_data, dtype=vertex_dtype, count=layout.vertex_count
    )
    points_world = np.column_stack(
        (
            vertex_records["x"].astype(np.float64),
            vertex_records["y"].astype(np.float64),
            vertex_records["z"].astype(np.float64),
        )
    )
    return _validate_points_world(
        cast(FloatArray, points_world), field_name=f"vertices at {mesh_ply_path}"
    )


def assign_nearest_scene_point_indices(
    points_world: FloatArray,
    scene_points_world: FloatArray,
    *,
    max_distance_meters: float,
) -> IntArray:
    """Assign each lifted point to its nearest ordered scene vertex index."""
    lifted_points = _validate_points_world(points_world, field_name="points_world")
    scene_points = _validate_points_world(
        scene_points_world, field_name="scene_points_world"
    )
    if not np.isfinite(max_distance_meters) or max_distance_meters <= 0.0:
        raise ToolInputError(
            "max_distance_meters must be a positive finite float: "
            f"value={max_distance_meters!r}"
        )
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        point_indices, distances = _assign_nearest_scene_point_indices_numpy(
            lifted_points, scene_points
        )
    else:
        tree = cKDTree(scene_points)
        raw_distances, raw_indices = tree.query(lifted_points, k=1)
        distances = cast(FloatArray, np.asarray(raw_distances, dtype=np.float64))
        point_indices = cast(IntArray, np.asarray(raw_indices, dtype=np.int64))
    _validate_nearest_scene_distances(
        distances,
        max_distance_meters=max_distance_meters,
    )
    return point_indices


def write_lift_npz(
    path: Path, points_world: FloatArray, *, point_indices: IntArray | None = None
) -> Path:
    """Write lifted world points to a compressed NPZ artifact."""
    valid_points = _validate_points_world(points_world, field_name="points_world")
    valid_point_indices = (
        _validate_point_indices(
            point_indices,
            expected_count=int(valid_points.shape[0]),
            field_name="point_indices",
        )
        if point_indices is not None
        else None
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if valid_point_indices is None:
            np.savez_compressed(path, points_world=valid_points)
        else:
            np.savez_compressed(
                path, points_world=valid_points, point_indices=valid_point_indices
            )
    except OSError as exc:
        raise ToolInputError(
            "could not write lifted points npz: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc
    return path


def _read_binary_ply_vertex_layout(
    handle: BinaryIO, mesh_ply_path: Path
) -> _PlyVertexLayout:
    first_line = _read_ply_header_line(handle, mesh_ply_path)
    if first_line != "ply":
        raise ToolInputError(
            "SceneFunc3D mesh PLY must start with 'ply': " f"path={mesh_ply_path}"
        )

    format_seen = False
    vertex_count = -1
    in_vertex_element = False
    dtype_fields: list[tuple[str, str]] = []
    for _line_index in range(_PLY_HEADER_MAX_LINES):
        line = _read_ply_header_line(handle, mesh_ply_path)
        if line == "end_header":
            break
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            if len(tokens) < 3 or tokens[1] != "binary_little_endian":
                raise ToolInputError(
                    "SceneFunc3D mesh PLY must use binary_little_endian format: "
                    f"path={mesh_ply_path}; line={line!r}"
                )
            format_seen = True
            continue
        if tokens[0] == "element":
            if len(tokens) != 3:
                raise ToolInputError(
                    "SceneFunc3D mesh PLY has malformed element line: "
                    f"path={mesh_ply_path}; line={line!r}"
                )
            in_vertex_element = tokens[1] == "vertex"
            if in_vertex_element:
                vertex_count = _parse_positive_int(
                    tokens[2], field_name="vertex_count", mesh_ply_path=mesh_ply_path
                )
            continue
        if in_vertex_element and tokens[0] == "property":
            dtype_fields.append(_parse_ply_property(tokens, mesh_ply_path))
            continue
    else:
        raise ToolInputError(
            "SceneFunc3D mesh PLY header is missing end_header: "
            f"path={mesh_ply_path}"
        )

    if not format_seen:
        raise ToolInputError(
            "SceneFunc3D mesh PLY is missing a format line: " f"path={mesh_ply_path}"
        )
    if vertex_count <= 0:
        raise ToolInputError(
            "SceneFunc3D mesh PLY is missing a positive vertex element count: "
            f"path={mesh_ply_path}"
        )
    field_names = {field_name for field_name, _dtype in dtype_fields}
    missing_fields = tuple(
        field_name for field_name in ("x", "y", "z") if field_name not in field_names
    )
    if missing_fields:
        raise ToolInputError(
            "SceneFunc3D mesh PLY vertex properties are missing XYZ fields: "
            f"path={mesh_ply_path}; missing={missing_fields}"
        )
    return _PlyVertexLayout(vertex_count=vertex_count, dtype_fields=tuple(dtype_fields))


def _read_ply_header_line(handle: BinaryIO, mesh_ply_path: Path) -> str:
    raw_line = handle.readline()
    if not raw_line:
        raise ToolInputError(
            "SceneFunc3D mesh PLY ended before end_header: " f"path={mesh_ply_path}"
        )
    try:
        return raw_line.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ToolInputError(
            "SceneFunc3D mesh PLY header must be ASCII: " f"path={mesh_ply_path}"
        ) from exc


def _parse_ply_property(tokens: list[str], mesh_ply_path: Path) -> tuple[str, str]:
    if len(tokens) != 3:
        raise ToolInputError(
            "SceneFunc3D mesh PLY only supports scalar vertex properties: "
            f"path={mesh_ply_path}; line={' '.join(tokens)!r}"
        )
    raw_dtype = tokens[1]
    dtype = _PLY_LITTLE_ENDIAN_DTYPES.get(raw_dtype)
    if dtype is None:
        raise ToolInputError(
            "SceneFunc3D mesh PLY has unsupported vertex property type: "
            f"path={mesh_ply_path}; property_type={raw_dtype!r}"
        )
    return (tokens[2], dtype)


def _parse_positive_int(raw_value: str, *, field_name: str, mesh_ply_path: Path) -> int:
    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise ToolInputError(
            "SceneFunc3D mesh PLY has a non-integer count: "
            f"path={mesh_ply_path}; field={field_name}; value={raw_value!r}"
        ) from exc
    if parsed_value <= 0:
        raise ToolInputError(
            "SceneFunc3D mesh PLY count must be positive: "
            f"path={mesh_ply_path}; field={field_name}; value={parsed_value}"
        )
    return parsed_value


def _assign_nearest_scene_point_indices_numpy(
    points_world: FloatArray, scene_points_world: FloatArray
) -> tuple[IntArray, FloatArray]:
    nearest_distances_sq = np.full(points_world.shape[0], np.inf, dtype=np.float64)
    nearest_indices = np.zeros(points_world.shape[0], dtype=np.int64)
    row_indices = np.arange(points_world.shape[0])
    for start_index in range(
        0, scene_points_world.shape[0], _FALLBACK_NEAREST_SCENE_CHUNK_SIZE
    ):
        end_index = min(
            start_index + _FALLBACK_NEAREST_SCENE_CHUNK_SIZE,
            scene_points_world.shape[0],
        )
        scene_chunk = scene_points_world[start_index:end_index]
        deltas = points_world[:, np.newaxis, :] - scene_chunk[np.newaxis, :, :]
        chunk_distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas)
        chunk_nearest_offsets = np.argmin(chunk_distances_sq, axis=1)
        chunk_nearest_distances_sq = chunk_distances_sq[
            row_indices, chunk_nearest_offsets
        ]
        should_update = chunk_nearest_distances_sq < nearest_distances_sq
        nearest_distances_sq[should_update] = chunk_nearest_distances_sq[should_update]
        nearest_indices[should_update] = (
            start_index + chunk_nearest_offsets[should_update]
        )
    distances = cast(FloatArray, np.sqrt(nearest_distances_sq))
    return cast(IntArray, nearest_indices), distances


def _validate_nearest_scene_distances(
    distances: FloatArray, *, max_distance_meters: float
) -> None:
    if not bool(np.all(np.isfinite(distances))):
        raise ToolInputError("nearest_scene_point_failed: non-finite distance")
    too_far = distances > max_distance_meters
    if bool(np.any(too_far)):
        first_bad_index = int(np.argmax(too_far))
        raise ToolInputError(
            "nearest_scene_point_too_far: "
            f"point_index={first_bad_index}; "
            f"distance_meters={float(distances[first_bad_index]):.6g}; "
            f"max_distance_meters={max_distance_meters:.6g}"
        )


def write_lift_ply(path: Path, points_world: FloatArray) -> Path:
    """Write lifted world points as an ASCII PLY point cloud."""
    valid_points = _validate_points_world(points_world, field_name="points_world")
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {valid_points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    vertex_lines = "\n".join(
        " ".join(_format_float(coordinate) for coordinate in point)
        for point in valid_points
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{header}{vertex_lines}\n", encoding="ascii")
    except OSError as exc:
        raise ToolInputError(
            "could not write lifted points ply: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc
    return path


def _coerce_float_array(
    value: FloatArray,
    *,
    field_name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    array = cast(FloatArray, np.asarray(value, dtype=np.float64))
    if array.shape != expected_shape:
        raise ToolInputError(
            f"{field_name} must have shape {expected_shape}: shape={array.shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ToolInputError(f"{field_name} must contain only finite values")
    return array


def _coerce_bool_mask(mask: BoolArray, *, field_name: str) -> BoolArray:
    mask_array = np.asarray(mask)
    return _coerce_binary_mask(mask_array, field_name=field_name)


def _coerce_binary_mask(
    mask_array: npt.NDArray[np.generic],
    *,
    field_name: str,
) -> BoolArray:
    if mask_array.ndim != 2:
        raise ToolInputError(f"{field_name} must be a 2D mask: ndim={mask_array.ndim}")
    if mask_array.dtype == np.dtype(np.bool_):
        return cast(BoolArray, mask_array.astype(np.bool_, copy=False))
    if np.issubdtype(mask_array.dtype, np.complexfloating):
        raise ToolInputError(f"{field_name} must be a binary numeric or bool mask")
    if not np.issubdtype(mask_array.dtype, np.number):
        raise ToolInputError(f"{field_name} must be a binary numeric or bool mask")
    numeric_mask = cast(FloatArray, np.asarray(mask_array, dtype=np.float64))
    if not bool(np.all(np.isfinite(numeric_mask))):
        raise ToolInputError(f"{field_name} must be a finite binary mask")
    is_binary = (numeric_mask == 0.0) | (numeric_mask == 1.0)
    if not bool(np.all(is_binary)):
        raise ToolInputError(f"{field_name} must be a binary mask with values 0 or 1")
    return cast(BoolArray, numeric_mask.astype(np.bool_, copy=False))


def _coerce_depth(depth_meters: FloatArray) -> FloatArray:
    depth = cast(FloatArray, np.asarray(depth_meters, dtype=np.float64))
    if depth.ndim != 2:
        raise ToolInputError(f"depth_meters must be 2D: ndim={depth.ndim}")
    return depth


def _validate_points_world(points_world: FloatArray, *, field_name: str) -> FloatArray:
    points = cast(FloatArray, np.asarray(points_world, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ToolInputError(
            f"{field_name} must have shape (N, 3): shape={points.shape}"
        )
    if points.shape[0] == 0:
        raise ToolInputError(f"{field_name} must contain at least one point")
    if not bool(np.all(np.isfinite(points))):
        raise ToolInputError(f"{field_name} must contain only finite values")
    return points


def _validate_point_indices(
    point_indices: IntArray,
    *,
    expected_count: int,
    field_name: str,
) -> IntArray:
    indices = np.asarray(point_indices)
    if indices.ndim != 1:
        raise ToolInputError(f"{field_name} must be a 1D array: ndim={indices.ndim}")
    if indices.shape[0] != expected_count:
        raise ToolInputError(
            f"{field_name} count must match points_world count: "
            f"indices={indices.shape[0]}; points={expected_count}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ToolInputError(f"{field_name} must contain integer scene point ids")
    if bool(np.any(indices < 0)):
        raise ToolInputError(f"{field_name} must contain non-negative scene point ids")
    return cast(IntArray, indices.astype(np.int64, copy=False))


def _format_float(value: np.float64) -> str:
    return format(float(value), ".15g")


__all__ = [
    "BoolArray",
    "CameraGeometry",
    "FloatArray",
    "IntArray",
    "assign_nearest_scene_point_indices",
    "backproject_mask_to_world",
    "load_mask_npz",
    "load_scene_mesh_vertices",
    "write_lift_npz",
    "write_lift_ply",
]
