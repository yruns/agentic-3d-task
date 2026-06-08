"""Lightweight triangle-mesh container and PLY loader.

Reads ScanNet ``*_vh_clean*.ply`` meshes with :mod:`plyfile` (a small,
pure-Python dependency) instead of the heavy ``open3d`` stack. Only the data
the BEV renderer needs is kept: vertex positions, per-vertex RGB colors and
triangle indices. Triangle normals are derived on demand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class TriangleMesh(BaseModel):
    """A colored triangle mesh in a single coordinate frame."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vertices: NDArray[np.float64]
    """Vertex positions, shape ``(V, 3)``."""

    colors: NDArray[np.float32]
    """Per-vertex RGB in ``[0, 1]``, shape ``(V, 3)``."""

    triangles: NDArray[np.int64]
    """Triangle vertex indices, shape ``(T, 3)``."""

    def triangle_normals(self) -> NDArray[np.float64]:
        """Unit face normals via the right-hand rule on vertex order."""
        tri = self.vertices[self.triangles]
        edge1 = tri[:, 1] - tri[:, 0]
        edge2 = tri[:, 2] - tri[:, 0]
        normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        unit: NDArray[np.float64] = normals / lengths
        return unit

    def transformed(self, matrix: NDArray[np.float64]) -> TriangleMesh:
        """Apply a 4x4 homogeneous transform to vertices (colors unchanged)."""
        if matrix.shape != (4, 4):
            raise ValueError(f"transform must be 4x4, got {matrix.shape}")
        homogeneous = np.concatenate(
            [self.vertices, np.ones((self.vertices.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        moved = (matrix @ homogeneous.T).T[:, :3]
        return TriangleMesh(
            vertices=moved, colors=self.colors, triangles=self.triangles
        )


def load_ply_mesh(path: Path) -> TriangleMesh:
    """Load a triangle mesh from a binary/ascii PLY file.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if it
    lacks vertex colors or triangular faces.
    """
    from plyfile import PlyData

    if not path.exists():
        raise FileNotFoundError(f"mesh not found: {path}")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not {"x", "y", "z"} <= set(names):
        raise ValueError(f"PLY {path} missing x/y/z vertex coordinates")
    if not {"red", "green", "blue"} <= set(names):
        raise ValueError(f"PLY {path} has no vertex colors")

    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float64
    )
    colors = (
        np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
            np.float32
        )
        / 255.0
    )

    triangles = _read_triangles(ply)
    return TriangleMesh(vertices=vertices, colors=colors, triangles=triangles)


def _read_triangles(ply_data: object) -> NDArray[np.int64]:
    """Extract an ``(T, 3)`` triangle index array from a PLY face element."""
    from plyfile import PlyData

    if not isinstance(ply_data, PlyData):  # pragma: no cover - defensive
        raise TypeError("expected a PlyData instance")
    if "face" not in {element.name for element in ply_data.elements}:
        raise ValueError("PLY has no 'face' element")

    faces = ply_data["face"].data
    field = "vertex_indices" if "vertex_indices" in (faces.dtype.names or ()) else None
    if field is None:
        for candidate in ("vertex_index", "vertex_indices"):
            if candidate in (faces.dtype.names or ()):
                field = candidate
                break
    if field is None:
        raise ValueError("PLY face element has no vertex index field")

    rows: list[NDArray[np.int64]] = []
    for face in faces[field]:
        indices = np.asarray(face, dtype=np.int64)
        if indices.shape[0] == 3:
            rows.append(indices)
        elif indices.shape[0] > 3:
            for k in range(1, indices.shape[0] - 1):
                rows.append(
                    np.array([indices[0], indices[k], indices[k + 1]], dtype=np.int64)
                )
    if not rows:
        raise ValueError("PLY contains no triangular faces")
    return np.stack(rows, axis=0)
