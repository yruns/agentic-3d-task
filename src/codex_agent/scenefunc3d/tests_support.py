"""Importable helpers for building SceneFun3D test artifacts on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends.lift_3d import FloatArray


def write_binary_ply_mesh(path: Path, vertices_world: FloatArray) -> Path:
    """Write ``vertices_world`` as a binary-little-endian XYZ PLY mesh."""
    vertices = np.asarray(vertices_world, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices_world must have shape (N, 3): {vertices.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))
    return path


__all__ = ["write_binary_ply_mesh"]
