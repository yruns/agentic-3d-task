"""Write a tiny on-disk SceneFunc3D scene for pipeline/loader tests.

Produces the exact raw layout the loaders expect: per-frame ``-rgb.jpg``,
``-depth.png`` (16-bit mm), ``-intrinsic.txt`` (3x3), ``<id>.txt`` (4x4 pose),
a binary-little-endian ``mesh.ply``, and ``source_frames.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene


@dataclass(frozen=True)
class SyntheticFrame:
    """One synthetic frame's pose and constant depth (metres)."""

    frame_id: str
    camera_to_world: np.ndarray
    depth_value_m: float


def write_synthetic_scene(
    scene_root: Path,
    *,
    frames: tuple[SyntheticFrame, ...],
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (40, 40),
) -> SceneFunc3dToolScene:
    """Write a minimal scene and return the loaded ``SceneFunc3dToolScene``."""
    raw_dir = scene_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    for frame in frames:
        Image.new("RGB", (width, height), (10, 20, 30)).save(
            raw_dir / f"{frame.frame_id}-rgb.jpg"
        )
        depth_mm = np.full(
            (height, width), int(round(frame.depth_value_m * 1000.0)), dtype=np.uint16
        )
        Image.fromarray(depth_mm).save(raw_dir / f"{frame.frame_id}-depth.png")
        _write_matrix(raw_dir / f"{frame.frame_id}-intrinsic.txt", intrinsics)
        _write_matrix(raw_dir / f"{frame.frame_id}.txt", frame.camera_to_world)
    _write_binary_ply(raw_dir / "mesh.ply", vertices_world)
    _write_source_frames_json(raw_dir / "source_frames.json", frames)
    return SceneFunc3dToolScene.load(scene_root)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    rows = "\n".join(
        " ".join(f"{value:.10f}" for value in row) for row in np.asarray(matrix)
    )
    path.write_text(rows + "\n", encoding="utf-8")


def _write_source_frames_json(path: Path, frames: tuple[SyntheticFrame, ...]) -> None:
    import json

    payload = [
        {
            "frame_id": frame.frame_id,
            "rgb": f"{frame.frame_id}-rgb.jpg",
            "depth": f"{frame.frame_id}-depth.png",
            "intrinsic": f"{frame.frame_id}-intrinsic.txt",
            "pose": f"{frame.frame_id}.txt",
        }
        for frame in frames
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_binary_ply(path: Path, vertices_world: np.ndarray) -> None:
    vertices = np.asarray(vertices_world, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))


__all__ = ["SyntheticFrame", "write_synthetic_scene"]
