"""Benchmark-aware scene BEV builders.

A :class:`SceneBEVBuilder` resolves the mesh / trajectory / intrinsic assets for
one scene of a given benchmark, renders the BEV (mesh in the raw scan frame is
moved into the axis-aligned object frame first), caches the PNG, and returns its
path. :class:`Nr3dSceneBEVBuilder` implements the NR3D / ScanNet data layout.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from keyframe.bev.config import (
    DEFAULT_SCENE_BEV_CONFIG,
    SceneBEVConfig,
    scannet_data_root_override,
)
from keyframe.bev.mesh import load_ply_mesh
from keyframe.bev.render import BEVMarker, render_scene_bev


class BEVScenePaths(BaseModel):
    """Resolved on-disk inputs needed to render one scene's BEV."""

    model_config = ConfigDict(frozen=True)

    mesh: Path
    trajectory: Path
    intrinsic: Path


class SceneBEVBuilder(ABC):
    """Render + cache a scene BEV for a specific benchmark data layout."""

    benchmark: str = "scannet"

    def __init__(self, config: SceneBEVConfig = DEFAULT_SCENE_BEV_CONFIG) -> None:
        self.config = config

    @abstractmethod
    def resolve_paths(self, scene_id: str, data_root: Path) -> BEVScenePaths:
        """Return the mesh / trajectory / intrinsic paths for the scene."""

    def resolve_axis_align(
        self, scene_id: str, data_root: Path
    ) -> NDArray[np.float64] | None:
        """Return the 4x4 raw->aligned matrix for the scene, or ``None``.

        ScanNet meshes live in the raw scan frame while trajectory poses and
        ConceptGraph objects live in the axis-aligned frame; the BEV needs the
        mesh moved into the aligned frame so everything overlays.
        """
        for candidate in self._axis_align_candidates(scene_id, data_root):
            if not candidate.exists():
                continue
            for line in candidate.read_text().splitlines():
                stripped = line.strip()
                if not stripped.startswith("axisAlignment"):
                    continue
                values = stripped.split("=", 1)[1].split()
                if len(values) == 16:
                    return np.array(values, dtype=np.float64).reshape(4, 4)
        return None

    def _axis_align_candidates(self, scene_id: str, data_root: Path) -> list[Path]:
        return [
            data_root.parent / "scannet_aux" / scene_id / f"{scene_id}.txt",
            data_root / scene_id / "raw" / f"{scene_id}.txt",
            data_root / scene_id / f"{scene_id}.txt",
        ]

    def build(
        self,
        *,
        scene_id: str,
        data_root: Path,
        markers: Sequence[BEVMarker],
        output_path: Path,
        highlight_ids: frozenset[int] = frozenset(),
        use_cache: bool = True,
    ) -> Path:
        """Render (or load cached) the scene BEV and return ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(scene_id, data_root, markers, highlight_ids)
        if use_cache and cache_path.exists():
            if cache_path.resolve() != output_path.resolve():
                output_path.write_bytes(cache_path.read_bytes())
            logger.info(f"[bev] cache hit {scene_id}: {cache_path.name}")
            return output_path

        paths = self.resolve_paths(scene_id, data_root)
        for asset in (paths.mesh, paths.trajectory, paths.intrinsic):
            if not asset.exists():
                raise FileNotFoundError(f"BEV asset missing for {scene_id}: {asset}")

        mesh = load_ply_mesh(paths.mesh)
        axis_align = self.resolve_axis_align(scene_id, data_root)
        if axis_align is not None:
            mesh = mesh.transformed(axis_align)
        poses = _load_poses(paths.trajectory)
        intrinsic = _load_intrinsic(paths.intrinsic)

        rendered = render_scene_bev(
            mesh,
            poses,
            intrinsic,
            markers,
            highlight_ids=highlight_ids,
            config=self.config,
        )
        bgr = cv2.cvtColor(rendered.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), bgr)
        logger.success(f"[bev] rendered {scene_id} -> {output_path}")
        return output_path

    def _cache_path(
        self,
        scene_id: str,
        data_root: Path,
        markers: Sequence[BEVMarker],
        highlight_ids: frozenset[int],
    ) -> Path:
        digest = self._render_hash(markers, highlight_ids)
        return data_root / scene_id / "bev_cache" / f"keyframe_bev_{digest}.png"

    def _render_hash(
        self, markers: Sequence[BEVMarker], highlight_ids: frozenset[int]
    ) -> str:
        marker_sig = sorted(
            (m.obj_id, m.category, tuple(round(v, 4) for v in m.position))
            for m in markers
        )
        payload = "|".join(
            [
                self.benchmark,
                self.config.model_dump_json(),
                repr(marker_sig),
                repr(sorted(highlight_ids)),
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


class Nr3dSceneBEVBuilder(SceneBEVBuilder):
    """NR3D / ScanNet BEV builder (prepared ``<root>/<scene>/{raw,conceptgraph}``)."""

    benchmark = "nr3d"

    def resolve_paths(self, scene_id: str, data_root: Path) -> BEVScenePaths:
        scene_dir = data_root / scene_id
        trajectory = _resolve_trajectory(scene_dir)
        intrinsic = scene_dir / "raw" / "intrinsic_color.txt"
        if not intrinsic.exists():
            intrinsic = scene_dir / "conceptgraph" / "intrinsic_color.txt"
        mesh = _find_scannet_mesh(scene_id, data_root)
        return BEVScenePaths(mesh=mesh, trajectory=trajectory, intrinsic=intrinsic)


def _resolve_trajectory(scene_dir: Path) -> Path:
    for candidate in (
        scene_dir / "raw" / "traj.txt",
        scene_dir / "conceptgraph" / "traj.txt",
    ):
        if candidate.exists():
            return candidate
    return scene_dir / "raw" / "traj.txt"


def _scannet_mesh_roots(data_root: Path) -> list[Path]:
    roots: list[Path] = []
    env_root = scannet_data_root_override()
    if env_root is not None:
        roots.append(env_root)
    for default in (
        data_root.parent / "scannet_aux_meshes",
        data_root.parent.parent / "scannetv2",
        data_root.parent.parent / "ScanNet" / "scans",
    ):
        if default not in roots:
            roots.append(default)
    return roots


def _find_scannet_mesh(scene_id: str, data_root: Path) -> Path:
    searched: list[Path] = []
    for root in _scannet_mesh_roots(data_root):
        for filename in (f"{scene_id}_vh_clean_2.ply", f"{scene_id}_vh_clean.ply"):
            candidate = root / scene_id / filename
            searched.append(candidate)
            if candidate.exists():
                return candidate
    joined = ", ".join(str(p) for p in searched)
    raise FileNotFoundError(
        f"ScanNet mesh for {scene_id} not found (searched: {joined})"
    )


def _load_poses(trajectory_path: Path) -> NDArray[np.float64]:
    """Load camera-to-world poses as an ``(N, 4, 4)`` array.

    Accepts both one-line-per-pose (16 values) and four-lines-per-pose layouts.
    """
    raw = np.loadtxt(str(trajectory_path), dtype=np.float64)
    if raw.size % 16 != 0:
        raise ValueError(
            f"trajectory {trajectory_path} has {raw.size} values, not a multiple of 16"
        )
    return raw.reshape(-1, 4, 4)


def _load_intrinsic(intrinsic_path: Path) -> NDArray[np.float64]:
    matrix = np.loadtxt(str(intrinsic_path), dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix[:3, :3]
    if matrix.shape == (3, 3):
        return matrix
    raise ValueError(
        f"intrinsic {intrinsic_path} must be 3x3 or 4x4, got {matrix.shape}"
    )
