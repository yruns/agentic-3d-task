"""Tests for the offline SceneFunc3D fusion re-scoring harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.evaluation.offline_fusion import (
    SampleFusionInput,
    _load_fragment_frames,
    _sample_ids_from_run_root,
    fuse_and_score_sample,
)


def _write_fragment(
    fragments_dir: Path, name: str, point_indices: tuple[int, ...]
) -> Path:
    fragment_dir = fragments_dir / name
    fragment_dir.mkdir(parents=True)
    npz_path = fragment_dir / "mask_data.npz"
    np.savez(npz_path, point_indices=np.asarray(point_indices, dtype=np.int64))
    return npz_path


def _line_scene(n: int = 40) -> np.ndarray:
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 0.39, n)  # 1 cm spacing
    return coords


def test_fuse_and_score_recovers_precision_over_union() -> None:
    scene = _line_scene()
    gt_ids = frozenset(range(0, 3))  # tight target well inside the anchor radius
    sample = SampleFusionInput(
        sample_id="420673::demo",
        motion_type="pinch_pull",
        frames_point_indices=(
            (0, 1, 2),
            (0, 1, 2, 30, 31, 32),
        ),
        anchor_point_indices=(0, 1, 2),
        scene_vertices=scene,
        gt_ids=gt_ids,
    )
    params = FusionParams(
        agreement_tau=0.5, cluster_link_eps_m=0.03, min_cluster_points=2
    )
    score = fuse_and_score_sample(sample, params)
    assert score.metrics.precision == 1.0  # spurious tail removed
    assert score.metrics.iou == 1.0


def test_load_fragment_frames_reads_fragments_sorted_by_name(tmp_path: Path) -> None:
    fragments_dir = tmp_path / "fragments"
    _write_fragment(fragments_dir, "b_frag", (3, 4))
    _write_fragment(fragments_dir, "a_frag", (0, 1, 2))
    frames = _load_fragment_frames(fragments_dir)
    assert frames == ((0, 1, 2), (3, 4))


def test_load_fragment_frames_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(SceneFunc3dDataError, match="fragments directory is missing"):
        _load_fragment_frames(tmp_path / "does_not_exist")


def test_load_fragment_frames_empty_dir_fails_closed(tmp_path: Path) -> None:
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    with pytest.raises(SceneFunc3dDataError, match="contains no"):
        _load_fragment_frames(fragments_dir)


def test_load_fragment_frames_missing_point_indices_key_raises(tmp_path: Path) -> None:
    fragment_dir = tmp_path / "fragments" / "frag"
    fragment_dir.mkdir(parents=True)
    np.savez(fragment_dir / "mask_data.npz", points_world=np.zeros((2, 3)))
    with pytest.raises(SceneFunc3dDataError, match="point_indices"):
        _load_fragment_frames(tmp_path / "fragments")


def test_sample_ids_from_run_root_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(SceneFunc3dDataError, match="run root is missing"):
        _sample_ids_from_run_root(tmp_path / "missing")


def test_sample_ids_from_run_root_empty_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SceneFunc3dDataError, match="no fragment samples"):
        _sample_ids_from_run_root(tmp_path)
