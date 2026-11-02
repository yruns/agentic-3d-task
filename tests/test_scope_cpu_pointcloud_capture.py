"""Validate fixed point-cloud snapshots without running a navigation pipeline."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TypedDict
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.scope import cpu_pointcloud_capture as capture_module
from scripts.scope.cpu_pointcloud_capture import (
    CaptureMetadata,
    DenoiseParameters,
    PointCloudInput,
    capture_denoise_inputs,
    load_capture,
)


@pytest.fixture
def upstream_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "upstream"
    source = root / "src/conceptgraph/slam/utils.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Source identity fixture; never executed.\n", encoding="utf-8")

    def git_head(command: Sequence[str], *, text: bool) -> str:
        assert tuple(command[:1]) == ("git",)
        assert text
        return "a" * 40 + "\n"

    monkeypatch.setattr(subprocess, "check_output", git_head)
    return root


def _parameters() -> DenoiseParameters:
    return DenoiseParameters(
        downsample_voxel_size=0.025,
        dbscan_remove_noise=True,
        dbscan_eps=0.1,
        dbscan_min_points=10,
        spatial_sim_type="overlap",
        device="cuda",
    )


def _input(object_id: int = 91) -> PointCloudInput:
    return PointCloudInput.from_arrays(
        object_id,
        np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]], dtype=np.float64),
        np.array([[0.2, 0.3, 0.4], [0.9, 0.8, 0.7]], dtype=np.float32),
    )


def test_default_disabled_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    capture_dir = tmp_path / "capture"
    assert (
        capture_denoise_inputs(
            (),
            _parameters(),
            capture_dir=capture_dir,
            upstream_root=tmp_path / "absent",
        )
        is None
    )
    assert not capture_dir.exists()


def test_capture_roundtrip_preserves_ids_arrays_parameters_and_hash(
    tmp_path: Path, upstream_root: Path
) -> None:
    capture_dir = tmp_path / "capture"
    inputs = (_input(91), _input(7))
    metadata = capture_denoise_inputs(
        inputs,
        _parameters(),
        capture_dir=capture_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    assert metadata is not None
    restored = load_capture(capture_dir)
    assert restored.metadata == metadata
    assert restored.metadata.parameters == _parameters()
    assert tuple(item.object_id for item in restored.inputs) == (91, 7)
    assert (
        metadata.archive_sha256
        == hashlib.sha256((capture_dir / "inputs.npz").read_bytes()).hexdigest()
    )
    for original, loaded in zip(inputs, restored.inputs):
        np.testing.assert_array_equal(original.points, loaded.points)
        np.testing.assert_array_equal(original.colors, loaded.colors)
        assert original.points.dtype == loaded.points.dtype
        assert original.colors.dtype == loaded.colors.dtype
        assert not loaded.points.flags.writeable
        assert not loaded.colors.flags.writeable
    assert {path.name for path in capture_dir.iterdir()} == {
        "inputs.npz",
        "capture.json",
    }
    with pytest.raises(FileExistsError):
        capture_denoise_inputs(
            inputs,
            _parameters(),
            capture_dir=capture_dir,
            upstream_root=upstream_root,
            enabled=True,
        )


def test_from_arrays_copies_original_without_mutating_it() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.zeros((2, 3), dtype=np.float64)
    captured = PointCloudInput.from_arrays(12, points, colors)
    assert points.flags.writeable and colors.flags.writeable
    points[0, 0] = 99.0
    colors[0, 0] = 0.75
    assert captured.points[0, 0] == 1.0
    assert captured.colors[0, 0] == 0.0


@pytest.mark.parametrize(
    "invalid_points",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 4)),
        np.array([[np.inf, 0.0, 0.0]]),
        np.array([[np.nan, 0.0, 0.0]]),
        np.zeros((2, 3), dtype=np.int64),
        np.array([["not numeric", "x", "y"]], dtype=object),
    ],
)
def test_rejects_invalid_point_arrays(invalid_points: object) -> None:
    with pytest.raises(ValueError):
        PointCloudInput.from_arrays(12, invalid_points, np.empty((0, 3)))


def test_rejects_boolean_id_and_mismatched_colors() -> None:
    with pytest.raises(ValueError):
        PointCloudInput.from_arrays(True, np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        PointCloudInput.from_arrays(12, np.zeros((2, 3)), np.zeros((1, 3)))


def test_tampered_archive_and_file_size_are_rejected(
    tmp_path: Path, upstream_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "capture"
    capture_denoise_inputs(
        (_input(),),
        _parameters(),
        capture_dir=capture_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    archive = capture_dir / "inputs.npz"
    original = archive.read_bytes()
    archive.write_bytes(original + b"tampered")
    with pytest.raises(ValueError):
        load_capture(capture_dir)
    archive.write_bytes(original)
    monkeypatch.setattr(capture_module, "MAX_CAPTURE_BYTES", 16)
    with pytest.raises(ValueError):
        load_capture(capture_dir)


def test_pickle_object_array_is_not_deserialized(
    tmp_path: Path, upstream_root: Path
) -> None:
    capture_dir = tmp_path / "capture"
    capture_denoise_inputs(
        (_input(),),
        _parameters(),
        capture_dir=capture_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    archive = capture_dir / "inputs.npz"
    np.savez(
        archive,
        points_0=np.array([["x", "y", "z"]], dtype=object),
        colors_0=np.zeros((1, 3)),
    )
    metadata_path = capture_dir / "capture.json"
    metadata = CaptureMetadata.model_validate_json(metadata_path.read_bytes())
    changed = metadata.model_copy(
        update={
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_bytes": archive.stat().st_size,
        }
    )
    metadata_path.write_text(changed.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError):
        load_capture(capture_dir)


def test_output_comparison_sorts_joint_coordinates_and_colors() -> None:
    original = _input()
    reordered = PointCloudInput.from_arrays(
        original.object_id, original.points[::-1], original.colors[::-1]
    )
    comparisons = capture_module.compare_pointcloud_outputs((original,), (reordered,))
    assert comparisons[0].exact_match
    mismatched = PointCloudInput.from_arrays(
        original.object_id, original.points, original.colors[::-1]
    )
    comparisons = capture_module.compare_pointcloud_outputs((original,), (mismatched,))
    assert not comparisons[0].exact_match
    assert comparisons[0].max_absolute_difference is not None


def test_output_comparison_rejects_changed_object_id() -> None:
    with pytest.raises(ValueError):
        capture_module.compare_pointcloud_outputs((_input(91),), (_input(7),))


class _NpyHeader(TypedDict):
    descr: str
    fortran_order: bool
    shape: tuple[int, int]


def test_large_declared_shape_with_short_payload_rejected_before_np_load(
    tmp_path: Path, upstream_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "capture"
    capture_denoise_inputs(
        (_input(),),
        _parameters(),
        capture_dir=capture_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    archive = capture_dir / "inputs.npz"
    huge_count = 10**12
    with ZipFile(archive, "w") as zipped:
        for name, rows, dtype in (
            ("points_0.npy", huge_count, "<f8"),
            ("colors_0.npy", 0, "<f4"),
        ):
            header = _NpyHeader(descr=dtype, fortran_order=False, shape=(rows, 3))
            with zipped.open(name, "w") as member:
                np.lib.format.write_array_header_1_0(member, dict(header))
    metadata_path = capture_dir / "capture.json"
    metadata = CaptureMetadata.model_validate_json(metadata_path.read_bytes())
    descriptor = metadata.objects[0].model_copy(
        update={"point_count": huge_count, "color_count": 0}
    )
    changed = metadata.model_copy(
        update={
            "objects": (descriptor,),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_bytes": archive.stat().st_size,
        }
    )
    metadata_path.write_text(changed.model_dump_json(), encoding="utf-8")

    def unexpected_np_load(path: Path, *, allow_pickle: bool) -> NoReturn:
        raise AssertionError("np.load must not run before bounded header validation")

    monkeypatch.setattr(np, "load", unexpected_np_load)
    with pytest.raises(ValueError):
        load_capture(capture_dir)
