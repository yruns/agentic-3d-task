"""Tests for the sidecar-backed per-frame proposer (Molmo->SAM->lift)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.pipeline_backends import _select_prompt_point
from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPoint


def _write_rgb(path: Path, size: tuple[int, int] = (40, 40)) -> None:
    Image.new("RGB", size, (0, 0, 0)).save(path)


def test_points_from_molmo_response_returns_empty_on_no_points() -> None:
    from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
    from codex_agent.scenefunc3d.tools.molmo_pointing import (
        points_from_molmo_response,
    )

    response = MolmoPointResponse(
        request_id="r",
        model_name="molmo",
        raw_text="no tags",
        image_points=(),
        latency_ms=0.0,
    )
    assert points_from_molmo_response(response, image_width=40, image_height=40) == ()


def test_sidecar_proposer_picks_nearest_molmo_point_and_smallest_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d import pipeline_backends
    from codex_agent.scenefunc3d.servers.schemas import (
        MolmoImagePoint,
        MolmoPointResponse,
    )
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult
    from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult

    scene_root = tmp_path / "scene"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    _write_rgb(raw / "000000-rgb.jpg")
    (raw / "000000-depth.png").write_bytes(b"")  # not read in this fake path
    (raw / "000000-intrinsic.txt").write_text("20 0 20\n0 20 20\n0 0 1\n")
    (raw / "000000.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg", "depth": "000000-depth.png",'
        ' "intrinsic": "000000-intrinsic.txt", "pose": "000000.txt"}]'
    )
    (raw / "mesh.ply").write_bytes(b"")  # raw_mesh_path passed through, not read here
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    scene = SceneFunc3dToolScene.load(scene_root)

    # Molmo returns two points; the one nearest (20,20) must be chosen.
    def _fake_request_molmo_point(settings, *, request_id, args):
        return MolmoPointResponse(
            request_id=request_id,
            model_name="molmo",
            raw_text="",
            image_points=(
                MolmoImagePoint(x_px=21.0, y_px=20.0, source="molmo", label="a"),
                MolmoImagePoint(x_px=39.0, y_px=39.0, source="molmo", label="b"),
            ),
            latency_ms=0.0,
        )

    captured_points: list[tuple[float, float]] = []

    def _fake_sam_mask(args, *, out_dir, backend_config_path):
        captured_points.append((args.points[0].x_px, args.points[0].y_px))
        big = tmp_path / "big.npz"
        small = tmp_path / "small.npz"
        big.write_bytes(b"")
        small.write_bytes(b"")
        return SamMaskResult(
            frame_id="000000",
            candidates=(
                SamCandidate("cbig", 0.9, 500, 10.0, big, big),
                SamCandidate("csmall", 0.8, 50, 1.0, small, small),
            ),
            contact_sheet_path=tmp_path / "sheet.jpg",
        )

    def _fake_lift(args, *, out_dir, raw_mesh_path):
        assert args.candidate_id == "csmall"
        assert Path(args.mask_path).name == "small.npz"
        npz = tmp_path / "fragments_000000_csmall.npz"
        np.savez_compressed(
            npz,
            points_world=np.array([[0.0, 0.0, 2.0]]),
            point_indices=np.array([3, 4, 5], dtype=np.int64),
        )
        return LiftMaskResult(
            frame_id="000000",
            candidate_id=args.candidate_id,
            lifted_point_count=3,
            mask_npz_path=npz,
            mask_ply_path=tmp_path / "x.ply",
            overlay_path=tmp_path / "x.txt",
        )

    monkeypatch.setattr(
        pipeline_backends, "request_molmo_point", _fake_request_molmo_point
    )
    monkeypatch.setattr(pipeline_backends, "sam_mask", _fake_sam_mask)
    monkeypatch.setattr(pipeline_backends, "lift_mask_to_3d", _fake_lift)
    monkeypatch.setattr(
        pipeline_backends, "load_backend_settings", lambda path: object()
    )

    proposer = pipeline_backends.SidecarFrameProposer(
        scene=scene,
        out_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backend.toml",
        affordance_concept="the handle",
        task_description="open the drawer",
    )
    geometry = CameraGeometry(
        intrinsics=np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        camera_to_world=np.eye(4),
    )
    proposal = proposer.propose(
        frame_id="000000", geometry=geometry, projected_anchor_xy=(20.0, 20.0)
    )
    assert proposal.point_indices == (3, 4, 5)
    assert proposal.molmo_fallback_used is False
    # SAM must have been prompted with the nearest Molmo point (21, 20).
    assert captured_points[0] == pytest.approx((21.0, 20.0))


def test_sidecar_proposer_falls_back_to_projected_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d import pipeline_backends
    from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult
    from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult

    scene_root = tmp_path / "scene"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    _write_rgb(raw / "000000-rgb.jpg")
    (raw / "000000-depth.png").write_bytes(b"")
    (raw / "000000-intrinsic.txt").write_text("20 0 20\n0 20 20\n0 0 1\n")
    (raw / "000000.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg", "depth": "000000-depth.png",'
        ' "intrinsic": "000000-intrinsic.txt", "pose": "000000.txt"}]'
    )
    (raw / "mesh.ply").write_bytes(b"")
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    scene = SceneFunc3dToolScene.load(scene_root)

    captured_points: list[tuple[float, float]] = []

    monkeypatch.setattr(
        pipeline_backends,
        "request_molmo_point",
        lambda settings, *, request_id, args: MolmoPointResponse(
            request_id=request_id,
            model_name="molmo",
            raw_text="",
            image_points=(),
            latency_ms=0.0,
        ),
    )

    def _fake_sam_mask(args, *, out_dir, backend_config_path):
        captured_points.append((args.points[0].x_px, args.points[0].y_px))
        npz = tmp_path / "m.npz"
        npz.write_bytes(b"")
        return SamMaskResult(
            frame_id="000000",
            candidates=(SamCandidate("c0", 0.8, 40, 1.0, npz, npz),),
            contact_sheet_path=tmp_path / "sheet.jpg",
        )

    def _fake_lift(args, *, out_dir, raw_mesh_path):
        npz = tmp_path / "frag.npz"
        np.savez_compressed(npz, point_indices=np.array([1], dtype=np.int64))
        return LiftMaskResult(
            frame_id="000000",
            candidate_id=args.candidate_id,
            lifted_point_count=1,
            mask_npz_path=npz,
            mask_ply_path=tmp_path / "x.ply",
            overlay_path=tmp_path / "x.txt",
        )

    monkeypatch.setattr(pipeline_backends, "sam_mask", _fake_sam_mask)
    monkeypatch.setattr(pipeline_backends, "lift_mask_to_3d", _fake_lift)
    monkeypatch.setattr(
        pipeline_backends, "load_backend_settings", lambda path: object()
    )

    proposer = pipeline_backends.SidecarFrameProposer(
        scene=scene,
        out_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backend.toml",
        affordance_concept="the handle",
        task_description="open the drawer",
    )
    geometry = CameraGeometry(
        intrinsics=np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        camera_to_world=np.eye(4),
    )
    proposal = proposer.propose(
        frame_id="000000", geometry=geometry, projected_anchor_xy=(15.0, 16.0)
    )
    assert proposal.molmo_fallback_used is True
    assert captured_points[0] == pytest.approx((15.0, 16.0))
    assert proposal.point_indices == (1,)


def test_select_prompt_point_uses_nearest_when_within_threshold() -> None:
    points = (
        MolmoPoint(x_px=21.0, y_px=20.0, source="molmo", label="a"),
        MolmoPoint(x_px=39.0, y_px=39.0, source="molmo", label="b"),
    )
    xy, fallback = _select_prompt_point(
        points, projected_anchor_xy=(20.0, 20.0), image_width=40, image_height=40
    )
    assert xy == (21.0, 20.0)
    assert fallback is False


def test_select_prompt_point_falls_back_when_all_molmo_too_far() -> None:
    # Both points are far from the projected anchor (0,0); diagonal of a 40x40
    # image is ~56.6, threshold 0.15*diag ~= 8.5, so a point at (39,39) is too far.
    points = (MolmoPoint(x_px=39.0, y_px=39.0, source="molmo", label="a"),)
    xy, fallback = _select_prompt_point(
        points, projected_anchor_xy=(0.0, 0.0), image_width=40, image_height=40
    )
    assert xy == (0.0, 0.0)
    assert fallback is True


def test_select_prompt_point_none_anchor_uses_first_molmo_point() -> None:
    points = (
        MolmoPoint(x_px=5.0, y_px=6.0, source="molmo", label="a"),
        MolmoPoint(x_px=7.0, y_px=8.0, source="molmo", label="b"),
    )
    xy, fallback = _select_prompt_point(
        points, projected_anchor_xy=None, image_width=40, image_height=40
    )
    assert xy == (5.0, 6.0)
    assert fallback is False


def test_select_prompt_point_none_anchor_no_points_returns_none_no_fallback() -> None:
    xy, fallback = _select_prompt_point(
        (), projected_anchor_xy=None, image_width=40, image_height=40
    )
    assert xy is None
    assert fallback is False  # nothing available is NOT an anchor fallback
