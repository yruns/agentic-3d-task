"""Tests for SceneFunc3D Molmo and SAM tool contracts."""

from __future__ import annotations

import json
import struct
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.__main__ import main
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import (
    MolmoBackendConfig,
    MolmoPointArgs,
    parse_molmo_points,
    run_molmo_backend,
)
from codex_agent.scenefunc3d.tools.sam_masking import (
    SamBackendConfig,
    SamCandidate,
    SamMaskArgs,
    SamMaskResult,
    run_sam_backend,
    sam_mask,
)


def test_parse_molmo_percent_point() -> None:
    points = parse_molmo_points(
        '<point x="81.0" y="61.9" alt="drawer knob">drawer knob</point>',
        image_width=1440,
        image_height=1920,
    )
    assert len(points) == 1
    assert points[0].x_px == 1166.4
    assert points[0].y_px == 1188.48
    assert points[0].label == "drawer knob"


def test_molmo_backend_missing_path_fails(tmp_path: Path) -> None:
    config = MolmoBackendConfig(
        model_name="MolmoPoint-8B", model_path=tmp_path / "missing"
    )

    with pytest.raises(ToolInputError, match="Molmo backend unavailable"):
        run_molmo_backend(config)


def test_molmo_backend_existing_path_is_not_fake_success(tmp_path: Path) -> None:
    model_path = tmp_path / "molmo-model"
    model_path.mkdir()
    config = MolmoBackendConfig(model_name="MolmoPoint-8B", model_path=model_path)

    with pytest.raises(
        ToolInputError, match="Molmo backend execution is not configured"
    ):
        run_molmo_backend(config)


def test_sam_backend_missing_path_fails(tmp_path: Path) -> None:
    config = SamBackendConfig(
        model_name="SAM2.1-Hiera-L", checkpoint_path=tmp_path / "missing.pt"
    )

    with pytest.raises(ToolInputError, match="SAM backend unavailable"):
        run_sam_backend(config)


def test_sam_backend_existing_path_is_not_fake_success(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sam.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    config = SamBackendConfig(
        model_name="SAM2.1-Hiera-L", checkpoint_path=checkpoint_path
    )

    with pytest.raises(ToolInputError, match="SAM backend execution is not configured"):
        run_sam_backend(config)


def test_molmo_point_args_rejects_missing_image_path(tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "frame_id": "000050",
        "image_path": str(tmp_path / "missing.jpg"),
        "prompt": "drawer handle",
        "image_width": 640,
        "image_height": 480,
    }

    with pytest.raises(ValidationError):
        MolmoPointArgs.model_validate(payload)


def test_molmo_point_args_derive_frame_id_from_view_crop_image_path(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "000056_crop_normalized_b7296fba91db.jpg"
    image_path.write_bytes(b"image")
    args = MolmoPointArgs.model_validate(
        {
            "image_path": str(image_path),
            "prompt": "drawer handle",
            "image_width": 640,
            "image_height": 480,
        }
    )

    assert args.frame_id == "000056"


def test_sam_mask_args_rejects_missing_image_path(tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "frame_id": "000050",
        "image_path": str(tmp_path / "missing.jpg"),
        "points": (
            {
                "x_px": 10.0,
                "y_px": 20.0,
                "source": '<point x="10" y="20">open</point>',
                "label": "open",
            },
        ),
    }

    with pytest.raises(ValidationError):
        SamMaskArgs.model_validate(payload)


def test_sam_mask_args_accept_single_point_alias_and_redundant_image_size(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "000056.jpg"
    image_path.write_bytes(b"image")
    args = SamMaskArgs.model_validate(
        {
            "frame_id": "000056",
            "image_path": str(image_path),
            "image_width": 1440,
            "image_height": 1920,
            "point": {
                "x_px": 613.18,
                "y_px": 1075.68,
                "label": "bottom drawer handle",
            },
        }
    )

    assert len(args.points) == 1
    assert args.points[0].x_px == 613.18
    assert args.points[0].label == "bottom drawer handle"


def test_cli_molmo_point_returns_recoverable_backend_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene(tmp_path)
    args_payload: dict[str, object] = {
        "frame_id": "000000",
        "image_path": str(image_path),
        "prompt": "drawer handle",
        "image_width": 640,
        "image_height": 480,
    }

    code = main(
        [
            "molmo_point",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(args_payload),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "molmo_point backend config is required" in payload["error"]


def test_cli_molmo_point_uses_configured_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["points"][0]["label"] == "handle"
    assert Path(payload["raw_text_path"]).exists()
    assert Path(payload["overlay_path"]).exists()


def test_cli_molmo_point_uses_sidecar_image_points_for_token_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    raw_text = (
        '<points coords="<POINT_1429><POINT_2759><POINT_2765>1<POINT_2758>">'
        "small dark round knob handle</point><|im_end|>"
    )
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": raw_text,
                "image_points": (
                    {
                        "x_px": 719.0,
                        "y_px": 930.0,
                        "source": raw_text,
                        "label": "small dark round knob handle",
                    },
                ),
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 1440,
                        "image_height": 1920,
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["points"] == [
        {
            "x_px": 719.0,
            "y_px": 930.0,
            "source": raw_text,
            "label": "small dark round knob handle",
        }
    ]
    assert Path(payload["raw_text_path"]).read_text(encoding="utf-8") == raw_text
    assert Path(payload["overlay_path"]).exists()


def test_cli_molmo_point_rejects_out_of_bounds_sidecar_image_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    raw_text = (
        '<points coords="<POINT_1429><POINT_2759>">'
        "small dark round knob handle</point><|im_end|>"
    )
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": raw_text,
                "image_points": (
                    {
                        "x_px": 100.0,
                        "y_px": 10.0,
                        "source": raw_text,
                        "label": "small dark round knob handle",
                    },
                ),
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "outside image bounds" in payload["error"]


def test_cli_molmo_point_rejects_raw_output_without_point_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": "I see a drawer handle, but no structured point.",
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "no <point> tags" in payload["error"]
    assert "raw_text_path" in payload["error"]


def test_cli_molmo_point_rejects_unsafe_frame_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    args_payload: dict[str, object] = {
        "frame_id": "../../escape",
        "image_path": str(image_path),
        "prompt": "drawer handle",
        "image_width": 100,
        "image_height": 80,
    }

    code = main(
        [
            "molmo_point",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(args_payload),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "frame_id" in payload["error"]


def test_cli_molmo_point_rejects_out_dir_outside_allowed_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                "--out-dir",
                str(tmp_path / "not-allowed-output"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "outside configured roots" in payload["error"]


def test_cli_sam_mask_returns_recoverable_backend_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene(tmp_path)
    args_payload: dict[str, object] = {
        "frame_id": "000000",
        "image_path": str(image_path),
        "points": [
            {
                "x_px": 10.0,
                "y_px": 20.0,
                "source": '<point x="10" y="20">drawer</point>',
                "label": "drawer",
            }
        ],
    }

    code = main(
        [
            "sam_mask",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(args_payload),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "sam_mask backend config is required" in payload["error"]


def test_cli_sam_mask_uses_configured_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    output_root = tmp_path / "out"
    mask_path = output_root / "server_masks" / "mask_00.npz"
    mask_path.parent.mkdir(parents=True)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:30, 10:30] = 1
    np.savez(mask_path, mask=mask)
    server = _start_json_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 2,
                        "coverage_percent": 50.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url=f"http://127.0.0.1:{server.server_port}",
    )
    try:
        code = main(
            [
                "sam_mask",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "points": [
                            {
                                "x_px": 10.0,
                                "y_px": 20.0,
                                "source": '<point x="10" y="20">drawer</point>',
                                "label": "drawer",
                            }
                        ],
                    }
                ),
                "--out-dir",
                str(output_root),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["candidates"][0]["candidate_id"] == "mask_00"
    assert Path(payload["candidates"][0]["mask_npz_path"]).exists()
    assert Path(payload["contact_sheet_path"]).exists()


def test_cli_sam_mask_contact_sheet_shows_all_candidate_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir, image_path = _write_cli_scene_with_real_image(tmp_path)
    output_root = tmp_path / "out"
    mask_dir = output_root / "server_masks"
    mask_dir.mkdir(parents=True)
    first_mask_path = mask_dir / "mask_00.npz"
    second_mask_path = mask_dir / "mask_01.npz"
    first_mask = np.zeros((80, 100), dtype=np.uint8)
    first_mask[20:30, 10:30] = 1
    second_mask = np.zeros((80, 100), dtype=np.uint8)
    second_mask[40:50, 60:80] = 1
    np.savez(first_mask_path, mask=first_mask)
    np.savez(second_mask_path, mask=second_mask)
    server = _start_json_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(first_mask_path),
                        "pixel_count": 200,
                        "coverage_percent": 2.0,
                    },
                    {
                        "candidate_id": "mask_01",
                        "score": 0.72,
                        "mask_npz_path": str(second_mask_path),
                        "pixel_count": 200,
                        "coverage_percent": 2.0,
                    },
                ],
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url=f"http://127.0.0.1:{server.server_port}",
    )
    try:
        code = main(
            [
                "sam_mask",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "points": [
                            {
                                "x_px": 10.0,
                                "y_px": 20.0,
                                "source": '<point x="10" y="20">drawer</point>',
                                "label": "drawer",
                            }
                        ],
                    }
                ),
                "--out-dir",
                str(output_root),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert [candidate["candidate_id"] for candidate in payload["candidates"]] == [
        "mask_00",
        "mask_01",
    ]
    with Image.open(payload["contact_sheet_path"]) as contact_sheet:
        assert contact_sheet.size[0] == 200
        assert contact_sheet.size[1] > 80


def test_sam_mask_rejects_npz_missing_mask_key(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    _, image_path = _write_cli_scene_with_real_image(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    mask_path = output_root / "mask_00.npz"
    np.savez(mask_path, not_mask=np.array([[0, 1], [1, 0]], dtype=np.uint8))
    server = _start_json_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 2,
                        "coverage_percent": 50.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url=f"http://127.0.0.1:{server.server_port}",
    )
    args = SamMaskArgs.model_validate(
        {
            "frame_id": "000000",
            "image_path": image_path,
            "points": [
                {
                    "x_px": 10.0,
                    "y_px": 20.0,
                    "source": '<point x="10" y="20">drawer</point>',
                    "label": "drawer",
                },
            ],
        }
    )
    try:
        with pytest.raises(ToolInputError, match="missing required key 'mask'"):
            sam_mask(args, out_dir=output_root, backend_config_path=config_path)
    finally:
        server.shutdown()
        server.server_close()


def test_sam_mask_rejects_mask_shape_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    _, image_path = _write_cli_scene_with_real_image(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    mask_path = output_root / "mask_00.npz"
    np.savez(mask_path, mask=np.array([[0, 1], [1, 0]], dtype=np.uint8))
    server = _start_json_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 2,
                        "coverage_percent": 50.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url=f"http://127.0.0.1:{server.server_port}",
    )
    args = SamMaskArgs.model_validate(
        {
            "frame_id": "000000",
            "image_path": image_path,
            "points": [
                {
                    "x_px": 10.0,
                    "y_px": 20.0,
                    "source": '<point x="10" y="20">drawer</point>',
                    "label": "drawer",
                },
            ],
        }
    )
    try:
        with pytest.raises(ToolInputError, match="shape"):
            sam_mask(args, out_dir=output_root, backend_config_path=config_path)
    finally:
        server.shutdown()
        server.server_close()


def test_cli_lift_mask_writes_deterministic_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir, _ = _write_cli_scene_with_real_image(tmp_path)
    _write_binary_scene_mesh(
        scene_dir / "conceptgraph" / "mesh.ply",
        points=((10.0, 0.0, 0.0), (1.0, 0.0, 2.0)),
    )
    mask_path = tmp_path / "mask.npz"
    depth_path = tmp_path / "depth.png"
    intrinsics_path = tmp_path / "intrinsics.txt"
    pose_path = tmp_path / "pose.txt"
    np.savez(mask_path, mask=np.array([[0, 1], [0, 0]], dtype=np.uint8))
    Image.fromarray(np.array([[0, 2000], [0, 0]], dtype=np.uint16)).save(depth_path)
    intrinsics_path.write_text("2 0 0\n0 2 0\n0 0 1\n", encoding="utf-8")
    pose_path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
    args_payload: dict[str, object] = {
        "frame_id": "000000",
        "candidate_id": "mask_00",
        "mask_path": str(mask_path),
        "depth_path": str(depth_path),
        "intrinsics_path": str(intrinsics_path),
        "pose_path": str(pose_path),
    }

    code = main(
        [
            "lift_mask_to_3d",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(args_payload),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["frame_id"] == "000000"
    assert payload["candidate_id"] == "mask_00"
    assert payload["lifted_point_count"] == 1
    assert Path(payload["mask_npz_path"]).exists()
    with np.load(Path(payload["mask_npz_path"])) as archive:
        np.testing.assert_array_equal(
            archive["point_indices"], np.array([1], dtype=np.int64)
        )
    assert Path(payload["mask_ply_path"]).read_text(encoding="ascii").startswith("ply")
    assert (
        Path(payload["overlay_path"])
        .read_text(encoding="utf-8")
        .startswith("frame_id=000000\n")
    )


def test_parse_molmo_points_preserves_tag_order() -> None:
    points = parse_molmo_points(
        (
            '<point x="10" y="20" alt="first">first</point>'
            '<point x="30" y="40" alt="second">second</point>'
        ),
        image_width=200,
        image_height=100,
    )

    assert tuple(point.label for point in points) == ("first", "second")
    assert tuple(point.x_px for point in points) == (20.0, 60.0)
    assert tuple(point.y_px for point in points) == (20.0, 40.0)


def test_parse_molmo_points_returns_empty_tuple_for_no_tags() -> None:
    assert (
        parse_molmo_points("no point tags here", image_width=200, image_height=100)
        == ()
    )


def test_parse_molmo_points_rejects_unmatched_point_fragment() -> None:
    with pytest.raises(SceneFunc3dDataError, match="malformed Molmo point tag"):
        parse_molmo_points(
            '<point x="10" y="20" alt="open">',
            image_width=200,
            image_height=100,
        )


@pytest.mark.parametrize(
    ("raw_text", "expected_message"),
    [
        ('<point y="20" alt="open">open</point>', "missing Molmo point x"),
        ('<point x="10" alt="open">open</point>', "missing Molmo point y"),
    ],
)
def test_parse_molmo_points_rejects_missing_coordinates(
    raw_text: str, expected_message: str
) -> None:
    with pytest.raises(SceneFunc3dDataError, match=expected_message):
        parse_molmo_points(raw_text, image_width=200, image_height=100)


def test_parse_molmo_points_rejects_malformed_percent_value() -> None:
    with pytest.raises(SceneFunc3dDataError, match="invalid Molmo point x percent"):
        parse_molmo_points(
            '<point x="not-a-number" y="40" alt="bad">bad</point>',
            image_width=200,
            image_height=100,
        )


@pytest.mark.parametrize(
    ("raw_text", "expected_message"),
    [
        ('<point x="101" y="40" alt="bad">bad</point>', "invalid Molmo point x"),
        ('<point x="10" y="-1" alt="bad">bad</point>', "invalid Molmo point y"),
    ],
)
def test_parse_molmo_points_rejects_out_of_range_coordinates(
    raw_text: str, expected_message: str
) -> None:
    with pytest.raises(SceneFunc3dDataError, match=expected_message):
        parse_molmo_points(raw_text, image_width=200, image_height=100)


def test_parse_molmo_points_rejects_valid_and_malformed_mixed_output() -> None:
    with pytest.raises(SceneFunc3dDataError, match="malformed Molmo point tag"):
        parse_molmo_points(
            (
                '<point x="10" y="20" alt="first">first</point>'
                '<point x="30" y="40" alt="second">'
            ),
            image_width=200,
            image_height=100,
        )


@pytest.mark.parametrize(
    ("image_width", "image_height", "expected_message"),
    [
        (0, 100, "image_width must be positive"),
        (200, 0, "image_height must be positive"),
    ],
)
def test_parse_molmo_points_rejects_invalid_image_dimensions(
    image_width: int, image_height: int, expected_message: str
) -> None:
    with pytest.raises(SceneFunc3dDataError, match=expected_message):
        parse_molmo_points(
            '<point x="10" y="20" alt="open">open</point>',
            image_width=image_width,
            image_height=image_height,
        )


@pytest.mark.parametrize("bad_x_px", [True, "10"])
def test_sam_mask_args_rejects_coerced_point_x_values(
    tmp_path: Path, bad_x_px: object
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"frame")
    payload: dict[str, object] = {
        "frame_id": "000050",
        "image_path": str(image_path),
        "points": (
            {
                "x_px": bad_x_px,
                "y_px": 20.0,
                "source": '<point x="10" y="20">open</point>',
                "label": "open",
            },
        ),
    }

    with pytest.raises(ValidationError):
        SamMaskArgs.model_validate(payload)


def test_sam_result_payload() -> None:
    result = SamMaskResult(
        frame_id="000050",
        candidates=(
            SamCandidate(
                candidate_id="mask_00",
                score=0.82,
                pixel_count=1119,
                coverage_percent=0.0405,
                mask_npz_path=Path("/tmp/mask_00.npz"),
                overlay_path=Path("/tmp/mask_00.jpg"),
            ),
        ),
        contact_sheet_path=Path("/tmp/sam_candidates.jpg"),
    )
    payload = result.to_payload()
    candidates = cast(list[dict[str, object]], payload["candidates"])
    assert candidates[0]["candidate_id"] == "mask_00"
    assert candidates[0]["mask_npz_path"] == "/tmp/mask_00.npz"


def _write_cli_scene(root: Path) -> tuple[Path, Path]:
    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    image_path = raw_dir / "000000-rgb.png"
    image_path.write_bytes(b"not-a-real-image")
    return scene_dir, image_path


def _write_cli_scene_with_real_image(root: Path) -> tuple[Path, Path]:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    conceptgraph_dir = scene_dir / "conceptgraph"
    raw_dir.mkdir(parents=True)
    conceptgraph_dir.mkdir()
    image_path = raw_dir / "000000-rgb.png"
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(image_path)
    return scene_dir, image_path


def _write_binary_scene_mesh(
    path: Path, *, points: tuple[tuple[float, float, float], ...]
) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for x_value, y_value, z_value in points:
            handle.write(
                struct.pack(
                    "<fffBBB",
                    x_value,
                    y_value,
                    z_value,
                    0,
                    0,
                    0,
                )
            )
    return path


def _write_backend_config(tmp_path: Path, *, molmo_url: str, sam_url: str) -> Path:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    output_root = tmp_path / "out"
    output_root.mkdir(exist_ok=True)
    config_path.write_text(
        f"""
molmo_url = "{molmo_url}"
sam_url = "{sam_url}"
request_timeout_seconds = 2.0
artifact_staging_root = "{output_root}"
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{output_root}"]
""",
        encoding="utf-8",
    )
    return config_path


def _start_json_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
