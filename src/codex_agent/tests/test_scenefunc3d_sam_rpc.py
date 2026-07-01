"""Tests for SceneFunc3D SAM RPC backend integration."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.config import SceneFunc3dBackendSettings
from codex_agent.scenefunc3d.backends.sam_rpc import request_sam_masks
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.sam_masking import SamMaskArgs


def test_request_sam_masks_returns_candidate_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    server = _start_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 12,
                        "coverage_percent": 15.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server.server_port}")
    args = _args(image_path)
    try:
        response = request_sam_masks(
            settings,
            request_id="req-1",
            args=args,
            staging_dir=staging_dir,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.candidates[0].candidate_id == "mask_00"
    assert response.candidates[0].mask_npz_path == mask_path


def test_request_sam_masks_supports_base_path(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    server = _start_server(
        {
            "/s/export-token/sam/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 12,
                        "coverage_percent": 15.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(
        tmp_path,
        sam_url=f"http://127.0.0.1:{server.server_port}/s/export-token/sam",
    )
    try:
        response = request_sam_masks(
            settings,
            request_id="req-path",
            args=_args(image_path),
            staging_dir=staging_dir,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.request_id == "req-path"
    assert response.candidates[0].candidate_id == "mask_00"


def test_request_sam_masks_server_down_is_recoverable(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server({})
    server_port = server.server_port
    server.shutdown()
    server.server_close()
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server_port}")
    args = _args(image_path)

    with pytest.raises(ToolInputError, match="sidecar HTTP request failed"):
        request_sam_masks(
            settings,
            request_id="req-2",
            args=args,
            staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
        )


def test_request_sam_masks_rejects_request_id_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    server = _start_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": "other-req",
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 12,
                        "coverage_percent": 15.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server.server_port}")
    args = _args(image_path)
    try:
        with pytest.raises(ToolInputError, match="request_id mismatch"):
            request_sam_masks(
                settings,
                request_id="req-1",
                args=args,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
    finally:
        server.shutdown()
        server.server_close()


def test_request_sam_masks_rejects_empty_candidate_response(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [],
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server.server_port}")
    try:
        with pytest.raises(ToolInputError, match="no mask candidates"):
            request_sam_masks(
                settings,
                request_id="req-empty",
                args=_args(image_path),
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
    finally:
        server.shutdown()
        server.server_close()


def test_request_sam_masks_rejects_image_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    settings = SceneFunc3dBackendSettings(
        molmo_url="http://127.0.0.1:8711",
        sam_url="http://127.0.0.1:8712",
        request_timeout_seconds=2.0,
        artifact_staging_root=tmp_path,
        allowed_image_roots=(tmp_path / "allowed-images",),
        allowed_output_roots=(tmp_path / "out",),
    )

    with pytest.raises(ToolInputError, match="outside configured roots"):
        request_sam_masks(
            settings,
            request_id="req-1",
            args=_args(image_path),
            staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
        )


def test_request_sam_masks_rejects_staging_dir_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    settings = _settings(tmp_path, sam_url="http://127.0.0.1:8712")

    with pytest.raises(ToolInputError, match="outside configured roots"):
        request_sam_masks(
            settings,
            request_id="req-1",
            args=_args(image_path),
            staging_dir=tmp_path / "not-output" / "sam" / "000050" / "candidates",
        )


def _settings(tmp_path: Path, *, sam_url: str) -> SceneFunc3dBackendSettings:
    output_root = tmp_path / "out"
    output_root.mkdir(exist_ok=True)
    return SceneFunc3dBackendSettings(
        molmo_url="http://127.0.0.1:8711",
        sam_url=sam_url,
        request_timeout_seconds=2.0,
        artifact_staging_root=output_root,
        allowed_image_roots=(tmp_path,),
        allowed_output_roots=(output_root,),
    )


def _args(image_path: Path) -> SamMaskArgs:
    return SamMaskArgs.model_validate(
        {
            "frame_id": "000050",
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


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
