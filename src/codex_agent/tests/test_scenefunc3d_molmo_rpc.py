"""Tests for SceneFunc3D Molmo RPC backend integration."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.config import SceneFunc3dBackendSettings
from codex_agent.scenefunc3d.backends.molmo_rpc import request_molmo_point
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPointArgs


def test_request_molmo_point_returns_raw_text(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "image_points": (
                    {
                        "x_px": 50.0,
                        "y_px": 40.0,
                        "source": '<point x="50" y="50">handle</point>',
                        "label": "handle",
                    },
                ),
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, molmo_url=f"http://127.0.0.1:{server.server_port}")
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )
    try:
        response = request_molmo_point(settings, request_id="req-1", args=args)
    finally:
        server.shutdown()
        server.server_close()

    assert response.raw_text == '<point x="50" y="50">handle</point>'
    assert response.image_points[0].x_px == 50.0
    assert response.image_points[0].label == "handle"


def test_request_molmo_point_supports_base_path(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server(
        {
            "/s/export-token/molmo/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "image_points": (
                    {
                        "x_px": 50.0,
                        "y_px": 40.0,
                        "source": '<point x="50" y="50">handle</point>',
                        "label": "handle",
                    },
                ),
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}/s/export-token/molmo",
    )
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )
    try:
        response = request_molmo_point(settings, request_id="req-path", args=args)
    finally:
        server.shutdown()
        server.server_close()

    assert response.request_id == "req-path"
    assert response.image_points[0].label == "handle"


def test_request_molmo_point_server_down_is_recoverable(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server({})
    server_port = server.server_port
    server.shutdown()
    server.server_close()
    settings = _settings(tmp_path, molmo_url=f"http://127.0.0.1:{server_port}")
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )

    with pytest.raises(ToolInputError, match="sidecar HTTP request failed"):
        request_molmo_point(settings, request_id="req-2", args=args)


def test_request_molmo_point_rejects_request_id_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server(
        {
            "/v1/point": lambda payload: {
                "request_id": "other-req",
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, molmo_url=f"http://127.0.0.1:{server.server_port}")
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )
    try:
        with pytest.raises(ToolInputError, match="request_id mismatch"):
            request_molmo_point(settings, request_id="req-1", args=args)
    finally:
        server.shutdown()
        server.server_close()


def test_request_molmo_point_rejects_image_outside_allowed_roots(
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
        allowed_output_roots=(tmp_path,),
    )
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )

    with pytest.raises(ToolInputError, match="outside configured roots"):
        request_molmo_point(settings, request_id="req-1", args=args)


def _settings(tmp_path: Path, *, molmo_url: str) -> SceneFunc3dBackendSettings:
    return SceneFunc3dBackendSettings(
        molmo_url=molmo_url,
        sam_url="http://127.0.0.1:8712",
        request_timeout_seconds=2.0,
        artifact_staging_root=tmp_path,
        allowed_image_roots=(tmp_path,),
        allowed_output_roots=(tmp_path,),
    )


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
