"""Tests for SceneFunc3D SAM RPC backend integration."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

from codex_agent.scenefunc3d.backends import sam_rpc
from codex_agent.scenefunc3d.backends.config import (
    HttpHeader,
    SceneFunc3dBackendSettings,
)
from codex_agent.scenefunc3d.backends.sam_rpc import request_sam_masks
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.servers.schemas import (
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.sam_masking import SamMaskArgs


class _LocalNpzSamRunner:
    model_name = "fake-sam2"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        np = pytest.importorskip("numpy")
        staging_dir = request.require_staging_dir()
        staging_dir.mkdir(parents=True, exist_ok=True)
        mask_path = staging_dir / "mask_00.npz"
        mask = np.array(
            [[True, False, True], [False, False, True]],
            dtype=np.bool_,
        )
        np.savez_compressed(mask_path, mask=mask)
        return (
            SamMaskCandidateResponse(
                candidate_id="mask_00",
                score=0.91,
                mask_npz_path=mask_path,
                pixel_count=3,
                coverage_percent=50.0,
            ),
        )


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


def test_sam_server_response_returns_rle_without_server_local_path(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    image_path.write_bytes(b"image")
    server = _start_server(_build_routes(_LocalNpzSamRunner()))
    try:
        payload = _post_json(
            f"http://127.0.0.1:{server.server_port}/v1/masks",
            payload={
                "request_id": "req-rle",
                "image_path": str(image_path),
                "points": [
                    {
                        "x_px": 10.0,
                        "y_px": 20.0,
                        "label": "drawer",
                        "source": '<point x="10" y="20">drawer</point>',
                    }
                ],
                "staging_dir": str(staging_dir),
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    candidates = cast(list[dict[str, object]], payload["candidates"])
    candidate = candidates[0]
    mask_rle = cast(dict[str, object], candidate["mask_rle"])
    assert "mask_npz_path" not in candidate
    assert mask_rle == {
        "encoding": "row_major_counts",
        "height": 2,
        "width": 3,
        "counts": [0, 1, 1, 1, 2, 1],
    }
    assert candidate["pixel_count"] == 3
    assert candidate["coverage_percent"] == 50.0


def test_request_sam_masks_sends_inline_image_to_remote_https_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = b"jpeg bytes"
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(image_bytes)
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    captured_payload: dict[str, object] = {}

    def fake_post_json(
        url: str,
        *,
        payload: dict[str, object],
        response_model: type[SamMaskResponse],
        timeout_seconds: float,
        request_headers: tuple[HttpHeader, ...] = (),
    ) -> SamMaskResponse:
        captured_payload.update(payload)
        return response_model.model_validate(
            {
                "request_id": str(payload["request_id"]),
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.91,
                        "mask_npz_path": mask_path,
                        "pixel_count": 12,
                        "coverage_percent": 15.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        )

    monkeypatch.setattr(sam_rpc, "post_json", fake_post_json)
    settings = _settings(tmp_path, sam_url="https://sidecar.example/sam")

    response = request_sam_masks(
        settings,
        request_id="req-inline",
        args=_args(image_path),
        staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
    )

    assert response.request_id == "req-inline"
    assert "image_path" not in captured_payload
    assert "staging_dir" not in captured_payload
    inline_image = cast(dict[str, object], captured_payload["image"])
    assert inline_image["filename"] == "frame.jpg"
    assert inline_image["mime_type"] == "image/jpeg"
    assert inline_image["sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert base64.b64decode(str(inline_image["data_base64"])) == image_bytes


def test_request_sam_masks_keeps_image_path_for_local_http_backend(
    tmp_path: Path,
) -> None:
    captured_payloads: list[dict[str, object]] = []
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"

    def record_backend_request(payload: dict[str, object]) -> dict[str, object]:
        captured_payloads.append(payload)
        return {
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

    server = _start_server({"/v1/masks": record_backend_request})
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server.server_port}")
    try:
        request_sam_masks(
            settings,
            request_id="req-local",
            args=_args(image_path),
            staging_dir=staging_dir,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert captured_payloads[0]["image_path"] == str(image_path)
    assert captured_payloads[0]["staging_dir"] == str(staging_dir)
    assert "image" not in captured_payloads[0]


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


def _post_json(url: str, *, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response_payload: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise AssertionError("expected JSON object response")
    return dict(response_payload)
