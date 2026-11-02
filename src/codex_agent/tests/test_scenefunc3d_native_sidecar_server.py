"""Tests for the native single-process SceneFunc3D sidecar server."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np

from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.servers.molmo_point_server import (
    MolmoRunnerPointResult,
)
from codex_agent.scenefunc3d.servers.schemas import (
    MolmoImagePoint,
    MolmoPointRequest,
    SamMaskCandidateResponse,
    SamMaskRequest,
)


class _FakeMolmoRunner:
    model_name: str = "fake-molmo"

    def point(self, request: MolmoPointRequest) -> MolmoRunnerPointResult:
        request.require_image_path()
        raw_text = f'<point x="25" y="50">{request.prompt}</point>'
        return MolmoRunnerPointResult(
            raw_text=raw_text,
            image_points=(
                MolmoImagePoint(
                    x_px=float(request.image_width) * 0.25,
                    y_px=float(request.image_height) * 0.5,
                    source=raw_text,
                    label=request.prompt,
                ),
            ),
        )


class _FakeSamRunner:
    model_name: str = "fake-sam"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        staging_dir = request.require_staging_dir()
        request.require_image_path()
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
                score=0.875,
                mask_npz_path=mask_path,
                pixel_count=3,
                coverage_percent=50.0,
            ),
        )


class _OutOfMemoryMolmoRunner:
    model_name: str = "fake-molmo"

    def point(self, request: MolmoPointRequest) -> MolmoRunnerPointResult:
        raise RuntimeError("CUDA out of memory")


class _RecordingMolmoRunner(_FakeMolmoRunner):
    seen_image_path: Path | None = None
    image_existed_during_call: bool = False

    def point(self, request: MolmoPointRequest) -> MolmoRunnerPointResult:
        image_path = request.require_image_path()
        self.seen_image_path = image_path
        self.image_existed_during_call = image_path.is_file()
        return super().point(request)


class _RecordingSamRunner(_FakeSamRunner):
    seen_image_path: Path | None = None
    seen_staging_dir: Path | None = None
    image_existed_during_call: bool = False

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        image_path = request.require_image_path()
        self.seen_image_path = image_path
        self.seen_staging_dir = request.require_staging_dir()
        self.image_existed_during_call = image_path.is_file()
        return super().masks(request)


def test_combined_server_exposes_aggregate_and_component_health() -> None:
    from codex_agent.scenefunc3d.servers.scenefunc_sidecar_server import (
        SceneFuncSidecarRunners,
        build_routes,
    )

    runners = SceneFuncSidecarRunners(
        molmo_runner=_FakeMolmoRunner(),
        sam_runner=_FakeSamRunner(),
    )

    with _running_server(build_routes(runners)) as base_url:
        aggregate_health = _get_json(f"{base_url}/health")
        molmo_health = _get_json(f"{base_url}/molmo/health")
        sam_health = _get_json(f"{base_url}/sam/health")

    assert aggregate_health["status"] == "ok"
    assert aggregate_health["molmo"] == {
        "status": "ok",
        "model_name": "fake-molmo",
        "model_loaded": True,
    }
    assert aggregate_health["sam"] == {
        "status": "ok",
        "model_name": "fake-sam",
        "model_loaded": True,
    }
    assert molmo_health["model_name"] == "fake-molmo"
    assert sam_health["model_name"] == "fake-sam"


def test_combined_server_can_mount_under_export_base_path() -> None:
    from codex_agent.scenefunc3d.servers.scenefunc_sidecar_server import (
        SceneFuncSidecarRunners,
        build_routes,
    )

    runners = SceneFuncSidecarRunners(
        molmo_runner=_FakeMolmoRunner(),
        sam_runner=_FakeSamRunner(),
    )

    with _running_server(
        build_routes(runners, base_paths=("/s/export-token",))
    ) as base_url:
        aggregate_health = _get_json(f"{base_url}/s/export-token/health")
        molmo_health = _get_json(f"{base_url}/s/export-token/molmo/health")

    assert aggregate_health["status"] == "ok"
    assert molmo_health["model_name"] == "fake-molmo"


def test_combined_server_routes_molmo_and_sam_requests(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.scenefunc_sidecar_server import (
        SceneFuncSidecarRunners,
        build_routes,
    )

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    runners = SceneFuncSidecarRunners(
        molmo_runner=_FakeMolmoRunner(),
        sam_runner=_FakeSamRunner(),
    )

    with _running_server(build_routes(runners)) as base_url:
        molmo_response = _post_json(
            f"{base_url}/molmo/v1/point",
            payload={
                "request_id": "molmo-1",
                "image_path": str(image_path),
                "prompt": "green dial",
                "image_width": 200,
                "image_height": 100,
            },
        )
        sam_response = _post_json(
            f"{base_url}/sam/v1/masks",
            payload={
                "request_id": "sam-1",
                "image_path": str(image_path),
                "points": [
                    {
                        "x_px": 50.0,
                        "y_px": 50.0,
                        "label": "green dial",
                        "source": "fake",
                    }
                ],
                "staging_dir": str(tmp_path / "sam-output"),
            },
        )

    assert molmo_response["request_id"] == "molmo-1"
    assert molmo_response["model_name"] == "fake-molmo"
    assert molmo_response["image_points"] == [
        {
            "x_px": 50.0,
            "y_px": 50.0,
            "source": '<point x="25" y="50">green dial</point>',
            "label": "green dial",
        }
    ]
    assert cast(float, molmo_response["latency_ms"]) >= 0.0
    assert sam_response["request_id"] == "sam-1"
    assert sam_response["model_name"] == "fake-sam"
    assert cast(float, sam_response["latency_ms"]) >= 0.0
    sam_candidates = cast(list[dict[str, object]], sam_response["candidates"])
    assert sam_candidates[0]["candidate_id"] == "mask_00"
    assert "mask_npz_path" not in sam_candidates[0]
    assert sam_candidates[0]["mask_rle"] == {
        "encoding": "row_major_counts",
        "height": 2,
        "width": 3,
        "counts": [0, 1, 1, 1, 2, 1],
    }


def test_combined_server_materializes_inline_molmo_and_sam_requests() -> None:
    from codex_agent.scenefunc3d.servers.scenefunc_sidecar_server import (
        SceneFuncSidecarRunners,
        build_routes,
    )

    molmo_runner = _RecordingMolmoRunner()
    sam_runner = _RecordingSamRunner()
    runners = SceneFuncSidecarRunners(
        molmo_runner=molmo_runner,
        sam_runner=sam_runner,
    )

    with _running_server(build_routes(runners)) as base_url:
        molmo_response = _post_json(
            f"{base_url}/molmo/v1/point",
            payload={
                "request_id": "molmo-inline",
                "image": _inline_image_payload(b"molmo image bytes"),
                "prompt": "green dial",
                "image_width": 200,
                "image_height": 100,
            },
        )
        sam_response = _post_json(
            f"{base_url}/sam/v1/masks",
            payload={
                "request_id": "sam-inline",
                "image": _inline_image_payload(b"sam image bytes"),
                "points": [
                    {
                        "x_px": 50.0,
                        "y_px": 50.0,
                        "label": "green dial",
                        "source": "fake",
                    }
                ],
            },
        )

    assert molmo_response["request_id"] == "molmo-inline"
    assert sam_response["request_id"] == "sam-inline"
    assert molmo_runner.image_existed_during_call is True
    assert sam_runner.image_existed_during_call is True
    assert molmo_runner.seen_image_path is not None
    assert sam_runner.seen_image_path is not None
    assert sam_runner.seen_staging_dir is not None
    assert not molmo_runner.seen_image_path.exists()
    assert not sam_runner.seen_image_path.exists()
    assert not sam_runner.seen_staging_dir.exists()


def test_combined_server_maps_gpu_resource_errors_to_503(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.scenefunc_sidecar_server import (
        SceneFuncSidecarRunners,
        build_routes,
    )

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    runners = SceneFuncSidecarRunners(
        molmo_runner=_OutOfMemoryMolmoRunner(),
        sam_runner=_FakeSamRunner(),
    )

    with _running_server(build_routes(runners)) as base_url:
        error_payload = _post_json_expect_error(
            f"{base_url}/molmo/v1/point",
            payload={
                "request_id": "molmo-oom",
                "image_path": str(image_path),
                "prompt": "green dial",
                "image_width": 200,
                "image_height": 100,
            },
            expected_status=503,
        )

    assert error_payload == {"error": "molmo_resource_unavailable"}


@contextmanager
def _running_server(routes: dict[str, JsonRoute]) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_json_handler(routes),
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        payload: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("expected JSON object response")
    return dict(payload)


def _post_json(url: str, *, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        response_payload: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise AssertionError("expected JSON object response")
    return dict(response_payload)


def _post_json_expect_error(
    url: str,
    *,
    payload: dict[str, object],
    expected_status: int,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=5.0)
    except urllib.error.HTTPError as exc:
        assert exc.code == expected_status
        error_payload: object = json.loads(exc.read().decode("utf-8"))
        if not isinstance(error_payload, dict):
            raise AssertionError("expected JSON object error response") from exc
        return dict(error_payload)
    raise AssertionError("expected HTTP error")


def _inline_image_payload(image_bytes: bytes) -> dict[str, object]:
    return {
        "filename": "frame.jpg",
        "mime_type": "image/jpeg",
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "data_base64": base64.b64encode(image_bytes).decode("ascii"),
    }
