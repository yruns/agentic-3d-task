"""Tests for shared SceneFunc3D sidecar HTTP schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    MolmoImagePoint,
    MolmoPointRequest,
    MolmoPointResponse,
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
    SamPointPrompt,
)


def test_molmo_point_schema_round_trip(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    request = MolmoPointRequest(
        request_id="req-1",
        image_path=image_path,
        prompt="point to the handle",
        image_width=640,
        image_height=480,
    )
    response = MolmoPointResponse(
        request_id=request.request_id,
        model_name="MolmoPoint-8B",
        raw_text='<point x="50" y="50">handle</point>',
        image_points=(
            MolmoImagePoint(
                x_px=320.0,
                y_px=240.0,
                source='<point x="50" y="50">handle</point>',
                label="handle",
            ),
        ),
        latency_ms=12,
    )

    assert request.model_dump(mode="json")["image_path"] == str(image_path)
    assert response.model_dump(mode="json")["raw_text"].startswith("<point")
    assert response.model_dump(mode="json")["image_points"][0]["x_px"] == 320.0


def test_molmo_point_request_rejects_extra_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    payload: dict[str, object] = {
        "request_id": "req-1",
        "image_path": str(image_path),
        "prompt": "point to the handle",
        "image_width": 640,
        "image_height": 480,
        "unexpected": "value",
    }

    with pytest.raises(ValidationError):
        MolmoPointRequest.model_validate(payload)


def test_molmo_point_request_rejects_string_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    payload: dict[str, object] = {
        "request_id": "req-1",
        "image_path": str(image_path),
        "prompt": "point to the handle",
        "image_width": "640",
        "image_height": "480",
    }

    with pytest.raises(ValidationError):
        MolmoPointRequest.model_validate(payload)


def test_molmo_point_response_accepts_empty_raw_text() -> None:
    response = MolmoPointResponse(
        request_id="req-1",
        model_name="MolmoPoint-8B",
        raw_text="",
        latency_ms=12,
    )

    assert response.raw_text == ""


def test_molmo_point_response_rejects_infinite_latency() -> None:
    with pytest.raises(ValidationError):
        MolmoPointResponse(
            request_id="req-1",
            model_name="MolmoPoint-8B",
            raw_text="",
            latency_ms=float("inf"),
        )


def test_sam_mask_schema_round_trip(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    mask_path = staging_dir / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    request = SamMaskRequest(
        request_id="req-2",
        image_path=image_path,
        points=(
            SamPointPrompt(
                x_px=10.0,
                y_px=20.0,
                label="handle",
                source='<point x="1" y="2">handle</point>',
            ),
        ),
        staging_dir=staging_dir,
    )
    response = SamMaskResponse(
        request_id=request.request_id,
        model_name="SAM2.1-Hiera-L",
        candidates=(
            SamMaskCandidateResponse(
                candidate_id="mask_00",
                score=0.9,
                mask_npz_path=mask_path,
                pixel_count=11,
                coverage_percent=0.5,
            ),
        ),
        latency_ms=34,
    )

    assert request.model_dump(mode="json")["points"][0]["label"] == "handle"
    assert (
        response.model_dump(mode="json")["candidates"][0]["candidate_id"] == "mask_00"
    )


def test_sam_point_prompt_defaults_empty_label_and_source() -> None:
    point = SamPointPrompt(x_px=10.0, y_px=20.0)

    assert point.label == ""
    assert point.source == ""


def test_sam_point_prompt_rejects_string_coordinates() -> None:
    payload: dict[str, object] = {
        "x_px": "10.0",
        "y_px": "20.0",
    }

    with pytest.raises(ValidationError):
        SamPointPrompt.model_validate(payload)


def test_sam_point_prompt_rejects_integer_coordinates() -> None:
    payload: dict[str, object] = {
        "x_px": 10,
        "y_px": 20.0,
    }

    with pytest.raises(ValidationError):
        SamPointPrompt.model_validate(payload)


def test_sam_point_prompt_rejects_infinite_coordinates() -> None:
    with pytest.raises(ValidationError):
        SamPointPrompt(x_px=float("inf"), y_px=20.0)


def test_sam_mask_request_accepts_missing_staging_dir_path(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "will-be-created"

    request = SamMaskRequest(
        request_id="req-4",
        image_path=image_path,
        points=(SamPointPrompt(x_px=10.0, y_px=20.0),),
        staging_dir=staging_dir,
    )

    assert request.staging_dir == staging_dir


def test_sam_mask_candidate_rejects_out_of_range_score(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")

    with pytest.raises(ValidationError):
        SamMaskCandidateResponse(
            candidate_id="mask_00",
            score=1.1,
            mask_npz_path=mask_path,
            pixel_count=11,
            coverage_percent=0.5,
        )


def test_sam_mask_candidate_rejects_out_of_range_coverage(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")

    with pytest.raises(ValidationError):
        SamMaskCandidateResponse(
            candidate_id="mask_00",
            score=0.9,
            mask_npz_path=mask_path,
            pixel_count=11,
            coverage_percent=100.1,
        )


def test_sam_mask_candidate_rejects_string_pixel_count(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    payload: dict[str, object] = {
        "candidate_id": "mask_00",
        "score": 0.9,
        "mask_npz_path": mask_path,
        "pixel_count": "1",
        "coverage_percent": 0.5,
    }

    with pytest.raises(ValidationError):
        SamMaskCandidateResponse.model_validate(payload)


def test_sam_mask_candidate_requires_existing_mask_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SamMaskCandidateResponse(
            candidate_id="mask_00",
            score=0.9,
            mask_npz_path=tmp_path / "missing.npz",
            pixel_count=11,
            coverage_percent=0.5,
        )


def test_sam_mask_response_rejects_infinite_latency(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask_00.npz"
    mask_path.write_bytes(b"mask")

    with pytest.raises(ValidationError):
        SamMaskResponse(
            request_id="req-2",
            model_name="SAM2.1-Hiera-L",
            candidates=(
                SamMaskCandidateResponse(
                    candidate_id="mask_00",
                    score=0.9,
                    mask_npz_path=mask_path,
                    pixel_count=11,
                    coverage_percent=0.5,
                ),
            ),
            latency_ms=float("inf"),
        )


def test_health_response() -> None:
    payload = HealthResponse(
        status="ok",
        model_name="MolmoPoint-8B",
        model_loaded=True,
    )

    assert payload.model_dump(mode="json")["status"] == "ok"


def test_health_response_rejects_non_ok_status() -> None:
    payload: dict[str, object] = {
        "status": "starting",
        "model_name": "MolmoPoint-8B",
        "model_loaded": False,
    }

    with pytest.raises(ValidationError):
        HealthResponse.model_validate(payload)


def test_sam_request_requires_point(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    with pytest.raises(ValidationError):
        SamMaskRequest(
            request_id="req-3",
            image_path=image_path,
            points=(),
            staging_dir=staging_dir,
        )
