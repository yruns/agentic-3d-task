"""Tests for shared SceneFunc3D sidecar HTTP schemas."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    InlineImagePayload,
    MolmoImagePoint,
    MolmoPointRequest,
    MolmoPointResponse,
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
    SamPointPrompt,
)


def test_bool_mask_rle_codec_round_trips_2d_bool_mask() -> None:
    np = pytest.importorskip("numpy")
    from codex_agent.scenefunc3d.backends.mask_codec import (
        decode_bool_mask_rle,
        encode_bool_mask_rle,
    )

    mask = np.array(
        [
            [False, True, True, False],
            [False, False, True, False],
        ],
        dtype=np.bool_,
    )

    payload = encode_bool_mask_rle(mask)
    decoded_mask = decode_bool_mask_rle(payload)

    assert payload.height == 2
    assert payload.width == 4
    assert payload.counts == (1, 2, 3, 1, 1)
    assert decoded_mask.dtype == np.bool_
    np.testing.assert_array_equal(decoded_mask, mask)


def test_bool_mask_rle_codec_rejects_invalid_payloads() -> None:
    from codex_agent.scenefunc3d.backends.mask_codec import (
        MaskRlePayload,
        decode_bool_mask_rle,
    )

    invalid_payloads = (
        MaskRlePayload(height=2, width=2, counts=(0, -1, 5)),
        MaskRlePayload(height=2, width=2, counts=(0, 2)),
        MaskRlePayload(height=0, width=2, counts=(0,)),
    )

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            decode_bool_mask_rle(payload)


def test_bool_mask_rle_codec_rejects_non_2d_encode_input() -> None:
    np = pytest.importorskip("numpy")
    from codex_agent.scenefunc3d.backends.mask_codec import encode_bool_mask_rle

    mask = np.zeros((1, 2, 3), dtype=np.bool_)

    with pytest.raises(ValueError, match="2D"):
        encode_bool_mask_rle(mask)


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


def test_inline_image_payload_rejects_sha256_mismatch() -> None:
    image_bytes = b"not really an image"
    payload = _inline_image_payload(image_bytes).model_dump(mode="json")

    with pytest.raises(ValidationError, match="sha256"):
        InlineImagePayload.model_validate(
            {
                **payload,
                "sha256": hashlib.sha256(b"different").hexdigest(),
            }
        )


def test_molmo_point_request_accepts_inline_image_without_image_path() -> None:
    request = MolmoPointRequest(
        request_id="req-inline",
        image=_inline_image_payload(b"image"),
        prompt="point to the handle",
        image_width=640,
        image_height=480,
    )

    assert request.image_source == "inline"
    assert request.require_inline_image().filename == "frame.jpg"
    assert "image_path" not in request.model_dump(mode="json", exclude_none=True)


def test_molmo_point_request_rejects_missing_image() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        MolmoPointRequest(
            request_id="req-missing",
            prompt="point to the handle",
            image_width=640,
            image_height=480,
        )


def test_molmo_point_request_rejects_both_image_sources(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(ValidationError, match="exactly one"):
        MolmoPointRequest(
            request_id="req-both",
            image_path=image_path,
            image=_inline_image_payload(b"image"),
            prompt="point to the handle",
            image_width=640,
            image_height=480,
        )


def test_molmo_point_request_rejects_nonexistent_image_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        MolmoPointRequest(
            request_id="req-missing-file",
            image_path=tmp_path / "missing.jpg",
            prompt="point to the handle",
            image_width=640,
            image_height=480,
        )


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


def test_sam_mask_request_accepts_inline_image_without_image_path() -> None:
    request = SamMaskRequest(
        request_id="req-inline",
        image=_inline_image_payload(b"image"),
        points=(SamPointPrompt(x_px=10.0, y_px=20.0),),
    )

    assert request.image_source == "inline"
    assert request.staging_dir is None
    assert request.require_inline_image().filename == "frame.jpg"
    assert "image_path" not in request.model_dump(mode="json", exclude_none=True)


def test_sam_mask_request_rejects_missing_image(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        SamMaskRequest(
            request_id="req-missing",
            points=(SamPointPrompt(x_px=10.0, y_px=20.0),),
            staging_dir=tmp_path / "staging",
        )


def test_sam_mask_request_rejects_both_image_sources(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(ValidationError, match="exactly one"):
        SamMaskRequest(
            request_id="req-both",
            image_path=image_path,
            image=_inline_image_payload(b"image"),
            points=(SamPointPrompt(x_px=10.0, y_px=20.0),),
            staging_dir=tmp_path / "staging",
        )


def test_sam_mask_request_rejects_nonexistent_image_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SamMaskRequest(
            request_id="req-missing-file",
            image_path=tmp_path / "missing.jpg",
            points=(SamPointPrompt(x_px=10.0, y_px=20.0),),
            staging_dir=tmp_path / "staging",
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


def test_sam_mask_candidate_accepts_rle_without_npz_path() -> None:
    candidate = SamMaskCandidateResponse.model_validate(
        {
            "candidate_id": "mask_00",
            "score": 0.9,
            "mask_rle": {
                "encoding": "row_major_counts",
                "height": 2,
                "width": 3,
                "counts": (1, 2, 3),
            },
            "pixel_count": 2,
            "coverage_percent": 33.33333333333333,
        }
    )

    assert candidate.mask_npz_path is None
    assert candidate.mask_rle is not None
    assert candidate.mask_rle.counts == (1, 2, 3)
    assert "mask_npz_path" not in candidate.model_dump(
        mode="json",
        exclude_none=True,
    )


def test_sam_mask_candidate_accepts_remote_mask_path(tmp_path: Path) -> None:
    candidate = SamMaskCandidateResponse(
        candidate_id="mask_00",
        score=0.9,
        mask_npz_path=tmp_path / "missing.npz",
        pixel_count=11,
        coverage_percent=0.5,
    )

    assert candidate.mask_npz_path == tmp_path / "missing.npz"
    assert candidate.mask_npz_base64 == ""
    assert candidate.mask_npz_sha256 == ""


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


def _inline_image_payload(image_bytes: bytes) -> InlineImagePayload:
    return InlineImagePayload(
        filename="frame.jpg",
        mime_type="image/jpeg",
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        data_base64=base64.b64encode(image_bytes).decode("ascii"),
    )
