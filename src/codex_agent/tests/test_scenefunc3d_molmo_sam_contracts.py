"""Tests for SceneFunc3D Molmo and SAM tool contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.tools.molmo_pointing import parse_molmo_points
from codex_agent.scenefunc3d.tools.sam_masking import (
    SamCandidate,
    SamMaskArgs,
    SamMaskResult,
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
def test_sam_mask_args_rejects_coerced_point_x_values(bad_x_px: object) -> None:
    payload: dict[str, object] = {
        "frame_id": "000050",
        "image_path": "/tmp/frame.jpg",
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
                overlay_path=Path("/tmp/mask_00.jpg"),
            ),
        ),
        contact_sheet_path=Path("/tmp/sam_candidates.jpg"),
    )
    assert result.to_payload()["candidates"][0]["candidate_id"] == "mask_00"
