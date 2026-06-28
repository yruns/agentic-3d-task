"""Tests for SceneFunc3D Molmo and SAM tool contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.tools.molmo_pointing import parse_molmo_points
from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult


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


def test_parse_molmo_points_rejects_malformed_percent_value() -> None:
    with pytest.raises(SceneFunc3dDataError, match="invalid Molmo point x percent"):
        parse_molmo_points(
            '<point x="not-a-number" y="40" alt="bad">bad</point>',
            image_width=200,
            image_height=100,
        )


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
