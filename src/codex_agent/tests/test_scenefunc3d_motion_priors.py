"""Tests for SceneFunc3D motion-type affordance-size priors."""

from __future__ import annotations

import pytest

from codex_agent.scenefunc3d.backends.motion_priors import (
    MotionSizePrior,
    motion_size_prior,
)


def test_small_control_motion_has_tight_prior() -> None:
    prior = motion_size_prior("key_press")
    assert isinstance(prior, MotionSizePrior)
    assert prior.max_radius_m == pytest.approx(0.12)
    assert prior.max_bbox_extent_m == pytest.approx(0.20)


def test_pull_motion_has_larger_prior_than_button() -> None:
    button = motion_size_prior("tip_push")
    pull = motion_size_prior("pinch_pull")
    assert pull.max_radius_m > button.max_radius_m


def test_unknown_motion_type_falls_back_to_default() -> None:
    prior = motion_size_prior("totally_unknown_motion")
    assert prior == motion_size_prior("")
    assert prior.max_radius_m == pytest.approx(0.25)


def test_motion_type_is_normalized_case_insensitively() -> None:
    assert motion_size_prior("KEY_PRESS") == motion_size_prior("key_press")
