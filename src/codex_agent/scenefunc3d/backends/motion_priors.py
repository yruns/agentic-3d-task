"""Affordance-size priors keyed by SceneFun3D motion type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSizePrior:
    """Upper bounds on the physical size of an affordance for one motion type."""

    max_radius_m: float
    max_bbox_extent_m: float


_DEFAULT_PRIOR = MotionSizePrior(max_radius_m=0.25, max_bbox_extent_m=0.45)

# Initial priors; calibrate against the offline sweep (see plan Task 8).
_PRIORS_BY_MOTION: dict[str, MotionSizePrior] = {
    "rotate": MotionSizePrior(0.12, 0.20),
    "key_press": MotionSizePrior(0.12, 0.20),
    "tip_push": MotionSizePrior(0.12, 0.20),
    "hook_turn": MotionSizePrior(0.12, 0.20),
    "pinch_pull": MotionSizePrior(0.20, 0.35),
    "hook_pull": MotionSizePrior(0.20, 0.35),
    "foot_push": MotionSizePrior(0.20, 0.35),
    "plug_in": MotionSizePrior(0.15, 0.25),
    "unplug": MotionSizePrior(0.15, 0.25),
}


def motion_size_prior(motion_type: str) -> MotionSizePrior:
    """Return the affordance-size prior for a SceneFun3D motion type."""
    return _PRIORS_BY_MOTION.get(motion_type.strip().lower(), _DEFAULT_PRIOR)


__all__ = ["MotionSizePrior", "motion_size_prior"]
