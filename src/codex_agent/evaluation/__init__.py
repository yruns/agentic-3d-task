"""Evaluation harness for running tasks over benchmark folds."""

from __future__ import annotations

from .nr3d_runner import (
    Nr3dRunSummary,
    Nr3dSampleResult,
    run_one_sample,
    run_samples,
)
from .sample_ids import load_sample_ids

__all__ = [
    "load_sample_ids",
    "run_samples",
    "run_one_sample",
    "Nr3dRunSummary",
    "Nr3dSampleResult",
]
