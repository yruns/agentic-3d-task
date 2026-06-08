"""NR3D visual-grounding task built on the Codex Agent SDK runtime.

This is the first task family wired into :class:`codex_agent.runtime.CodexAgentRuntime`.
It selects one object proposal for a referring expression over a prepared 3D
scene (an EmbodiedScan-style proposal pool) and scores it with oriented 3D IoU.
"""

from __future__ import annotations

from .geometry import compute_oriented_iou_3d, oriented_bbox_to_corners
from .grounding import (
    Nr3dGroundingDecision,
    Nr3dGroundingOutcome,
    Nr3dGroundingTask,
)
from .proposals import Proposal, ProposalPool
from .sample import Nr3dSample, Nr3dSampleId, Nr3dScene

__all__ = [
    "compute_oriented_iou_3d",
    "oriented_bbox_to_corners",
    "Proposal",
    "ProposalPool",
    "Nr3dSample",
    "Nr3dSampleId",
    "Nr3dScene",
    "Nr3dGroundingDecision",
    "Nr3dGroundingOutcome",
    "Nr3dGroundingTask",
]
