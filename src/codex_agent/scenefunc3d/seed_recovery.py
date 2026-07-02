"""Recover a confirmed 2D SAM candidate mask from a run's ``events.jsonl``.

The Phase-2c agent turn confirms a seed ``(frame_id, candidate_id)`` after
reviewing the SAM candidate mask on the original RGB. The run's tool event log
already recorded the successful ``sam_mask`` that produced that candidate. This
module resolves the candidate's 2D mask NPZ so ``run_single_sample`` can
deterministically lift it and freeze a ``TargetAnchor``. It reads the log
standalone (no ``runner`` imports) to avoid an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StringConstraints

from ..errors import CodexResponseError

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]

_SAM_TOOL_NAME = "sam_mask"


class _SamCandidateEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: NonEmptyString
    mask_npz_path: NonEmptyString


class _SamEventResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    candidates: tuple[_SamCandidateEvent, ...] = Field(min_length=1)


@dataclass(frozen=True)
class SeedSamCandidate:
    """The confirmed seed's 2D SAM candidate mask NPZ."""

    frame_id: str
    candidate_id: str
    mask_npz_path: Path


def recover_seed_sam_candidate(
    events_path: Path, *, seed_frame_id: str, seed_candidate_id: str
) -> SeedSamCandidate:
    """Return the confirmed seed's last successful SAM candidate mask.

    Raises:
        CodexResponseError: If the event log is missing/unreadable, or no
            successful ``sam_mask`` event on ``seed_frame_id`` contains
            ``seed_candidate_id``.
    """
    if not events_path.is_file():
        raise CodexResponseError(
            "SceneFun3D tool event log is missing while recovering the confirmed "
            f"seed SAM candidate: events_path={events_path}; "
            f"seed_frame_id={seed_frame_id!r}; seed_candidate_id={seed_candidate_id!r}"
        )
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFun3D tool event log while recovering the seed "
            f"SAM candidate: events_path={events_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    matched_npz_path: Path | None = None
    for line_number, raw_line in enumerate(event_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        candidate_path = _match_seed_sam_line(
            line,
            events_path=events_path,
            line_number=line_number,
            seed_frame_id=seed_frame_id,
            seed_candidate_id=seed_candidate_id,
        )
        if candidate_path is not None:
            matched_npz_path = candidate_path

    if matched_npz_path is None:
        raise CodexResponseError(
            "no successful sam_mask event on the confirmed seed frame contains the "
            f"confirmed candidate: events_path={events_path}; "
            f"seed_frame_id={seed_frame_id!r}; seed_candidate_id={seed_candidate_id!r}"
        )
    return SeedSamCandidate(
        frame_id=seed_frame_id,
        candidate_id=seed_candidate_id,
        mask_npz_path=matched_npz_path,
    )


def _match_seed_sam_line(
    line: str,
    *,
    events_path: Path,
    line_number: int,
    seed_frame_id: str,
    seed_candidate_id: str,
) -> Path | None:
    try:
        event: object = loads(line)
    except JSONDecodeError as exc:
        raise CodexResponseError(
            "SceneFun3D tool event line is not valid JSON while recovering the "
            f"seed SAM candidate: events_path={events_path}; line={line_number}; "
            f"error={exc}"
        ) from exc
    if not isinstance(event, dict):
        return None
    if event.get("tool_name") != _SAM_TOOL_NAME or event.get("status") != "success":
        return None
    try:
        result = _SamEventResult.model_validate(event.get("result"))
    except ValidationError as exc:
        raise CodexResponseError(
            "successful sam_mask event does not match the expected result schema: "
            f"events_path={events_path}; line={line_number}; error={exc}"
        ) from exc
    if result.frame_id != seed_frame_id:
        return None
    for candidate in result.candidates:
        if candidate.candidate_id == seed_candidate_id:
            return Path(candidate.mask_npz_path).expanduser().resolve()
    return None


__all__ = ["SeedSamCandidate", "recover_seed_sam_candidate"]
