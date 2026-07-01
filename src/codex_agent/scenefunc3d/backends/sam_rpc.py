"""SAM sidecar RPC client for SceneFunc3D tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codex_agent.scenefunc3d.servers.schemas import SamMaskResponse
from codex_agent.scenefunc3d.tools.models import ToolInputError

from .config import SceneFunc3dBackendSettings, ensure_path_under_roots
from .http_client import post_json

if TYPE_CHECKING:
    from pathlib import Path

    from codex_agent.scenefunc3d.tools.sam_masking import SamMaskArgs


def request_sam_masks(
    settings: SceneFunc3dBackendSettings,
    *,
    request_id: str,
    args: SamMaskArgs,
    staging_dir: Path,
) -> SamMaskResponse:
    """Call the configured local SAM sidecar and validate its response."""
    image_path = ensure_path_under_roots(
        args.image_path,
        roots=settings.allowed_image_roots,
        field_name="image_path",
    )
    artifact_staging_dir = ensure_path_under_roots(
        staging_dir,
        roots=settings.allowed_output_roots,
        field_name="staging_dir",
    )
    response = post_json(
        f"{settings.sam_url.rstrip('/')}/v1/masks",
        payload={
            "request_id": request_id,
            "image_path": str(image_path),
            "points": [point.to_payload() for point in args.points],
            "staging_dir": str(artifact_staging_dir),
        },
        response_model=SamMaskResponse,
        timeout_seconds=settings.request_timeout_seconds,
        request_headers=settings.request_headers,
    )
    if response.request_id != request_id:
        raise ToolInputError(
            "SAM sidecar response request_id mismatch: "
            f"expected={request_id!r}; received={response.request_id!r}"
        )
    if not response.candidates:
        raise ToolInputError(
            "SAM sidecar returned no mask candidates: " f"request_id={request_id!r}"
        )
    return response


__all__ = ["request_sam_masks"]
