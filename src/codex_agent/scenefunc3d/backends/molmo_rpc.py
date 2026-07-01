"""Molmo sidecar RPC client for SceneFunc3D tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
from codex_agent.scenefunc3d.tools.models import ToolInputError

from .config import SceneFunc3dBackendSettings, ensure_path_under_roots
from .http_client import post_json

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPointArgs


def request_molmo_point(
    settings: SceneFunc3dBackendSettings,
    *,
    request_id: str,
    args: MolmoPointArgs,
) -> MolmoPointResponse:
    """Call the configured local Molmo sidecar and validate its response."""
    image_path = ensure_path_under_roots(
        args.image_path,
        roots=settings.allowed_image_roots,
        field_name="image_path",
    )
    response = post_json(
        f"{settings.molmo_url.rstrip('/')}/v1/point",
        payload={
            "request_id": request_id,
            "image_path": str(image_path),
            "prompt": args.prompt,
            "image_width": args.image_width,
            "image_height": args.image_height,
        },
        response_model=MolmoPointResponse,
        timeout_seconds=settings.request_timeout_seconds,
        request_headers=settings.request_headers,
    )
    if response.request_id != request_id:
        raise ToolInputError(
            "Molmo sidecar response request_id mismatch: "
            f"expected={request_id!r}; received={response.request_id!r}"
        )
    return response


__all__ = ["request_molmo_point"]
