"""Molmo sidecar RPC client for SceneFunc3D tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
from codex_agent.scenefunc3d.tools.models import ToolInputError

from .config import SceneFunc3dBackendSettings, ensure_path_under_roots
from .http_client import post_json
from .image_payload import build_inline_image_payload, is_remote_backend_url

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPointArgs


def request_molmo_point(
    settings: SceneFunc3dBackendSettings,
    *,
    request_id: str,
    args: MolmoPointArgs,
) -> MolmoPointResponse:
    """Call the configured local Molmo sidecar and validate its response."""
    payload: dict[str, object] = {
        "request_id": request_id,
        "prompt": args.prompt,
        "image_width": args.image_width,
        "image_height": args.image_height,
    }
    if is_remote_backend_url(settings.molmo_url):
        payload["image"] = build_inline_image_payload(
            args.image_path,
            allowed_roots=settings.allowed_image_roots,
        ).model_dump(mode="json")
    else:
        image_path = ensure_path_under_roots(
            args.image_path,
            roots=settings.allowed_image_roots,
            field_name="image_path",
        )
        payload["image_path"] = str(image_path)

    response = post_json(
        f"{settings.molmo_url.rstrip('/')}/v1/point",
        payload=payload,
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
