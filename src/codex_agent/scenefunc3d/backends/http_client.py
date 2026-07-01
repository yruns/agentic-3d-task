"""JSON HTTP client for local SceneFunc3D sidecar backends."""

from __future__ import annotations

import json
import math
from json import JSONDecodeError
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from codex_agent.scenefunc3d.backends.config import HttpHeader
from codex_agent.scenefunc3d.tools.models import ToolInputError

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def post_json(
    url: str,
    *,
    payload: dict[str, object],
    response_model: type[ResponseT],
    timeout_seconds: float,
    request_headers: tuple[HttpHeader, ...] = (),
) -> ResponseT:
    """POST a JSON object and validate the sidecar response with Pydantic.

    Raises:
        ToolInputError: If the request cannot be encoded, the sidecar is
            unavailable, the request times out, or the response is not valid
            JSON matching ``response_model``.
    """
    sanitized_url = _sanitize_url(url)
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise ToolInputError(
            "sidecar HTTP request has invalid timeout: "
            f"url={sanitized_url}; error_type=invalid_timeout"
        )

    request_body = _encode_request_payload(payload, sanitized_url=sanitized_url)
    request = Request(
        url,
        data=request_body,
        headers=_json_request_headers(request_headers),
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except HTTPError as exc:
        error_body = exc.read()
        raise ToolInputError(
            "sidecar HTTP request failed: "
            f"url={sanitized_url}; error_type=http_error; "
            f"HTTP {exc.code}; response_bytes={len(error_body)}"
        ) from exc
    except TimeoutError as exc:
        raise ToolInputError(
            "sidecar HTTP request failed: "
            f"url={sanitized_url}; error_type=sidecar_request_timeout"
        ) from exc
    except URLError as exc:
        reason_type = exc.reason.__class__.__name__
        raise ToolInputError(
            "sidecar HTTP request failed: "
            f"url={sanitized_url}; error_type=sidecar_request_failed; "
            f"reason_type={reason_type}"
        ) from exc
    except OSError as exc:
        reason_type = exc.__class__.__name__
        raise ToolInputError(
            "sidecar HTTP request failed: "
            f"url={sanitized_url}; error_type=sidecar_request_failed; "
            f"reason_type={reason_type}"
        ) from exc

    decoded_response = _decode_response_body(
        response_body,
        sanitized_url=sanitized_url,
    )
    try:
        return response_model.model_validate(decoded_response)
    except ValidationError as exc:
        raise ToolInputError(
            "sidecar HTTP response failed validation: "
            f"url={sanitized_url}; error_type=invalid_response_schema; "
            f"error_count={exc.error_count()}"
        ) from exc


def _encode_request_payload(
    payload: dict[str, object],
    *,
    sanitized_url: str,
) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolInputError(
            "sidecar HTTP request payload is not JSON serializable: "
            f"url={sanitized_url}; error_type=invalid_request_payload"
        ) from exc


def _json_request_headers(request_headers: tuple[HttpHeader, ...]) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    for header in request_headers:
        headers[header.name] = header.value
    return headers


def _decode_response_body(response_body: bytes, *, sanitized_url: str) -> object:
    try:
        response_text = response_body.decode("utf-8")
        return json.loads(response_text)
    except UnicodeDecodeError as exc:
        raise ToolInputError(
            "sidecar HTTP response was not valid JSON: "
            f"url={sanitized_url}; error_type=invalid_response_json; "
            f"response_bytes={len(response_body)}"
        ) from exc
    except JSONDecodeError as exc:
        raise ToolInputError(
            "sidecar HTTP response was not valid JSON: "
            f"url={sanitized_url}; error_type=invalid_response_json; "
            f"response_bytes={len(response_body)}"
        ) from exc


def _sanitize_url(url: str) -> str:
    parsed_url = urlsplit(url)
    scheme = parsed_url.scheme or "<missing-scheme>"
    hostname = parsed_url.hostname or "<missing-host>"
    credentials = "<credentials>@" if parsed_url.username or parsed_url.password else ""
    try:
        port = parsed_url.port
    except ValueError:
        port_text = ":<invalid-port>"
    else:
        port_text = f":{port}" if port is not None else ""
    path = parsed_url.path or "/"
    return f"{scheme}://{credentials}{hostname}{port_text}{path}"


__all__ = ["post_json"]
