"""Loss-aware text and image translation for Gemini Chat messages."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
from urllib.parse import urlsplit

from codex_adapter.domain.json_codec import JsonObject, JsonValue
from codex_adapter.domain.models import ModelCapabilities

from .gemini_common import (
    array_value,
    invalid,
    object_value,
    reject_extra,
    string_value,
    unsupported,
)

_DATA_IMAGE = re.compile(
    r"^data:image/[a-zA-Z0-9][a-zA-Z0-9.+-]*;base64,([A-Za-z0-9+/]*={0,2})$"
)


def _data_url(url: str) -> None:
    match = _DATA_IMAGE.fullmatch(url)
    if match is None:
        raise invalid("image_url")
    try:
        decoded = base64.b64decode(match.group(1), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise invalid("image_url") from exc
    if not decoded:
        raise invalid("image_url")


def _https_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise invalid("image_url") from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise invalid("image_url")
    if port is not None and not 1 <= port <= 65535:
        raise invalid("image_url")
    if hostname.lower() == "localhost" or hostname.lower().endswith(
        (".localhost", ".local")
    ):
        raise invalid("image_url")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise invalid("image_url")


def _image_url(value: JsonValue) -> str:
    url = string_value(value, "image_url")
    if any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 or char == "\\"
        for char in url
    ):
        raise invalid("image_url")
    if url.startswith("data:"):
        _data_url(url)
    else:
        _https_url(url)
    return url


def _image_part(part: JsonObject, capabilities: ModelCapabilities) -> JsonObject:
    if not capabilities.images:
        raise unsupported("image input")
    reject_extra(part, {"type", "image_url", "detail"}, "image input field")
    raw_url = part.get("image_url")
    if isinstance(raw_url, dict):
        reject_extra(raw_url, {"url", "detail"}, "image_url field")
        if "detail" in part and "detail" in raw_url:
            raise invalid("image_url.detail")
        detail = part.get("detail", raw_url.get("detail"))
        url = _image_url(raw_url.get("url"))
    else:
        detail = part.get("detail")
        url = _image_url(raw_url)
    image: JsonObject = {"url": url}
    if detail is not None:
        if not isinstance(detail, str) or detail not in {"auto", "low", "high"}:
            raise invalid("image_url.detail")
        image = {**image, "detail": detail}
    return {"type": "image_url", "image_url": image}


def _text_part(part: JsonObject) -> JsonObject:
    reject_extra(part, {"type", "text"}, "text input field")
    return {
        "type": "text",
        "text": string_value(part.get("text"), "input text", empty=True),
    }


def message_content(
    value: JsonValue, capabilities: ModelCapabilities, *, allow_images: bool
) -> JsonValue:
    """Translate input text and validated image URLs without mutating input."""
    if isinstance(value, str):
        return value
    parts = array_value(value, "message content")
    if not parts:
        raise invalid("message content")
    converted: tuple[JsonObject, ...] = ()
    for value in parts:
        part = object_value(value, "content part")
        if part.get("type") in ("input_text", "text"):
            converted = (*converted, _text_part(part))
        elif part.get("type") in ("input_image", "image_url") and allow_images:
            converted = (*converted, _image_part(part, capabilities))
        else:
            raise unsupported("content part")
    if all(part["type"] == "text" for part in converted):
        return "".join(str(part["text"]) for part in converted)
    return list(converted)
