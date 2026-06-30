from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SanitizedStateStats:
    removed_encrypted_content_count: int
    dropped_empty_reasoning_count: int
    removed_previous_response_id: bool


def sanitize_encrypted_state(
    raw_body: JsonObject,
) -> tuple[JsonObject, SanitizedStateStats]:
    sanitized, stats = _sanitize_encrypted_state_value(raw_body, top_level=True)
    if not isinstance(sanitized, dict):
        sanitized = {}
    return sanitized, SanitizedStateStats(
        removed_encrypted_content_count=int(stats["removed_encrypted_content_count"]),
        dropped_empty_reasoning_count=int(stats["dropped_empty_reasoning_count"]),
        removed_previous_response_id=bool(stats["removed_previous_response_id"]),
    )


def is_invalid_encrypted_content(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return b"invalid_encrypted_content" in body or b"Encrypted content" in body
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        values = [error.get("code"), error.get("message"), error.get("type")]
        return any(
            "invalid_encrypted_content" in str(value)
            for value in values
            if value is not None
        )
    return "invalid_encrypted_content" in json.dumps(payload, ensure_ascii=False)


class _DropSanitizedItem:
    pass


_DROP_SANITIZED_ITEM = _DropSanitizedItem()


def _sanitize_encrypted_state_value(
    value: Any, *, top_level: bool = False
) -> tuple[Any, JsonObject]:
    stats: JsonObject = {
        "removed_encrypted_content_count": 0,
        "dropped_empty_reasoning_count": 0,
        "removed_previous_response_id": False,
    }

    def merge(child_stats: JsonObject) -> None:
        stats["removed_encrypted_content_count"] += int(
            child_stats.get("removed_encrypted_content_count") or 0
        )
        stats["dropped_empty_reasoning_count"] += int(
            child_stats.get("dropped_empty_reasoning_count") or 0
        )
        stats["removed_previous_response_id"] = bool(
            stats["removed_previous_response_id"]
            or child_stats.get("removed_previous_response_id")
        )

    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for item in value:
            sanitized_item, child_stats = _sanitize_encrypted_state_value(item)
            merge(child_stats)
            if sanitized_item is _DROP_SANITIZED_ITEM:
                continue
            sanitized_list.append(sanitized_item)
        return sanitized_list, stats

    if not isinstance(value, dict):
        return value, stats

    sanitized_dict: JsonObject = {}
    for key, item_value in value.items():
        if key == "encrypted_content":
            stats["removed_encrypted_content_count"] += 1
            continue
        if top_level and key == "previous_response_id":
            stats["removed_previous_response_id"] = True
            continue
        sanitized_item, child_stats = _sanitize_encrypted_state_value(item_value)
        merge(child_stats)
        if sanitized_item is _DROP_SANITIZED_ITEM:
            continue
        sanitized_dict[key] = sanitized_item

    if _is_empty_reasoning_item(sanitized_dict):
        stats["dropped_empty_reasoning_count"] += 1
        return _DROP_SANITIZED_ITEM, stats
    return sanitized_dict, stats


def _is_empty_reasoning_item(item: JsonObject) -> bool:
    if item.get("type") != "reasoning":
        return False
    readable_keys = set(item) - {"id", "type", "status"}
    if not readable_keys:
        return True
    for key in readable_keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return False
    return True
