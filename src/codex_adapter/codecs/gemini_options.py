"""Validate Responses options and construct stable Gemini Chat options."""

from __future__ import annotations

from codex_adapter.domain.json_codec import JsonObject, JsonValue
from codex_adapter.domain.models import ModelCapabilities, RegisteredModel
from codex_adapter.domain.state import ToolName

from .gemini_common import (
    array_value,
    boolean_value,
    invalid,
    number_value,
    object_value,
    reject_extra,
    string_value,
    unsupported,
)
from .gemini_tools import checked_name, translate_tool_choice, translate_tools

_ALLOWED = {
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "text",
    "include",
    "store",
    "stream",
    "metadata",
    "client_metadata",
    "prompt_cache_key",
    "service_tier",
    "max_output_tokens",
    "temperature",
    "top_p",
    "user",
    "truncation",
    "previous_response_id",
    "conversation",
}


def _local_metadata(body: JsonObject) -> None:
    for key in ("metadata", "client_metadata"):
        value = body.get(key)
        if value is None:
            continue
        metadata = object_value(value, key)
        if any(not isinstance(item, str) for item in metadata.values()):
            raise invalid(key)
    if body.get("prompt_cache_key") is not None:
        string_value(body["prompt_cache_key"], "prompt_cache_key")


def _local_options(body: JsonObject) -> None:
    reject_extra(body, _ALLOWED, "request field")
    for field in ("previous_response_id", "conversation", "service_tier"):
        if body.get(field) is not None:
            raise unsupported(field)
    if body.get("truncation") not in (None, "disabled"):
        raise unsupported("truncation")
    if body.get("store") is not None and boolean_value(body["store"], "store"):
        raise unsupported("store")
    if "stream" in body:
        boolean_value(body["stream"], "stream")
    include = body.get("include")
    if include is not None:
        entries = array_value(include, "include")
        if entries not in ([], ["reasoning.encrypted_content"]):
            raise unsupported("include")
    _local_metadata(body)


def _output_format(
    value: JsonValue, capabilities: ModelCapabilities
) -> JsonObject | None:
    if value is None:
        return None
    text = object_value(value, "text")
    reject_extra(text, {"format"}, "text field")
    if "format" not in text:
        return None
    format_value = object_value(text["format"], "text.format")
    if format_value.get("type") == "text":
        reject_extra(format_value, {"type"}, "text.format field")
        return None
    if format_value.get("type") != "json_schema" or not capabilities.structured_output:
        raise unsupported("text.format")
    reject_extra(
        format_value,
        {"type", "name", "schema", "strict", "description"},
        "text.format field",
    )
    schema: JsonObject = {
        "name": checked_name(format_value.get("name"), "text.format.name"),
        "schema": object_value(format_value.get("schema"), "text.format.schema"),
    }
    if "strict" in format_value:
        schema = {
            **schema,
            "strict": boolean_value(format_value["strict"], "text.format.strict"),
        }
    if "description" in format_value:
        schema = {
            **schema,
            "description": string_value(
                format_value["description"], "text.format.description", empty=True
            ),
        }
    return {"type": "json_schema", "json_schema": schema}


def _reasoning(value: JsonValue, capabilities: ModelCapabilities) -> JsonObject:
    reasoning = {} if value is None else object_value(value, "reasoning")
    reject_extra(reasoning, {"effort", "summary"}, "reasoning field")
    if reasoning.get("summary") not in (None, "none"):
        raise unsupported("reasoning.summary")
    effort = reasoning.get("effort", capabilities.default_reasoning_effort)
    if effort is None:
        return {}
    if not isinstance(effort, str) or effort not in capabilities.reasoning_efforts:
        raise unsupported("reasoning.effort")
    return {"reasoning_effort": effort}


def _numeric_options(body: JsonObject, capabilities: ModelCapabilities) -> JsonObject:
    maximum = body.get("max_output_tokens", capabilities.default_max_output_tokens)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise invalid("max_output_tokens")
    options: JsonObject = {"max_tokens": maximum}
    for field, upper in (("temperature", 2.0), ("top_p", 1.0)):
        if field in body:
            options = {
                **options,
                field: number_value(body[field], field, low=0.0, high=upper),
            }
    return options


def translate_options(
    body: JsonObject, model: RegisteredModel
) -> tuple[JsonObject, tuple[ToolName, ...]]:
    """Return canonicalizable model-affecting options and reversible tool names."""
    _local_options(body)
    if model.protocol != "gemini_chat" or body.get("model") != model.id:
        raise invalid("model")
    tools, names = translate_tools(body.get("tools", []), model.capabilities)
    options: JsonObject = {
        "model": model.id,
        "stream": False,
        **_numeric_options(body, model.capabilities),
    }
    if tools:
        options = {**options, "tools": list(tools)}
    if body.get("tool_choice") is not None:
        options = {
            **options,
            "tool_choice": translate_tool_choice(body["tool_choice"], names),
        }
    if "parallel_tool_calls" in body:
        parallel = boolean_value(body["parallel_tool_calls"], "parallel_tool_calls")
        if parallel and not model.capabilities.parallel_tools:
            raise unsupported("parallel_tool_calls")
        options = {**options, "parallel_tool_calls": parallel}
    options = {**options, **_reasoning(body.get("reasoning"), model.capabilities)}
    output_format = _output_format(body.get("text"), model.capabilities)
    if output_format is not None:
        options = {**options, "response_format": output_format}
    if body.get("user") is not None:
        options = {**options, "user": string_value(body["user"], "user")}
    return options, names
