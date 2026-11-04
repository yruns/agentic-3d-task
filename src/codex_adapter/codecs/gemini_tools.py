"""Loss-aware function and namespace tool translation for ModelHub Chat."""

from __future__ import annotations

import hashlib
import re

from codex_adapter.domain.json_codec import JsonObject, JsonValue
from codex_adapter.domain.models import ModelCapabilities
from codex_adapter.domain.state import ToolName

from .gemini_common import (
    array_value,
    boolean_value,
    invalid,
    object_value,
    reject_extra,
    string_value,
    unsupported,
)

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def checked_name(value: JsonValue, field: str) -> str:
    name = string_value(value, field)
    if _NAME.fullmatch(name) is None:
        raise invalid(field)
    return name


def _upstream_name(name: str, namespace: str | None) -> str:
    if namespace is None:
        return name
    digest = hashlib.sha256(f"{namespace}\0{name}".encode()).hexdigest()[:16]
    return f"ns_{namespace[:18]}_{name[:22]}_{digest}"


def _parameters(value: JsonValue) -> JsonObject:
    parameters = object_value(value, "tools.parameters")
    if "type" in parameters and parameters["type"] != "object":
        raise invalid("tools.parameters.type")
    if "properties" in parameters and not isinstance(parameters["properties"], dict):
        raise invalid("tools.parameters.properties")
    if "required" in parameters:
        required = array_value(parameters["required"], "tools.parameters.required")
        if any(not isinstance(item, str) for item in required) or len(
            set(required)
        ) != len(required):
            raise invalid("tools.parameters.required")
    return parameters


def _function_description(
    tool: JsonObject, namespace_description: str | None
) -> str | None:
    description = tool.get("description")
    if description is not None:
        description = string_value(description, "tools.description", empty=True)
    if namespace_description is None:
        return description
    if description is None:
        return f"Namespace description: {namespace_description}"
    return f"Namespace description: {namespace_description}\n\n{description}"


def _function_tool(
    tool: JsonObject, namespace: str | None, namespace_description: str | None
) -> tuple[JsonObject, ToolName]:
    reject_extra(
        tool,
        {"type", "name", "description", "parameters", "strict", "namespace"},
        "tool field",
    )
    if tool.get("type") != "function":
        raise unsupported("tool type")
    name = checked_name(tool.get("name"), "tools.name")
    declared_namespace = tool.get("namespace")
    if declared_namespace is not None:
        declared_namespace = checked_name(declared_namespace, "tools.namespace")
        if namespace is not None and declared_namespace != namespace:
            raise invalid("tools.namespace")
        namespace = declared_namespace
    upstream = _upstream_name(name, namespace)
    function: JsonObject = {
        "name": upstream,
        "parameters": _parameters(tool.get("parameters")),
    }
    description = _function_description(tool, namespace_description)
    if description is not None:
        function = {**function, "description": description}
    if "strict" in tool:
        function = {**function, "strict": boolean_value(tool["strict"], "tools.strict")}
    return {"type": "function", "function": function}, ToolName(
        upstream, name, namespace
    )


def _namespace_tools(
    tool: JsonObject, capabilities: ModelCapabilities
) -> tuple[tuple[JsonObject, ToolName], ...]:
    if not capabilities.namespace_tools:
        raise unsupported("namespace tools")
    reject_extra(tool, {"type", "name", "description", "tools"}, "namespace field")
    namespace = checked_name(tool.get("name"), "tools.namespace")
    description = tool.get("description")
    if description is not None:
        description = string_value(description, "tools.description", empty=True)
    children = array_value(tool.get("tools"), "namespace.tools")
    if not children:
        raise invalid("namespace.tools")
    return tuple(
        _function_tool(object_value(child, "namespace.tool"), namespace, description)
        for child in children
    )


def translate_tools(
    value: JsonValue, capabilities: ModelCapabilities
) -> tuple[list[JsonObject], tuple[ToolName, ...]]:
    tools = array_value(value, "tools")
    if tools and not capabilities.function_tools:
        raise unsupported("function tools")
    entries: tuple[tuple[JsonObject, ToolName], ...] = ()
    for value in tools:
        tool = object_value(value, "tool")
        if tool.get("type") == "namespace":
            entries = (*entries, *_namespace_tools(tool, capabilities))
        else:
            entry = _function_tool(tool, None, None)
            if entry[1].namespace is not None and not capabilities.namespace_tools:
                raise unsupported("namespace tools")
            entries = (*entries, entry)
    names = tuple(entry[1] for entry in entries)
    if len({(name.namespace, name.name) for name in names}) != len(names):
        raise invalid("duplicate tools")
    if len({name.upstream_name for name in names}) != len(names):
        raise invalid("colliding tool names")
    return [entry[0] for entry in entries], names


def translate_tool_choice(value: JsonValue, names: tuple[ToolName, ...]) -> JsonValue:
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise unsupported("tool_choice")
        if value == "required" and not names:
            raise invalid("tool_choice")
        return value
    choice = object_value(value, "tool_choice")
    reject_extra(choice, {"type", "name", "namespace"}, "tool_choice field")
    if choice.get("type") != "function":
        raise unsupported("tool_choice type")
    name = checked_name(choice.get("name"), "tool_choice.name")
    namespace = choice.get("namespace")
    if namespace is not None:
        namespace = checked_name(namespace, "tool_choice.namespace")
    matches = tuple(
        item for item in names if item.name == name and item.namespace == namespace
    )
    if len(matches) != 1:
        raise invalid("tool_choice")
    return {"type": "function", "function": {"name": matches[0].upstream_name}}
