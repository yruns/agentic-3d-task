"""Public model capability descriptions, with no credential or endpoint data."""

from __future__ import annotations

from codex_adapter.domain.json_codec import JsonObject
from codex_adapter.domain.models import RegisteredModel


def describe_model(model: RegisteredModel) -> JsonObject:
    """Describe declared capabilities; this is not an upstream health claim."""
    capability = model.capabilities
    return {
        "model": model.id,
        "protocol": model.protocol,
        "codec_version": model.codec_version,
        "capabilities": {
            "structured_output": capability.structured_output,
            "images": capability.images,
            "function_tools": capability.function_tools,
            "namespace_tools": capability.namespace_tools,
            "parallel_tools": capability.parallel_tools,
            "reasoning_summary": capability.reasoning_summary,
            "reasoning_efforts": list(capability.reasoning_efforts),
            "default_reasoning_effort": capability.default_reasoning_effort,
            "context_window": capability.context_window,
            "compact": model.protocol == "responses",
            "streaming": "native" if model.protocol == "responses" else "buffered_sse",
        },
    }
