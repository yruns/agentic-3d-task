"""Capability-driven model catalog used by the actual Codex SDK subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from .config import CodexAgentConfig
from .errors import CodexConfigError
from .models import CodexTurnRequest
from .provider_environment import toml_literal
from .provider_http import ADAPTER_PROVIDER
from .provider_profile import AdapterModelProfile, discover_adapter_model

_CATALOG_FILENAME = "adapter-model-catalog.json"
_TASK_INSTRUCTIONS = (
    "Complete the user's task using the provided evidence, attached skills, and "
    "available tools. Do not assume you are editing source code. Respect the "
    "requested output schema and return only the final structured answer."
)
_DISABLED_GEMINI_FEATURES = (
    "async_compaction",
    "context_management",
    "code_mode",
    "stable_environment_tools",
    "deferred_executor",
    "multi_agent",
    "multi_agent_v2",
    "multi_agent_v2_dynamic_tools",
    "agent_message_board",
    "apps",
    "plugins",
    "hooks",
    "plugin_hooks",
    "goals",
    "tool_search",
    "tool_suggest",
    "skill_search",
    "js_repl",
    "sleep_tool",
    "external_agent_memory_import",
)


class _ReasoningPreset(TypedDict):
    effort: str
    description: str


class _ModelMessages(TypedDict):
    instructions_template: str


class _TruncationPolicy(TypedDict):
    mode: Literal["bytes"]
    limit: int


class CatalogModel(TypedDict):
    slug: str
    display_name: str
    description: None
    default_reasoning_level: str | None
    supported_reasoning_levels: list[_ReasoningPreset]
    shell_type: str
    visibility: str
    supported_in_api: bool
    priority: int
    upgrade: None
    model_messages: _ModelMessages
    include_skills_usage_instructions: bool
    include_plugin_usage_instructions: bool
    include_apps_usage_instructions: bool
    supports_reasoning_summary_parameter: bool
    default_reasoning_summary: str
    support_verbosity: bool
    default_verbosity: None
    apply_patch_tool_type: None
    truncation_policy: _TruncationPolicy
    supports_image_detail_original: bool
    context_window: None
    max_context_window: None
    auto_compact_token_limit: None
    effective_context_window_percent: int
    experimental_supported_tools: list[str]
    input_modalities: list[str]
    node_repl_disabled: bool
    tool_mode: str
    multi_agent_version: str


class ModelCatalog(TypedDict):
    models: list[CatalogModel]


def build_model_catalog(profile: AdapterModelProfile) -> ModelCatalog:
    """Build Gemini metadata without GPT fallback reasoning or automatic compact.

    In pinned core 0.163.0 both context windows and the automatic limit must be
    null. A large sentinel limit would be clamped to 90% of a positive window.
    The real upstream context window remains in the capability document, not in
    this core catalog, because Gemini does not implement Codex compaction.
    """
    capability = profile.capabilities
    model: CatalogModel = {
        "slug": profile.model,
        "display_name": profile.model,
        "description": None,
        "default_reasoning_level": capability.default_reasoning_effort,
        "supported_reasoning_levels": [
            {"effort": effort, "description": effort}
            for effort in capability.reasoning_efforts
        ],
        "shell_type": "unified_exec" if capability.function_tools else "disabled",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "upgrade": None,
        "model_messages": {"instructions_template": _TASK_INSTRUCTIONS},
        "include_skills_usage_instructions": True,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": capability.reasoning_summary,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": False,
        "context_window": None,
        "max_context_window": None,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"] if capability.images else ["text"],
        "node_repl_disabled": True,
        "tool_mode": "direct",
        "multi_agent_version": "disabled",
    }
    return {"models": [model]}


def prepare_provider_catalog(
    config: CodexAgentConfig, run_home: Path, request: CodexTurnRequest
) -> tuple[str, ...]:
    """Validate before SDK startup; write catalogs only into isolated run homes."""
    if config.model_provider != ADAPTER_PROVIDER:
        return ()
    profile = discover_adapter_model(config)
    if request.image_paths and not profile.capabilities.images:
        raise CodexConfigError(
            f"model={profile.model!r} does not declare images support"
        )
    if config.sandbox == "workspace_write" and not profile.capabilities.function_tools:
        raise CodexConfigError(
            f"model={profile.model!r} does not declare function_tools required by tool tasks"
        )
    if profile.protocol == "responses":
        # GPT keeps the SDK's native metadata, never a Gemini/custom fallback.
        return ()
    path = run_home / _CATALOG_FILENAME
    try:
        path.write_text(
            json.dumps(build_model_catalog(profile), ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise CodexConfigError(
            f"Could not write adapter model catalog: path={path}"
        ) from exc
    return _catalog_overrides(profile, path)


def _catalog_overrides(profile: AdapterModelProfile, path: Path) -> tuple[str, ...]:
    capability = profile.capabilities
    return (
        f"model_catalog_json={toml_literal(str(path))}",
        'model_reasoning_summary="none"',
        "model_post_turn_compact_threshold_percent=0",
        'web_search="disabled"',
        'approval_policy="never"',
        "tools.update_plan.enabled=false",
        "tools.experimental_request_user_input.enabled=false",
        f"features.shell_tool={str(capability.function_tools).lower()}",
        f"features.unified_exec={str(capability.function_tools).lower()}",
        f"features.view_image={str(capability.function_tools and capability.images).lower()}",
        *(f"features.{feature}=false" for feature in _DISABLED_GEMINI_FEATURES),
    )
