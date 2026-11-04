"""Tool-free ALFRED actor isolation on the registered native Responses route.

The task-specific catalog controls the SDK's exposed tools, not the model name
or transport. Requests retain the exact registered model and use Responses end
to end. This first actor supports the adapter's Responses protocol explicitly;
other protocols are rejected before the SDK starts.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.provider_catalog import ModelCatalog, build_model_catalog
from codex_agent.provider_environment import toml_literal
from codex_agent.provider_http import ADAPTER_PROVIDER
from codex_agent.provider_profile import AdapterModelProfile, discover_adapter_model
from codex_agent.runtime import CodexAgentRuntime

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - compatibility with the project's Python 3.10 support.
    import tomli as tomllib

_CATALOG_FILENAME = "alfred_actor_catalog.json"
_DISABLED_ACTOR_FEATURES = (
    "shell_tool",
    "view_image",
    "multi_agent",
    "multi_agent_v2",
    "multi_agent_v2_dynamic_tools",
    "agent_message_board",
    "apps",
    "plugins",
    "hooks",
    "plugin_hooks",
    "skill_search",
    "tool_search",
    "tool_suggest",
    "js_repl",
    "code_mode",
    "stable_environment_tools",
    "deferred_executor",
    "external_agent_memory_import",
    "goals",
    "sleep_tool",
    "current_time_reminder",
    "send_message_to_user_async",
)


def build_actor_catalog(profile: AdapterModelProfile) -> ModelCatalog:
    """Preserve registered capabilities while removing native coding tool access.

    The shared catalog builder constructs SDK metadata; using that data structure
    does not invoke its Gemini codec or change the native Responses wire route.
    In SDK 0.163.0, disabling the removed ``apply_patch_freeform`` feature does
    not remove apply_patch. Its model-catalog declaration must be null instead.
    """
    if profile.protocol != "responses":
        raise CodexConfigError(
            "ALFRED tool-free actor requires the native Responses protocol; "
            f"model={profile.model!r} advertises protocol={profile.protocol!r}"
        )
    if not profile.capabilities.images:
        raise CodexConfigError("ALFRED actor requires declared images capability")
    catalog = build_model_catalog(profile)
    model = catalog["models"][0]
    model["shell_type"] = "disabled"
    model["apply_patch_tool_type"] = None
    model["include_skills_usage_instructions"] = False
    model["include_plugin_usage_instructions"] = False
    model["include_apps_usage_instructions"] = False
    model["experimental_supported_tools"] = []
    model["multi_agent_version"] = "disabled"
    model["node_repl_disabled"] = True
    model["tool_mode"] = "direct"
    model["model_messages"] = {
        "instructions_template": (
            "Control the embodied task using only the supplied instructions, "
            "observations and images. Return the requested structured decision. "
            "All actions and evidence queries are executed by the caller."
        )
    }
    return catalog


class AlfredPolicyRuntime(CodexAgentRuntime):
    """Run independent image-only decisions with no host tools or ambient apps.

    One instance serves a sequential episode stream. It is intentionally not
    thread-safe: each turn owns one isolated catalog path until cleanup. Use a
    separate runtime instance for another concurrent episode worker.
    """

    def __init__(self, config: CodexAgentConfig | None = None) -> None:
        super().__init__(config)
        self._actor_catalog_path: Path | None = None

    def _prepare_run_home(self) -> Path:
        if self.config.model_provider != ADAPTER_PROVIDER:
            raise CodexConfigError(
                "ALFRED actor requires model_provider='codex_adapter'"
            )
        if self.config.sandbox != "read_only" or self.config.sandbox_network_access:
            raise CodexConfigError(
                "ALFRED actor requires a read_only sandbox without network access"
            )
        if not self.config.restrict_skills_to_project:
            raise CodexConfigError(
                "ALFRED actor requires isolated HOME/skill discovery"
            )
        if self._actor_catalog_path is not None:
            raise CodexConfigError(
                "ALFRED actor runtime cannot prepare overlapping turns"
            )
        _validate_actor_config(self.config.codex_home / "config.toml")
        profile = discover_adapter_model(self.config)
        catalog = build_actor_catalog(profile)
        run_home = super()._prepare_run_home()
        try:
            # Validate the actual copied config too, so an edit between discovery
            # and copying cannot introduce an inherited MCP server or plugin.
            _validate_actor_config(run_home / "config.toml")
            path = run_home / _CATALOG_FILENAME
            path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            self._actor_catalog_path = path
            return run_home
        except BaseException:
            # The base driver calls this hook before its cleanup try/finally.
            # Remove incomplete homes even when successful turn traces are kept.
            shutil.rmtree(run_home)
            raise

    def _build_config_overrides(self) -> tuple[str, ...]:
        if self._actor_catalog_path is None:
            raise CodexConfigError(
                "ALFRED actor run home must be prepared before SDK configuration"
            )
        return super()._build_config_overrides() + (
            *(f"features.{feature}=false" for feature in _DISABLED_ACTOR_FEATURES),
            "tools.update_plan.enabled=false",
            "tools.experimental_request_user_input.enabled=false",
            'web_search="disabled"',
            "project_doc_max_bytes=0",
            f"model_catalog_json={toml_literal(str(self._actor_catalog_path))}",
        )

    def _cleanup_run_home(self, run_home: Path) -> None:
        try:
            super()._cleanup_run_home(run_home)
        finally:
            self._actor_catalog_path = None


def _validate_actor_config(path: Path) -> None:
    if not path.is_file():
        raise CodexConfigError(f"ALFRED Codex config is missing: {path}")
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    def contains_host_tools(section: Mapping[str, object]) -> bool:
        return any(
            (key in {"mcp_servers", "plugins"} and bool(value))
            or (isinstance(value, dict) and contains_host_tools(value))
            for key, value in section.items()
        )

    if contains_host_tools(document):
        raise CodexConfigError(
            "ALFRED actor requires a Codex home without MCP servers or plugins, "
            "including inherited profiles"
        )
