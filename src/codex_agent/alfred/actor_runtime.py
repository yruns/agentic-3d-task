"""Tool-enabled ALFRED actor runtime on the registered codex_adapter route.

Each ALFRED episode runs as one Codex Agent SDK turn with ``shell`` and
``view_image`` enabled in a ``workspace_write`` sandbox so the agent can invoke
``python -m codex_agent.alfred.tools`` and inspect rendered frames/BEV/crops,
while ambient plugins, MCP servers, web search, and external skills remain
strictly disabled.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from codex_agent.config import DEFAULT_PROJECT_ROOT, CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.provider_catalog import ModelCatalog, build_model_catalog
from codex_agent.provider_environment import toml_literal
from codex_agent.provider_http import ADAPTER_PROVIDER
from codex_agent.provider_profile import AdapterModelProfile, discover_adapter_model
from codex_agent.runtime import CodexAgentRuntime

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

_CATALOG_FILENAME = "alfred_actor_catalog.json"
_DISABLED_AMBIENT_FEATURES = (
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
    "deferred_executor",
    "external_agent_memory_import",
    "goals",
    "sleep_tool",
    "current_time_reminder",
    "send_message_to_user_async",
)


def build_actor_catalog(profile: AdapterModelProfile) -> ModelCatalog:
    """Build SDK model catalog enabling shell + view_image while disabling apply_patch/ambient tools."""
    if not profile.capabilities.images:
        raise CodexConfigError("ALFRED actor requires declared images capability")
    catalog = build_model_catalog(profile)
    model = catalog["models"][0]
    model["apply_patch_tool_type"] = None
    model["include_skills_usage_instructions"] = False
    model["include_plugin_usage_instructions"] = False
    model["include_apps_usage_instructions"] = False
    model["multi_agent_version"] = "disabled"
    model["node_repl_disabled"] = True
    return catalog


class AlfredPolicyRuntime(CodexAgentRuntime):
    """Run one interactive tool-enabled Codex turn per ALFRED episode."""

    def __init__(self, config: CodexAgentConfig | None = None) -> None:
        super().__init__(config)
        self._actor_catalog_path: Path | None = None

    def _prepare_run_home(self) -> Path:
        if self.config.model_provider != ADAPTER_PROVIDER:
            raise CodexConfigError(
                "ALFRED actor requires model_provider='codex_adapter'"
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
            _validate_actor_config(run_home / "config.toml")
            path = run_home / _CATALOG_FILENAME
            path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            self._actor_catalog_path = path
            return run_home
        except BaseException:
            shutil.rmtree(run_home)
            raise

    def _build_turn_env(
        self, run_home: Path, *, session_id: str | None = None
    ) -> dict[str, str]:
        env = super()._build_turn_env(run_home, session_id=session_id)
        src_dir = str((DEFAULT_PROJECT_ROOT / "src").resolve())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
        no_proxy_items = "127.0.0.1,localhost,::1"
        env["no_proxy"] = no_proxy_items
        env["NO_PROXY"] = no_proxy_items
        return env

    def _build_config_overrides(self) -> tuple[str, ...]:
        if self._actor_catalog_path is None:
            raise CodexConfigError(
                "ALFRED actor run home must be prepared before SDK configuration"
            )
        return super()._build_config_overrides() + (
            *(f"features.{feature}=false" for feature in _DISABLED_AMBIENT_FEATURES),
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
