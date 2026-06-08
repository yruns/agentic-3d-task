"""``CodexAgentRuntime`` — a task-agnostic driver for the Codex Agent SDK.

The runtime owns everything that is specific to *running a Codex turn* and
nothing about any particular task:

* isolating ``CODEX_HOME`` per turn (the Codex app-server keeps SQLite state in
  ``CODEX_HOME``; sharing one home across thread workers races its migrations),
* wiring ModelHub prefix-cache + log-id headers for prompt-prefix reuse,
* attaching skills / prompt / images and requesting a structured answer,
* issuing a single finalization re-ask when the answer is unusable.

Task logic is injected through the :class:`codex_agent.tasks.base.CodexTask`
protocol, so the same runtime can drive NR3D visual grounding today and other
task families later.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger

from .config import CodexAgentConfig, sanitize_session_id
from .errors import CodexConfigError, CodexTurnError
from .models import (
    CodexTaskResult,
    CodexTurnMetadata,
    CodexTurnRequest,
    CodexTurnResult,
)
from .tasks.base import CodexTask, OutcomeT_co

#: Env vars the Codex config provider reads for ModelHub prefix-cache headers.
MODELHUB_EXTRA_HEADER_ENV = "CODEX_AGENT_MODELHUB_EXTRA_HEADER"
MODELHUB_LOGID_ENV = "CODEX_AGENT_MODELHUB_LOGID"

#: Files copied from the base ``CODEX_HOME`` into each per-turn run home.
_RUN_HOME_SEED_FILES: tuple[str, ...] = (
    "config.toml",
    "installation_id",
    ".personality_migration",
)


class CodexAgentRuntime:
    """Drive single Codex Agent SDK turns and parse them into task outcomes."""

    def __init__(self, config: CodexAgentConfig | None = None) -> None:
        self.config = config or CodexAgentConfig()

    def execute(self, task: CodexTask[OutcomeT_co]) -> CodexTaskResult[OutcomeT_co]:
        """Run ``task`` end to end: build a turn, run it, parse the outcome."""
        request = task.build_turn_request()
        turn = self.run_turn(request, response_validator=task.is_valid_response)
        outcome = task.parse_response(turn.final_response)
        return CodexTaskResult(task_name=task.task_name, outcome=outcome, turn=turn)

    def run_turn(
        self,
        request: CodexTurnRequest,
        *,
        response_validator: Callable[[str], bool] | None = None,
        max_finalization_retries: int = 1,
    ) -> CodexTurnResult:
        """Run one Codex turn and return its raw final response + metadata.

        Args:
            request: The prompt, attachments, and output schema for the turn.
            response_validator: Optional predicate; when it rejects the model's
                answer, the runtime issues a finalization re-ask on the same
                thread (so prior context is retained).
            max_finalization_retries: Maximum number of finalization re-asks.

        Raises:
            CodexConfigError: Invalid arguments, missing attachments, or the
                ``openai-codex`` dependency not being installed.
            CodexTurnError: The turn completed without a final response.
        """
        if max_finalization_retries < 0:
            raise CodexConfigError("max_finalization_retries must be non-negative")
        self._validate_request(request)
        codex = self._import_codex()
        run_home = self._prepare_run_home()
        attempts: list[Any] = []
        try:
            with codex.Codex(
                config=codex.CodexConfig(
                    config_overrides=self._build_config_overrides(),
                    cwd=str(self.config.project_root),
                    env=self._build_turn_env(run_home),
                )
            ) as client:
                sandbox = self._sandbox(codex)
                thread = client.thread_start(
                    model=self.config.model,
                    model_provider=self.config.model_provider,
                    sandbox=sandbox,
                    cwd=str(self.config.project_root),
                )
                output_schema = dict(request.output_schema)
                result = thread.run(
                    self._build_turn_input(codex, request),
                    cwd=str(self.config.project_root),
                    output_schema=output_schema,
                    sandbox=sandbox,
                )
                attempts.append(result)
                retries = 0
                while (
                    response_validator is not None
                    and retries < max_finalization_retries
                    and not self._response_ok(result.final_response, response_validator)
                ):
                    logger.info(
                        "codex turn response unusable; finalization re-ask {}/{}",
                        retries + 1,
                        max_finalization_retries,
                    )
                    result = thread.run(
                        [
                            codex.TextInput(
                                self._finalization_prompt(result.final_response)
                            )
                        ],
                        cwd=str(self.config.project_root),
                        output_schema=output_schema,
                        sandbox=sandbox,
                    )
                    attempts.append(result)
                    retries += 1
        finally:
            self._cleanup_run_home(run_home)

        final_response = result.final_response
        if final_response is None:
            raise CodexTurnError(
                "Codex turn completed without a final_response; "
                f"status={_status_str(result.status)} error={result.error!r}"
            )
        return CodexTurnResult(
            final_response=str(final_response),
            metadata=self._build_metadata(result, attempts, run_home),
        )

    def _validate_request(self, request: CodexTurnRequest) -> None:
        for skill in request.skills:
            if not skill.path.exists():
                raise CodexConfigError(
                    f"Codex skill file is missing: {skill.name} at {skill.path}"
                )
        for image in request.image_paths:
            if not image.exists():
                raise CodexConfigError(f"Codex turn image is missing: {image}")

    def _build_turn_input(
        self, codex: ModuleType, request: CodexTurnRequest
    ) -> list[Any]:
        items: list[Any] = [
            codex.SkillInput(name=skill.name, path=str(skill.path))
            for skill in request.skills
        ]
        items.append(codex.TextInput(request.prompt))
        items.extend(
            codex.LocalImageInput(path=str(path)) for path in request.image_paths
        )
        return items

    def _prepare_run_home(self) -> Path:
        """Create an isolated ``CODEX_HOME`` copy for one app-server process."""
        home = self.config.codex_home
        if not home.exists():
            raise CodexConfigError(f"CODEX_HOME does not exist: {home}")
        if not (home / "config.toml").exists():
            raise CodexConfigError(
                f"CODEX_HOME is missing config.toml: {home / 'config.toml'}"
            )
        run_home = home / "runs" / uuid.uuid4().hex
        if run_home.exists():
            shutil.rmtree(run_home)
        run_home.mkdir(parents=True, exist_ok=True)
        for name in _RUN_HOME_SEED_FILES:
            source = home / name
            if source.exists():
                shutil.copy2(source, run_home / name)
        return run_home

    def _cleanup_run_home(self, run_home: Path) -> None:
        if self.config.keep_run_home:
            return
        runs_root = self.config.codex_home / "runs"
        try:
            run_home.relative_to(runs_root)
        except ValueError:
            return
        _remove_tree_best_effort(run_home)

    def _build_turn_env(self, run_home: Path) -> dict[str, str]:
        env: dict[str, str] = {"CODEX_HOME": str(run_home)}
        if not self.config.enable_prefix_cache:
            return env
        chat_run_id = sanitize_session_id(
            f"{self.config.prefix_cache_session_id}_{uuid.uuid4().hex[:12]}"
        )
        extra = {
            "session_id": self.config.prefix_cache_session_id,
            "source": "codex_agent_sdk",
            "chat_run_id": chat_run_id,
        }
        env[MODELHUB_EXTRA_HEADER_ENV] = json.dumps(
            extra, ensure_ascii=False, separators=(",", ":")
        )
        env[MODELHUB_LOGID_ENV] = f"codexsdk_{chat_run_id}_{int(time.time() * 1000)}"
        return env

    def _build_config_overrides(self) -> tuple[str, ...]:
        if not self.config.enable_prefix_cache:
            return ()
        provider_key = f"model_providers.{_toml_key_part(self.config.model_provider)}"
        return (
            f"{provider_key}.env_http_headers.extra="
            f"{_toml_literal(MODELHUB_EXTRA_HEADER_ENV)}",
            f"{provider_key}.env_http_headers.X-TT-LOGID="
            f"{_toml_literal(MODELHUB_LOGID_ENV)}",
        )

    def _sandbox(self, codex: ModuleType) -> Any:
        # Boundary to the untyped Codex SDK enum; members match SandboxMode.
        return getattr(codex.Sandbox, self.config.sandbox)

    def _finalization_prompt(self, previous_response: str | None) -> str:
        base = (
            "The previous turn did not return the required structured final "
            "answer. Using the evidence already present in this thread, return "
            "only one JSON object that matches the requested output schema. Do "
            "not include markdown, prose, or tool commands outside the JSON."
        )
        preview = _truncate(previous_response or "", 600)
        if preview:
            return f"{base}\nPrevious non-JSON response: {preview}"
        return base

    def _build_metadata(
        self, result: Any, attempts: Sequence[Any], run_home: Path
    ) -> CodexTurnMetadata:
        return CodexTurnMetadata(
            turn_id=result.id,
            status=_status_str(result.status),
            duration_ms=result.duration_ms,
            usage=_dump_usage(result.usage),
            run_home=str(run_home),
            attempts=tuple(
                {
                    "turn_id": item.id,
                    "status": _status_str(item.status),
                    "duration_ms": item.duration_ms,
                    "has_final_response": bool(item.final_response),
                }
                for item in attempts
            ),
        )

    @staticmethod
    def _response_ok(text: str | None, validator: Callable[[str], bool]) -> bool:
        if not text:
            return False
        return bool(validator(text))

    @staticmethod
    def _import_codex() -> ModuleType:
        try:
            import openai_codex
        except ImportError as exc:
            raise CodexConfigError(
                "openai-codex is required for CodexAgentRuntime. Install it with "
                "`uv pip install openai-codex` or `uv pip install -e '.[codex]'`."
            ) from exc
        return openai_codex


def _remove_tree_best_effort(path: Path, *, attempts: int = 4) -> None:
    """Delete a per-turn run home, tolerating the Codex app-server cleanup race.

    The Codex app-server clones plugins into ``<run_home>/.tmp`` and may still be
    finalizing that directory as the turn returns, which makes a single
    ``shutil.rmtree`` fail with ``Directory not empty``. Removing the scratch
    home is housekeeping, so retry briefly and then log instead of failing an
    otherwise-successful turn.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if attempt == attempts - 1:
                logger.warning(
                    "could not remove codex run home {} after {} attempts: {}",
                    path,
                    attempts,
                    exc,
                )
                return
            time.sleep(0.25 * (attempt + 1))


def _status_str(status: Any) -> str | None:
    if status is None:
        return None
    return str(getattr(status, "value", status))


def _dump_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump(mode="json", exclude_none=True)
        return dict(dumped) if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(usage, Mapping):
        return dict(usage)
    return {"value": str(usage)}


def _truncate(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _toml_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_key_part(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "CodexAgentRuntime",
    "MODELHUB_EXTRA_HEADER_ENV",
    "MODELHUB_LOGID_ENV",
]
