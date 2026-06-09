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
import threading
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

#: Sub-directory of each run home used as an isolated ``HOME`` so the Codex
#: app-server does not discover user/global skills under the real home.
_ISOLATED_HOME_DIRNAME = ".home"

#: Bundled ``.system`` skills the Codex app-server installs into every run home.
#: They are irrelevant to focused task agents and are disabled (via the
#: ``skills/config/write`` RPC) when ``restrict_skills_to_project`` is set, so the
#: model is not tempted to read them. HOME isolation removes the much larger
#: user/global skill set; these five are the only ones that live inside CODEX_HOME.
_BUNDLED_SYSTEM_SKILLS: tuple[str, ...] = (
    "imagegen",
    "openai-docs",
    "plugin-creator",
    "skill-creator",
    "skill-installer",
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
                if self.config.restrict_skills_to_project:
                    self._disable_ambient_system_skills(client)
                sandbox = self._sandbox(codex)
                thread = client.thread_start(
                    model=self.config.model,
                    model_provider=self.config.model_provider,
                    sandbox=sandbox,
                    cwd=str(self.config.project_root),
                )
                output_schema = dict(request.output_schema)
                result = self._run_turn_bounded(
                    thread,
                    self._build_turn_input(codex, request),
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
                    result = self._run_turn_bounded(
                        thread,
                        [
                            codex.TextInput(
                                self._finalization_prompt(
                                    result.final_response, output_schema
                                )
                            )
                        ],
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

    def _run_turn_bounded(
        self,
        thread: Any,
        turn_input: list[Any],
        *,
        output_schema: Mapping[str, Any],
        sandbox: Any,
    ) -> Any:
        """Run one turn, interrupting it if it exceeds ``turn_timeout_s``.

        Without a budget this is a plain blocking ``thread.run``. With a budget,
        the turn is started via ``thread.turn`` (which exposes a handle), and a
        watchdog requests interruption once the wall-clock budget elapses so a
        runaway tool loop cannot stall the whole run. The (partial) result is
        returned; the caller's finalization re-ask then collects a usable answer
        from the evidence already gathered.
        """
        cwd = str(self.config.project_root)
        schema = dict(output_schema)
        timeout = self.config.turn_timeout_s
        if timeout <= 0:
            return thread.run(
                turn_input, cwd=cwd, output_schema=schema, sandbox=sandbox
            )
        handle = thread.turn(turn_input, cwd=cwd, output_schema=schema, sandbox=sandbox)
        interrupted = threading.Event()

        def _interrupt() -> None:
            interrupted.set()
            try:
                handle.interrupt()
            except Exception as exc:  # interruption is best-effort
                logger.warning("codex turn interrupt failed: {}", exc)

        timer = threading.Timer(timeout, _interrupt)
        timer.start()
        try:
            result = handle.run()
        finally:
            timer.cancel()
        if interrupted.is_set():
            logger.warning(
                "codex turn exceeded {}s budget and was interrupted", timeout
            )
        return result

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
        if self.config.restrict_skills_to_project:
            (run_home / _ISOLATED_HOME_DIRNAME).mkdir(parents=True, exist_ok=True)
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
        if self.config.restrict_skills_to_project:
            # Isolate HOME so the app-server cannot discover user/global skills
            # (``~/.agents/skills``, ``~/.codex/superpowers``); only the project's
            # own repo skills remain. Merged onto os.environ by the SDK, so PATH
            # / PYTHONPATH and other inherited vars are preserved.
            env["HOME"] = str(run_home / _ISOLATED_HOME_DIRNAME)
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
        # Focused task agents must not inherit the repo's AGENTS.md (coding-agent
        # rules) — injecting it makes the model behave like a coding agent and
        # loop re-reading project docs instead of solving the task.
        overrides: list[str] = ["project_doc_max_bytes=0"]
        if self.config.enable_prefix_cache:
            provider_key = (
                f"model_providers.{_toml_key_part(self.config.model_provider)}"
            )
            overrides.append(
                f"{provider_key}.env_http_headers.extra="
                f"{_toml_literal(MODELHUB_EXTRA_HEADER_ENV)}"
            )
            overrides.append(
                f"{provider_key}.env_http_headers.X-TT-LOGID="
                f"{_toml_literal(MODELHUB_LOGID_ENV)}"
            )
        if (
            self.config.sandbox == "workspace_write"
            and self.config.sandbox_network_access
        ):
            overrides.append("sandbox_workspace_write.network_access=true")
        if self.config.reasoning_effort:
            overrides.append(f"model_reasoning_effort={self.config.reasoning_effort}")
        return tuple(overrides)

    def _disable_ambient_system_skills(self, client: Any) -> None:
        """Disable the bundled ``.system`` skills for this turn.

        HOME isolation (see :meth:`_build_turn_env`) stops the app-server from
        discovering user/global skills, but it still installs a few bundled
        ``.system`` skills into the run home. We disable them through the
        official ``skills/config/write`` RPC so the model's skill catalog is
        limited to the project's own skills. This is best-effort hardening:
        a failure is logged but never aborts the turn (HOME isolation already
        removes the bulk of the ambient skills).
        """
        raw_client = getattr(client, "_client", None)
        request = getattr(raw_client, "request", None)
        if request is None:
            logger.warning(
                "codex client exposes no request channel; "
                "cannot disable ambient system skills"
            )
            return
        try:
            from openai_codex.generated.v2_all import SkillsConfigWriteResponse
        except ImportError as exc:
            logger.warning("cannot import SkillsConfigWriteResponse: {}", exc)
            return
        for name in _BUNDLED_SYSTEM_SKILLS:
            try:
                request(
                    "skills/config/write",
                    {"enabled": False, "name": name},
                    response_model=SkillsConfigWriteResponse,
                )
            except Exception as exc:  # hardening RPC; degrade with visibility
                logger.warning("could not disable system skill {}: {}", name, exc)

    def _sandbox(self, codex: ModuleType) -> Any:
        # Boundary to the untyped Codex SDK enum; members match SandboxMode.
        return getattr(codex.Sandbox, self.config.sandbox)

    def _finalization_prompt(
        self, previous_response: str | None, output_schema: Mapping[str, Any]
    ) -> str:
        base = (
            "Stop gathering evidence now. Do NOT run any more tools or shell "
            "commands. Using only the evidence already present in this thread, "
            "return one JSON object that matches the requested output schema as "
            "your entire reply — no markdown, prose, or tool commands."
        )
        required = _required_keys(output_schema)
        if required:
            base += f"\nThe JSON object must include these keys: {required}."
        preview = _truncate(previous_response or "", 600)
        if preview:
            return f"{base}\nPrevious non-JSON response: {preview}"
        return base

    def _build_metadata(
        self, result: Any, attempts: Sequence[Any], run_home: Path
    ) -> CodexTurnMetadata:
        input_tokens, cached_input_tokens = _extract_token_counts(result.usage)
        return CodexTurnMetadata(
            turn_id=result.id,
            status=_status_str(result.status),
            duration_ms=result.duration_ms,
            usage=_dump_usage(result.usage),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
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


def _required_keys(output_schema: Mapping[str, Any]) -> list[str]:
    required = output_schema.get("required")
    if isinstance(required, list):
        return [str(key) for key in required]
    return []


def _status_str(status: Any) -> str | None:
    if status is None:
        return None
    return str(getattr(status, "value", status))


def _extract_token_counts(usage: Any) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, cached_input_tokens)`` from a Codex usage object.

    Reads ``usage.last`` (a ``TokenUsageBreakdown``); returns ``(None, None)``
    when the adapter/model did not report usage for the turn.
    """
    last = getattr(usage, "last", None)
    if last is None:
        return None, None
    input_tokens = getattr(last, "input_tokens", None)
    cached_input_tokens = getattr(last, "cached_input_tokens", None)
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(cached_input_tokens) if cached_input_tokens is not None else None,
    )


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
