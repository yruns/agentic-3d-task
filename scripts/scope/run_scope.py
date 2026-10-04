"""Run pinned upstream SCOPE with an explicit, fail-closed model transport.

The upstream algorithm and prompts remain untouched. Its two global model
clients are replaced after import so transport failures cannot be swallowed by
the upstream fallback paths. No upstream API secret is copied to Linux.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import Unpack

_UPSTREAM_API_MODULES = (
    "src.eval_utils_gpt_goatbench",
    "src.potential_estimation_gpt_goal",
)
# Permit two SDK transport retries, then surface the final error to FailClosed.
_MODEL_TRANSPORT_MAX_RETRIES: Final[int] = 2
# Leave a response margin above the Mac bridge's 180-second upstream timeout.
_MODEL_TRANSPORT_TIMEOUT_SECONDS: Final[float] = 240.0


class BridgeConnection(BaseModel):
    """Validated private connection settings for the Linux loopback bridge."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    endpoint: AnyHttpUrl
    bridge_token: SecretStr
    expected_model: str = Field(min_length=1)


class CompletionPort(Protocol):
    """The non-streaming completion operation used by upstream SCOPE."""

    def create(
        self, **parameters: Unpack[CompletionCreateParamsNonStreaming]
    ) -> ChatCompletion: ...


@dataclass(frozen=True)
class FailClosedCompletions:
    """Abort the CLI on infrastructure failure rather than inventing a score."""

    delegate: CompletionPort
    expected_model: str

    def create(
        self, **parameters: Unpack[CompletionCreateParamsNonStreaming]
    ) -> ChatCompletion:
        try:
            completion = self.delegate.create(**parameters)
        except openai.APIError as error:
            status = getattr(error, "status_code", None)
            description = (
                f"HTTP {status}" if status is not None else type(error).__name__
            )
            raise SystemExit(f"SCOPE model transport failed: {description}") from None
        if completion.model != self.expected_model:
            raise SystemExit("SCOPE model transport returned an unexpected model")
        if (
            not completion.choices
            or not completion.choices[0].message.content
            or not completion.choices[0].message.content.strip()
        ):
            raise SystemExit("SCOPE model transport returned empty assistant content")
        return completion


@dataclass(frozen=True)
class ChatEndpoint:
    """Expose the upstream client's chat.completions access path."""

    completions: FailClosedCompletions


@dataclass(frozen=True)
class ScopeClient:
    """Minimal injected client, without implicit initialization."""

    chat: ChatEndpoint


class ScopeApiModule(Protocol):
    """Concrete client attribute at the external Python-module boundary."""

    client: OpenAI | ScopeClient


@dataclass(frozen=True)
class RunSettings:
    """CLI paths and scene/episode selection for one upstream launch."""

    upstream_root: Path
    cfg_file: Path
    bridge_config: Path
    split: int
    start_ratio: float
    end_ratio: float

    def __post_init__(self) -> None:
        for path in (self.cfg_file, self.bridge_config):
            if not path.is_file():
                raise ValueError(f"required configuration file does not exist: {path}")
        entrypoint = self.upstream_root / "run_goatbench_evaluation.py"
        if not entrypoint.is_file():
            raise ValueError(f"upstream SCOPE entrypoint does not exist: {entrypoint}")
        if not 1 <= self.split <= 10:
            raise ValueError(f"split={self.split}; allowed range is 1..10")
        if not 0 <= self.start_ratio < self.end_ratio <= 1:
            raise ValueError(
                "scene ratios must satisfy 0 <= start_ratio < end_ratio <= 1"
            )


def run_upstream(settings: RunSettings) -> None:
    """Inject the validated bridge client and execute the original entrypoint."""
    connection = BridgeConnection.model_validate_json(
        settings.bridge_config.read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "requested_upstream_model": "gpt-4o-2024-11-20",
                "actual_model": connection.expected_model,
                "split": settings.split,
                "source": str(settings.upstream_root),
            }
        ),
        flush=True,
    )
    sys.path.insert(0, str(settings.upstream_root))
    os.chdir(settings.upstream_root)
    with OpenAI(
        base_url=str(connection.endpoint),
        api_key=connection.bridge_token.get_secret_value(),
        max_retries=_MODEL_TRANSPORT_MAX_RETRIES,
        timeout=_MODEL_TRANSPORT_TIMEOUT_SECONDS,
    ) as client:
        injected = ScopeClient(
            ChatEndpoint(
                FailClosedCompletions(
                    client.chat.completions, expected_model=connection.expected_model
                )
            )
        )
        for module_name in _UPSTREAM_API_MODULES:
            module = cast(ScopeApiModule, importlib.import_module(module_name))
            original_client = module.client
            if not isinstance(original_client, OpenAI):
                raise TypeError(f"unexpected upstream client type in {module_name}")
            original_client.close()
            module.client = injected
        sys.argv = [
            str(settings.upstream_root / "run_goatbench_evaluation.py"),
            "-cf",
            str(settings.cfg_file),
            "--split",
            str(settings.split),
            "--start_ratio",
            str(settings.start_ratio),
            "--end_ratio",
            str(settings.end_ratio),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")


def main() -> None:
    """Parse the CLI and delegate to the upstream execution service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--split", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--start-ratio", type=float, default=0)
    parser.add_argument("--end-ratio", type=float, default=1)
    args = parser.parse_args()
    settings = RunSettings(
        upstream_root=args.upstream_root.resolve(),
        cfg_file=args.cfg_file.resolve(),
        bridge_config=args.bridge_config.resolve(),
        split=args.split,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
    )
    run_upstream(settings)


if __name__ == "__main__":
    main()
