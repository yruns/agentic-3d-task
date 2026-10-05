"""Serve SCOPE's non-streaming chat requests through a local ModelHub bridge.

Credentials belong in a private JSON config, never in this script or the ledger.
The upstream HTTP client and metadata sink are injected for offline verification.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated, Literal, Protocol
from uuid import uuid4

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from scripts.scope.prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )
elif __package__:
    from .prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )
else:
    from prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )

MAX_REQUEST_BYTES = 16 * 1024 * 1024
SOURCE_MODEL = "gpt-4o-2024-11-20"
TARGET_MODEL: Literal["gemini-3.8-flash"] = "gemini-3.8-flash"
RequestModel = Literal["gpt-4o-2024-11-20", "gemini-3.8-flash"]
ErrorType = Literal[
    "invalid_request_error", "authentication_error", "upstream_error", "bridge_error"
]


class ThinkingConfig(BaseModel):
    """ModelHub-specific thinking controls."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    budget_tokens: int = Field(default=2000, ge=0, le=32768)
    include_thoughts: bool = True


class BridgeConfig(BaseModel):
    """Validate the private configuration without disclosing its values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    endpoint: str
    api_key: SecretStr = Field(min_length=1)
    model: Literal["gemini-3.8-flash"] = TARGET_MODEL
    bridge_token: SecretStr = Field(min_length=1)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    ledger_path: Path | None = None
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, endpoint: str) -> str:
        TypeAdapter(AnyHttpUrl).validate_python(endpoint)
        try:
            parsed = httpx.URL(endpoint)
        except httpx.InvalidURL as exc:
            raise ValueError("endpoint_invalid") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("endpoint_invalid")
        return endpoint


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str


class ImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        if url.startswith("data:image/") and ";base64," in url:
            return url
        TypeAdapter(AnyHttpUrl).validate_python(url)
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL as exc:
            raise ValueError("image_url_invalid") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("image_url_invalid")
        return url


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["image_url"]
    image_url: ImageReference


ContentPart = Annotated[TextContent | ImageContent, Field(discriminator="type")]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "developer", "user", "assistant"]
    content: str | Annotated[list[ContentPart], Field(min_length=1)]


class ChatCompletionRequest(BaseModel):
    """SCOPE's supported OpenAI request fields; unknown inputs fail closed."""

    model_config = ConfigDict(extra="forbid", strict=True)
    model: RequestModel
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)


class TokenCounts(BaseModel):
    """Only numeric billing metadata is eligible for the local ledger."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_prompt_tokens: int | None = Field(default=None, ge=0)


class PromptTokenDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)
    cached_tokens: int | None = Field(default=None, ge=0)


class CompletionTokenDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class ProviderUsage(TokenCounts):
    """Extract billing counters without modifying the raw provider usage."""

    prompt_tokens_details: PromptTokenDetails | None = None
    completion_tokens_details: CompletionTokenDetails | None = None

    def ledger_counts(self) -> TokenCounts:
        reasoning_tokens = self.reasoning_tokens
        if reasoning_tokens is None and self.completion_tokens_details is not None:
            reasoning_tokens = self.completion_tokens_details.reasoning_tokens
        cached_prompt_tokens = self.cached_prompt_tokens
        if cached_prompt_tokens is None and self.prompt_tokens_details is not None:
            cached_prompt_tokens = self.prompt_tokens_details.cached_tokens
        return TokenCounts(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )


@dataclass(frozen=True)
class CompletionValidationContext:
    """Permit an empty set only when the original candidate list is empty."""

    allows_empty_object_prefilter: bool


class CompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    role: Literal["assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, content: str, info: ValidationInfo) -> str:
        if not content.strip():
            context = info.context
            if not (
                isinstance(context, CompletionValidationContext)
                and context.allows_empty_object_prefilter
            ):
                raise ValueError("completion_content_empty")
        return content


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    index: int | None = Field(default=None, ge=0)
    message: CompletionMessage
    finish_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_empty_answer_finished(self) -> CompletionChoice:
        if not self.message.content.strip() and self.finish_reason != "stop":
            raise ValueError("empty_prefilter_response_not_finished")
        return self


class CompletionEnvelope(BaseModel):
    """Validate success while retaining the provider's raw response fields."""

    model_config = ConfigDict(extra="allow", strict=True)
    model: str = Field(min_length=1)
    choices: list[CompletionChoice] = Field(min_length=1)
    usage: ProviderUsage | None = None
    id: str | None = None
    created: int | None = Field(default=None, ge=0)


class LedgerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model: str
    elapsed_ms: float = Field(ge=0)
    status: int = Field(ge=100, le=599)
    tokens: TokenCounts = Field(default_factory=TokenCounts)
    thinking_budget_tokens: int = Field(ge=0)
    response_id_synthesized: bool = False
    metadata_synthesized: bool = False


class RecordSink(Protocol):
    def record(self, record: LedgerRecord) -> None:
        """Persist call metadata; failures must be reported to the caller."""


class SuppressNetworkLogs(logging.Filter):
    """HTTP-library diagnostics can include the upstream query credential."""

    def filter(self, record: logging.LogRecord) -> bool:
        return False


def suppress_network_logs() -> None:
    for name in (
        "httpx",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        logger = logging.getLogger(name)
        if not any(
            isinstance(existing, SuppressNetworkLogs) for existing in logger.filters
        ):
            logger.addFilter(SuppressNetworkLogs())


class JsonlLedger:
    """Append metadata atomically across request threads with private permissions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def record(self, record: LedgerRecord) -> None:
        payload = record.model_dump_json(exclude_none=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as ledger_file:
                ledger_file.write(payload)


@dataclass(frozen=True)
class BridgeResponse:
    status_code: int
    body: bytes


def error_response(status: int, code: str, error_type: ErrorType) -> BridgeResponse:
    """Use fixed error codes, never exception text or upstream response bodies."""
    return BridgeResponse(
        status,
        json.dumps(
            {
                "error": {
                    "message": code,
                    "type": error_type,
                    "param": None,
                    "code": code,
                }
            }
        ).encode("utf-8"),
    )


class BridgeService:
    """Translate and forward one request; credentials remain inside this boundary."""

    def __init__(
        self, config: BridgeConfig, *, client: httpx.Client, ledger: RecordSink
    ) -> None:
        suppress_network_logs()
        self.config = config
        self._client = client
        self._ledger = ledger

    def close(self) -> None:
        self._client.close()

    def record_failure(
        self, status: int, code: str, error_type: ErrorType
    ) -> BridgeResponse:
        response = error_response(status, code, error_type)
        return self._record(
            response, self.config.model, time.perf_counter(), TokenCounts()
        )

    def _record(
        self,
        response: BridgeResponse,
        model: str,
        started: float,
        tokens: TokenCounts,
        *,
        response_id_synthesized: bool = False,
        metadata_synthesized: bool = False,
    ) -> BridgeResponse:
        record = LedgerRecord(
            model=model,
            elapsed_ms=max(0.0, (time.perf_counter() - started) * 1000),
            status=response.status_code,
            tokens=tokens,
            thinking_budget_tokens=self.config.thinking.budget_tokens,
            response_id_synthesized=response_id_synthesized,
            metadata_synthesized=metadata_synthesized,
        )
        try:
            self._ledger.record(record)
        except OSError:
            return error_response(500, "ledger_write_failed", "bridge_error")
        return response

    def forward(
        self,
        request: ChatCompletionRequest,
        *,
        supports_empty_object_prefilter: bool = False,
    ) -> BridgeResponse:
        started = time.perf_counter()
        request_id = uuid4().hex
        payload = request.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        payload["model"] = self.config.model
        payload["stream"] = False
        payload["thinking"] = self.config.thinking.model_dump()
        endpoint = httpx.URL(self.config.endpoint).copy_set_param(
            "ak", self.config.api_key.get_secret_value()
        )
        try:
            upstream = self._client.post(
                endpoint,
                json=payload,
                headers={"X-TT-LOGID": request_id},
                timeout=self.config.timeout_seconds,
            )
        except httpx.TimeoutException:
            response = error_response(504, "upstream_timeout", "upstream_error")
            return self._record(response, self.config.model, started, TokenCounts())
        except httpx.RequestError:
            response = error_response(
                502, "upstream_connection_failed", "upstream_error"
            )
            return self._record(response, self.config.model, started, TokenCounts())

        if upstream.status_code != 200:
            status = 429 if upstream.status_code == 429 else 502
            response = error_response(status, "upstream_http_error", "upstream_error")
            return self._record(response, self.config.model, started, TokenCounts())
        try:
            completion = CompletionEnvelope.model_validate_json(
                upstream.content,
                context=CompletionValidationContext(
                    supports_empty_object_prefilter
                    and allows_empty_object_prefilter_response(payload["messages"])
                ),
            )
            provider_payload = TypeAdapter(dict[str, JsonValue]).validate_json(
                upstream.content
            )
            if "error" in provider_payload:
                return self._record(
                    error_response(502, "upstream_invalid_response", "upstream_error"),
                    self.config.model,
                    started,
                    TokenCounts(),
                )
        except ValidationError:
            response = error_response(
                502, "upstream_invalid_response", "upstream_error"
            )
            return self._record(response, self.config.model, started, TokenCounts())

        synthesized = "id" not in provider_payload
        metadata_missing = synthesized or "created" not in provider_payload
        if synthesized:
            provider_payload["id"] = "chatcmpl-" + request_id
        if "created" not in provider_payload:
            provider_payload["created"] = int(time.time())
        provider_choices = provider_payload["choices"]
        if isinstance(provider_choices, list):
            for index, choice in enumerate(provider_choices):
                if isinstance(choice, dict) and "index" not in choice:
                    choice["index"] = index
                    metadata_missing = True
        body = (
            json.dumps(provider_payload, ensure_ascii=False).encode("utf-8")
            if metadata_missing
            else upstream.content
        )
        return self._record(
            BridgeResponse(200, body),
            completion.model,
            started,
            (
                completion.usage.ledger_counts()
                if completion.usage is not None
                else TokenCounts()
            ),
            response_id_synthesized=synthesized,
            metadata_synthesized=metadata_missing,
        )


class BridgeServer(ThreadingHTTPServer):
    """Wait for active request threads before closing the injected HTTP client."""

    def __init__(self, address: tuple[str, int], service: BridgeService) -> None:
        self.service = service
        super().__init__(address, BridgeRequestHandler)


class BridgeRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        # BaseHTTPRequestHandler includes request text in its access log.
        return

    def _service(self) -> BridgeService:
        assert isinstance(self.server, BridgeServer)
        return self.server.service

    def _send(self, response: BridgeResponse) -> None:
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response.body)

    def _fail(
        self, status: int, code: str, error_type: ErrorType = "invalid_request_error"
    ) -> None:
        self._send(self._service().record_failure(status, code, error_type))

    def do_GET(self) -> None:
        if self.path != "/health":
            self._fail(404, "route_not_found")
            return
        self._send(
            BridgeResponse(
                200, json.dumps({"status": "ok", "model": TARGET_MODEL}).encode("utf-8")
            )
        )

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._fail(404, "route_not_found")
            return
        authorization = self.headers.get("Authorization", "")
        expected = "Bearer " + self._service().config.bridge_token.get_secret_value()
        if not hmac.compare_digest(
            authorization.encode("utf-8"), expected.encode("utf-8")
        ):
            self._fail(401, "bridge_authentication_failed", "authentication_error")
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._fail(400, "transfer_encoding_unsupported")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self._fail(411, "content_length_required")
            return
        try:
            content_length = int(lengths[0])
        except ValueError:
            self._fail(400, "content_length_invalid")
            return
        if content_length < 0:
            self._fail(400, "content_length_invalid")
            return
        if content_length > MAX_REQUEST_BYTES:
            self._fail(413, "request_too_large")
            return
        self.connection.settimeout(30.0)
        try:
            body = self.rfile.read(content_length)
        except TimeoutError:
            self._fail(408, "request_read_timeout")
            return
        if len(body) != content_length:
            self._fail(400, "request_body_incomplete")
            return
        try:
            request = ChatCompletionRequest.model_validate_json(body)
        except ValidationError:
            self._fail(400, "request_invalid")
            return
        self._send(
            self._service().forward(
                request,
                supports_empty_object_prefilter=(
                    self.headers.get(EMPTY_PREFILTER_HEADER) == "1"
                ),
            )
        )


def load_config(path: Path) -> BridgeConfig:
    """Read a private configuration; callers sanitize validation diagnostics."""
    return BridgeConfig.model_validate_json(path.read_bytes())


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port_must_be_integer") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port_out_of_range")
    return port


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", default=8788, type=parse_port)
    parser.add_argument("--ledger", type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
    except (OSError, ValidationError, httpx.InvalidURL):
        print("bridge_config_invalid", file=sys.stderr)
        return 2
    ledger_path = (
        arguments.ledger
        or config.ledger_path
        or arguments.config.with_suffix(".requests.jsonl")
    )
    service = BridgeService(
        config, client=httpx.Client(), ledger=JsonlLedger(ledger_path)
    )
    try:
        with BridgeServer((arguments.host, arguments.port), service) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except OSError:
        print("bridge_server_failed", file=sys.stderr)
        return 1
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
