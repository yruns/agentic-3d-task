"""Upload exact data URIs once, then send small concurrent completion requests."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Lock, Semaphore
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from scripts.scope.modelhub_bridge import ChatCompletionRequest, ImageContent
    from scripts.scope.modelhub_image_cache import image_cache_uri
elif __package__:
    from .modelhub_bridge import ChatCompletionRequest, ImageContent
    from .modelhub_image_cache import image_cache_uri
else:
    from modelhub_bridge import ChatCompletionRequest, ImageContent
    from modelhub_image_cache import image_cache_uri


class _UploadLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    concurrency: int = Field(ge=1, le=64)


class ImageUploadError(Exception):
    """A sanitized upload HTTP failure, never a fabricated model response."""

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__("image_upload_failed")
        self.status = status
        self.body = body


class ImageUploadAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    uri: str
    stored_bytes: int = Field(gt=0)


class ImageCachingTransport(httpx.BaseTransport):
    """Bound only image uploads; model requests use the delegate concurrently."""

    def __init__(
        self, delegate: httpx.BaseTransport, *, upload_concurrency: int = 1
    ) -> None:
        limit = _UploadLimit(concurrency=upload_concurrency)
        self._delegate = delegate
        self._uploads = Semaphore(limit.concurrency)
        self._lock = Lock()
        self._uploaded: dict[str, Future[None]] = {}

    def close(self) -> None:
        self._delegate.close()

    def _ensure_uploaded(self, data_uri: str, request: httpx.Request) -> str:
        uri = image_cache_uri(data_uri)
        with self._lock:
            existing = self._uploaded.get(uri)
            if existing is None:
                existing = Future()
                self._uploaded[uri] = existing
                owns_upload = True
            else:
                owns_upload = False
        if not owns_upload:
            existing.result()
            return uri
        try:
            with self._uploads:
                body = data_uri.encode("utf-8")
                headers = httpx.Headers(request.headers)
                headers["Content-Type"] = "text/plain; charset=utf-8"
                headers["Content-Length"] = str(len(body))
                headers.pop("Transfer-Encoding", None)
                digest = uri.removeprefix("scope-cache://")
                upload = httpx.Request(
                    "PUT",
                    request.url.copy_with(path=f"/v1/images/{digest}"),
                    headers=headers,
                    content=body,
                    extensions=request.extensions,
                )
                response = self._delegate.handle_request(upload)
                try:
                    content = response.read()
                    if response.status_code != 200:
                        raise ImageUploadError(response.status_code, content)
                    acknowledged = ImageUploadAcknowledgement.model_validate_json(
                        content
                    )
                    if acknowledged.uri != uri or acknowledged.stored_bytes != len(
                        body
                    ):
                        raise ValueError("image_upload_acknowledgement_mismatch")
                finally:
                    response.close()
            existing.set_result(None)
        except BaseException as error:
            # Wake every waiter even for cancellation; preserve the original error.
            existing.set_exception(error)
            with self._lock:
                self._uploaded.pop(uri, None)
            raise
        return uri

    def _replace_images(
        self, completion: ChatCompletionRequest, request: httpx.Request
    ) -> None:
        for message in completion.messages:
            if isinstance(message.content, str):
                continue
            for part in message.content:
                if isinstance(part, ImageContent) and part.image_url.url.startswith(
                    "data:image/"
                ):
                    part.image_url.url = self._ensure_uploaded(
                        part.image_url.url, request
                    )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return self._delegate.handle_request(request)
        original = request.read()
        for attempt in range(2):
            completion = ChatCompletionRequest.model_validate_json(original)
            try:
                self._replace_images(completion, request)
            except ImageUploadError as error:
                return httpx.Response(error.status, content=error.body, request=request)
            body = completion.model_dump_json(
                exclude_unset=True, exclude_none=True
            ).encode("utf-8")
            headers = httpx.Headers(request.headers)
            headers["Content-Length"] = str(len(body))
            headers.pop("Transfer-Encoding", None)
            cached = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=body,
                extensions=request.extensions,
            )
            response = self._delegate.handle_request(cached)
            if response.status_code != 409 or attempt == 1:
                return response
            # A cache miss is rejected before any provider call. Re-upload once.
            response.read()
            response.close()
            with self._lock:
                for message in completion.messages:
                    if isinstance(message.content, str):
                        continue
                    for part in message.content:
                        if isinstance(part, ImageContent):
                            self._uploaded.pop(part.image_url.url, None)
        raise AssertionError("bounded_cache_retry_unreachable")
