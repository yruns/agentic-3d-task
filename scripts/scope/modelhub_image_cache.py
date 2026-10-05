"""Cache original image data URIs without decoding or changing their content.

One store owns one cache directory. Its lock serializes disk/index operations
across request threads; atomic replacement keeps each published entry complete.
Access timestamps preserve LRU ordering when the bridge opens the cache again.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from threading import Lock
from time import time_ns
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_IMAGE_DATA_URI_BYTES: Final[int] = 16 * 1024 * 1024
DEFAULT_CACHE_MAX_BYTES: Final[int] = 4 * 1024 * 1024 * 1024
_CACHE_PREFIX: Final[str] = "scope-cache://"
_CACHE_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMAGE_PREFIX = re.compile(r"data:image/[A-Za-z0-9][A-Za-z0-9.+-]*;base64,")


class ImageCacheMissError(LookupError):
    """A valid cache reference has no currently retained disk entry."""


class _CacheLimits(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, hide_input_in_errors=True)
    max_bytes: int = Field(gt=0)


@dataclass(frozen=True)
class _StoredImage:
    digest: str
    byte_count: int
    accessed_at_ns: int


def _encode_image_data_uri(data_uri: str) -> bytes:
    if not isinstance(data_uri, str):
        raise ValueError("image_data_uri_must_be_string")
    prefix = _IMAGE_PREFIX.match(data_uri)
    if prefix is None or prefix.end() == len(data_uri):
        raise ValueError("image_data_uri_requires_image_base64_prefix_and_payload")
    if len(data_uri) > MAX_IMAGE_DATA_URI_BYTES:
        raise ValueError("image_data_uri_exceeds_16_mib")
    try:
        encoded = data_uri.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("image_data_uri_must_be_valid_utf8") from None
    if len(encoded) > MAX_IMAGE_DATA_URI_BYTES:
        raise ValueError("image_data_uri_exceeds_16_mib")
    return encoded


def image_cache_uri(data_uri: str) -> str:
    """Return the SHA256 reference for the complete, validated UTF-8 data URI."""
    encoded = _encode_image_data_uri(data_uri)
    return _CACHE_PREFIX + hashlib.sha256(encoded).hexdigest()


def is_cache_uri(uri: str) -> bool:
    """Recognize only a complete lowercase SHA256 cache reference."""
    return (
        isinstance(uri, str)
        and uri.startswith(_CACHE_PREFIX)
        and _CACHE_DIGEST.fullmatch(uri[len(_CACHE_PREFIX) :]) is not None
    )


def _cache_digest(uri: str) -> str:
    if not is_cache_uri(uri):
        raise ValueError("image_cache_uri_requires_lowercase_sha256")
    return uri[len(_CACHE_PREFIX) :]


class ImageCacheStore:
    """A bounded disk cache with thread-safe, persisted LRU access ordering.

    Entries are opaque UTF-8 strings, limited to 16 MiB each. The capacity counts
    published entry bytes; a temporary atomic-write file may briefly use one extra
    entry's disk space. Invalid input raises ValueError without exposing images.
    """

    def __init__(self, root: Path, max_bytes: int = DEFAULT_CACHE_MAX_BYTES) -> None:
        if not isinstance(root, Path):
            raise ValueError("image_cache_root_must_be_path")
        try:
            limits = _CacheLimits(max_bytes=max_bytes)
        except ValidationError:
            raise ValueError("image_cache_max_bytes_must_be_positive_integer") from None
        self._root = root
        self._max_bytes = limits.max_bytes
        self._lock = Lock()
        self._entries: OrderedDict[str, _StoredImage] = OrderedDict()
        self._byte_count = 0
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._restore_index()
        self._evict_to_capacity()

    def _restore_index(self) -> None:
        entries: list[_StoredImage] = []
        for path in self._root.iterdir():
            if _CACHE_DIGEST.fullmatch(path.name) is None:
                continue
            status = os.stat(path, follow_symlinks=False)
            if S_ISREG(status.st_mode):
                entries.append(
                    _StoredImage(path.name, status.st_size, status.st_mtime_ns)
                )
        for entry in sorted(
            entries, key=lambda entry: (entry.accessed_at_ns, entry.digest)
        ):
            self._entries[entry.digest] = entry
            self._byte_count += entry.byte_count

    def _write_image(self, digest: str, encoded: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(dir=self._root, prefix=".image-")
        try:
            with os.fdopen(descriptor, "wb") as image_file:
                os.fchmod(image_file.fileno(), 0o600)
                image_file.write(encoded)
                image_file.flush()
                os.fsync(image_file.fileno())
            os.replace(temporary_name, self._root / digest)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def _mark_recent(self, digest: str, byte_count: int) -> None:
        accessed_at_ns = time_ns()
        os.utime(self._root / digest, ns=(accessed_at_ns, accessed_at_ns))
        self._entries[digest] = _StoredImage(digest, byte_count, accessed_at_ns)
        self._entries.move_to_end(digest)

    def _evict_to_capacity(self) -> None:
        while self._byte_count > self._max_bytes:
            oldest = next(iter(self._entries.values()))
            (self._root / oldest.digest).unlink(missing_ok=True)
            del self._entries[oldest.digest]
            self._byte_count -= oldest.byte_count

    def put(self, uri: str, data_uri: str) -> None:
        """Atomically publish matching original content and evict oldest entries.

        Reject an entry larger than the capacity before changing retained data.
        """
        digest = _cache_digest(uri)
        encoded = _encode_image_data_uri(data_uri)
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise ValueError("image_cache_uri_sha256_mismatch")
        if len(encoded) > self._max_bytes:
            raise ValueError("image_cache_entry_exceeds_max_bytes")
        with self._lock:
            self._write_image(digest, encoded)
            previous = self._entries.get(digest)
            if previous is not None:
                self._byte_count -= previous.byte_count
            self._byte_count += len(encoded)
            self._mark_recent(digest, len(encoded))
            self._evict_to_capacity()

    def get(self, uri: str) -> str:
        """Read and verify original content, or raise ImageCacheMissError."""
        digest = _cache_digest(uri)
        with self._lock:
            entry = self._entries.get(digest)
            if entry is None:
                raise ImageCacheMissError("image_cache_miss: " + digest)
            try:
                with (self._root / digest).open("rb") as image_file:
                    encoded = image_file.read(MAX_IMAGE_DATA_URI_BYTES + 1)
            except FileNotFoundError:
                del self._entries[digest]
                self._byte_count -= entry.byte_count
                raise ImageCacheMissError("image_cache_miss: " + digest) from None
            if len(encoded) > MAX_IMAGE_DATA_URI_BYTES:
                raise ValueError("image_cache_disk_entry_exceeds_16_mib")
            if hashlib.sha256(encoded).hexdigest() != digest:
                raise ValueError("image_cache_disk_entry_sha256_mismatch")
            try:
                data_uri = encoded.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("image_cache_disk_entry_must_be_utf8") from None
            _encode_image_data_uri(data_uri)
            self._mark_recent(digest, len(encoded))
            return data_uri
