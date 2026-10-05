"""Verify exact image URI transport caching without decoding image payloads."""

from __future__ import annotations

import hashlib
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from scripts.scope.modelhub_image_cache import (
    ImageCacheMissError,
    ImageCacheStore,
    image_cache_uri,
    is_cache_uri,
)

_DATA_URI_PREFIX = "data:image/png;base64,"
_ORIGINAL_DATA_URI = _DATA_URI_PREFIX + "bm90IGFuIGltYWdl\nΔ"
_MAX_DATA_URI_BYTES = 16 * 1024 * 1024


def _expected_cache_uri(data_uri: str) -> str:
    return "scope-cache://" + hashlib.sha256(data_uri.encode("utf-8")).hexdigest()


def _exercise_concurrent_cache(
    store: ImageCacheStore, data_uri: str, barrier: Barrier
) -> tuple[str, ...]:
    uri = image_cache_uri(data_uri)
    barrier.wait(timeout=10)
    restored: list[str] = []
    for _ in range(8):
        store.put(uri, data_uri)
        restored.append(store.get(uri))
    return tuple(restored)


def test_cache_uri_hashes_complete_original_utf8_data_uri() -> None:
    uri = image_cache_uri(_ORIGINAL_DATA_URI)

    assert uri == _expected_cache_uri(_ORIGINAL_DATA_URI)
    assert is_cache_uri(uri)


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "scope-cache://" + "a" * 63,
        "scope-cache://" + "a" * 65,
        "scope-cache://" + "A" * 64,
        "scope-cache://" + "g" * 64,
        "SCOPE-CACHE://" + "a" * 64,
        " scope-cache://" + "a" * 64,
        "scope-cache://" + "a" * 64 + "/",
        "scope-cache://" + "a" * 64 + "?query=1",
        "https://example.com/" + "a" * 64,
    ],
)
def test_invalid_cache_uri_is_rejected_for_reads_and_writes(
    tmp_path: Path, invalid_uri: str
) -> None:
    store = ImageCacheStore(tmp_path / "cache")

    assert not is_cache_uri(invalid_uri)
    with pytest.raises(ValueError):
        store.put(invalid_uri, _ORIGINAL_DATA_URI)
    with pytest.raises(ValueError):
        store.get(invalid_uri)


def test_cache_preserves_original_bytes_and_private_disk_permissions(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    store = ImageCacheStore(cache_root)
    uri = image_cache_uri(_ORIGINAL_DATA_URI)

    store.put(uri, _ORIGINAL_DATA_URI)

    assert store.get(uri) == _ORIGINAL_DATA_URI
    payload_files = [
        path
        for path in cache_root.rglob("*")
        if path.is_file() and path.read_bytes() == _ORIGINAL_DATA_URI.encode("utf-8")
    ]
    assert len(payload_files) == 1
    assert stat.S_IMODE(payload_files[0].stat().st_mode) == 0o600


def test_missing_valid_cache_key_raises_cache_miss(tmp_path: Path) -> None:
    store = ImageCacheStore(tmp_path / "cache")

    with pytest.raises(ImageCacheMissError):
        store.get(image_cache_uri(_ORIGINAL_DATA_URI))


def test_hash_mismatch_does_not_replace_an_existing_value(tmp_path: Path) -> None:
    store = ImageCacheStore(tmp_path / "cache")
    uri = image_cache_uri(_ORIGINAL_DATA_URI)
    store.put(uri, _ORIGINAL_DATA_URI)
    different_data_uri = _DATA_URI_PREFIX + "a different original payload"

    with pytest.raises(ValueError):
        store.put(uri, different_data_uri)

    assert store.get(uri) == _ORIGINAL_DATA_URI
    with pytest.raises(ImageCacheMissError):
        store.get(image_cache_uri(different_data_uri))


@pytest.mark.parametrize(
    "invalid_data_uri",
    [
        "",
        "https://example.com/image.png",
        "data:text/plain;base64,YQ==",
        "data:image/;base64,YQ==",
        "data:image/png,YQ==",
        "data:image/png;base64YQ==",
        "data:image/png;base64,",
    ],
)
def test_malformed_or_empty_data_uri_is_rejected(
    tmp_path: Path, invalid_data_uri: str
) -> None:
    store = ImageCacheStore(tmp_path / "cache")

    with pytest.raises(ValueError):
        image_cache_uri(invalid_data_uri)
    with pytest.raises(ValueError):
        store.put(_expected_cache_uri(invalid_data_uri), invalid_data_uri)


def test_full_utf8_uri_limit_accepts_16_mib_and_rejects_one_extra_byte(
    tmp_path: Path,
) -> None:
    store = ImageCacheStore(tmp_path / "cache")
    payload_bytes = _MAX_DATA_URI_BYTES - len(_DATA_URI_PREFIX.encode("utf-8"))
    data_uri = _DATA_URI_PREFIX + "Δ" * (payload_bytes // 2)
    assert len(data_uri.encode("utf-8")) == _MAX_DATA_URI_BYTES
    uri = image_cache_uri(data_uri)

    store.put(uri, data_uri)

    assert store.get(uri) == data_uri
    oversized_data_uri = data_uri + "A"
    with pytest.raises(ValueError):
        image_cache_uri(oversized_data_uri)
    with pytest.raises(ValueError):
        store.put(_expected_cache_uri(oversized_data_uri), oversized_data_uri)
    assert store.get(uri) == data_uri


def test_concurrent_same_key_reads_and_writes_never_observe_partial_payloads(
    tmp_path: Path,
) -> None:
    store = ImageCacheStore(tmp_path / "cache")
    uri = image_cache_uri(_ORIGINAL_DATA_URI)
    workers = 8
    barrier = Barrier(workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _exercise_concurrent_cache, store, _ORIGINAL_DATA_URI, barrier
            )
            for _ in range(workers)
        ]
        restored = tuple(future.result(timeout=10) for future in futures)

    assert restored == ((_ORIGINAL_DATA_URI,) * 8,) * workers
    assert store.get(uri) == _ORIGINAL_DATA_URI


def test_lru_capacity_refreshes_recency_on_get(tmp_path: Path) -> None:
    data_uris = tuple(
        _DATA_URI_PREFIX + payload for payload in ("YQ==", "Yg==", "Yw==")
    )
    first, second, third = tuple(image_cache_uri(value) for value in data_uris)
    item_size = len(data_uris[0].encode("utf-8"))
    store = ImageCacheStore(tmp_path / "cache", max_bytes=2 * item_size)
    store.put(first, data_uris[0])
    store.put(second, data_uris[1])
    assert store.get(first) == data_uris[0]

    store.put(third, data_uris[2])

    with pytest.raises(ImageCacheMissError):
        store.get(second)
    assert store.get(first) == data_uris[0]
    assert store.get(third) == data_uris[2]


def test_item_larger_than_capacity_is_rejected_without_evicting_old_items(
    tmp_path: Path,
) -> None:
    original = _DATA_URI_PREFIX + "YQ=="
    oversized = _DATA_URI_PREFIX + "YWJjZA=="
    original_uri = image_cache_uri(original)
    oversized_uri = image_cache_uri(oversized)
    store = ImageCacheStore(tmp_path / "cache", max_bytes=len(original.encode("utf-8")))
    store.put(original_uri, original)

    with pytest.raises(ValueError):
        store.put(oversized_uri, oversized)

    assert store.get(original_uri) == original
    with pytest.raises(ImageCacheMissError):
        store.get(oversized_uri)


def test_reopening_cache_root_restores_existing_payload(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    uri = image_cache_uri(_ORIGINAL_DATA_URI)
    ImageCacheStore(cache_root).put(uri, _ORIGINAL_DATA_URI)

    reopened = ImageCacheStore(cache_root)

    assert reopened.get(uri) == _ORIGINAL_DATA_URI


@pytest.mark.parametrize("invalid_capacity", [0, -1, True])
def test_invalid_capacity_is_rejected(tmp_path: Path, invalid_capacity: int) -> None:
    with pytest.raises(ValueError):
        ImageCacheStore(tmp_path / "cache", max_bytes=invalid_capacity)
