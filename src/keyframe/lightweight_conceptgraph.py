"""Loading of ConceptGraph object maps, with a lightweight sidecar cache.

The original ConceptGraph ``*_post.pkl.gz`` files embed per-detection masks and
full point clouds that can expand to many GB during ``pickle.load``. Keyframe
selection only needs object metadata, boxes, centroids and (optionally) CLIP
features, so this module writes a stripped sidecar cache (``*.light.pkl.gz``)
that is safe to load at high concurrency, and exposes
:func:`load_scene_objects` which returns strongly-typed ``SceneObject`` lists.
"""

from __future__ import annotations

import gzip
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from keyframe.models.scene import SceneObject

LIGHTWEIGHT_FORMAT_VERSION = "conceptgraph_lightweight_objects_v1"

#: Heavy fields stripped from the sidecar cache.
DROP_FIELDS: frozenset[str] = frozenset({"mask", "pcd_np", "pcd_color_np"})

#: Fields kept in the sidecar cache (those consumed by keyframe selection).
KEEP_FIELDS: frozenset[str] = frozenset(
    {
        "bbox_np",
        "class_name",
        "conf",
        "xyxy",
        "num_detections",
        "is_background",
        "clip_ft",
        "image_idx",
    }
)

# A raw ConceptGraph object is an untyped pickle dict; ``object`` (not ``Any``)
# keeps it at arm's length until ``SceneObject.from_conceptgraph`` validates it.
RawObject = dict[str, object]


def load_scene_objects(
    pcd_path: Path,
    *,
    prefer_lightweight: bool = True,
    ensure_lightweight: bool = False,
) -> list[SceneObject]:
    """Load a scene's objects as ``SceneObject`` instances.

    Objects without resolvable geometry (no point cloud, centroid, or bbox) are
    skipped. Object ids are assigned contiguously in load order.
    """
    objects: list[SceneObject] = []
    for raw in load_conceptgraph_objects(
        pcd_path,
        prefer_lightweight=prefer_lightweight,
        ensure_lightweight=ensure_lightweight,
    ):
        scene_object = SceneObject.from_conceptgraph(len(objects), raw)
        if scene_object.centroid is None:
            continue
        objects.append(scene_object)
    return objects


def load_conceptgraph_objects(
    pcd_path: Path,
    *,
    prefer_lightweight: bool = True,
    ensure_lightweight: bool = False,
) -> list[RawObject]:
    """Load the raw object dicts from a ConceptGraph pickle (cache-preferring)."""
    payload = _load_payload(
        pcd_path,
        prefer_lightweight=prefer_lightweight,
        ensure_lightweight=ensure_lightweight,
    )
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{pcd_path} must contain an 'objects' list")
    return [obj for obj in objects if isinstance(obj, dict)]


def lightweight_pcd_path(pcd_path: Path) -> Path:
    """Return the stripped sidecar path for a ConceptGraph object pickle."""
    path = Path(pcd_path)
    name = path.name
    if name.endswith(".light.pkl.gz"):
        return path
    if name.endswith(".pkl.gz"):
        return path.with_name(f"{name[:-7]}.light.pkl.gz")
    return path.with_suffix(f"{path.suffix}.light.pkl.gz")


def ensure_lightweight_conceptgraph_cache(pcd_path: Path) -> Path:
    """Create the stripped sidecar cache if missing (serialized across workers)."""
    return write_lightweight_conceptgraph_cache(pcd_path, overwrite=False)


def write_lightweight_conceptgraph_cache(
    pcd_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a stripped sidecar cache and return its path."""
    return _with_cache_build_lock(
        pcd_path,
        lambda: _write_cache_unlocked(
            pcd_path, output_path=output_path, overwrite=overwrite
        ),
    )


def strip_object(obj: RawObject) -> RawObject:
    """Keep only the fields needed by keyframe selection, plus a centroid."""
    stripped: RawObject = {
        key: value for key, value in obj.items() if key in KEEP_FIELDS
    }
    centroid = _object_centroid(obj)
    if centroid is not None:
        stripped["centroid"] = centroid.astype(np.float32, copy=False)
    return stripped


def _load_payload(
    pcd_path: Path,
    *,
    prefer_lightweight: bool,
    ensure_lightweight: bool,
) -> dict[str, object]:
    path = Path(pcd_path)
    light_path = lightweight_pcd_path(path)
    if prefer_lightweight and ensure_lightweight and not light_path.exists():
        ensure_lightweight_conceptgraph_cache(path)
    if prefer_lightweight and light_path.exists():
        return _load_pickle_gz(light_path)
    return _load_pickle_gz(path)


def _write_cache_unlocked(
    pcd_path: Path,
    *,
    output_path: Path | None,
    overwrite: bool,
) -> Path:
    source_path = Path(pcd_path)
    target_path = output_path or lightweight_pcd_path(source_path)
    if target_path.exists() and not overwrite:
        return target_path

    payload = _load_pickle_gz(source_path)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{source_path} must contain an 'objects' list")

    light_payload: dict[str, object] = {
        "format_version": LIGHTWEIGHT_FORMAT_VERSION,
        "source": str(source_path),
        "dropped_fields": sorted(DROP_FIELDS),
        "objects": [strip_object(obj) for obj in objects if isinstance(obj, dict)],
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    with gzip.open(tmp_path, "wb") as handle:
        pickle.dump(light_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(target_path)
    return target_path


def _with_cache_build_lock(pcd_path: Path, build: Callable[[], Path]) -> Path:
    lock_path = _cache_lock_path(Path(pcd_path))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX platforms
            return build()
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            return build()
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _cache_lock_path(pcd_path: Path) -> Path:
    # Lock next to the cache file (its directory is necessarily writable since
    # that is where the sidecar cache is written).
    cache_path = lightweight_pcd_path(Path(pcd_path))
    return cache_path.with_name(f"{cache_path.name}.lock")


def _object_centroid(obj: RawObject) -> NDArray[np.float64] | None:
    raw_centroid = obj.get("centroid")
    if raw_centroid is not None:
        array = np.asarray(raw_centroid, dtype=np.float64).reshape(-1)
        if array.size >= 3:
            return array[:3]
    for key in ("pcd_np", "bbox_np"):
        value = obj.get(key)
        if value is not None:
            array = np.asarray(value, dtype=np.float64)
            if array.size and array.ndim >= 2 and array.shape[-1] >= 3:
                return array.reshape(-1, array.shape[-1])[:, :3].mean(axis=0)  # type: ignore[no-any-return]
    return None


def _load_pickle_gz(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ConceptGraph pickle: {path}")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dict payload")
    return payload
