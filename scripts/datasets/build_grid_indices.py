"""Offline precomputation CLI for depth-verified BEV grid visibility indices.

Walks prepared ConceptGraph scene directories under ``--data-root`` and builds
(or refreshes) ``conceptgraph/indices/grid_visibility_index.pkl`` using each
scene's 3D objects, camera trajectory, intrinsics, and raw depth maps.

Usage::

    python -m scripts.datasets.build_grid_indices \\
        --data-root /home/ysh/Datasets/open-eqa/processed_v1/scannet \\
        --dataset openeqa --stride 5 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from keyframe.keyframe_selector import KeyframeSelector


def discover_conceptgraph_dirs(data_root: Path) -> list[Path]:
    """Return sorted ``conceptgraph/`` directories under ``data_root``."""
    if (data_root / "pcd_saves").is_dir():
        return [data_root]
    if (data_root / "conceptgraph" / "pcd_saves").is_dir():
        return [data_root / "conceptgraph"]
    return sorted(
        path
        for path in data_root.glob("*/conceptgraph")
        if (path / "pcd_saves").is_dir()
    )


def _build_one_scene(
    cg_dir: Path,
    dataset: str,
    stride: int,
    overwrite_grid: bool,
    overwrite_visibility: bool,
) -> str:
    vis_path, grid_path = KeyframeSelector.build_offline_indices(
        cg_dir,
        stride=stride,
        dataset=dataset,
        overwrite_visibility=overwrite_visibility,
        overwrite_grid=overwrite_grid,
    )
    return f"{cg_dir.parent.name}: {vis_path.name}, {grid_path.name}"


def build_indices_for_root(
    data_root: Path,
    *,
    dataset: str = "openeqa",
    stride: int = 5,
    overwrite: bool = False,
    overwrite_visibility: bool = False,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    """Build ``grid_visibility_index.pkl`` for scenes under ``data_root``.

    Returns:
        ``(built_count, skipped_count)``.
    """
    scene_dirs = discover_conceptgraph_dirs(data_root)
    if limit is not None:
        scene_dirs = scene_dirs[:limit]

    todo: list[Path] = []
    skipped = 0
    for cg_dir in scene_dirs:
        vis_path = cg_dir / "indices" / "visibility_index.pkl"
        grid_path = cg_dir / "indices" / "grid_visibility_index.pkl"
        if (
            vis_path.is_file()
            and grid_path.is_file()
            and not overwrite
            and not overwrite_visibility
        ):
            skipped += 1
            continue
        todo.append(cg_dir)

    if not todo:
        return 0, skipped

    built = 0
    if workers <= 1 or len(todo) == 1:
        for cg_dir in todo:
            summary = _build_one_scene(
                cg_dir, dataset, stride, overwrite, overwrite_visibility
            )
            built += 1
            logger.info(f"[build_grid_indices] ({built}/{len(todo)}) {summary}")
        return built, skipped

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _build_one_scene,
                cg_dir,
                dataset,
                stride,
                overwrite,
                overwrite_visibility,
            ): cg_dir
            for cg_dir in todo
        }
        for future in as_completed(futures):
            summary = future.result()
            built += 1
            logger.info(f"[build_grid_indices] ({built}/{len(todo)}) {summary}")

    return built, skipped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build depth-verified BEV grid visibility indices for prepared scenes."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing <scene>/conceptgraph directories.",
    )
    parser.add_argument(
        "--dataset",
        choices=("nr3d", "openeqa"),
        default="openeqa",
        help="Dataset layout preset (default: openeqa).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Trajectory sampling stride matching KeyframeSelector (default: 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild indices/grid_visibility_index.pkl even when it already exists.",
    )
    parser.add_argument(
        "--overwrite-visibility",
        action="store_true",
        help="Also rebuild indices/visibility_index.pkl when it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of scenes to process.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    built, skipped = build_indices_for_root(
        args.data_root,
        dataset=args.dataset,
        stride=args.stride,
        overwrite=args.overwrite,
        overwrite_visibility=args.overwrite_visibility,
        limit=args.limit,
        workers=args.workers,
    )
    print(f"Done: built={built}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
