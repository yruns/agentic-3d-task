"""Write a minimal scored SceneFun3D dataset root for scorer tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codex_agent.scenefunc3d.tests_support import write_binary_ply_mesh


def write_scoring_scene(
    data_root: Path,
    *,
    visit_id: str = "421254",
    desc_id: str = "desc-a",
    annot_id: str = "annot-a",
    gt_point_ids: tuple[int, ...] = (0, 1, 2),
    vertex_count: int = 16,
) -> Path:
    """Write descriptions/motions/annotations + a raw binary mesh PLY."""
    scene_dir = data_root / visit_id
    (scene_dir / "raw").mkdir(parents=True, exist_ok=True)
    (scene_dir / f"{visit_id}_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "descriptions": [
                    {
                        "desc_id": desc_id,
                        "annot_id": annot_id,
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / f"{visit_id}_motions.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": annot_id,
                        "motion_type": "pinch_pull",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / f"{visit_id}_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "annotations": [{"annot_id": annot_id, "indices": list(gt_point_ids)}],
            }
        ),
        encoding="utf-8",
    )
    write_binary_ply_mesh(
        scene_dir / "raw" / "mesh.ply",
        np.random.default_rng(0).random((vertex_count, 3)).astype(np.float32),
    )
    return scene_dir
