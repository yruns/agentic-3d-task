#!/usr/bin/env python3
"""Evaluate NR3D Stage-1 KeyframeSelector target-id frame coverage (hit@K).

Measures whether the top-K keyframes returned by ``KeyframeSelector`` include at
least one frame where the ground-truth ``target_id`` is depth-visible according
to ``conceptgraph/indices/visibility_index.pkl`` (``object_to_views[target_id]``).

Optionally accepts ``--compare-with <baseline_results.json>`` to produce a
paired sample-by-sample comparison against a saved baseline run JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from codex_agent.nr3d.sample import DEFAULT_PACK_NAME, load_sample
from keyframe.keyframe_selector import KeyframeSelector
from keyframe.llm.client import LLMClient


@dataclass(frozen=True)
class CoverageSampleSpec:
    """Input specification for one NR3D sample in the fold."""

    sample_id: str
    scene_id: str
    target_id: int
    category: str
    query: str
    is_easy: bool
    is_view_dep: bool


@dataclass
class SampleCoverageRecord:
    """Per-sample evaluation outcome."""

    sample_id: str
    scene_id: str
    target_id: int
    category: str
    query: str
    is_easy: bool
    is_view_dep: bool
    status: str = "no_evidence"
    pred_frames: list[int] = field(default_factory=list)
    gt_frames: list[int] = field(default_factory=list)
    n_scene_views: int = 0
    pred_target_ids: list[int] = field(default_factory=list)
    target_grids: list[list[int]] = field(default_factory=list)
    has_viewpoint_contexts: bool = False
    hit_at_1: bool = False
    hit_at_2: bool = False
    hit_at_3: bool = False
    recall_at_1: float = 0.0
    recall_at_2: float = 0.0
    recall_at_3: float = 0.0
    first_hit_rank: int | None = None
    target_obj_top1_hit: bool = False
    target_obj_top5_hit: bool = False
    duration_ms: int = 0
    error: str | None = None


def load_fold_specs(
    fold_path: Path, data_root: Path, *, pack_name: str = DEFAULT_PACK_NAME
) -> list[CoverageSampleSpec]:
    """Load sample specifications from a fold JSON file."""
    raw = json.loads(fold_path.read_text(encoding="utf-8"))
    entries = raw["sample_ids"] if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in fold file {fold_path}")

    specs: list[CoverageSampleSpec] = []
    for item in entries:
        if isinstance(item, Mapping):
            sample_id = str(item["sample_id"])
            is_easy = bool(item.get("is_easy", False))
            is_view_dep = bool(item.get("is_view_dep", False))
        else:
            sample_id = str(item)
            is_easy = False
            is_view_dep = False
        sample = load_sample(data_root, sample_id, pack_name=pack_name)
        specs.append(
            CoverageSampleSpec(
                sample_id=sample.sample_id,
                scene_id=sample.scene_id,
                target_id=sample.target_id,
                category=sample.category,
                query=sample.query,
                is_easy=is_easy,
                is_view_dep=is_view_dep,
            )
        )
    return specs


def evaluate_one_sample(
    spec: CoverageSampleSpec,
    selector: KeyframeSelector,
    *,
    k: int = 3,
    use_visual_context: bool = False,
) -> SampleCoverageRecord:
    """Evaluate a single NR3D sample and compute frame/object coverage metrics."""
    t0 = time.monotonic()
    gt_views = [int(vid) for vid, _ in selector.object_to_views.get(spec.target_id, [])]
    gt_set = set(gt_views)
    n_scene_views = max(len(selector.camera_poses), len(selector.view_to_objects), 1)

    try:
        result = selector.select_keyframes_v2(
            spec.query,
            k=k,
            use_visual_context=use_visual_context,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        pred_frames = [int(vid) for vid in result.keyframe_indices[:k]]
        pred_target_ids = [int(o.obj_id) for o in result.target_objects]
        status = result.metadata.status.value

        has_vp = False
        if result.metadata.hypothesis_output is not None:
            has_vp = any(
                bool(h.grounding_query.viewpoint_contexts)
                for h in result.metadata.hypothesis_output.hypotheses
            )

        def hit_k(top_n: int) -> bool:
            return bool(set(pred_frames[:top_n]) & gt_set)

        def recall_k(top_n: int) -> float:
            if not gt_set:
                return 0.0
            denom = min(top_n, len(gt_set))
            return len(set(pred_frames[:top_n]) & gt_set) / denom if denom > 0 else 0.0

        first_rank: int | None = None
        for idx, fid in enumerate(pred_frames[:3], start=1):
            if fid in gt_set:
                first_rank = idx
                break

        return SampleCoverageRecord(
            sample_id=spec.sample_id,
            scene_id=spec.scene_id,
            target_id=spec.target_id,
            category=spec.category,
            query=spec.query,
            is_easy=spec.is_easy,
            is_view_dep=spec.is_view_dep,
            status=status,
            pred_frames=pred_frames,
            gt_frames=gt_views,
            n_scene_views=n_scene_views,
            pred_target_ids=pred_target_ids,
            target_grids=[[r, c] for r, c in result.target_grids],
            has_viewpoint_contexts=has_vp,
            hit_at_1=hit_k(1),
            hit_at_2=hit_k(2),
            hit_at_3=hit_k(3),
            recall_at_1=recall_k(1),
            recall_at_2=recall_k(2),
            recall_at_3=recall_k(3),
            first_hit_rank=first_rank,
            target_obj_top1_hit=bool(
                pred_target_ids and pred_target_ids[0] == spec.target_id
            ),
            target_obj_top5_hit=(spec.target_id in pred_target_ids[:5]),
            duration_ms=elapsed_ms,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(f"sample {spec.sample_id} failed ({type(exc).__name__}): {exc}")
        return SampleCoverageRecord(
            sample_id=spec.sample_id,
            scene_id=spec.scene_id,
            target_id=spec.target_id,
            category=spec.category,
            query=spec.query,
            is_easy=spec.is_easy,
            is_view_dep=spec.is_view_dep,
            status="error",
            gt_frames=gt_views,
            n_scene_views=n_scene_views,
            duration_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize_records(records: Sequence[SampleCoverageRecord]) -> dict[str, Any]:
    """Compute aggregate coverage metrics matching the NR3D stage-1 coverage schema."""
    n_total = len(records)
    measurable = [r for r in records if r.gt_frames]
    n_measurable = len(measurable)
    n_unmeasurable = n_total - n_measurable
    n_error = sum(1 for r in records if r.error is not None)
    n_empty = sum(1 for r in measurable if not r.pred_frames)

    def mean_bool(items: Sequence[SampleCoverageRecord], attr: str) -> float:
        if not items:
            return 0.0
        return round(sum(1 for r in items if getattr(r, attr)) / len(items), 4)

    def mean_float(items: Sequence[SampleCoverageRecord], attr: str) -> float:
        if not items:
            return 0.0
        return round(sum(float(getattr(r, attr)) for r in items) / len(items), 4)

    easy = [r for r in measurable if r.is_easy]
    hard = [r for r in measurable if not r.is_easy]
    view_dep = [r for r in measurable if r.is_view_dep]
    view_indep = [r for r in measurable if not r.is_view_dep]
    hit3_ranks = [r.first_hit_rank for r in measurable if r.first_hit_rank is not None]
    gt_covs = [len(r.gt_frames) / max(1, r.n_scene_views) for r in measurable]
    status_counts = dict(sorted(Counter(r.status for r in records).items()))

    return {
        "n_total": n_total,
        "n_measurable": n_measurable,
        "n_unmeasurable_no_gt_frames": n_unmeasurable,
        "n_error": n_error,
        "n_empty_pred_frames": n_empty,
        "hit_at_1": mean_bool(measurable, "hit_at_1"),
        "mean_recall_at_1": mean_float(measurable, "recall_at_1"),
        "hit_at_2": mean_bool(measurable, "hit_at_2"),
        "mean_recall_at_2": mean_float(measurable, "recall_at_2"),
        "hit_at_3": mean_bool(measurable, "hit_at_3"),
        "mean_recall_at_3": mean_float(measurable, "recall_at_3"),
        "mean_first_hit_rank_when_hit_at_3": (
            round(sum(hit3_ranks) / len(hit3_ranks), 4) if hit3_ranks else 0.0
        ),
        "target_obj_top1_hit": mean_bool(measurable, "target_obj_top1_hit"),
        "target_obj_top5_hit": mean_bool(measurable, "target_obj_top5_hit"),
        "n_easy": len(easy),
        "hit_at_1_easy": mean_bool(easy, "hit_at_1"),
        "hit_at_2_easy": mean_bool(easy, "hit_at_2"),
        "hit_at_3_easy": mean_bool(easy, "hit_at_3"),
        "n_hard": len(hard),
        "hit_at_1_hard": mean_bool(hard, "hit_at_1"),
        "hit_at_2_hard": mean_bool(hard, "hit_at_2"),
        "hit_at_3_hard": mean_bool(hard, "hit_at_3"),
        "n_view_dep": len(view_dep),
        "hit_at_1_view_dep": mean_bool(view_dep, "hit_at_1"),
        "hit_at_2_view_dep": mean_bool(view_dep, "hit_at_2"),
        "hit_at_3_view_dep": mean_bool(view_dep, "hit_at_3"),
        "n_view_indep": len(view_indep),
        "hit_at_1_view_indep": mean_bool(view_indep, "hit_at_1"),
        "hit_at_2_view_indep": mean_bool(view_indep, "hit_at_2"),
        "hit_at_3_view_indep": mean_bool(view_indep, "hit_at_3"),
        "mean_gt_coverage": round(sum(gt_covs) / len(gt_covs), 4) if gt_covs else 0.0,
        "n_with_viewpoint_contexts": sum(
            1 for r in records if r.has_viewpoint_contexts
        ),
        "parse_status_counts": status_counts,
    }


def _record_from_dict(row: Mapping[str, Any]) -> SampleCoverageRecord:
    allowed = SampleCoverageRecord.__dataclass_fields__.keys()
    filtered = {k: v for k, v in row.items() if k in allowed}
    return SampleCoverageRecord(**filtered)


def compare_records(
    baseline_records: Sequence[SampleCoverageRecord],
    current_records: Sequence[SampleCoverageRecord],
) -> dict[str, Any]:
    """Compute paired sample-by-sample comparison between a baseline JSON and current records."""
    base_by_id = {r.sample_id: r for r in baseline_records}
    curr_by_id = {r.sample_id: r for r in current_records}
    common_ids = [sid for sid in base_by_id if sid in curr_by_id]

    both_hit3 = 0
    both_miss3 = 0
    current_only_hit3: list[str] = []
    baseline_only_hit3: list[str] = []
    identical_frames = 0

    both_hit1 = 0
    current_only_hit1 = 0
    baseline_only_hit1 = 0

    for sid in common_ids:
        base = base_by_id[sid]
        curr = curr_by_id[sid]
        if base.pred_frames == curr.pred_frames:
            identical_frames += 1
        if base.hit_at_3 and curr.hit_at_3:
            both_hit3 += 1
        elif not base.hit_at_3 and not curr.hit_at_3:
            both_miss3 += 1
        elif curr.hit_at_3 and not base.hit_at_3:
            current_only_hit3.append(sid)
        else:
            baseline_only_hit3.append(sid)

        if base.hit_at_1 and curr.hit_at_1:
            both_hit1 += 1
        elif curr.hit_at_1 and not base.hit_at_1:
            current_only_hit1 += 1
        elif base.hit_at_1 and not curr.hit_at_1:
            baseline_only_hit1 += 1

    n = max(len(common_ids), 1)
    base_sum = summarize_records(baseline_records)
    curr_sum = summarize_records(current_records)

    return {
        "n_common": len(common_ids),
        "hit_at_3": {
            "baseline": base_sum["hit_at_3"],
            "current": curr_sum["hit_at_3"],
            "delta_pp": round((curr_sum["hit_at_3"] - base_sum["hit_at_3"]) * 100, 2),
            "both_hit": both_hit3,
            "both_miss": both_miss3,
            "current_only_hit": len(current_only_hit3),
            "baseline_only_hit": len(baseline_only_hit3),
            "net_samples": len(current_only_hit3) - len(baseline_only_hit3),
            "current_only_sample_ids": current_only_hit3,
            "baseline_only_sample_ids": baseline_only_hit3,
        },
        "hit_at_1": {
            "baseline": base_sum["hit_at_1"],
            "current": curr_sum["hit_at_1"],
            "delta_pp": round((curr_sum["hit_at_1"] - base_sum["hit_at_1"]) * 100, 2),
            "both_hit": both_hit1,
            "current_only_hit": current_only_hit1,
            "baseline_only_hit": baseline_only_hit1,
            "net_samples": current_only_hit1 - baseline_only_hit1,
        },
        "target_obj_top1_hit": {
            "baseline": base_sum["target_obj_top1_hit"],
            "current": curr_sum["target_obj_top1_hit"],
            "delta_pp": round(
                (curr_sum["target_obj_top1_hit"] - base_sum["target_obj_top1_hit"])
                * 100,
                2,
            ),
        },
        "empty_pred_frames": {
            "baseline": base_sum["n_empty_pred_frames"],
            "current": curr_sum["n_empty_pred_frames"],
            "delta": curr_sum["n_empty_pred_frames"] - base_sum["n_empty_pred_frames"],
        },
        "identical_pred_frame_set_ratio": round(identical_frames / n, 4),
    }


def run_evaluation(
    specs: Sequence[CoverageSampleSpec],
    selectors_by_scene: Mapping[str, KeyframeSelector],
    *,
    output_dir: Path,
    k: int = 3,
    workers: int = 16,
    use_visual_context: bool = False,
) -> list[SampleCoverageRecord]:
    """Run coverage evaluation across all ``specs`` with per-sample JSONL checkpointing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.jsonl"
    completed: dict[str, SampleCoverageRecord] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is None:
                completed[str(row["sample_id"])] = _record_from_dict(row)
        logger.info(
            f"Resumed {len(completed)}/{len(specs)} completed samples from {checkpoint_path}"
        )

    write_lock = threading.Lock()
    done_count = len(completed)
    total_count = len(specs)
    t_start = time.monotonic()

    for retry_round in range(5):
        pending = [
            s
            for s in specs
            if s.sample_id not in completed or completed[s.sample_id].error is not None
        ]
        if not pending:
            break
        round_workers = workers if retry_round == 0 else min(8, workers)
        if retry_round > 0:
            logger.info(
                f"Retry round {retry_round} for {len(pending)} failed samples (workers={round_workers})"
            )
            time.sleep(3.0)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=round_workers)
        try:
            with checkpoint_path.open("a", encoding="utf-8") as ckpt_handle:
                future_to_spec = {
                    pool.submit(
                        evaluate_one_sample,
                        spec,
                        selectors_by_scene[spec.scene_id],
                        k=k,
                        use_visual_context=use_visual_context,
                    ): spec
                    for spec in pending
                }
                remaining = set(future_to_spec)
                while remaining:
                    done_batch, remaining = concurrent.futures.wait(
                        remaining,
                        timeout=120.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done_batch:
                        logger.warning(
                            f"{len(remaining)} futures stalled >120s; abandoning stuck sockets for retry"
                        )
                        for fut in remaining:
                            spec = future_to_spec[fut]
                            sel = selectors_by_scene[spec.scene_id]
                            gt_views = [
                                int(vid)
                                for vid, _ in sel.object_to_views.get(
                                    spec.target_id, []
                                )
                            ]
                            n_views = max(
                                len(sel.camera_poses), len(sel.view_to_objects), 1
                            )
                            completed[spec.sample_id] = SampleCoverageRecord(
                                sample_id=spec.sample_id,
                                scene_id=spec.scene_id,
                                target_id=spec.target_id,
                                category=spec.category,
                                query=spec.query,
                                is_easy=spec.is_easy,
                                is_view_dep=spec.is_view_dep,
                                status="error",
                                gt_frames=gt_views,
                                n_scene_views=n_views,
                                error="TimeoutError: socket stalled >120s",
                            )
                        break
                    for future in done_batch:
                        record = future.result()
                        with write_lock:
                            completed[record.sample_id] = record
                            if record.error is None:
                                ckpt_handle.write(json.dumps(asdict(record)) + "\n")
                                ckpt_handle.flush()
                            if retry_round == 0:
                                done_count += 1
                            if done_count % 25 == 0 or done_count == total_count:
                                elapsed = time.monotonic() - t_start
                                sofar = list(completed.values())
                                h1 = (
                                    sum(1 for r in sofar if r.hit_at_1)
                                    / len(sofar)
                                    * 100
                                )
                                h3 = (
                                    sum(1 for r in sofar if r.hit_at_3)
                                    / len(sofar)
                                    * 100
                                )
                                errs = sum(1 for r in sofar if r.error is not None)
                                logger.info(
                                    f"Progress {done_count}/{total_count} "
                                    f"(hit@1={h1:.1f}%, hit@3={h3:.1f}%, err={errs}, {elapsed:.1f}s)"
                                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    ordered = [completed[s.sample_id] for s in specs]
    summary = summarize_records(ordered)
    full_payload = {
        "k": k,
        "use_visual_context": use_visual_context,
        "summary": summary,
        "per_sample": [asdict(r) for r in ordered],
    }
    (output_dir / "results.json").write_text(
        json.dumps(full_payload, indent=2), encoding="utf-8"
    )
    return ordered


def _format_comparison_table(
    base_sum: Mapping[str, Any],
    curr_sum: Mapping[str, Any],
    comp: Mapping[str, Any],
) -> str:
    """Format a human-readable Markdown comparison table."""
    lines = [
        "=== NR3D Keyframe Coverage Comparison ===",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        ("Overall hit@1", "hit_at_1"),
        ("Overall hit@2", "hit_at_2"),
        ("Overall hit@3", "hit_at_3"),
        ("Easy hit@3", "hit_at_3_easy"),
        ("Hard hit@3", "hit_at_3_hard"),
        ("View-Dep hit@3", "hit_at_3_view_dep"),
        ("View-Indep hit@3", "hit_at_3_view_indep"),
        ("Mean Recall@3", "mean_recall_at_3"),
        ("Target Obj Top-1 Hit", "target_obj_top1_hit"),
        ("Target Obj Top-5 Hit", "target_obj_top5_hit"),
    ]
    for label, key in rows:
        lv = float(base_sum[key]) * 100
        uv = float(curr_sum[key]) * 100
        dv = uv - lv
        sign = "+" if dv >= 0 else ""
        lines.append(f"| {label} | {lv:.2f}% | {uv:.2f}% | {sign}{dv:.2f} pp |")

    le = int(base_sum["n_empty_pred_frames"])
    ue = int(curr_sum["n_empty_pred_frames"])
    de = ue - le
    lines.append(f"| Empty pred frames | {le} | {ue} | {de:+d} |")
    lines.append(
        f"| Paired hit@3 (+current / -baseline / net) | "
        f"+{comp['hit_at_3']['current_only_hit']} | "
        f"-{comp['hit_at_3']['baseline_only_hit']} | "
        f"{comp['hit_at_3']['net_samples']:+d} samples |"
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NR3D Stage-1 KeyframeSelector target-id frame coverage."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ysh/Datasets/nr3d/processed_v1/scannet"),
        help="NR3D processed_v1 scannet root directory.",
    )
    parser.add_argument(
        "--fold",
        type=Path,
        default=Path(
            "/home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json"
        ),
        help="Path to sample IDs fold JSON file.",
    )
    parser.add_argument(
        "--pack-name",
        type=str,
        default=DEFAULT_PACK_NAME,
        help="Prepared NR3D pack directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write checkpoints and summary JSONs.",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Optional path to a saved baseline results.json for paired comparison.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional LLM model name override from configs/llm.toml.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of keyframes to retrieve per query.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples to evaluate.",
    )
    parser.add_argument(
        "--use-visual-context",
        action="store_true",
        help="Attach rendered top-down BEV image to QueryParser.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = load_fold_specs(args.fold, args.data_root, pack_name=args.pack_name)
    if args.limit is not None and args.limit > 0:
        specs = specs[: args.limit]
    scene_ids = sorted({s.scene_id for s in specs})
    logger.info(
        f"Loaded {len(specs)} samples across {len(scene_ids)} scenes from {args.fold}"
    )

    llm_client = LLMClient.from_toml()
    resolved_model = args.model or llm_client.config.default_model
    logger.info(f"Using LLM model: {resolved_model!r} (workers={args.workers})")

    selectors_by_scene: dict[str, KeyframeSelector] = {}
    for sid in scene_ids:
        cg_dir = args.data_root / sid / "conceptgraph"
        selectors_by_scene[sid] = KeyframeSelector.from_scene_path(
            cg_dir,
            llm_client=llm_client,
            model=resolved_model,
            dataset="nr3d",
        )
    logger.success(f"Preloaded {len(selectors_by_scene)} KeyframeSelector instances")

    records = run_evaluation(
        specs,
        selectors_by_scene,
        output_dir=args.output_dir,
        k=args.k,
        workers=args.workers,
        use_visual_context=args.use_visual_context,
    )
    curr_sum = summarize_records(records)
    print(json.dumps(curr_sum, indent=2))

    if args.compare_with is not None and args.compare_with.is_file():
        base_payload = json.loads(args.compare_with.read_text(encoding="utf-8"))
        base_records = [
            _record_from_dict(r) for r in base_payload.get("per_sample", [])
        ]
        base_sum = summarize_records(base_records)
        comp = compare_records(base_records, records)
        table_md = _format_comparison_table(base_sum, curr_sum, comp)
        (args.output_dir / "comparison_report.md").write_text(
            table_md + "\n", encoding="utf-8"
        )
        print("\n" + table_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
