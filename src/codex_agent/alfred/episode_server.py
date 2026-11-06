"""Local HTTP JSON RPC server exposing an active EpisodeExecutor to CLI tools.

During a single Codex turn for one ALFRED episode, ``EpisodeExecutor`` runs
:class:`EpisodeToolServer` on ``127.0.0.1:0`` (ephemeral loopback port) and
writes ``session_endpoint.json`` inside the episode artifact directory. When
Codex invokes ``python -m codex_agent.alfred.tools <tool> --episode-dir ...``,
the CLI forwards the tool request over loopback HTTP (with proxies explicitly
disabled) to this server, which serializes execution through a lock and returns
either a compact :func:`compute_scene_delta` or visual inspection payload.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from .delta import format_visible_objects_now
from .exploration import suggest_frontiers
from .geometry import GridCell

if TYPE_CHECKING:
    from .execution import EpisodeExecutor

SESSION_ENDPOINT_FILENAME = "session_endpoint.json"


class EpisodeToolError(Exception):
    """Recoverable tool input or state error returned to the agent as JSON."""


class EpisodeToolServer:
    """Thread-safe loopback HTTP server bound to one EpisodeExecutor."""

    def __init__(self, executor: EpisodeExecutor) -> None:
        self.executor = executor
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="alfred-episode-tool-server",
            daemon=True,
        )

    @property
    def url(self) -> str:
        raw_host, port = self._server.server_address[:2]
        host = (
            raw_host.decode("utf-8") if isinstance(raw_host, bytes) else str(raw_host)
        )
        return f"http://{host}:{port}"

    def start(self) -> Path:
        """Start serving and write session_endpoint.json in output_dir and project_root."""
        self._thread.start()
        payload = json.dumps({"url": self.url}, indent=2) + "\n"
        endpoint_path = self.executor.output_dir / SESSION_ENDPOINT_FILENAME
        endpoint_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint_path.write_text(payload, encoding="utf-8")
        if (
            self.executor.workspace_dir is not None
            and self.executor.workspace_dir.is_dir()
        ):
            (self.executor.workspace_dir / SESSION_ENDPOINT_FILENAME).write_text(
                payload, encoding="utf-8"
            )
        runtime_cfg = getattr(self.executor.runtime, "config", None)
        proj_root = getattr(runtime_cfg, "project_root", None)
        if isinstance(proj_root, Path) and proj_root.is_dir():
            (proj_root / SESSION_ENDPOINT_FILENAME).write_text(
                payload, encoding="utf-8"
            )
        return endpoint_path

    def close(self) -> None:
        """Shut down the loopback server and remove session_endpoint.json."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        endpoint_path = self.executor.output_dir / SESSION_ENDPOINT_FILENAME
        endpoint_path.unlink(missing_ok=True)
        if self.executor.workspace_dir is not None:
            (self.executor.workspace_dir / SESSION_ENDPOINT_FILENAME).unlink(
                missing_ok=True
            )
        runtime_cfg = getattr(self.executor.runtime, "config", None)
        proj_root = getattr(runtime_cfg, "project_root", None)
        if isinstance(proj_root, Path):
            (proj_root / SESSION_ENDPOINT_FILENAME).unlink(missing_ok=True)

    def __enter__(self) -> EpisodeToolServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def dispatch_tool(
        self, tool_name: str, args: dict[str, object]
    ) -> dict[str, object]:
        """Execute one tool call under the episode lock and persist to tool_calls.jsonl."""
        with self._lock:
            self.executor.decisions += 1
            decision_idx = self.executor.decisions
            frame_before = self.executor.memory._latest_frame_id
            actions_before = self.executor.actions
            started = time.monotonic()
            try:
                if (
                    tool_name != "finish_episode"
                    and self.executor.decisions > self.executor.config.max_decisions
                ):
                    raise EpisodeToolError(
                        f"Decision budget ({self.executor.config.max_decisions}) reached. "
                        "Call finish_episode immediately and emit your final JSON response."
                    )
                if tool_name == "navigate_to":
                    response = self._handle_navigate_to(args)
                elif tool_name == "explore_frontier":
                    response = self._handle_explore_frontier(args)
                elif tool_name == "scan_surroundings":
                    response = self._handle_scan_surroundings(args)
                elif tool_name == "interact_object":
                    response = self._handle_interact_object(args)
                elif tool_name == "list_objects":
                    response = self._handle_list_objects(args)
                elif tool_name == "keyframe_selector":
                    response = self._handle_keyframe_selector(args)
                elif tool_name == "view_frame":
                    response = self._handle_view_frame(args)
                elif tool_name == "view_crop":
                    response = self._handle_view_crop(args)
                elif tool_name == "view_bev":
                    response = self._handle_view_bev(args)
                elif tool_name == "finish_episode":
                    response = self._handle_finish_episode(args)
                else:
                    raise EpisodeToolError(f"Unknown ALFRED tool: {tool_name!r}")
            except EpisodeToolError as exc:
                error_payload: dict[str, object] = {"error": str(exc)}
                self._append_tool_call_record(
                    decision_index=decision_idx,
                    tool_name=tool_name,
                    args=args,
                    duration_ms=round((time.monotonic() - started) * 1000.0, 1),
                    frame_before=frame_before,
                    frame_after=self.executor.memory._latest_frame_id,
                    actions_before=actions_before,
                    actions_after=self.executor.actions,
                    response=error_payload,
                )
                raise
            self._append_tool_call_record(
                decision_index=decision_idx,
                tool_name=tool_name,
                args=args,
                duration_ms=round((time.monotonic() - started) * 1000.0, 1),
                frame_before=frame_before,
                frame_after=self.executor.memory._latest_frame_id,
                actions_before=actions_before,
                actions_after=self.executor.actions,
                response=response,
            )
            return response

    def _append_tool_call_record(
        self,
        *,
        decision_index: int,
        tool_name: str,
        args: dict[str, object],
        duration_ms: float,
        frame_before: int,
        frame_after: int,
        actions_before: int,
        actions_after: int,
        response: dict[str, object],
    ) -> None:
        self.executor.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "decision_index": decision_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": args,
            "duration_ms": duration_ms,
            "source_frame": frame_before,
            "result_frame": frame_after,
            "actions_before": actions_before,
            "actions_after": actions_after,
            "response": response,
        }
        with (self.executor.output_dir / "tool_calls.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        status_str = str(
            response.get("status")
            or response.get("tool")
            or ("error" if "error" in response else "ok")
        )
        msg_str = str(response.get("message") or response.get("error") or "")[:120]
        self.executor._tool_call_summaries.append(
            f"d{decision_index:02d} [f{frame_before}->f{frame_after}] "
            f"{tool_name}({json.dumps(args, ensure_ascii=False)}) -> {status_str}: {msg_str}"
        )

    def _handle_navigate_to(self, args: dict[str, object]) -> dict[str, object]:
        obj_id_raw = args.get("object_id")
        region_raw = args.get("region") or args.get("cell")
        max_steps_raw = args.get("max_steps", 32)
        obj_id = _optional_int(obj_id_raw, "object_id")
        region = _optional_region(region_raw, "region")
        max_steps = _optional_int(max_steps_raw, "max_steps") or 32
        if (obj_id is None) == (region is None):
            raise EpisodeToolError(
                "navigate_to requires exactly one of 'object_id' (int) or 'region' ([x, y])."
            )
        return self.executor.macro.navigate_to(
            object_id=obj_id, region=region, max_steps=max_steps
        )

    def _handle_explore_frontier(self, args: dict[str, object]) -> dict[str, object]:
        region = _optional_region(args.get("region") or args.get("cell"), "region")
        stop_raw = args.get("stop_on_categories") or args.get("categories") or []
        if isinstance(stop_raw, str):
            stop_cats = [stop_raw]
        elif isinstance(stop_raw, list) and all(isinstance(x, str) for x in stop_raw):
            stop_cats = list(stop_raw)
        else:
            raise EpisodeToolError(
                "stop_on_categories must be a list of category strings."
            )
        max_steps = _optional_int(args.get("max_steps", 20), "max_steps") or 20
        return self.executor.macro.explore_frontier(
            region=region,
            stop_on_categories=stop_cats,
            max_steps=max_steps,
        )

    def _handle_scan_surroundings(self, args: dict[str, object]) -> dict[str, object]:
        pitches_raw = args.get("pitches", [30.0, 0.0])
        if isinstance(pitches_raw, (int, float)) and not isinstance(pitches_raw, bool):
            pitches = [float(pitches_raw)]
        elif isinstance(pitches_raw, list) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in pitches_raw
        ):
            pitches = [float(x) for x in pitches_raw]
        else:
            raise EpisodeToolError("pitches must be a list of numbers, e.g. [30, 0].")
        return self.executor.macro.scan_surroundings(pitches=pitches)

    def _handle_interact_object(self, args: dict[str, object]) -> dict[str, object]:
        action = args.get("action")
        if not isinstance(action, str) or not action.strip():
            raise EpisodeToolError(
                "interact_object requires 'action' (pickup, put, open, close, "
                "toggle_on, toggle_off, slice, clean_in_sink, heat_in_microwave, cool_in_fridge)."
            )
        obj_id = _optional_int(args.get("object_id"), "object_id")
        if obj_id is None:
            raise EpisodeToolError("interact_object requires integer 'object_id'.")
        try:
            return self.executor.macro.interact_object(action=action, object_id=obj_id)
        except ValueError as exc:
            raise EpisodeToolError(str(exc)) from exc

    def _handle_list_objects(self, args: dict[str, object]) -> dict[str, object]:
        memory = self.executor.memory
        query_raw = args.get("query")
        query = str(query_raw).strip().casefold() if query_raw is not None else ""
        cats_raw = args.get("categories") or args.get("category")
        if isinstance(cats_raw, str):
            cat_filter = {cats_raw.strip().casefold()}
        elif isinstance(cats_raw, list):
            cat_filter = {str(x).strip().casefold() for x in cats_raw if str(x).strip()}
        else:
            cat_filter = set()

        region_filter = _optional_region(args.get("region"), "region")
        only_visible = bool(args.get("only_visible_now", False))
        ids_raw = args.get("object_ids")
        id_filter: set[int] | None = None
        if isinstance(ids_raw, list):
            id_filter = set()
            for raw_id in ids_raw:
                parsed = _optional_int(raw_id, "object_ids")
                if parsed is not None:
                    try:
                        id_filter.add(memory.resolve_object_id(parsed))
                    except KeyError:
                        continue

        current_ids = {
            memory.resolve_object_id(oid)
            for oid in memory.current_detection_objects.values()
        }
        agent_pos = np.asarray(memory.agent_position or (0.0, 0.0, 0.0))
        q_tokens = [tok for tok in re.split(r"[\s,|]+", query) if tok]
        scored_items: list[tuple[int, float, int, dict[str, object]]] = []
        for obj_id, node in sorted(memory.objects.items()):
            if id_filter is not None and obj_id not in id_filter:
                continue
            if only_visible and obj_id not in current_ids:
                continue
            if cat_filter:
                candidate_labels = [
                    node.category.casefold(),
                    *(ev.casefold() for ev in node.category_evidence),
                ]
                if not any(
                    cf == lbl or (len(cf) >= 3 and (cf in lbl or lbl in cf))
                    for cf in cat_filter
                    for lbl in candidate_labels
                ):
                    continue
            obj_region = (
                memory.world_to_region(float(node.centroid[0]), float(node.centroid[1]))
                if len(node.points)
                else None
            )
            if region_filter is not None and obj_region != region_filter:
                continue
            searchable = (
                f"{node.object_id} {node.category} "
                f"{' '.join(node.category_evidence.keys())} {node.state}"
            ).casefold()
            matched_tokens = 0
            if q_tokens:
                matched_tokens = sum(1 for tok in q_tokens if tok in searchable)
                if matched_tokens == 0:
                    continue

            dist_m = round(float(np.linalg.norm(node.centroid - agent_pos)), 2)
            recent_frames = [
                view.frame_id
                for view in node.views
                if view.frame_id > node.stale_through_frame
            ][-4:]
            scored_items.append(
                (
                    -matched_tokens,
                    dist_m,
                    node.object_id,
                    {
                        "object_id": node.object_id,
                        "category": node.category,
                        "category_evidence": dict(node.category_evidence),
                        "distance_m": dist_m,
                        "region": list(obj_region) if obj_region is not None else None,
                        "centroid": [round(float(v), 3) for v in node.centroid],
                        "geometry_valid": node.geometry_valid,
                        "provisional": node.provisional,
                        "state": node.state,
                        "visible_now": obj_id in current_ids,
                        "action_states": {
                            name: state.value
                            for name, state in node.action_states.items()
                        },
                        "best_frames": recent_frames,
                    },
                )
            )

        scored_items.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        items = [entry[3] for entry in scored_items]
        return {
            "tool": "list_objects",
            "count": len(items),
            "held_object_id": memory.held_object_id,
            "objects": items,
        }

    def _handle_keyframe_selector(self, args: dict[str, object]) -> dict[str, object]:
        memory = self.executor.memory
        query = str(args.get("query", "")).strip()
        k = max(1, min(4, _optional_int(args.get("k", 3), "k") or 3))
        grids_raw = args.get("target_grids") or args.get("grid_cells") or []
        target_cells: list[GridCell] = []
        if isinstance(grids_raw, list):
            for item in grids_raw:
                parsed_reg = _optional_region(item, "target_grids")
                if parsed_reg is not None:
                    target_cells.append(parsed_reg)

        matched_ids: list[int] = []
        ids_raw = args.get("object_ids")
        if isinstance(ids_raw, list):
            for raw_id in ids_raw:
                parsed_id = _optional_int(raw_id, "object_ids")
                if parsed_id is not None:
                    try:
                        matched_ids.append(memory.resolve_object_id(parsed_id))
                    except KeyError:
                        continue

        if query:
            q_tokens = [
                tok
                for tok in query.casefold().replace(",", " ").split()
                if len(tok) >= 3
                and tok
                not in {"the", "and", "near", "with", "from", "into", "under", "onto"}
            ]
            for obj_id, node in memory.objects.items():
                text = f"{node.category} {' '.join(node.category_evidence)}".casefold()
                if any(tok in text for tok in q_tokens):
                    matched_ids.append(obj_id)

        selected_records = memory.retrieve_frames(
            object_ids=tuple(sorted(set(matched_ids))),
            cells=tuple(target_cells),
            limit=k,
            include_stale=bool(args.get("include_stale", False)),
        )
        if not selected_records and memory.frames:
            selected_records = memory.retrieve_frames(limit=k, include_stale=True)

        frame_ids = [rec.frame_id for rec in selected_records]
        rendered = self._render_frames_and_sheet(
            frame_ids, annotate=True, prefix="keyframe"
        )
        rendered["tool"] = "keyframe_selector"
        rendered["query"] = query
        rendered["grounded_object_ids"] = sorted(set(matched_ids))
        return rendered

    def _handle_view_frame(self, args: dict[str, object]) -> dict[str, object]:
        memory = self.executor.memory
        annotate = bool(args.get("annotate", True))
        fids_raw = args.get("frame_ids") or args.get("frame_id")
        frame_ids: list[int] = []
        if fids_raw == "current" or fids_raw is None:
            if memory._latest_frame_id >= 0:
                frame_ids = [memory._latest_frame_id]
        elif isinstance(fids_raw, int) and not isinstance(fids_raw, bool):
            frame_ids = [fids_raw]
        elif isinstance(fids_raw, list):
            for item in fids_raw:
                if item == "current":
                    frame_ids.append(memory._latest_frame_id)
                else:
                    parsed = _optional_int(item, "frame_ids")
                    if parsed is not None:
                        frame_ids.append(parsed)

        valid_ids = [fid for fid in frame_ids if fid in memory.frames][:4]
        if not valid_ids and memory._latest_frame_id in memory.frames:
            valid_ids = [memory._latest_frame_id]
        rendered = self._render_frames_and_sheet(
            valid_ids, annotate=annotate, prefix="view_frame"
        )
        rendered["tool"] = "view_frame"
        return rendered

    def _handle_view_crop(self, args: dict[str, object]) -> dict[str, object]:
        memory = self.executor.memory
        obj_id = _optional_int(args.get("object_id"), "object_id")
        frame_id = _optional_int(args.get("frame_id"), "frame_id")
        bbox_raw = args.get("bbox")
        box_xyxy: tuple[float, float, float, float] | None = None
        category = "region"

        if obj_id is not None:
            try:
                resolved_id = memory.resolve_object_id(obj_id)
            except KeyError as exc:
                raise EpisodeToolError(f"Unknown object_id={obj_id}") from exc
            node = memory.objects[resolved_id]
            category = node.category
            if frame_id is None:
                # Prefer current frame if object is visible now, else highest-area view.
                current_dets = [
                    idx
                    for idx, oid in memory.current_detection_objects.items()
                    if memory.resolve_object_id(oid) == resolved_id
                ]
                if current_dets:
                    frame_id = memory._latest_frame_id
                    box_xyxy = memory.current_detection_boxes[current_dets[0]]
                elif node.views:
                    best_view = max(
                        node.views,
                        key=lambda v: (v.pixel_area * v.confidence, v.frame_id),
                    )
                    frame_id = best_view.frame_id
                    box_xyxy = best_view.box_xyxy
            else:
                matching_views = [v for v in node.views if v.frame_id == frame_id]
                if matching_views:
                    box_xyxy = matching_views[-1].box_xyxy

        if frame_id is None:
            frame_id = memory._latest_frame_id
        if frame_id not in memory.frames:
            raise EpisodeToolError(f"Frame {frame_id} does not exist in memory.")

        frame_rec = memory.frames[frame_id]
        with Image.open(frame_rec.rgb_path) as raw_img:
            rgb = raw_img.convert("RGB")
            width, height = rgb.size

            if box_xyxy is None and isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                vals = [float(x) for x in bbox_raw]
                if max(vals) <= 1.05:
                    box_xyxy = (
                        vals[0] * width,
                        vals[1] * height,
                        vals[2] * width,
                        vals[3] * height,
                    )
                else:
                    box_xyxy = (vals[0], vals[1], vals[2], vals[3])

            if box_xyxy is None:
                raise EpisodeToolError(
                    "view_crop requires either an observed object_id or frame_id + bbox=[x0,y0,x1,y1]."
                )

            x1, y1, x2, y2 = box_xyxy
            bw, bh = max(8.0, x2 - x1), max(8.0, y2 - y1)
            margin_x, margin_y = 0.25 * bw + 12.0, 0.25 * bh + 12.0
            cx1 = max(0, int(math.floor(x1 - margin_x)))
            cy1 = max(0, int(math.floor(y1 - margin_y)))
            cx2 = min(width, int(math.ceil(x2 + margin_x)))
            cy2 = min(height, int(math.ceil(y2 + margin_y)))
            cropped = rgb.crop((cx1, cy1, cx2, cy2))
            scale = 480.0 / max(1, max(cropped.size))
            resized_crop = cropped.resize(
                (
                    max(64, int(round(cropped.width * scale))),
                    max(64, int(round(cropped.height * scale))),
                ),
                Image.Resampling.LANCZOS,
            )

            context_img = rgb.resize((480, 480), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(context_img)
            sx, sy = 480.0 / width, 480.0 / height
            draw.rectangle(
                (x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#ffdd00", width=3
            )
            label = f"#{obj_id} {category}" if obj_id is not None else category
            draw.text((max(4, x1 * sx), max(4, y1 * sy - 16)), label, fill="#ffdd00")

            canvas = Image.new(
                "RGB",
                (resized_crop.width + context_img.width + 24, 520),
                "#1e242b",
            )
            canvas_draw = ImageDraw.Draw(canvas)
            canvas_draw.text(
                (12, 8),
                f"Frame {frame_id} | High-Res Unoccluded Crop ({label}) + Context Frame",
                fill="#ffffff",
            )
            canvas.paste(resized_crop, (8, 32))
            canvas.paste(context_img, (resized_crop.width + 16, 32))

            crops_dir = self.executor.output_dir / "tool_artifacts"
            crops_dir.mkdir(parents=True, exist_ok=True)
            out_path = (
                crops_dir
                / f"crop_f{frame_id:04d}_obj{obj_id if obj_id is not None else 'box'}_{self.executor.decisions:03d}.png"
            )
            canvas.save(out_path)
            published = self.executor.publish_tool_artifact(out_path)

        return {
            "tool": "view_crop",
            "frame_id": frame_id,
            "object_id": obj_id,
            "category": category,
            "image_path": str(published),
        }

    def _handle_view_bev(self, args: dict[str, object]) -> dict[str, object]:
        del args
        bev_dir = self.executor.output_dir / "tool_artifacts"
        bev_dir.mkdir(parents=True, exist_ok=True)
        out_path = bev_dir / f"bev_{self.executor.decisions:03d}.png"
        self.executor.memory.render_bev(out_path)
        published_bev = self.executor.publish_tool_artifact(out_path)
        frontiers = [
            {
                "region": list(c.region),
                "world_xy": [
                    round(c.world_position[0], 2),
                    round(c.world_position[1], 2),
                ],
                "path_cost_m": round(c.path_cost_m, 2),
                "info_gain_m2": round(c.information_gain_m2, 2),
            }
            for c in suggest_frontiers(self.executor.memory, limit=5)
        ]
        return {
            "tool": "view_bev",
            "frame_id": self.executor.memory._latest_frame_id,
            "image_path": str(published_bev),
            "top_frontiers": frontiers,
        }

    def _handle_finish_episode(self, args: dict[str, object]) -> dict[str, object]:
        summary = str(args.get("summary", "")).strip()
        self.executor.finish_requested = True
        self.executor.finish_summary = summary
        return {
            "tool": "finish_episode",
            "status": "episode_marked_complete",
            "summary": summary,
            "actions_taken": self.executor.actions,
            "failures": self.executor.failures,
            "held_object_id": self.executor.memory.held_object_id,
            "verified_macros_completed": sorted(self.executor.completed_macros),
        }

    def _render_frames_and_sheet(
        self, frame_ids: list[int], *, annotate: bool, prefix: str
    ) -> dict[str, object]:
        memory = self.executor.memory
        out_dir = self.executor.output_dir / "tool_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered_images: list[Image.Image] = []
        frames_payload: list[dict[str, object]] = []

        for fid in frame_ids:
            rec = memory.frames[fid]
            with Image.open(rec.rgb_path) as src:
                img = src.convert("RGB").resize((540, 540), Image.Resampling.BILINEAR)
                orig_w, orig_h = src.size
            draw = ImageDraw.Draw(img)
            visible_objs: list[dict[str, object]] = []
            for obj_id in rec.object_ids:
                resolved = memory.resolve_object_id(obj_id)
                if resolved not in memory.objects:
                    continue
                node = memory.objects[resolved]
                views = [v for v in node.views if v.frame_id == fid]
                if not views:
                    continue
                v = views[-1]
                x1, y1, x2, y2 = v.box_xyxy
                bbox_norm = [
                    round(x1 / max(1, orig_w), 3),
                    round(y1 / max(1, orig_h), 3),
                    round(x2 / max(1, orig_w), 3),
                    round(y2 / max(1, orig_h), 3),
                ]
                visible_objs.append(
                    {
                        "object_id": resolved,
                        "category": node.category,
                        "bbox_norm": bbox_norm,
                        "confidence": round(v.confidence, 3),
                    }
                )
                if annotate:
                    sx, sy = 540.0 / max(1, orig_w), 540.0 / max(1, orig_h)
                    draw.rectangle(
                        (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                        outline="#ffdd00",
                        width=2,
                    )
                    draw.text(
                        (max(2, x1 * sx), max(2, y1 * sy)),
                        f"#{resolved}:{node.category}",
                        fill="#000000",
                        stroke_width=1,
                        stroke_fill="#ffdd00",
                    )

            draw.rectangle((0, 0, 540, 22), fill="#1e242b")
            draw.text(
                (6, 4),
                f"Frame {fid} | pos=({rec.camera_position[0]:.2f}, {rec.camera_position[1]:.2f})",
                fill="#ffffff",
            )
            fpath = out_dir / f"{prefix}_f{fid:04d}_{self.executor.decisions:03d}.png"
            img.save(fpath)
            pub_fpath = self.executor.publish_tool_artifact(fpath)
            rendered_images.append(img)
            frames_payload.append(
                {
                    "frame_id": fid,
                    "image_path": str(pub_fpath),
                    "camera_position": [round(x, 3) for x in rec.camera_position],
                    "visible_objects": visible_objs,
                }
            )

        result: dict[str, object] = {
            "frames": frames_payload,
        }
        if len(rendered_images) == 1:
            result["image_path"] = frames_payload[0]["image_path"]
        elif len(rendered_images) >= 2:
            cols = min(2, len(rendered_images))
            rows = math.ceil(len(rendered_images) / cols)
            sheet = Image.new("RGB", (cols * 548 + 8, rows * 548 + 8), "#1e242b")
            for idx, tile in enumerate(rendered_images):
                r, c = divmod(idx, cols)
                sheet.paste(tile, (8 + c * 548, 8 + r * 548))
            sheet_path = (
                out_dir / f"{prefix}_contact_sheet_{self.executor.decisions:03d}.png"
            )
            sheet.save(sheet_path)
            pub_sheet = self.executor.publish_tool_artifact(sheet_path)
            result["contact_sheet"] = str(pub_sheet)
            result["image_path"] = str(pub_sheet)
        elif self.executor.memory._latest_frame_id >= 0:
            result["visible_objects_now"] = format_visible_objects_now(memory)
        return result

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/tool":
                    self._send_json(404, {"error": "unknown_route"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise EpisodeToolError("Request body must be a JSON object.")
                    tool_name = str(body.get("tool", ""))
                    raw_args = body.get("args", {})
                    if not isinstance(raw_args, dict):
                        raise EpisodeToolError("'args' must be a JSON object.")
                    payload = server_ref.dispatch_tool(tool_name, raw_args)
                    self._send_json(200, payload)
                except EpisodeToolError as exc:
                    self._send_json(200, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

            def log_message(self, format: str, *args: object) -> None:
                """Suppress default HTTP access logging."""

            def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return _Handler


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EpisodeToolError(f"Parameter {name!r} must be an integer, got {value!r}.")
    return value


def _optional_region(value: object, name: str) -> GridCell | None:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return (int(value[0]), int(value[1]))
    raise EpisodeToolError(
        f"Parameter {name!r} must be a 2-integer region [x, y], got {value!r}."
    )


__all__ = [
    "EpisodeToolError",
    "EpisodeToolServer",
    "SESSION_ENDPOINT_FILENAME",
]
