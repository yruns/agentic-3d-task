"""Schematic object footprints over observed navigation and coverage layers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import IncrementalSceneMemory


def render_memory_bev(memory: IncrementalSceneMemory, path: Path) -> Path:
    """Render fixed world-region labels, fine occupancy, frontiers and heading.

    The image window may expand, but no object, fine cell or semantic region is
    renumbered. Blue coverage is distinct from white free space. Gray includes
    unseen space and dark cells are occupied; these are estimates, not a mesh.
    """
    from PIL import Image, ImageDraw

    spatial_map = memory.spatial_map
    cells = spatial_map.known_cells | spatial_map.seen_cells
    if not cells or spatial_map.agent_position is None:
        raise ValueError("BEV requires an integrated observation")
    resolution = spatial_map.config.fine_cell_size_m
    region_size = spatial_map.config.region_size_m
    min_x = (
        math.floor(min(cell[0] * resolution for cell in cells) / region_size)
        * region_size
        - region_size
    )
    max_x = (
        math.ceil(max((cell[0] + 1) * resolution for cell in cells) / region_size)
        * region_size
        + region_size
    )
    min_y = (
        math.floor(min(cell[1] * resolution for cell in cells) / region_size)
        * region_size
        - region_size
    )
    max_y = (
        math.ceil(max((cell[1] + 1) * resolution for cell in cells) / region_size)
        * region_size
        + region_size
    )
    pixels_per_m = min(180.0, 850.0 / max(max_x - min_x, max_y - min_y))
    margin, header = 48, 80
    map_width = math.ceil((max_x - min_x) * pixels_per_m)
    map_height = math.ceil((max_y - min_y) * pixels_per_m)
    legend_rows = min(24, len(memory.objects))
    width = max(740, map_width + 2 * margin)
    height = header + map_height + margin + 20 * (legend_rows + 2)
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text(
        (12, 10),
        f"Observed scene | frame {spatial_map.latest_frame_id} | fixed {region_size:g} m region IDs",
        fill="#202c39",
    )
    draw.text(
        (12, 30),
        "White: free | dark: occupied | blue: seen only | gray: unknown",
        fill="#35465c",
    )
    draw.text(
        (12, 50),
        "Green: frontier | red arrow: agent | orange box: committed exploration stance",
        fill="#35465c",
    )

    def pixel(x: float, y: float) -> tuple[int, int]:
        return margin + round((x - min_x) * pixels_per_m), header + round(
            (max_y - y) * pixels_per_m
        )

    draw.rectangle(
        (margin, header, margin + map_width, header + map_height), fill="#cbd0d5"
    )
    colors = {"free": "#ffffff", "occupied": "#424b58", "unknown": "#c6e5f5"}
    for cell in cells:
        x, y = cell[0] * resolution, cell[1] * resolution
        left, top = pixel(x, y + resolution)
        right, bottom = pixel(x + resolution, y)
        draw.rectangle(
            (left, top, right, bottom), fill=colors[spatial_map.cell_state(cell)]
        )
    for cell in spatial_map.frontier_cells():
        center = pixel(*spatial_map.cell_center(cell))
        draw.ellipse(
            (center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1), fill="#38a66d"
        )

    first_col, last_col = math.floor(min_x / region_size), math.ceil(
        max_x / region_size
    )
    first_row, last_row = math.floor(min_y / region_size), math.ceil(
        max_y / region_size
    )
    label_stride = max(1, math.ceil(28 / (region_size * pixels_per_m)))
    for col in range(first_col, last_col + 1):
        left, _ = pixel(col * region_size, max_y)
        draw.line((left, header, left, header + map_height), fill="#9aa7b5", width=1)
        if col % label_stride == 0:
            center_x, _ = pixel((col + 0.5) * region_size, max_y)
            draw.text((center_x - 5, header - 16), str(col), fill="#25374b")
    for row in range(first_row, last_row + 1):
        _, top = pixel(min_x, row * region_size)
        draw.line((margin, top, margin + map_width, top), fill="#9aa7b5", width=1)
        if row % label_stride == 0:
            _, center_y = pixel(min_x, (row + 0.5) * region_size)
            draw.text((7, center_y - 6), str(row), fill="#25374b")

    for node in memory.objects.values():
        if not node.geometry_valid:
            continue
        bounds = node.bounds
        left, top = pixel(float(bounds[0, 0]), float(bounds[1, 1]))
        right, bottom = pixel(float(bounds[1, 0]), float(bounds[0, 1]))
        draw.rectangle((left, top, right, bottom), outline="#a76715", width=2)
        x, y = pixel(float(node.centroid[0]), float(node.centroid[1]))
        draw.text((x + 2, y - 10), str(node.object_id), fill="#8c3f0b")
    progress = spatial_map.exploration
    if progress.goal is not None:
        x, y = pixel(*spatial_map.cell_center(progress.goal))
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline="#f08b21", width=3)
    agent_x, agent_y = pixel(*spatial_map.agent_position[:2])
    direction_x, direction_y, _ = spatial_map.camera_direction
    length = max(1e-6, math.hypot(direction_x, direction_y))
    tip_x = agent_x + round(18 * direction_x / length)
    tip_y = agent_y - round(18 * direction_y / length)
    draw.ellipse((agent_x - 5, agent_y - 5, agent_x + 5, agent_y + 5), fill="#d73742")
    draw.line((agent_x, agent_y, tip_x, tip_y), fill="#d73742", width=4)
    draw.ellipse((tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3), fill="#d73742")
    legend_y = header + map_height + 16
    for index, node in enumerate(list(memory.objects.values())[:legend_rows]):
        suffix = "" if node.geometry_valid else f" ({node.state}; historical position)"
        draw.text(
            (12, legend_y + 20 * index),
            f"{node.object_id}: {node.category}{suffix}",
            fill="#263749",
        )
    if len(memory.objects) > legend_rows:
        draw.text(
            (12, legend_y + 20 * legend_rows),
            f"{len(memory.objects) - legend_rows} more observed instances in the catalog",
            fill="#263749",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.resolve()
