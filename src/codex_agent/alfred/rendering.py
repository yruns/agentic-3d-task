"""BEV visualization of incrementally observed geometry and navigation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import IncrementalSceneMemory


def render_memory_bev(memory: IncrementalSceneMemory, path: Path) -> Path:
    """Draw a bounded-size BEV with fixed world-cell labels and an object legend.

    Empty space outside the observed extent is not drawn as explored free space.
    Blue coverage means seen, white means free, dark gray means occupied, and
    unobserved cells remain gray. Object positions are drawn only while valid.
    """
    from PIL import Image, ImageDraw

    cells = set(memory.grid_to_views) | memory.free_cells | memory.occupied_cells
    if not cells:
        raise ValueError("cannot render BEV before receiving an observation")
    min_x = min(cell[0] for cell in cells) - 1
    max_x = max(cell[0] for cell in cells) + 1
    min_y = min(cell[1] for cell in cells) - 1
    max_y = max(cell[1] for cell in cells) + 1
    cols, rows = max_x - min_x + 1, max_y - min_y + 1
    scale = max(2, min(30, 700 // max(cols, rows)))
    margin, header = 44, 68
    width = max(620, cols * scale + 2 * margin)
    legend_height = 18 * min(20, len(memory.objects)) + 30
    height = rows * scale + header + margin + legend_height
    image = Image.new("RGB", (width, height), "#f7f7f7")
    draw = ImageDraw.Draw(image)
    draw.text(
        (12, 8),
        f"Incremental scene memory | frame {max(memory.frames)}",
        fill="black",
    )
    draw.text(
        (12, 26),
        f"Cell = floor(world XY / {memory.config.cell_size_m:g} m), Z up",
        fill="black",
    )
    draw.text(
        (12, 44),
        "White free | dark occupied | blue seen only | gray unknown",
        fill="black",
    )

    def rectangle(cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
        left = margin + (cell_x - min_x) * scale
        top = header + (max_y - cell_y) * scale
        return left, top, left + scale, top + scale

    colors = {"free": "#ffffff", "occupied": "#525b65", "unknown": "#cbd0d5"}
    for cell_x in range(min_x, max_x + 1):
        for cell_y in range(min_y, max_y + 1):
            cell = (cell_x, cell_y)
            state = memory.cell_state(cell)
            color = colors[state]
            if state == "unknown" and cell in memory.grid_to_views:
                color = "#c6e5f5"
            draw.rectangle(rectangle(*cell), fill=color, outline="#aeb7c0")
    label_stride = max(1, 32 // scale)
    for cell_x in range(min_x, max_x + 1, label_stride):
        left, _, _, _ = rectangle(cell_x, max_y)
        draw.text((left, header - 14), str(cell_x), fill="black")
    for cell_y in range(min_y, max_y + 1, label_stride):
        _, top, _, _ = rectangle(min_x, cell_y)
        draw.text((3, top), str(cell_y), fill="black")
    for node in memory.objects.values():
        if not node.geometry_valid:
            continue
        cell = memory.world_to_cell(*node.centroid[:2])
        left, top, right, bottom = rectangle(*cell)
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        draw.ellipse(
            (center_x - 4, center_y - 4, center_x + 4, center_y + 4), fill="#d78c10"
        )
        draw.text((center_x + 4, center_y - 8), str(node.object_id), fill="#9e3b00")
    if memory.agent_position is not None:
        cell = memory.world_to_cell(*memory.agent_position[:2])
        left, top, right, bottom = rectangle(*cell)
        draw.ellipse((left + 2, top + 2, right - 2, bottom - 2), fill="#d6293e")
        draw.text((left + 4, top + 3), "A", fill="white")
    legend_y = header + rows * scale + 24
    for node in list(memory.objects.values())[:20]:
        suffix = "" if node.geometry_valid else f" [{node.state}; location stale]"
        draw.text(
            (12, legend_y), f"{node.object_id}: {node.category}{suffix}", fill="#222222"
        )
        legend_y += 18
    if len(memory.objects) > 20:
        draw.text(
            (12, legend_y),
            f"+ {len(memory.objects) - 20} more objects in catalog",
            fill="#222222",
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.resolve()
