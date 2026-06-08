---
name: nr3d-codex-sdk
description: Select one object proposal for a single NR3D referring expression from a prepared 3D scene proposal pool, using the Codex Agent SDK.
---

# NR3D Codex SDK Visual Grounding

You are selecting exactly one object proposal for a single NR3D referring
expression over a prepared 3D scene.

## Rules

- Choose exactly one `proposal_id` from the provided proposal pool.
- Use `proposal_id: -1` only when the described target is absent from the pool.
- A top-down BEV image of the scene is attached. Use it together with the
  printed proposal metadata (the printed proposal ids match the scene).
- Do not use benchmark ground-truth fields; none are provided.
- Do not create, edit, or delete files.
- Return only the requested JSON object — no markdown, prose, or commentary
  outside the JSON.

## Reasoning Priorities

1. Match the target category and enriched category first.
2. Apply the spatial language in the query — especially `near`, `closest`,
   `farthest`, `left`, `right`, `above`, `below`, `by`, `between`, `next to`,
   and `close to` — against proposal centers and sizes.
3. Use proposal centers `(cx, cy, cz)` and sizes `(dx, dy, dz)` as 3D evidence,
   and the BEV layout to disambiguate left/right/front/back and adjacency.
4. Use each proposal's compact note for visual attributes such as shape, color,
   material, and object role.
5. If several proposals remain plausible, pick the best-supported one and lower
   `confidence` accordingly; record the competing candidates in `uncertainties`.
