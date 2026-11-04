"""The NR3D tool-using playbook, inlined into the turn prompt.

Why this lives in the prompt instead of a Codex *skill* file:

The Codex skill mechanism advertises a skill as ``<skill><path>…</path></skill>``
and expects the model to *read that file when needed*. Once Codex's app-server
windows the context (it drops images and older tool outputs after big images
enter the turn), a "lost" model re-runs its default orientation action — reading
the advertised ``SKILL.md`` path — over and over, an unbounded loop (see
``docs/codex_agent/skill_loop_and_reasoning_dropped_20260609.md``). Inlining the
playbook into the prompt keeps the guidance inside the always-present, cached
prompt prefix and removes the re-readable on-disk bait entirely.

Keep this in sync with ``.agents/skills/nr3d-codex-tools/SKILL.md`` (kept only
for manual/legacy ``--skill-path`` use); the inline constant is the source of
truth for the default tool-mode run.
"""

from __future__ import annotations

#: Tool names the inline playbook must document (used by a test to keep them in
#: sync with the CLI dispatcher and to catch a tool added without guidance).
NR3D_TOOL_NAMES: tuple[str, ...] = (
    "inspect_proposal",
    "list_scene_proposals",
    "keyframe_selector",
    "select_by_proposal",
    "mark_frame_with_bbox",
    "list_frame_proposals",
    "compare_proposals_spatial",
    "compare_candidates_to_anchors",
    "view_bev",
)

NR3D_TOOLS_PLAYBOOK = """\
Playbook (everything you need is here — do NOT look for a skill or any other \
file):

Tool catalog
- inspect_proposal — full untruncated detail for one id (color, description, \
location, nearby_objects, bbox_3d_9dof, every frame it appears in). \
args: {"proposal_id": 15}
- list_scene_proposals — filter the pool by category or 3D XY region. \
args: {"category": "chair"} or {"region_bev": [xmin, ymin, xmax, ymax]} or \
{"limit": 20}
- select_by_proposal — fetch up to 4 spatially-diverse, high-visibility frames \
showing given proposal ids, plus per_proposal_best_frames and \
requested_left_to_right order. Set require_all=true for co-visible target+anchor \
frames (auto-falls back to maximal partial co-visibility when >2 ids cannot all \
fit in one view); set require_all=false for balanced coverage across all \
candidate ids. \
args: {"proposal_ids": [15, 23], "require_all": true, "k": 3}
- mark_frame_with_bbox — draw labeled 2D boxes for ids/categories on one frame \
(frame_id) or a batch of up to 4 frames (frame_ids, composing a single \
contact_sheet), AND automatically write crops_sheet_path (high-res unoccluded \
crops of the target objects from raw RGB). Open image_path and/or \
crops_sheet_path with view_image! \
args: {"frame_id": 212, "ids": [15, 23]} or \
{"frame_ids": [45, 63], "ids": [12, 13]}
- list_frame_proposals — all proposals visible in one frame, ordered left->right \
by 2D box center. args: {"frame_id": 212}
- compare_proposals_spatial — deterministic 3D + co-visible 2D + virtual-viewer \
ranking. Supports relations: closest_to, near, next_to, farthest_from, above, \
below, left_of, right_of, in_front_of, behind, between, opposite. Pass \
anchor_id when relative to an anchor, or omit anchor_id when comparing a group \
directly (e.g. "facing the monitors, the one on the right", or "the middle \
window" with relation="between"). Pass observer_id when the query specifies \
standing at / looking from another object (e.g. "looking from the door #11"). \
Returns both co-visible 2D frame votes and 3D virtual-observer rankings \
(facing_left_to_right_ids, facing_front_to_back_ids). \
args: {"candidate_ids": [15, 16, 17], "anchor_id": 23, "relation": "closest_to"} \
or {"candidate_ids": [12, 13], "observer_id": 0, "relation": "right_of"}
- compare_candidates_to_anchors — rank candidates against multiple anchors at \
once (e.g. relation="between" for "between the bed and the nightstand" using \
exact 3D segment distance, or relation="near" when the anchor itself has multiple \
instances). \
args: {"candidate_ids": [15, 16], "anchor_ids": [23, 24], "relation": "between"}
- view_bev — render the top-down map with highlighted 3D oriented footprints + \
labels and return highlighted_objects (3D size/pos + 2D BEV pixel coords). \
args: {"highlight": [15, 23]}
- keyframe_selector — language -> up to 3 first-person frames when the anchor is \
not in the proposal catalog. args: {"query": "the trash can by the tv", "k": 3}

Critical disambiguation rules
1. Trust 3D `category` and `size/center` over noisy `enriched/note` text: \
Offline `enriched` / `note` text is generated from 2D crops and is frequently \
polluted by a foreground occluder (e.g. a chair in front of a cabinet, or a \
hairdryer cord next to a towel). NEVER eliminate a proposal of the target \
`category` or output `proposal_id: -1` just because its `note` describes a \
nearby/occluding object! Inspect `crops_sheet_path` from `mark_frame_with_bbox` \
and verify its 3D `size` and `center`.
2. Resolve view-dependent language ("facing X", "looking from Y", "left/right", \
"front/behind"):
   - "Facing X" (including "facing the foot of the bed" or "facing the monitors") \
ALWAYS means standing in the room looking AT X from the outside (never standing \
inside X looking outward).
   - Call `compare_proposals_spatial` (with `observer_id` if looking from a \
specific object, or `anchor_id` if relative to an anchor) and check BOTH \
co-visible frame votes (`supporting_frame_counts`) AND the 3D virtual-viewer \
order (`facing_left_to_right_ids` / `facing_front_to_back_ids`).
   - Fetch co-visible frame(s) with `select_by_proposal`, run \
`mark_frame_with_bbox`, and `view_image` to confirm visually from the camera's \
perspective.
3. Resolve "between / middle / aligned row" queries:
   - Between two anchors -> `compare_candidates_to_anchors` with `relation="between"`.
   - Middle of a row of N same-category objects (e.g. "the middle window left of \
the chalkboard") -> filter to the subset on the same wall/line (check `center` \
coordinates and `view_bev(highlight=[...])`), then pick the geometric middle one.

Recommended loop
1. Shortlist all proposals matching the target category from the printed catalog.
2. Run `compare_proposals_spatial` / `compare_candidates_to_anchors` to rank \
candidates by the query's spatial or metric relation.
3. Fetch the best co-visible or balanced frames with `select_by_proposal`, \
annotate with `mark_frame_with_bbox` (passing `frame_ids` to batch up to 4 \
frames at once), and call `view_image` on `image_path` (and `crops_sheet_path` \
when appearance/color is subtle).
4. Once visual and spatial evidence confirm the winner, emit the final JSON \
immediately within ~6-10 tool calls."""


__all__ = ["NR3D_TOOLS_PLAYBOOK", "NR3D_TOOL_NAMES"]
