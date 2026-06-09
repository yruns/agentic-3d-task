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
- inspect_proposal — full detail for one id (color, description, nearby objects, \
every frame it appears in). args: {"proposal_id": 15}
- list_scene_proposals — filter the pool. args: {"category": "chair"} or \
{"region_bev": [xmin, ymin, xmax, ymax]} or {"limit": 20}
- keyframe_selector — language -> up to 3 first-person frames for a free-form \
description, when the catalog shortlist is unclear. args: {"query": "the trash \
can by the tv", "k": 3}
- select_by_proposal — frames that show given ids; set require_all to get a \
frame where target AND anchor are co-visible. args: {"proposal_ids": [15, 23], \
"require_all": true, "k": 3}
- mark_frame_with_bbox — draw labeled boxes for ids/categories on a frame, then \
view_image the returned image_path. args: {"frame_id": 212, "ids": [15, 23]}
- list_frame_proposals — which proposals are in a frame, ordered left->right by \
2D position. args: {"frame_id": 212}
- compare_proposals_spatial — rank candidates against one anchor. args: \
{"candidate_ids": [15, 16, 17], "anchor_id": 23, "relation": "closest_to"}. \
Relations: closest_to, near, next_to, farthest_from, above, below, left_of, \
right_of. Metric relations use 3D centers; left_of/right_of use votes over \
co-visible frames (shared_frame_counts, supporting_frame_counts).
- compare_candidates_to_anchors — rank candidates against several plausible \
anchors at once and check agreement (globally_consistent_top1, \
anchor_disagreement). args: {"candidate_ids": [15,16], "anchor_ids": [23,24], \
"relation": "near"}
- view_bev — top-down map; highlight ids/categories to tie first-person frames \
to scene layout. args: {"highlight": [15, 23]}

Recommended loop
1. Parse the query: target category, anchor(s), relation, attributes.
2. Shortlist candidates from the catalog (inspect_proposal / list_scene_proposals \
for color/description).
3. If one candidate is obvious from appearance, jump to step 5 to confirm.
4. Resolve spatial language:
   - metric (near / closest / above / below / between) -> compare_proposals_spatial \
(or compare_candidates_to_anchors when the anchor itself is ambiguous);
   - view-dependent (left / right / in front of / behind) -> fetch a co-visible \
frame with select_by_proposal(require_all=true), then mark_frame_with_bbox + \
view_image, and read left<->right from the pixels.
5. Confirm the winner: mark_frame_with_bbox the top candidate (with its anchor) \
on a frame that shows it, view_image it once, and check color / shape / relation \
match the query.
6. Emit the final JSON. Lower confidence and list rivals in uncertainties when \
more than one candidate survives.

Tool budget — be decisive
- Aim to decide within about 6-10 tool calls; a unique target needs only one \
confirming frame. View at most one or two images, then commit.
- Never repeat a tool with the same arguments and never re-view an image you have \
already seen. If a result is empty or errors, change approach instead of \
retrying identically.
- Once you have inspected the top candidates and confirmed with one co-visible \
frame or one spatial comparison, output the final JSON immediately. Do not keep \
gathering evidence to chase certainty on an already well-supported answer."""


__all__ = ["NR3D_TOOLS_PLAYBOOK", "NR3D_TOOL_NAMES"]
