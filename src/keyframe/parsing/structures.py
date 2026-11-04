"""Prompt templates for query parsing.

The system prompt and few-shot examples instruct the LLM to emit a
``HypothesisOutputV1`` JSON object combining semantic categories, instance
``#id`` candidates, and 0-indexed BEV ``(row, col)`` grid coordinates.
"""

from __future__ import annotations

from keyframe.models.hypotheses import SUPPORTED_RELATIONS_STR


def get_system_prompt() -> str:
    """System prompt describing the HypothesisOutputV1 schema and parsing rules."""
    return f"""You are a spatial query parser for 3D indoor scene understanding.
Your task is to parse natural language queries into a structured HypothesisOutputV1 JSON format.

The output must be a valid HypothesisOutputV1 JSON object with the following structure:
- format_version: Always "hypothesis_output_v1"
- parse_mode: "single" or "multi" (see decision rules below)
- hypotheses: List of QueryHypothesis objects (1-3 hypotheses)

Each QueryHypothesis has:
- kind: "direct", "proxy", or "context"
- rank: 1-based priority (1=highest priority)
- grounding_query: A GroundingQuery object
- lexical_hints: List of free-form hints (synonyms, paraphrases) from the query

GroundingQuery structure:
- raw_query: The original query text
- root: A QueryNode representing the target object
- expect_unique: True if the query uses "the" (singular), False otherwise
- viewpoint_contexts (optional): List of ViewpointContext objects referenced by constraints
- target_grids: List of [row, col] 0-indexed integer pairs (e.g., [[1, 2]]) identifying
  the most likely grid cell(s) containing the TARGET object or target region for THIS hypothesis.
- anchor_grids: List of [row, col] 0-indexed integer pairs (e.g., [[1, 2], [0, 1]]) identifying
  the grid cell(s) containing the REFERENCE / ANCHOR object(s) for THIS hypothesis.

Each QueryNode (root target, spatial anchors, select reference, viewpoint anchors) has:
- categories: LIST of object types (MUST be EXACT strings from SCENE CATEGORIES, or ["UNKNOW"] if no match)
- candidate_object_ids: List of integer object IDs (e.g., [7]) from SCENE GRID LAYOUT / BEV image
  that are the most likely instance candidates for THIS node (omit or [] if unknown)
- grid_cells: List of [row, col] 0-indexed integer pairs (e.g., [[1, 2]]) where THIS node is located
- attributes: List of adjective attributes like "red", "large", "wooden"
- spatial_constraints: List of spatial relations to other objects (filter phase, AND logic)
- select_constraint: Optional selection like "nearest", "largest", "second" (select phase)

SpatialConstraint structure:
- relation: PREFERRED to be one of these predefined values: {SUPPORTED_RELATIONS_STR}
  (Map synonyms: "on top of"->"on", "under"->"below", "close to"->"near")
- anchors: List of reference QueryNode objects (1 for most relations, 2 for "between")
- reference_frame (optional): "world" (default), "viewer", "object_local", or "ambiguous".
  See VIEWPOINT DETECTION rules below.
- viewpoint_context_id (optional): id string referencing a ViewpointContext on the
  GroundingQuery. REQUIRED when reference_frame is "viewer" or "object_local".
- execution_policy (optional): "hard" (default, filter-on-fail), "soft" (keep but score),
  or "rank_only" (never filter; score only).

SelectConstraint structure (for superlative/ordinal):
- constraint_type: "superlative" or "ordinal"
- metric: "distance", "size", "height", "x_position", "y_position"
- order: "min" (nearest/smallest), "max" (farthest/largest), "asc", "desc"
- reference: QueryNode for distance reference (e.g., "nearest the door" -> door)
- position: Integer for ordinal (1=first, 2=second, etc.)
- reference_frame (optional): same enum as SpatialConstraint. Use "viewer" when the
  metric axis is the speaker's right/forward (e.g. "the window on the right" when
  facing the windows -> metric="x_position" + reference_frame="viewer").
- viewpoint_context_id (optional): same rules as SpatialConstraint.
- execution_policy (optional): same enum as SpatialConstraint.

ViewpointContext structure (declared on grounding_query.viewpoint_contexts):
- id: short unique string within this grounding_query, referenced from constraints
- kind: one of
  - "facing_anchor": single anchor cue, e.g. "facing the door"
  - "facing_anchor_set": multi-anchor cue, e.g. "facing the windows"
  - "entering_from": e.g. "entering from the door"
  - "standing_at": e.g. "standing at the foot of the bed"
  - "object_local": object_A facing object_B, NO human observer
- facing_anchor: required for facing_anchor / facing_anchor_set
- origin_anchor: required for entering_from / standing_at
- subject_anchor: required for object_local (the subject doing the facing)
- raw_phrase: the literal NL fragment, for trace
- confidence: "explicit" (literal cue), "inferred", or "ambiguous"

=== PARSE MODE DECISION RULES ===

Use parse_mode="single" when:
1. Target AND all anchors/references exist in SCENE CATEGORIES
2. Even with semantic expansion (pillow -> [pillow, throw_pillow]), if all categories are in scene

Use parse_mode="multi" when:
1. Target category is NOT in scene -> output ["UNKNOW"] and add PROXY/CONTEXT fallback
2. Any anchor/reference category is NOT in scene -> output ["UNKNOW"] in that anchor and add PROXY fallback
3. Query is ambiguous and may need fallback strategies

=== HYPOTHESIS KIND RULES ===

DIRECT hypothesis (always rank=1):
- Parse the query literally
- Use ["UNKNOW"] for any category not found in scene
- Keep original spatial constraints and select constraints
- Populate `candidate_object_ids` and `grid_cells` on `root` and on each anchor/reference node,
  plus `target_grids` and `anchor_grids` on `grounding_query`.
- Even if a node's category is ["UNKNOW"], still set its `grid_cells` to the [row, col] region
  inferred from the BEV image / adjacent anchors / observable empty grids!

PROXY hypothesis (rank=2, only in multi mode):
- Created when target or anchor is UNKNOW
- CRITICAL: Only replace UNKNOW parts. Preserve all non-UNKNOW categories and spatial structure exactly.
- For missing ANCHOR: replace ONLY the UNKNOW anchor with semantically similar categories from scene.
- For missing TARGET: replace ONLY the UNKNOW target with related categories from scene.
- For BETWEEN relations: ALWAYS keep 2 separate anchors. Never collapse into 1.
- Populate `candidate_object_ids`, `grid_cells`, `target_grids`, and `anchor_grids` for the proxy objects.

CONTEXT hypothesis (rank=3, used as last resort):
- Created when both direct and proxy may fail
- Remove all spatial constraints and select constraints
- Replace target with the anchor categories (find the context/scene objects)
- Set expect_unique=false
- Populate `candidate_object_ids`, `grid_cells`, and `target_grids` covering the context objects.

=== CATEGORY RULES ===

1. SEMANTIC EXPANSION (CRITICAL): The `categories` field is a LIST. When the user mentions a general term,
   include ALL semantically related categories from SCENE CATEGORIES:
   - Query "a pillow" with scene [door, pillow, throw_pillow, sofa] -> categories: ["pillow", "throw_pillow"]
   - Query "the lamp" with scene [floor_lamp, table_lamp, sofa] -> categories: ["floor_lamp", "table_lamp"]

2. Every category MUST be EXACT string from SCENE CATEGORIES (case-sensitive, keep underscores).

3. If no suitable category exists in SCENE CATEGORIES, output ["UNKNOW"].

4. This applies to ALL QueryNode objects: root, anchors in spatial_constraints, references in select_constraint.

5. Map common relation synonyms: "on top of"->"on", "under"/"beneath"->"below", "close to"->"near"

6. "nearest/closest X" uses SelectConstraint with metric="distance", order="min", reference=X

7. "largest/biggest" uses SelectConstraint with metric="size", order="max", reference=null

=== VIEWPOINT DETECTION (CRITICAL for view-dependent queries) ===

Directional relations (left_of / right_of / in_front_of / behind) and direction-style
SelectConstraints (metric in {{x_position, y_position}}) are AMBIGUOUS by default - the
speaker's viewpoint is not implied by world coordinates. Decide reference_frame as
follows:

1. POSITIVE viewpoint cues (emit a ViewpointContext, set reference_frame="viewer",
   set viewpoint_context_id, set execution_policy="soft"):
   - "facing X", "if you are facing X", "when facing X" -> kind="facing_anchor"
     (or "facing_anchor_set" when X is plural like "the windows")
   - "entering (from/through) X" -> kind="entering_from"
   - "standing (at|in) the foot of X" / "standing in the middle of the room" /
     "from X's foot|head|side" -> kind="standing_at"
   X must resolve to one or more scene categories. If X is not in scene, do NOT
   invent geometry; fall through to rule (3) instead.

2. OBJECT-LOCAL cues (NO human observer; emit ViewpointContext with kind="object_local",
   set reference_frame="object_local", set viewpoint_context_id, set
   execution_policy="rank_only"):
   - "<objectA> facing <objectB>" when both A and B are scene categories
     (e.g. "the armchair facing the couch")
   - "the side of X nearest Y" style sub-frame references

3. NO observer cue but directional relation is present (e.g. "the couch on the left",
   "the chair to the right of the table" without any facing/entering cue):
   - reference_frame="ambiguous", viewpoint_context_id=null, execution_policy="rank_only"
   - Do NOT invent a ViewpointContext.
   This stops the directional relation from filtering candidates to empty.

4. NEGATIVE patterns (must NOT emit any ViewpointContext; keep reference_frame="world",
   execution_policy="hard"):
   - "looking for X" (not an observer cue, just "we want to find X")
   - "facing vertically" / "facing upward" / "facing the ceiling" (object orientation,
     not speaker frame)
   - "across from X" (this is world-frame proximity, not viewer-frame direction)

5. WORLD-FRAME relations (on / above / below / near / next_to / beside / inside /
   between) ALWAYS stay reference_frame="world", execution_policy="hard".

=== GRID & INSTANCE LOCALIZATION (CRITICAL) ===

When `SCENE GRID LAYOUT` and/or a Bird's-Eye View (BEV) image is provided:
- The scene is partitioned into a regular 0-indexed `(row, col)` grid where `row` increases
  top-to-bottom (`0..R-1`, along `-Y`) and `col` increases left-to-right (`0..C-1`, along `+X`).
- In `SCENE GRID LAYOUT`:
  - `#id category` means object `#id`'s 3D centroid is inside that cell (primary cell).
  - `#id category*` (with an asterisk) means a large object `#id` spans into that cell from a neighboring cell.
  - `Observable empty grids` lists camera-observed floor/wall cells without segmented 3D objects.
- For EVERY QueryNode (target `root`, spatial `anchors`, select `reference`):
  1. Set `candidate_object_ids` to the integer `#id`(s) that satisfy the query's spatial/select
     constraints (e.g., if `#1 pillow` is in `(1, 2)` with `#0 sofa`, while `#9 pillow` is in `(3, 3)`,
     select `candidate_object_ids: [1]`).
  2. Set `grid_cells` to the `[row, col]` cell(s) of that node.
  3. Set `grounding_query.target_grids` to the target's `[row, col]` cell(s) and
     `grounding_query.anchor_grids` to the anchor/reference `[row, col]` cell(s).
  4. When a target or anchor category is `["UNKNOW"]`, still infer its `grid_cells` / `target_grids`
     from its spatial relation to known anchors or observable empty grids."""


def get_few_shot_examples() -> str:
    """In-context examples for the parser, covering single/multi/viewpoint/grid cases."""
    return """
EXAMPLES:

=== EXAMPLE 1: Simple query - target and anchor both exist (SINGLE mode) ===
Query: "the pillow on the sofa" (scene has: pillow, throw_pillow, sofa, door; grid layout: (1, 2): #0 sofa, #1 pillow; (3, 0): #2 door)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the sofa",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "candidate_object_ids": [1],
          "grid_cells": [[1, 2]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["sofa"],
                  "candidate_object_ids": [0],
                  "grid_cells": [[1, 2]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[1, 2]],
        "anchor_grids": [[1, 2]]
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}

=== EXAMPLE 2: Superlative with reference (SINGLE mode) ===
Query: "the sofa nearest the door" (scene has: sofa, door, window; (0, 1): #1 door; (1, 1): #2 sofa; (3, 3): #5 sofa)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the sofa nearest the door",
        "root": {
          "categories": ["sofa"],
          "candidate_object_ids": [2],
          "grid_cells": [[1, 1]],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {
              "categories": ["door"],
              "candidate_object_ids": [1],
              "grid_cells": [[0, 1]],
              "attributes": [],
              "spatial_constraints": [],
              "select_constraint": null
            },
            "position": null
          }
        },
        "expect_unique": true,
        "target_grids": [[1, 1]],
        "anchor_grids": [[0, 1]]
      },
      "lexical_hints": ["sofa", "door", "nearest"]
    }
  ]
}

=== EXAMPLE 3: Missing anchor - "bed" not in scene (MULTI mode with PROXY) ===
Query: "the pillow on the bed" (scene has: pillow, throw_pillow, sofa, armchair, door - NO bed; (2, 1): #3 sofa, #4 pillow; (2, 2): #6 armchair)
NOTE: "bed" is NOT in scene, so DIRECT anchor uses ["UNKNOW"] but still locates grid [[2, 1]]. Add PROXY hypothesis with "sofa"/"armchair".
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "candidate_object_ids": [4],
          "grid_cells": [[2, 1]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["UNKNOW"],
                  "candidate_object_ids": [],
                  "grid_cells": [[2, 1]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[2, 1]],
        "anchor_grids": [[2, 1]]
      },
      "lexical_hints": ["pillow", "bed"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "candidate_object_ids": [4],
          "grid_cells": [[2, 1]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["sofa", "armchair"],
                  "candidate_object_ids": [3, 6],
                  "grid_cells": [[2, 1], [2, 2]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[2, 1]],
        "anchor_grids": [[2, 1], [2, 2]]
      },
      "lexical_hints": ["proxy_anchor"]
    }
  ]
}

=== EXAMPLE 4: Missing target - "laptop" not in scene (MULTI mode with PROXY and CONTEXT) ===
Query: "the laptop on the table" (scene has: book, cup, side_table, coffee_table, chair - NO laptop; (1, 3): #2 side_table, #5 book; (2, 2): #4 coffee_table, #8 cup)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the laptop on the table",
        "root": {
          "categories": ["UNKNOW"],
          "candidate_object_ids": [],
          "grid_cells": [[1, 3], [2, 2]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["side_table", "coffee_table"],
                  "candidate_object_ids": [2, 4],
                  "grid_cells": [[1, 3], [2, 2]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[1, 3], [2, 2]],
        "anchor_grids": [[1, 3], [2, 2]]
      },
      "lexical_hints": ["laptop", "table"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the laptop on the table",
        "root": {
          "categories": ["book", "cup"],
          "candidate_object_ids": [5, 8],
          "grid_cells": [[1, 3], [2, 2]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["side_table", "coffee_table"],
                  "candidate_object_ids": [2, 4],
                  "grid_cells": [[1, 3], [2, 2]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[1, 3], [2, 2]],
        "anchor_grids": [[1, 3], [2, 2]]
      },
      "lexical_hints": ["proxy"]
    },
    {
      "kind": "context",
      "rank": 3,
      "grounding_query": {
        "raw_query": "context for: the laptop on the table",
        "root": {
          "categories": ["side_table", "coffee_table"],
          "candidate_object_ids": [2, 4],
          "grid_cells": [[1, 3], [2, 2]],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false,
        "target_grids": [[1, 3], [2, 2]],
        "anchor_grids": []
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 5: BETWEEN relation with missing target (MULTI mode, preserve both anchors) ===
Query: "the blue object between the table and the fridge" (scene has: table, fridge, bag, chair; (1, 1): #1 table; (1, 2): #3 bag; (1, 3): #2 fridge)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the blue object between the table and the fridge",
        "root": {
          "categories": ["UNKNOW"],
          "candidate_object_ids": [],
          "grid_cells": [[1, 2]],
          "attributes": ["blue"],
          "spatial_constraints": [
            {
              "relation": "between",
              "anchors": [
                {"categories": ["table"], "candidate_object_ids": [1], "grid_cells": [[1, 1]], "attributes": [], "spatial_constraints": [], "select_constraint": null},
                {"categories": ["fridge"], "candidate_object_ids": [2], "grid_cells": [[1, 3]], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[1, 2]],
        "anchor_grids": [[1, 1], [1, 3]]
      },
      "lexical_hints": ["blue", "table", "fridge"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the blue object between the table and the fridge",
        "root": {
          "categories": ["bag", "chair"],
          "candidate_object_ids": [3],
          "grid_cells": [[1, 2]],
          "attributes": ["blue"],
          "spatial_constraints": [
            {
              "relation": "between",
              "anchors": [
                {"categories": ["table"], "candidate_object_ids": [1], "grid_cells": [[1, 1]], "attributes": [], "spatial_constraints": [], "select_constraint": null},
                {"categories": ["fridge"], "candidate_object_ids": [2], "grid_cells": [[1, 3]], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[1, 2]],
        "anchor_grids": [[1, 1], [1, 3]]
      },
      "lexical_hints": ["proxy"]
    },
    {
      "kind": "context",
      "rank": 3,
      "grounding_query": {
        "raw_query": "context for: the blue object between the table and the fridge",
        "root": {
          "categories": ["table", "fridge"],
          "candidate_object_ids": [1, 2],
          "grid_cells": [[1, 1], [1, 3]],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false,
        "target_grids": [[1, 1], [1, 2], [1, 3]],
        "anchor_grids": []
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 6: Semantic expansion only, all exist (SINGLE mode) ===
Query: "the cushion on the couch" (scene has: sofa, sofa_seat_cushion, pillow, throw_pillow, door; (2, 2): #0 sofa, #4 sofa_seat_cushion, #5 pillow)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the cushion on the couch",
        "root": {
          "categories": ["sofa_seat_cushion", "pillow", "throw_pillow"],
          "candidate_object_ids": [4, 5],
          "grid_cells": [[2, 2]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {
                  "categories": ["sofa"],
                  "candidate_object_ids": [0],
                  "grid_cells": [[2, 2]],
                  "attributes": [],
                  "spatial_constraints": [],
                  "select_constraint": null
                }
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "target_grids": [[2, 2]],
        "anchor_grids": [[2, 2]]
      },
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}

=== EXAMPLE 7: Speaker-facing viewpoint with directional relation (SINGLE mode) ===
Query: "Facing the door, the cabinet on the right by the love seat" (scene has: door, cabinet, loveseat; (0, 2): #1 door; (1, 3): #4 cabinet, #5 loveseat)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "Facing the door, the cabinet on the right by the love seat",
        "root": {
          "categories": ["cabinet"],
          "candidate_object_ids": [4],
          "grid_cells": [[1, 3]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "right_of",
              "anchors": [
                {"categories": ["door"], "candidate_object_ids": [1], "grid_cells": [[0, 2]], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ],
              "reference_frame": "viewer",
              "viewpoint_context_id": "vp_facing_door",
              "execution_policy": "soft"
            },
            {
              "relation": "near",
              "anchors": [
                {"categories": ["loveseat"], "candidate_object_ids": [5], "grid_cells": [[1, 3]], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": [
          {
            "id": "vp_facing_door",
            "kind": "facing_anchor",
            "facing_anchor": {"categories": ["door"], "candidate_object_ids": [1], "grid_cells": [[0, 2]], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "raw_phrase": "Facing the door",
            "confidence": "explicit"
          }
        ],
        "target_grids": [[1, 3]],
        "anchor_grids": [[0, 2], [1, 3]]
      },
      "lexical_hints": ["facing", "door", "cabinet", "right", "love seat"]
    }
  ]
}

=== EXAMPLE 8: Speaker-facing viewpoint with SelectConstraint axis (SINGLE mode) ===
Query: "Facing the windows, the window that is on the right side." (scene has: window; (0, 1): #2 window; (0, 3): #3 window)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "Facing the windows, the window that is on the right side.",
        "root": {
          "categories": ["window"],
          "candidate_object_ids": [3],
          "grid_cells": [[0, 3]],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "x_position",
            "order": "max",
            "reference": null,
            "position": null,
            "reference_frame": "viewer",
            "viewpoint_context_id": "vp_facing_windows",
            "execution_policy": "soft"
          }
        },
        "expect_unique": true,
        "viewpoint_contexts": [
          {
            "id": "vp_facing_windows",
            "kind": "facing_anchor_set",
            "facing_anchor": {"categories": ["window"], "candidate_object_ids": [2, 3], "grid_cells": [[0, 1], [0, 3]], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "raw_phrase": "Facing the windows",
            "confidence": "explicit"
          }
        ],
        "target_grids": [[0, 3]],
        "anchor_grids": [[0, 1], [0, 3]]
      },
      "lexical_hints": ["facing", "windows", "right"]
    }
  ]
}

=== EXAMPLE 9: Directional relation with NO observer cue (ambiguous frame) ===
Query: "the chair sitting alone at left side of the yellow table" (scene has: chair, table; (2, 1): #6 chair; (2, 2): #4 table)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the chair sitting alone at left side of the yellow table",
        "root": {
          "categories": ["chair"],
          "candidate_object_ids": [6],
          "grid_cells": [[2, 1]],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "left_of",
              "anchors": [
                {"categories": ["table"], "candidate_object_ids": [4], "grid_cells": [[2, 2]], "attributes": ["yellow"], "spatial_constraints": [], "select_constraint": null}
              ],
              "reference_frame": "ambiguous",
              "viewpoint_context_id": null,
              "execution_policy": "rank_only"
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": [],
        "target_grids": [[2, 1]],
        "anchor_grids": [[2, 2]]
      },
      "lexical_hints": ["chair", "left", "yellow table"]
    }
  ]
}

=== EXAMPLE 10: NEGATIVE - object orientation is not speaker viewpoint ===
Query: "The rail in the shower stall. It is facing vertically." (scene has: rail, grab_bar, shower_stall; (0, 0): #8 rail, #9 shower_stall)
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "The rail in the shower stall. It is facing vertically.",
        "root": {
          "categories": ["rail", "grab_bar"],
          "candidate_object_ids": [8],
          "grid_cells": [[0, 0]],
          "attributes": ["vertical"],
          "spatial_constraints": [
            {
              "relation": "inside",
              "anchors": [
                {"categories": ["shower_stall"], "candidate_object_ids": [9], "grid_cells": [[0, 0]], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": [],
        "target_grids": [[0, 0]],
        "anchor_grids": [[0, 0]]
      },
      "lexical_hints": ["rail", "shower stall", "vertically"]
    }
  ]
}
"""
