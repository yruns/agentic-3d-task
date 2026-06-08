"""Prompt templates for query parsing.

The system prompt and few-shot examples instruct the LLM to emit a
``HypothesisOutputV1`` JSON object. They are kept verbatim from the source
pipeline because their exact wording is tuned for parsing quality.
"""

from __future__ import annotations

from keyframe.models.hypotheses import SUPPORTED_RELATIONS_STR


def get_system_prompt() -> str:
    """System prompt describing the HypothesisOutputV1 schema and parsing rules."""
    return f"""You are a spatial query parser for 3D scene understanding.
Your task is to parse natural language queries into a structured HypothesisOutputV1 JSON format.

The output must be a valid HypothesisOutputV1 with the following structure:
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

Each QueryNode has:
- categories: LIST of object types (MUST be EXACT strings from SCENE CATEGORIES, or ["UNKNOW"] if no match)
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
- metric: "distance", "size", "height", "x_position", etc.
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

PROXY hypothesis (rank=2, only in multi mode):
- Created when target or anchor is UNKNOW
- CRITICAL: Only replace UNKNOW parts. Preserve all non-UNKNOW categories and spatial structure exactly.
- For missing ANCHOR: replace ONLY the UNKNOW anchor with semantically similar categories from scene.
  Keep all other anchors unchanged. (e.g., "between table and bed" where bed is missing ->
  anchor1=["table"] stays, anchor2=["sofa","armchair"] replaces bed. Both anchors preserved.)
- For missing TARGET: replace ONLY the UNKNOW target with related categories from scene.
  Keep ALL spatial constraints and anchors exactly as in the DIRECT hypothesis.
- For BETWEEN relations: ALWAYS keep 2 separate anchors. Never collapse into 1.

CONTEXT hypothesis (rank=3, used as last resort):
- Created when both direct and proxy may fail
- Remove all spatial constraints and select constraints
- Replace target with the anchor categories (find the context/scene objects)
- Set expect_unique=false

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
   - Statements about who is "looking at" something when there is no relation to
     evaluate directionally

5. WORLD-FRAME relations (on / above / below / near / next_to / beside / inside /
   between) ALWAYS stay reference_frame="world", execution_policy="hard". The only
   exception is when a spatial constraint is genuinely speaker-relative AND uses one
   of those words colloquially; in practice, leave them as world.

When you emit ViewpointContext(s), put them in grounding_query.viewpoint_contexts as
a list, and reference them by id from the constraints that need them. The same
context may be referenced by multiple constraints in one grounding_query.

=== VISUAL CONTEXT (if image provided) ===

When a Bird's Eye View (BEV) image is provided:
- Each object is shown as a labeled circle at its centroid position
- Labels follow format "NNN: category" (e.g., "001: sofa", "002: pillow")
- Object IDs correspond to the SCENE CATEGORIES list
- Use this visual context to understand spatial relationships
- Reference the image to verify "near", "on", "between" relationships
- Use object positions to resolve ambiguous references
- Note: In the BEV, Y-axis increases downward (image coordinates)"""


def get_few_shot_examples() -> str:
    """In-context examples for the parser, covering single/multi/viewpoint cases."""
    return """
EXAMPLES:

=== EXAMPLE 1: Simple query - target and anchor both exist (SINGLE mode) ===
Query: "the pillow on the sofa" (scene has: pillow, throw_pillow, sofa, door)
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}

=== EXAMPLE 2: Superlative with reference (SINGLE mode) ===
Query: "the sofa nearest the door" (scene has: sofa, door, window)
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
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "position": null
          }
        },
        "expect_unique": true
      },
      "lexical_hints": ["sofa", "door", "nearest"]
    }
  ]
}

=== EXAMPLE 3: Missing anchor - "bed" not in scene (MULTI mode with PROXY) ===
Query: "the pillow on the bed" (scene has: pillow, throw_pillow, sofa, armchair, door - NO bed)
NOTE: "bed" is NOT in scene, so anchor uses ["UNKNOW"]. Add PROXY hypothesis with "sofa" as proxy anchor.
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa", "armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy_anchor"]
    }
  ]
}

=== EXAMPLE 4: Missing target - "laptop" not in scene (MULTI mode with PROXY and CONTEXT) ===
Query: "the laptop on the table" (scene has: book, cup, side_table, coffee_table, chair - NO laptop)
NOTE: "laptop" is NOT in scene. PROXY tries related objects like "book". CONTEXT falls back to anchor.
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table", "coffee_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
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
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 5: BETWEEN relation with missing target (MULTI mode, preserve both anchors) ===
Query: "the blue object between the table and the fridge" (scene has: table, fridge, bag, chair - NO "blue object" match)
NOTE: Target is UNKNOW. PROXY replaces target with scene objects ("bag"). BOTH anchors "table" and "fridge" are preserved exactly.
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
          "attributes": ["blue"],
          "spatial_constraints": [
            {
              "relation": "between",
              "anchors": [
                {"categories": ["table"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
                {"categories": ["fridge"], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
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
          "attributes": ["blue"],
          "spatial_constraints": [
            {
              "relation": "between",
              "anchors": [
                {"categories": ["table"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
                {"categories": ["fridge"], "attributes": [], "spatial_constraints": [], "select_constraint": null}
              ]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
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
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false
      },
      "lexical_hints": ["context"]
    }
  ]
}

=== EXAMPLE 6: Semantic expansion only, all exist (SINGLE mode) ===
Query: "the cushion on the couch" (scene has: sofa, sofa_seat_cushion, pillow, throw_pillow, door)
NOTE: "cushion" expands to all cushion-like categories; "couch" maps to "sofa". All exist -> SINGLE mode.
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}

=== EXAMPLE 7: Speaker-facing viewpoint with directional relation (SINGLE mode) ===
Query: "Facing the door, the cabinet on the right by the love seat" (scene has: door, cabinet, loveseat)
NOTE: "Facing the door" creates a ViewpointContext. The right_of relation uses reference_frame="viewer"
with execution_policy="soft"; near(loveseat) remains world/hard by default.
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "right_of",
              "anchors": [{"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null}],
              "reference_frame": "viewer",
              "viewpoint_context_id": "vp_facing_door",
              "execution_policy": "soft"
            },
            {
              "relation": "near",
              "anchors": [{"categories": ["loveseat"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": [
          {
            "id": "vp_facing_door",
            "kind": "facing_anchor",
            "facing_anchor": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "raw_phrase": "Facing the door",
            "confidence": "explicit"
          }
        ]
      },
      "lexical_hints": ["facing", "door", "cabinet", "right", "love seat"]
    }
  ]
}

=== EXAMPLE 8: Speaker-facing viewpoint with SelectConstraint axis (SINGLE mode) ===
Query: "Facing the windows, the window that is on the right side." (scene has: window)
NOTE: "on the right side" becomes a viewer-frame SelectConstraint over x_position.
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
            "facing_anchor": {"categories": ["window"], "attributes": [], "spatial_constraints": [], "select_constraint": null},
            "raw_phrase": "Facing the windows",
            "confidence": "explicit"
          }
        ]
      },
      "lexical_hints": ["facing", "windows", "right"]
    }
  ]
}

=== EXAMPLE 9: Directional relation with NO observer cue (ambiguous frame) ===
Query: "the chair sitting alone at left side of the yellow table" (scene has: chair, table)
NOTE: There is no facing/entering/standing cue. Do not fabricate a ViewpointContext; use rank_only.
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
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "left_of",
              "anchors": [{"categories": ["table"], "attributes": ["yellow"], "spatial_constraints": [], "select_constraint": null}],
              "reference_frame": "ambiguous",
              "viewpoint_context_id": null,
              "execution_policy": "rank_only"
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": []
      },
      "lexical_hints": ["chair", "left", "yellow table"]
    }
  ]
}

=== EXAMPLE 10: NEGATIVE - object orientation is not speaker viewpoint ===
Query: "The rail in the shower stall. It is facing vertically." (scene has: rail, grab_bar, shower_stall)
NOTE: "facing vertically" describes object orientation, not observer viewpoint. Keep world-frame defaults.
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
          "attributes": ["vertical"],
          "spatial_constraints": [
            {
              "relation": "inside",
              "anchors": [{"categories": ["shower_stall"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true,
        "viewpoint_contexts": []
      },
      "lexical_hints": ["rail", "shower stall", "vertically"]
    }
  ]
}
"""
