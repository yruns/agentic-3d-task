"""Structured representation of nested spatial queries.

These pydantic models are the canonical parsed form of a natural-language
query. They support arbitrary nesting of spatial constraints and selection
operations, e.g. "the pillow on the sofa nearest the door":

    QueryNode(
        categories=["pillow"],
        spatial_constraints=[
            SpatialConstraint(
                relation="on",
                anchors=[
                    QueryNode(
                        categories=["sofa"],
                        select_constraint=SelectConstraint(
                            constraint_type=ConstraintType.SUPERLATIVE,
                            metric="distance",
                            order="min",
                            reference=QueryNode(categories=["door"]),
                        ),
                    )
                ],
            )
        ],
    )
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_GRID_STR_RE = re.compile(r"^\s*\(?[rR]?\s*(\d+)\s*[,;:\s]\s*[cC]?\s*(\d+)\s*\)?\s*$")
_INLINE_ID_CAT_RE = re.compile(r"^\s*#(\d+)(?:\s*[:,-]?\s*(.*))?$")
_INT_TOKEN_RE = re.compile(r"#?(\d+)")


def _coerce_grid_cell(item: object) -> tuple[int, int]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        r, c = item[0], item[1]
        if isinstance(r, (int, float)) and isinstance(c, (int, float)):
            return (int(r), int(c))
    if isinstance(item, Mapping):
        r = item.get("row", item.get("r"))
        c = item.get("col", item.get("c"))
        if isinstance(r, (int, float)) and isinstance(c, (int, float)):
            return (int(r), int(c))
    if isinstance(item, str):
        match = _GRID_STR_RE.match(item)
        if match is not None:
            return (int(match.group(1)), int(match.group(2)))
    raise ValueError(
        f"invalid grid coordinate {item!r}; expected [row, col] or {{'row': r, 'col': c}}"
    )


def _coerce_grid_list(value: object) -> list[tuple[int, int]]:
    """Normalize arbitrary LLM grid list representations into deduplicated (row, col) tuples."""
    if value is None or value == "":
        return []
    if isinstance(value, (str, Mapping)):
        items: Sequence[object] = [value]
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        return []

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in items:
        try:
            cell = _coerce_grid_cell(item)
        except ValueError:
            continue
        if cell not in seen:
            seen.add(cell)
            deduped.append(cell)
    return deduped


def _coerce_object_id_list(value: object) -> list[int]:
    """Normalize candidate object IDs (e.g. [7, '#3', '12']) into deduplicated ints."""
    if value is None or value == "":
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_items: Sequence[object] = [int(value)]
    elif isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = value
    else:
        return []

    deduped: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        oid: int | None = None
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            oid = int(item)
        elif isinstance(item, str):
            match = _INT_TOKEN_RE.search(item)
            if match is not None:
                oid = int(match.group(1))
        if oid is not None and oid >= 0 and oid not in seen:
            seen.add(oid)
            deduped.append(oid)
    return deduped


class ConstraintType(str, Enum):
    """Type of selection constraint."""

    SUPERLATIVE = "superlative"  # nearest, largest, highest, smallest
    COMPARATIVE = "comparative"  # closer than, larger than
    ORDINAL = "ordinal"  # first, second, third


class SpatialRelation(str, Enum):
    """Predefined spatial relations.

    View-independent relations (vertical + distance) support quick
    coordinate-based filtering. View-dependent (left/right/front/behind) and
    complex (inside/between) relations require full geometric reasoning.
    """

    # View-independent: vertical (Z-axis, gravity defines "up")
    ON = "on"
    ABOVE = "above"
    BELOW = "below"

    # View-independent: distance (Euclidean)
    NEAR = "near"
    NEXT_TO = "next_to"
    BESIDE = "beside"

    # View-dependent: horizontal (require observer viewpoint)
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"

    # Complex: containment / multi-object
    INSIDE = "inside"
    BETWEEN = "between"

    @classmethod
    def from_string(cls, value: str) -> SpatialRelation | None:
        """Convert a (possibly aliased) string to a relation, or None.

        None means the relation is not predefined and quick filtering cannot
        be applied (the executor falls back to full relation checking).
        """
        if not value:
            return None

        normalized = value.lower().strip().replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError:
            pass

        aliases: dict[str, SpatialRelation] = {
            "on_top_of": cls.ON,
            "upon": cls.ON,
            "atop": cls.ON,
            "resting_on": cls.ON,
            "over": cls.ABOVE,
            "higher_than": cls.ABOVE,
            "under": cls.BELOW,
            "beneath": cls.BELOW,
            "underneath": cls.BELOW,
            "lower_than": cls.BELOW,
            "left": cls.LEFT_OF,
            "to_the_left_of": cls.LEFT_OF,
            "right": cls.RIGHT_OF,
            "to_the_right_of": cls.RIGHT_OF,
            "front": cls.IN_FRONT_OF,
            "facing": cls.IN_FRONT_OF,
            "back": cls.BEHIND,
            "back_of": cls.BEHIND,
            "in_back_of": cls.BEHIND,
            "close_to": cls.NEAR,
            "nearby": cls.NEAR,
            "adjacent_to": cls.NEXT_TO,
            "adjacent": cls.NEXT_TO,
            "in": cls.INSIDE,
            "within": cls.INSIDE,
            "contained_in": cls.INSIDE,
            "in_between": cls.BETWEEN,
        }
        return aliases.get(normalized)

    def is_view_dependent(self) -> bool:
        """Whether this relation depends on the observer's viewpoint."""
        return self in {
            SpatialRelation.LEFT_OF,
            SpatialRelation.RIGHT_OF,
            SpatialRelation.IN_FRONT_OF,
            SpatialRelation.BEHIND,
        }

    def supports_quick_filter(self) -> bool:
        """Whether this relation supports quick coordinate-based filtering."""
        return self in {
            SpatialRelation.ON,
            SpatialRelation.ABOVE,
            SpatialRelation.BELOW,
            SpatialRelation.NEAR,
            SpatialRelation.NEXT_TO,
            SpatialRelation.BESIDE,
        }

    def get_filter_type(self) -> str | None:
        """Return the quick-filter family ("vertical"/"distance") or None."""
        if self in {SpatialRelation.ON, SpatialRelation.ABOVE, SpatialRelation.BELOW}:
            return "vertical"
        if self in {
            SpatialRelation.NEAR,
            SpatialRelation.NEXT_TO,
            SpatialRelation.BESIDE,
        }:
            return "distance"
        return None


SUPPORTED_RELATIONS: list[str] = [r.value for r in SpatialRelation]
SUPPORTED_RELATIONS_STR: str = ", ".join(SUPPORTED_RELATIONS)

# Relations whose semantics depend on the speaker / observer / object frame.
# When the parser routes one of these into a non-world reference_frame, the
# executor must honour the configured execution_policy.
DIRECTIONAL_RELATIONS: frozenset[str] = frozenset(
    {
        SpatialRelation.LEFT_OF.value,
        SpatialRelation.RIGHT_OF.value,
        SpatialRelation.IN_FRONT_OF.value,
        SpatialRelation.BEHIND.value,
    }
)


class ReferenceFrame(str, Enum):
    """Frame in which a spatial / select constraint is evaluated.

    WORLD reproduces global X/Y/Z semantics. VIEWER and OBJECT_LOCAL require a
    viewpoint context id. AMBIGUOUS marks a directional relation with no
    explicit observer cue so the executor can route it to a non-filtering
    policy without inventing a viewer pose.
    """

    WORLD = "world"
    VIEWER = "viewer"
    OBJECT_LOCAL = "object_local"
    AMBIGUOUS = "ambiguous"


class ViewpointKind(str, Enum):
    """Linguistic frame of a viewpoint context."""

    FACING_ANCHOR = "facing_anchor"
    FACING_ANCHOR_SET = "facing_anchor_set"
    ENTERING_FROM = "entering_from"
    STANDING_AT = "standing_at"
    OBJECT_LOCAL = "object_local"


class ExecutionPolicy(str, Enum):
    """How the executor treats a failing constraint.

    HARD: filter to empty (legacy). SOFT: keep candidates, scale score by
    satisfaction. RANK_ONLY: never filter; record per-candidate score for
    downstream rerankers.
    """

    HARD = "hard"
    SOFT = "soft"
    RANK_ONLY = "rank_only"


class ViewpointContext(BaseModel):
    """A named viewpoint referenced by spatial / select constraints.

    Exactly one of ``facing_anchor`` / ``origin_anchor`` / ``subject_anchor``
    is expected for a well-formed context; the parser-normalize layer
    downgrades unresolved contexts to a non-filtering policy rather than
    failing validation.
    """

    id: str = Field(
        ..., min_length=1, description="Unique id referenced by *_context_id fields"
    )
    kind: ViewpointKind = Field(
        ..., description="Linguistic frame this context represents"
    )
    facing_anchor: QueryNode | None = Field(
        default=None, description="Anchor for FACING_ANCHOR / FACING_ANCHOR_SET"
    )
    origin_anchor: QueryNode | None = Field(
        default=None, description="Anchor for ENTERING_FROM / STANDING_AT"
    )
    subject_anchor: QueryNode | None = Field(
        default=None, description="Subject object for OBJECT_LOCAL kind"
    )
    raw_phrase: str = Field(
        default="", description="Original natural-language fragment, for trace only"
    )
    confidence: Literal["explicit", "inferred", "ambiguous"] = Field(
        default="explicit",
        description="explicit = literal cue; inferred = deduced; ambiguous = undecided",
    )


class QueryNode(BaseModel):
    """A target/anchor node in the parsed query tree."""

    categories: list[str] = Field(
        default_factory=lambda: ["UNKNOW"],
        min_length=1,
        description="Object categories to search for, including related synonyms.",
    )
    candidate_object_ids: list[int] = Field(
        default_factory=list,
        description="Optional candidate object #ids from SCENE GRID LAYOUT / BEV.",
    )
    grid_cells: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Optional 0-indexed (row, col) grid coordinates for this node.",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Attribute filters like 'red', 'large', 'wooden'.",
    )
    spatial_constraints: list[SpatialConstraint] = Field(
        default_factory=list,
        description="Spatial constraints (AND logic between them).",
    )
    select_constraint: SelectConstraint | None = Field(
        default=None,
        description="Selection constraint like 'nearest', 'largest', 'second'.",
    )
    open_ended: bool = Field(
        default=False,
        description="True for 'what is near/on X?' queries with an unknown target; "
        "categories=['UNKNOW'] and spatial_constraints define the anchor.",
    )
    node_id: str = Field(
        default="", description="Unique identifier for tracking during execution."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_node_payload(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data: dict[str, Any] = dict(value)

        # Normalize candidate_object_ids aliases
        raw_ids: list[object] = []
        for key in (
            "candidate_object_ids",
            "object_ids",
            "obj_ids",
            "ids",
            "object_id",
            "obj_id",
        ):
            if key in data and data[key] is not None:
                raw_ids.extend(_coerce_object_id_list(data.pop(key)))

        # Normalize grid_cells aliases
        raw_grids: list[tuple[int, int]] = []
        for key in ("grid_cells", "grids", "target_grids", "cells"):
            if key in data and data[key] is not None:
                raw_grids.extend(_coerce_grid_list(data.pop(key)))

        # Normalize categories and extract inline '#id category' tokens
        raw_cats = data.get("categories", data.pop("category", None))
        if isinstance(raw_cats, str):
            raw_cat_list: list[object] = [raw_cats]
        elif isinstance(raw_cats, Sequence) and not isinstance(raw_cats, (str, bytes)):
            raw_cat_list = list(raw_cats)
        else:
            raw_cat_list = []

        cleaned_cats: list[str] = []
        for item in raw_cat_list:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                raw_ids.append(int(item))
                continue
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if not stripped:
                continue
            inline_match = _INLINE_ID_CAT_RE.match(stripped)
            if inline_match is not None:
                raw_ids.append(int(inline_match.group(1)))
                rest = (inline_match.group(2) or "").strip()
                if rest and rest not in cleaned_cats:
                    cleaned_cats.append(rest)
            elif stripped not in cleaned_cats:
                cleaned_cats.append(stripped)

        data["categories"] = cleaned_cats or ["UNKNOW"]
        data["candidate_object_ids"] = _coerce_object_id_list(raw_ids)
        data["grid_cells"] = _coerce_grid_list(raw_grids)
        return data

    @field_validator("candidate_object_ids", mode="before")
    @classmethod
    def _validate_candidate_ids(cls, value: object) -> list[int]:
        return _coerce_object_id_list(value)

    @field_validator("grid_cells", mode="before")
    @classmethod
    def _validate_grid_cells(cls, value: object) -> list[tuple[int, int]]:
        return _coerce_grid_list(value)

    @property
    def category(self) -> str:
        """Primary category (first in list)."""
        return self.categories[0] if self.categories else ""


class SpatialConstraint(BaseModel):
    """A single spatial relation between a node and its anchors."""

    relation: str = Field(
        ..., description=f"Spatial relation, one of: {SUPPORTED_RELATIONS_STR}"
    )
    anchors: list[QueryNode] = Field(
        default_factory=list,
        description="Reference objects. Usually 1, can be 2 for 'between'.",
    )
    reference_frame: ReferenceFrame = Field(
        default=ReferenceFrame.WORLD,
        description="Frame in which this relation is evaluated.",
    )
    viewpoint_context_id: str | None = Field(
        default=None,
        description="Id of a ViewpointContext; required for VIEWER / OBJECT_LOCAL.",
    )
    execution_policy: ExecutionPolicy = Field(
        default=ExecutionPolicy.HARD,
        description="How a failing constraint is handled.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_spatial_constraint(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data: dict[str, Any] = dict(value)
        if "anchors" not in data or data["anchors"] is None:
            single_anchor = data.pop("anchor", None)
            if isinstance(single_anchor, list):
                data["anchors"] = single_anchor
            elif single_anchor is not None:
                data["anchors"] = [single_anchor]
            else:
                data["anchors"] = []
        elif isinstance(data["anchors"], Mapping):
            data["anchors"] = [data["anchors"]]
        if isinstance(data.get("relation"), str):
            data["relation"] = data["relation"].strip().lower().replace(" ", "_")
        return data

    @property
    def relation_enum(self) -> SpatialRelation | None:
        """Normalized relation enum, or None if not predefined."""
        return SpatialRelation.from_string(self.relation)

    @property
    def supports_quick_filter(self) -> bool:
        """Whether this constraint can use quick coordinate-based filtering."""
        rel = self.relation_enum
        return rel is not None and rel.supports_quick_filter()

    @property
    def filter_type(self) -> str | None:
        """Quick-filter family for this constraint, or None."""
        rel = self.relation_enum
        return rel.get_filter_type() if rel is not None else None


_METRIC_ALIASES: dict[str, str] = {
    "dist": "distance",
    "proximity": "distance",
    "closeness": "distance",
    "volume": "size",
    "area": "size",
    "scale": "size",
    "z": "height",
    "z_position": "height",
    "elevation": "height",
    "altitude": "height",
    "x": "x_position",
    "horizontal": "x_position",
    "left_right": "x_position",
    "y": "y_position",
    "depth": "y_position",
    "front_back": "y_position",
}

_ORDER_ALIASES: dict[str, str] = {
    "nearest": "min",
    "closest": "min",
    "smallest": "min",
    "lowest": "min",
    "shortest": "min",
    "farthest": "max",
    "furthest": "max",
    "largest": "max",
    "biggest": "max",
    "highest": "max",
    "tallest": "max",
    "ascending": "asc",
    "first": "asc",
    "descending": "desc",
    "last": "desc",
}


class SelectConstraint(BaseModel):
    """A selection / ranking constraint applied to candidate matches."""

    constraint_type: ConstraintType = Field(
        default=ConstraintType.SUPERLATIVE,
        description="superlative / comparative / ordinal.",
    )
    metric: str = Field(
        default="distance",
        description="Metric: 'distance', 'size', 'height', 'x_position', 'y_position'.",
    )
    order: str = Field(default="min", description="Order: 'min', 'max', 'asc', 'desc'.")
    reference: QueryNode | None = Field(
        default=None,
        description="Reference object for distance comparisons.",
    )
    position: int | None = Field(
        default=None, description="Position for ordinal selection (1=first)."
    )
    reference_frame: ReferenceFrame = Field(
        default=ReferenceFrame.WORLD,
        description="Frame in which the metric axis is evaluated.",
    )
    viewpoint_context_id: str | None = Field(
        default=None,
        description="Id of a ViewpointContext; required for VIEWER / OBJECT_LOCAL.",
    )
    execution_policy: ExecutionPolicy = Field(
        default=ExecutionPolicy.HARD,
        description="How a failing selection is handled.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_select_constraint(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data: dict[str, Any] = dict(value)
        raw_metric = str(data.get("metric") or "distance").strip().lower()
        data["metric"] = _METRIC_ALIASES.get(raw_metric, raw_metric)
        raw_order = str(data.get("order") or "min").strip().lower()
        data["order"] = _ORDER_ALIASES.get(raw_order, raw_order)
        return data

    @model_validator(mode="after")
    def _validate_constraint(self) -> SelectConstraint:
        if self.constraint_type == ConstraintType.ORDINAL and self.position is None:
            self.position = 1
        return self


class GroundingQuery(BaseModel):
    """Top-level parsed query: a target node plus optional viewpoint and grid contexts."""

    raw_query: str = Field(default="", description="Original natural language query.")
    root: QueryNode = Field(
        ..., description="Root query node representing the target object."
    )
    expect_unique: bool = Field(
        default=True,
        description="True for 'the X' (single result), False for 'X' / 'Xs'.",
    )
    viewpoint_contexts: list[ViewpointContext] = Field(
        default_factory=list,
        description="Viewpoint contexts referenced by constraints in this query.",
    )
    target_grids: list[tuple[int, int]] = Field(
        default_factory=list,
        description="0-indexed (row, col) BEV grid coordinates for the target object/region.",
    )
    anchor_grids: list[tuple[int, int]] = Field(
        default_factory=list,
        description="0-indexed (row, col) BEV grid coordinates for reference anchor objects/regions.",
    )

    @field_validator("target_grids", "anchor_grids", mode="before")
    @classmethod
    def _normalize_grids_field(cls, value: object) -> list[tuple[int, int]]:
        return _coerce_grid_list(value)

    def get_all_categories(self) -> list[str]:
        """Every object category mentioned anywhere in the query."""
        categories: list[str] = []
        self._collect_categories(self.root, categories)
        for context in self.viewpoint_contexts:
            for anchor in (
                context.facing_anchor,
                context.origin_anchor,
                context.subject_anchor,
            ):
                if anchor is not None:
                    self._collect_categories(anchor, categories)
        return categories

    def _collect_categories(self, node: QueryNode, categories: list[str]) -> None:
        categories.extend(node.categories)
        for constraint in node.spatial_constraints:
            for anchor in constraint.anchors:
                self._collect_categories(anchor, categories)
        if node.select_constraint is not None and node.select_constraint.reference:
            self._collect_categories(node.select_constraint.reference, categories)

    def iter_constraints(
        self,
    ) -> Iterable[tuple[QueryNode, SpatialConstraint | SelectConstraint]]:
        """Yield every constraint paired with its owning node."""
        stack: list[QueryNode] = [self.root]
        while stack:
            node = stack.pop()
            for spatial in node.spatial_constraints:
                yield node, spatial
                stack.extend(spatial.anchors)
            if node.select_constraint is not None:
                yield node, node.select_constraint
                if node.select_constraint.reference is not None:
                    stack.append(node.select_constraint.reference)

    def _collect_descendant_anchor_grids(self) -> list[tuple[int, int]]:
        """Collect all grid_cells from anchor, reference, and viewpoint nodes."""
        cells: list[tuple[int, int]] = []
        stack: list[QueryNode] = []
        for sc in self.root.spatial_constraints:
            stack.extend(sc.anchors)
        if (
            self.root.select_constraint is not None
            and self.root.select_constraint.reference is not None
        ):
            stack.append(self.root.select_constraint.reference)
        for ctx in self.viewpoint_contexts:
            for vp_anchor in (ctx.facing_anchor, ctx.origin_anchor, ctx.subject_anchor):
                if vp_anchor is not None:
                    stack.append(vp_anchor)
        while stack:
            node = stack.pop()
            cells.extend(node.grid_cells)
            for sc in node.spatial_constraints:
                stack.extend(sc.anchors)
            if (
                node.select_constraint is not None
                and node.select_constraint.reference is not None
            ):
                stack.append(node.select_constraint.reference)
        return _coerce_grid_list(cells)

    @model_validator(mode="after")
    def _sync_and_heal_query(self) -> GroundingQuery:
        # 1. Sync root.grid_cells <-> target_grids
        merged_target = _coerce_grid_list([*self.root.grid_cells, *self.target_grids])
        self.target_grids = merged_target
        self.root.grid_cells = list(merged_target)

        # 2. Sync descendant anchor grid_cells -> anchor_grids
        descendant_grids = self._collect_descendant_anchor_grids()
        self.anchor_grids = _coerce_grid_list([*self.anchor_grids, *descendant_grids])

        # 3. Deduplicate viewpoint_contexts by id and heal dangling references
        deduped_contexts: list[ViewpointContext] = []
        known_ids: set[str] = set()
        for context in self.viewpoint_contexts:
            if context.id not in known_ids:
                known_ids.add(context.id)
                deduped_contexts.append(context)
        self.viewpoint_contexts = deduped_contexts

        non_world = {ReferenceFrame.VIEWER, ReferenceFrame.OBJECT_LOCAL}
        for _node, constraint in self.iter_constraints():
            ctx_id = constraint.viewpoint_context_id
            if constraint.reference_frame in non_world:
                if ctx_id is None or ctx_id not in known_ids:
                    if len(self.viewpoint_contexts) == 1:
                        constraint.viewpoint_context_id = self.viewpoint_contexts[0].id
                    else:
                        constraint.reference_frame = ReferenceFrame.AMBIGUOUS
                        constraint.viewpoint_context_id = None
                        constraint.execution_policy = ExecutionPolicy.RANK_ONLY
            elif ctx_id is not None and ctx_id not in known_ids:
                constraint.viewpoint_context_id = None
        return self


class HypothesisKind(str, Enum):
    """Type of hypothesis for open-world query parsing."""

    DIRECT = "direct"
    PROXY = "proxy"
    CONTEXT = "context"


class ParseMode(str, Enum):
    """Parsing mode for hypothesis output."""

    SINGLE = "single"
    MULTI = "multi"


class QueryHypothesis(BaseModel):
    """One executable hypothesis parsed from a user query."""

    kind: HypothesisKind = Field(default=HypothesisKind.DIRECT)
    rank: int = Field(default=1, ge=1, description="1-based priority rank.")
    grounding_query: GroundingQuery = Field(...)
    lexical_hints: list[str] = Field(
        default_factory=list,
        description="Free-form lexical hints (synonyms), not executable categories.",
    )


class HypothesisOutputV1(BaseModel):
    """Unified query-parsing output consumed by ``KeyframeSelector``.

    Supports deterministic single-result parsing and multi-hypothesis parsing
    for open-world fallback, automatically normalizing ranks and parse_mode.
    """

    model_config = ConfigDict(frozen=False)

    format_version: Literal["hypothesis_output_v1"] = "hypothesis_output_v1"
    parse_mode: ParseMode = Field(default=ParseMode.SINGLE)
    hypotheses: list[QueryHypothesis] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="before")
    @classmethod
    def _normalize_output_payload(cls, value: Any) -> Any:
        if isinstance(value, list):
            value = {"hypotheses": value}
        if not isinstance(value, Mapping):
            return value
        data: dict[str, Any] = dict(value)
        data["format_version"] = "hypothesis_output_v1"

        # If the LLM emitted a single GroundingQuery or QueryNode at the top level, wrap it
        if "hypotheses" not in data:
            if "grounding_query" in data:
                data["hypotheses"] = [
                    {
                        "kind": data.get("kind", "direct"),
                        "rank": 1,
                        "grounding_query": data["grounding_query"],
                    }
                ]
            elif "root" in data:
                data["hypotheses"] = [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": data.get("raw_query", ""),
                            "root": data["root"],
                            "expect_unique": data.get("expect_unique", True),
                            "target_grids": data.get("target_grids", []),
                            "anchor_grids": data.get("anchor_grids", []),
                        },
                    }
                ]

        raw_hyps = data.get("hypotheses")
        if isinstance(raw_hyps, list) and raw_hyps:
            normalized_hyps: list[Any] = []
            for idx, item in enumerate(raw_hyps[:3], start=1):
                if isinstance(item, MutableMapping):
                    copied = dict(item)
                    copied.setdefault("kind", "direct" if idx == 1 else "proxy")
                    copied.setdefault("rank", idx)
                    normalized_hyps.append(copied)
                else:
                    normalized_hyps.append(item)
            data["hypotheses"] = normalized_hyps
        return data

    @model_validator(mode="after")
    def _validate_and_heal_output(self) -> HypothesisOutputV1:
        # Sort by rank ascending (preserving input order on ties) and re-index 1..N
        ordered = sorted(
            enumerate(self.hypotheses[:3]),
            key=lambda pair: (pair[1].rank, pair[0]),
        )
        healed: list[QueryHypothesis] = []
        for new_rank, (_idx, hyp) in enumerate(ordered, start=1):
            hyp.rank = new_rank
            healed.append(hyp)
        self.hypotheses = healed

        if (
            len(self.hypotheses) == 1
            and self.hypotheses[0].kind == HypothesisKind.DIRECT
        ):
            self.parse_mode = ParseMode.SINGLE
        else:
            self.parse_mode = ParseMode.MULTI
        return self

    def ordered_hypotheses(self) -> list[QueryHypothesis]:
        """Hypotheses sorted by rank ascending."""
        return sorted(self.hypotheses, key=lambda h: h.rank)

    def validate_categories(self, scene_categories: Iterable[str]) -> None:
        """Raise if any executable category is not in scene categories or UNKNOW."""
        scene_set = set(scene_categories)
        for hypothesis in self.hypotheses:
            for category in hypothesis.grounding_query.get_all_categories():
                if category != "UNKNOW" and category not in scene_set:
                    raise ValueError(
                        f"Category '{category}' is not in scene categories or UNKNOW"
                    )

    def validate_no_mask_leak(self, hidden_categories: Iterable[str]) -> None:
        """Raise if any hidden category appears in an executable hypothesis."""
        hidden_set = set(hidden_categories)
        if not hidden_set:
            return
        for hypothesis in self.hypotheses:
            for category in hypothesis.grounding_query.get_all_categories():
                if category in hidden_set:
                    raise ValueError(f"Masked category leak detected: '{category}'")

    @classmethod
    def from_direct_query(cls, grounding_query: GroundingQuery) -> HypothesisOutputV1:
        """Build a single-direct output from one grounding query."""
        return cls(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=grounding_query,
                )
            ],
        )


# Resolve forward references between the recursively-nested models.
QueryNode.model_rebuild()
SpatialConstraint.model_rebuild()
SelectConstraint.model_rebuild()
ViewpointContext.model_rebuild()
GroundingQuery.model_rebuild()
QueryHypothesis.model_rebuild()
HypothesisOutputV1.model_rebuild()
