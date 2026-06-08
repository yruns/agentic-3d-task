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

from collections.abc import Iterable
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        ...,
        min_length=1,
        description="Object categories to search for, including related synonyms.",
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
        ..., description="Reference objects. Usually 1, can be 2 for 'between'."
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


class SelectConstraint(BaseModel):
    """A selection / ranking constraint applied to candidate matches."""

    constraint_type: ConstraintType = Field(
        ..., description="superlative / comparative / ordinal."
    )
    metric: str = Field(
        ...,
        description="Metric: 'distance', 'size', 'height', 'x_position', 'y_position'.",
    )
    order: str = Field(..., description="Order: 'min', 'max', 'asc', 'desc'.")
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

    @model_validator(mode="after")
    def _validate_constraint(self) -> SelectConstraint:
        if self.constraint_type == ConstraintType.ORDINAL and self.position is None:
            raise ValueError("Ordinal constraints require 'position' to be set")
        return self


class GroundingQuery(BaseModel):
    """Top-level parsed query: a target node plus optional viewpoint contexts."""

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

    @model_validator(mode="after")
    def _validate_viewpoint_references(self) -> GroundingQuery:
        known_ids = {context.id for context in self.viewpoint_contexts}
        if len(known_ids) != len(self.viewpoint_contexts):
            raise ValueError("viewpoint_contexts must have unique ids")

        non_world = {ReferenceFrame.VIEWER, ReferenceFrame.OBJECT_LOCAL}
        for _node, constraint in self.iter_constraints():
            ctx_id = constraint.viewpoint_context_id
            if ctx_id is not None and ctx_id not in known_ids:
                raise ValueError(f"viewpoint_context_id {ctx_id!r} is not declared")
            if constraint.reference_frame in non_world and ctx_id is None:
                raise ValueError(
                    f"reference_frame={constraint.reference_frame.value} "
                    "requires a viewpoint_context_id"
                )
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

    kind: HypothesisKind = Field(...)
    rank: int = Field(..., ge=1, description="1-based priority rank.")
    grounding_query: GroundingQuery = Field(...)
    lexical_hints: list[str] = Field(
        default_factory=list,
        description="Free-form lexical hints (synonyms), not executable categories.",
    )


class HypothesisOutputV1(BaseModel):
    """Unified query-parsing output consumed by ``KeyframeSelector``.

    Supports deterministic single-result parsing and multi-hypothesis parsing
    for open-world fallback.
    """

    model_config = ConfigDict(frozen=False)

    format_version: Literal["hypothesis_output_v1"] = "hypothesis_output_v1"
    parse_mode: ParseMode = Field(...)
    hypotheses: list[QueryHypothesis] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="after")
    def _validate_output(self) -> HypothesisOutputV1:
        ranks = [h.rank for h in self.hypotheses]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Hypothesis ranks must be unique")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("Hypothesis ranks must be contiguous and start from 1")
        if self.parse_mode == ParseMode.SINGLE:
            if len(self.hypotheses) != 1:
                raise ValueError("parse_mode='single' requires exactly one hypothesis")
            if self.hypotheses[0].kind != HypothesisKind.DIRECT:
                raise ValueError(
                    "parse_mode='single' requires hypothesis kind='direct'"
                )
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
