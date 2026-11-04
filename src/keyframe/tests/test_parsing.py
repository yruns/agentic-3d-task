"""Tests for the LLM query parser (with a fake client)."""

from __future__ import annotations

from keyframe.llm import LLMClient
from keyframe.parsing import QueryParser


def test_parser_returns_typed_hypothesis(fake_llm: LLMClient) -> None:
    parser = QueryParser(["sofa", "pillow", "door"], fake_llm)
    output = parser.parse("the pillow on the sofa")
    root = output.hypotheses[0].grounding_query.root
    assert root.category == "pillow"
    assert root.spatial_constraints[0].relation == "on"
    assert root.spatial_constraints[0].anchors[0].category == "sofa"


def test_parser_assigns_node_ids(fake_llm: LLMClient) -> None:
    parser = QueryParser(["sofa", "pillow", "door"], fake_llm)
    output = parser.parse("the pillow on the sofa")
    root = output.hypotheses[0].grounding_query.root
    assert root.node_id == "h1_root"
    assert root.spatial_constraints[0].anchors[0].node_id == "h1_root_sc0_a0"


def test_parser_includes_grid_context_in_prompt(fake_llm: LLMClient) -> None:
    grid_ctx = "Grid shape: 4 rows x 4 cols\n  (1, 2): #0 sofa, #1 pillow"
    parser = QueryParser(["sofa", "pillow", "door"], fake_llm, grid_context=grid_ctx)
    prompt = parser._build_prompt("the pillow on the sofa")
    assert "SCENE GRID LAYOUT:" in prompt
    assert "(1, 2): #0 sofa, #1 pillow" in prompt
