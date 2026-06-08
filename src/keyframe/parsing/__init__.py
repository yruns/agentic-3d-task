"""Natural-language query parsing into structured hypotheses."""

from __future__ import annotations

from keyframe.parsing.parser import QueryParser, parse_query
from keyframe.parsing.structures import get_few_shot_examples, get_system_prompt

__all__ = [
    "QueryParser",
    "parse_query",
    "get_system_prompt",
    "get_few_shot_examples",
]
