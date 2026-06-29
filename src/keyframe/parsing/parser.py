"""Natural-language query parser using LLM JSON output.

Converts a query such as "the pillow on the sofa nearest the door" into a
:class:`~keyframe.models.hypotheses.HypothesisOutputV1`. The LLM is prompted
with the full schema and few-shot examples plus the scene's category list, and
returns a JSON object that is validated into the typed model. Category
correctness is enforced downstream by ``KeyframeSelector._sanitize_categories``.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import MutableMapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict, cast

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from loguru import logger

from keyframe.llm.client import LLMClient
from keyframe.models.hypotheses import HypothesisOutputV1, QueryNode
from keyframe.parsing.structures import get_few_shot_examples, get_system_prompt


class _TextContentBlock(TypedDict):
    type: Literal["text"]
    text: str


class _ImageUrlPayload(TypedDict):
    url: str
    detail: Literal["low"]


class _ImageContentBlock(TypedDict):
    type: Literal["image_url"]
    image_url: _ImageUrlPayload


_MessageContentBlock = _TextContentBlock | _ImageContentBlock


class QueryParser:
    """Parse natural-language queries into ``HypothesisOutputV1`` via an LLM."""

    def __init__(
        self,
        scene_categories: Sequence[str],
        llm_client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.scene_categories = list(scene_categories)
        self._client = llm_client
        self._model = model or llm_client.config.default_model
        self._temperature = temperature

    @property
    def model(self) -> str:
        """Resolved model name used for parsing."""
        return self._model

    def parse(
        self,
        query: str,
        scene_images: Sequence[str | Path] = (),
    ) -> HypothesisOutputV1:
        """Parse a query, rotating across the model's key pool on failure."""
        attempts = max(2, self._client.key_count(self._model))
        last_error: Exception | None = None
        for attempt in range(attempts):
            chat_model = self._client.get_chat_model(
                self._model, temperature=self._temperature
            )
            try:
                parsed = self._do_parse(query, chat_model, scene_images)
                for hypothesis in parsed.hypotheses:
                    self._assign_node_ids(
                        hypothesis.grounding_query.root, f"h{hypothesis.rank}_root"
                    )
                logger.success(f"[QueryParser] parsed query {query!r}")
                return parsed
            except Exception as error:  # noqa: BLE001 - retried, re-raised below
                last_error = error
                logger.warning(
                    f"[QueryParser] attempt {attempt + 1}/{attempts} failed: {error}"
                )
        raise ValueError(
            f"Failed to parse query {query!r} after {attempts} attempts: {last_error}"
        )

    def _do_parse(
        self,
        query: str,
        chat_model: AzureChatOpenAI,
        scene_images: Sequence[str | Path],
    ) -> HypothesisOutputV1:
        prompt = self._build_prompt(query)
        response = chat_model.invoke(self._build_messages(prompt, scene_images))
        content = response.content
        text = content if isinstance(content, str) else str(content)
        return self._parse_json_response(text, query)

    def _build_messages(
        self, prompt: str, scene_images: Sequence[str | Path]
    ) -> list[HumanMessage]:
        if not scene_images:
            return [HumanMessage(content=prompt)]
        blocks: list[_MessageContentBlock] = [{"type": "text", "text": prompt}]
        for image_path in scene_images:
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_data_url(image_path),
                        "detail": "low",
                    },
                }
            )
        message_content = cast(list[str | dict[str, object]], blocks)
        return [HumanMessage(content=message_content)]

    def _parse_json_response(
        self, response_text: str, query: str
    ) -> HypothesisOutputV1:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        json_str = match.group(1).strip() if match else response_text.strip()
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise TypeError("parsed JSON response is not an object")
        return self._finalize_json(cast(MutableMapping[str, object], data), query)

    @staticmethod
    def _finalize_json(
        data: MutableMapping[str, object], query: str
    ) -> HypothesisOutputV1:
        """Fill defaults the LLM may omit, then validate into the typed model."""
        data.setdefault("format_version", "hypothesis_output_v1")
        hypotheses = data.get("hypotheses")
        if isinstance(hypotheses, list):
            for hypothesis in hypotheses:
                if not isinstance(hypothesis, MutableMapping):
                    continue
                grounding = hypothesis.get("grounding_query")
                if isinstance(grounding, MutableMapping) and not grounding.get(
                    "raw_query"
                ):
                    kind = hypothesis.get("kind")
                    prefix = (
                        "proxy for: "
                        if kind == "proxy"
                        else ("context for: " if kind == "context" else "")
                    )
                    grounding["raw_query"] = f"{prefix}{query}"
        return HypothesisOutputV1.model_validate(data)

    def _build_prompt(self, query: str) -> str:
        categories = ", ".join(sorted(set(self.scene_categories)))
        return (
            f"{get_system_prompt()}\n\n"
            f"SCENE CATEGORIES: [{categories}]\n\n"
            f"{get_few_shot_examples()}\n\n"
            f'Now parse this query:\nQuery: "{query}"\n\n'
            "Return ONLY the JSON object matching the HypothesisOutputV1 schema."
        )

    @staticmethod
    def _image_to_data_url(image_path: str | Path, max_size: int = 800) -> str:
        """Encode an image as a base64 data URL for multimodal input."""
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pillow is required for image input. Install the 'vision' extra."
            ) from error

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            image = image.resize(
                (int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS
            )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _assign_node_ids(self, node: QueryNode, prefix: str) -> None:
        """Recursively assign unique node ids for executor memoization."""
        node.node_id = prefix
        for sc_index, constraint in enumerate(node.spatial_constraints):
            for anchor_index, anchor in enumerate(constraint.anchors):
                self._assign_node_ids(anchor, f"{prefix}_sc{sc_index}_a{anchor_index}")
        if node.select_constraint is not None and node.select_constraint.reference:
            self._assign_node_ids(node.select_constraint.reference, f"{prefix}_sel_ref")


def parse_query(
    query: str,
    scene_categories: Sequence[str],
    llm_client: LLMClient,
    *,
    model: str | None = None,
) -> HypothesisOutputV1:
    """Convenience wrapper: build a parser and parse a single query."""
    return QueryParser(scene_categories, llm_client, model=model).parse(query)
