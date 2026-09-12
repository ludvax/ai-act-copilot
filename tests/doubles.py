"""Test doubles that keep the suite offline and deterministic."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from ai_act_copilot.embeddings.base import Vector, normalise
from ai_act_copilot.llm.base import LLMResult, Message
from ai_act_copilot.llm.pricing import Usage


class FakeEmbedder:
    """Hashing embedder: shared words produce overlapping vectors, no network involved."""

    def __init__(self, model: str = "fake-embed", dimensions: int = 64) -> None:
        self.model = model
        self.dimensions = dimensions
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in text.lower().split():
            bucket = hashlib.sha256(word.encode("utf-8")).digest()[0] % self.dimensions
            vector[bucket] += 1.0
        return normalise(vector)


class FakeLLM:
    """Scripted Claude stand-in: no network, no cost, deterministic answers."""

    def __init__(
        self,
        *,
        model: str = "fake-claude",
        payload: dict[str, Any] | None = None,
        text: str = "Prohibited practices are listed in Article 5.",
        stop_reason: str = "end_turn",
    ) -> None:
        self.model = model
        self.payload = (
            payload
            if payload is not None
            else {
                "answer": text,
                "citations": ["ai_act:art:5"],
                "abstained": False,
            }
        )
        self.text = text
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        max_tokens: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_format: type[BaseModel] | None = None,
    ) -> LLMResult:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        parsed = output_format(**self.payload) if output_format is not None else None
        usage = Usage(input_tokens=1200, output_tokens=180, cache_read_tokens=800)
        return LLMResult(
            text=self.text,
            model=self.model,
            stop_reason=self.stop_reason,
            usage=usage,
            cost_usd=0.0,
            content=[],
            parsed=parsed,
        )
