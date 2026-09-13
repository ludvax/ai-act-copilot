"""bge-m3 embeddings served by a local Ollama instance.

Local by choice: the corpus is public but the questions are not, embedding 1 900 chunks
costs nothing, and the model is multilingual, which the FR/EN corpus requires. bge-m3 uses
the same encoder for documents and queries, so no asymmetric prefixes are needed.
"""

import logging
import time
from collections.abc import Sequence

import httpx

from ai_act_copilot.embeddings.base import Vector, normalise
from ai_act_copilot.observability.tracing import observe, record_generation

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-m3"
DEFAULT_BASE_URL = "http://localhost:11434"


class EmbeddingBackendError(RuntimeError):
    """Raised with an actionable message when Ollama cannot serve embeddings."""


class OllamaEmbedder:
    """Calls the Ollama embedding endpoint, batching and retrying transient failures."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        *,
        batch_size: int = 16,
        timeout: float = 120.0,
        max_attempts: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = client

    @observe(name="embed-texts", as_type="embedding", capture_input=False, capture_output=False)
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(normalise(vector) for vector in self._embed_batch(batch))
        # The vectors themselves would be thousands of floats of noise in the UI; their
        # shape is what tells you whether this step did what it was asked.
        record_generation(
            model=self.model,
            input=list(texts) if len(texts) == 1 else f"{len(texts)} texts",
            output={"vectors": len(vectors), "dimensions": len(vectors[0]) if vectors else 0},
            model_parameters={"batch_size": self.batch_size},
            metadata={"backend": "ollama", "base_url": self.base_url},
        )
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model, "input": batch}
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                data = self._post(payload)
            except httpx.HTTPError as error:
                last_error = error
                if attempt == self.max_attempts:
                    break
                delay = 2.0 ** (attempt - 1)
                logger.warning(
                    "embedding attempt %d failed (%s); retrying in %.0fs", attempt, error, delay
                )
                time.sleep(delay)
                continue

            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise EmbeddingBackendError(
                    f"Ollama returned {len(embeddings or [])} embeddings for {len(batch)} inputs"
                )
            return [[float(value) for value in vector] for vector in embeddings]

        raise EmbeddingBackendError(
            f"could not reach Ollama at {self.base_url} ({last_error}). "
            f"Start Ollama, then run: ollama pull {self.model}"
        ) from last_error

    def _post(self, payload: dict[str, object]) -> dict[str, list[list[float]]]:
        url = f"{self.base_url}/api/embed"
        if self._client is not None:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return dict(response.json())
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return dict(response.json())
