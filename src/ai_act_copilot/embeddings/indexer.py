"""Embed the corpus into the vector store.

Every passage is hashed before it is embedded, so re-indexing after a chunking change only
pays for passages whose text actually changed. On this corpus that turns a full re-index
into seconds instead of minutes.
"""

import logging
from dataclasses import dataclass

from ai_act_copilot.config import Settings
from ai_act_copilot.embeddings.base import Embedder
from ai_act_copilot.embeddings.ollama import OllamaEmbedder
from ai_act_copilot.models import ChunkStrategy
from ai_act_copilot.observability.tracing import observe
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore, text_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexReport:
    model: str
    strategy: ChunkStrategy
    chunks: int
    embedded: int
    reused: int

    @property
    def cache_hit_rate(self) -> float:
        total = self.embedded + self.reused
        return self.reused / total if total else 0.0


def make_embedder(settings: Settings) -> Embedder:
    return OllamaEmbedder(settings.embedding_model, settings.ollama_base_url)


@observe(name="index", capture_input=False, capture_output=False)
def build_index(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    strategy: ChunkStrategy | None = None,
    batch_size: int = 64,
) -> IndexReport:
    """Embed every chunk of the chosen strategy, reusing cached vectors."""
    backend = embedder or make_embedder(settings)
    chunk_strategy = strategy or settings.chunk_strategy

    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        chunks = list(store.chunks(strategy=chunk_strategy))
        if not chunks:
            raise RuntimeError(
                f"no {chunk_strategy} chunks in {settings.database_path}; run `aiact ingest` first"
            )

        digests = [text_hash(chunk.embedding_text) for chunk in chunks]
        cached = vectors.cached_vectors(sorted(set(digests)), backend.model)

        pending: dict[str, str] = {}  # hash -> text, de-duplicated
        for chunk, digest in zip(chunks, digests, strict=True):
            if digest not in cached:
                pending.setdefault(digest, chunk.embedding_text)

        embedded = 0
        items = list(pending.items())
        for start in range(0, len(items), batch_size):
            window = items[start : start + batch_size]
            computed = backend.embed([text for _, text in window])
            vectors.store_vectors(
                backend.model,
                [(digest, vector) for (digest, _), vector in zip(window, computed, strict=True)],
            )
            embedded += len(window)
            logger.info("embedded %d/%d new passages", embedded, len(items))

        vectors.link_chunks(
            backend.model,
            [(chunk.chunk_id, digest) for chunk, digest in zip(chunks, digests, strict=True)],
        )

    return IndexReport(
        model=backend.model,
        strategy=chunk_strategy,
        chunks=len(chunks),
        embedded=embedded,
        reused=len(chunks) - embedded,
    )
