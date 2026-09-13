"""Hybrid retrieval: exact references, dense vectors and BM25, fused by rank.

Three signals are implemented - exact references, dense vectors and BM25 - and combined by
rank fusion. Which of them are on by default is a measured decision, not a habit: on this
corpus BM25 lowers ranking quality (MRR 0.88 -> 0.66 when added to dense), while resolving
explicit citations lifts it (0.88 -> 0.93). The default is therefore references + dense,
with BM25 one argument away for corpora where rare-term matching matters more.

Results are de-duplicated per provision so the same article does not occupy every slot in
English and French.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ai_act_copilot.config import Settings
from ai_act_copilot.embeddings.base import Embedder, Vector
from ai_act_copilot.models import Chunk, ChunkStrategy, Language
from ai_act_copilot.observability.tracing import observe, record_span
from ai_act_copilot.retrieval.base import (
    DEFAULT_WEIGHTS,
    RetrievedChunk,
    reciprocal_rank_fusion,
)
from ai_act_copilot.retrieval.reference_lookup import find_references
from ai_act_copilot.store.bm25 import BM25Index
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.text_analysis import analyse, detect_language
from ai_act_copilot.store.vector_index import VectorIndex
from ai_act_copilot.store.vectors import VectorStore

logger = logging.getLogger(__name__)

# Measured on the golden set (see README): adding BM25 to dense costs ranking quality on
# this corpus, so it is implemented and available but not in the default path.
DEFAULT_SIGNALS = ("reference", "dense")


@dataclass(slots=True)
class _LanguageIndex:
    """Everything needed to search one language, built once and kept in memory."""

    chunks: dict[str, Chunk]
    bm25: BM25Index
    vectors: VectorIndex
    by_provision: dict[str, list[str]]


class HybridRetriever:
    """Searches the corpus with all three signals and fuses the rankings."""

    def __init__(
        self,
        store: CorpusStore,
        vector_store: VectorStore,
        embedder: Embedder,
        *,
        settings: Settings,
        strategy: ChunkStrategy | None = None,
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = settings
        self.strategy = strategy or settings.chunk_strategy
        self._indexes: dict[Language, _LanguageIndex] = {}

    @observe(
        name="retrieve-passages", as_type="retriever", capture_input=False, capture_output=False
    )
    def search(
        self,
        query: str,
        *,
        language: Language | None = None,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Best passages for the query, best first."""
        target_language = language or detect_language(query)
        index = self._index_for(target_language)
        if not index.chunks:
            record_span(input=query, output=[], metadata={"language": target_language.value})
            return []

        limit = limit or self.settings.retrieval_top_k
        candidates = self.settings.retrieval_candidates

        requested = tuple(signals) if signals else DEFAULT_SIGNALS
        rankings: dict[str, list[str]] = {}
        if "reference" in requested:
            referenced = self._reference_hits(query, index, sources)
            if referenced:
                rankings["reference"] = referenced
        if "dense" in requested:
            rankings["dense"] = [
                chunk_id for chunk_id, _ in index.vectors.search(self._embed(query), candidates)
            ]
        if "bm25" in requested:
            rankings["bm25"] = [
                chunk_id
                for chunk_id, _ in index.bm25.search(analyse(query, target_language), candidates)
            ]

        fused = reciprocal_rank_fusion(
            rankings, k=self.settings.rrf_k, weights=DEFAULT_WEIGHTS if weights is None else weights
        )
        ordered = sorted(fused.items(), key=lambda item: (-sum(item[1].values()), item[0]))

        results: list[RetrievedChunk] = []
        seen_provisions: set[tuple[str, ...]] = set()
        for chunk_id, components in ordered:
            chunk = index.chunks.get(chunk_id)
            if chunk is None:
                continue
            if sources and chunk.source_id not in sources:
                continue
            if chunk.provision_ids in seen_provisions:
                continue  # the same provision already has a better passage
            seen_provisions.add(chunk.provision_ids)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=sum(components.values()),
                    rank=len(results) + 1,
                    components=dict(components),
                    matched_reference="reference" in components,
                )
            )
            if len(results) == limit:
                break

        # Which signal found what is the whole debugging story of a bad answer: a passage
        # that only BM25 liked reads very differently from one three signals agreed on.
        record_span(
            input=query,
            output=[
                {
                    "provision_ids": list(hit.chunk.provision_ids),
                    "signals": sorted(hit.components),
                    "score": round(hit.score, 4),
                    "header": hit.chunk.header,
                }
                for hit in results
            ],
            metadata={
                "language": target_language.value,
                "signals_requested": list(requested),
                "candidates_per_signal": candidates,
                "limit": limit,
                "strategy": self.strategy.value,
                "embedding_model": self.embedder.model,
                "sources": list(sources) if sources else None,
            },
        )
        return results

    def _embed(self, query: str) -> Vector:
        return self.embedder.embed([query])[0]

    def _reference_hits(
        self, query: str, index: _LanguageIndex, sources: Sequence[str] | None
    ) -> list[str]:
        """Chunks of provisions the question cites explicitly."""
        hits: list[str] = []
        for reference in find_references(query):
            candidates = (
                [reference.provision_id()]
                if reference.source_id
                else [
                    f"{source_id}:{reference.kind}:{reference.number}"
                    for source_id in (sources or self._known_sources(index))
                ]
            )
            for provision_id in filter(None, candidates):
                hits.extend(index.by_provision.get(provision_id, []))
        return hits

    @staticmethod
    def _known_sources(index: _LanguageIndex) -> list[str]:
        return sorted({chunk.source_id for chunk in index.chunks.values()})

    def _index_for(self, language: Language) -> _LanguageIndex:
        if language in self._indexes:
            return self._indexes[language]

        chunks = list(self.store.chunks(strategy=self.strategy, language=language))
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        by_provision: dict[str, list[str]] = {}
        for chunk in chunks:
            for provision_id in chunk.provision_ids:
                by_provision.setdefault(provision_id, []).append(chunk.chunk_id)

        bm25 = BM25Index.build(
            [(chunk.chunk_id, analyse(chunk.embedding_text, language)) for chunk in chunks]
        )
        ids, matrix = self.vector_store.matrix(
            model=self.embedder.model, strategy=self.strategy, language=language
        )
        logger.info(
            "built %s index for %s: %d chunks, %d vectors",
            self.strategy,
            language,
            len(chunks),
            len(ids),
        )
        index = _LanguageIndex(by_id, bm25, VectorIndex(ids, matrix), by_provision)
        self._indexes[language] = index
        return index
