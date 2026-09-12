"""Shared retrieval types and rank fusion."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ai_act_copilot.models import Chunk, Language

DEFAULT_RRF_K = 60
# Signals are not equally trustworthy on this corpus; see the measurement in the README.
DEFAULT_WEIGHTS: dict[str, float] = {"reference": 2.0, "dense": 1.0, "bm25": 0.4}


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk with the evidence for why it was retrieved."""

    chunk: Chunk
    score: float
    rank: int
    components: dict[str, float] = field(default_factory=dict)
    matched_reference: bool = False

    @property
    def provision_ids(self) -> tuple[str, ...]:
        return self.chunk.provision_ids


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Combine rankings by reciprocal rank: 1/(k + rank).

    Rank fusion rather than score fusion, because a BM25 score and a cosine similarity are
    not on the same scale and normalising them is guesswork. Only the order matters, and k
    damps the influence of the very top positions so one retriever cannot dominate.
    """
    contributions: dict[str, dict[str, float]] = {}
    for name, ranked_ids in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for position, identifier in enumerate(ranked_ids, start=1):
            contributions.setdefault(identifier, {})[name] = weight / (k + position)
    return contributions


class Retriever(Protocol):
    """What the generation layer needs from retrieval, and nothing more."""

    def search(
        self,
        query: str,
        *,
        language: Language | None = None,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...
