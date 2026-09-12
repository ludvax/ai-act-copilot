"""Okapi BM25, written here rather than pulled in as a dependency.

It is forty lines, it makes the ranking auditable, and an FDE interview will ask what the
knobs do: k1 controls how fast term frequency saturates, b how strongly long passages are
penalised. The index is rebuilt from SQLite at search time - at this corpus size that costs
milliseconds.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(slots=True)
class BM25Index:
    """An inverted index over pre-analysed documents."""

    ids: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    average_length: float = 0.0
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    @classmethod
    def build(
        cls,
        documents: Sequence[tuple[str, Sequence[str]]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> "BM25Index":
        index = cls(k1=k1, b=b)
        for position, (identifier, tokens) in enumerate(documents):
            index.ids.append(identifier)
            index.lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                index.postings.setdefault(term, []).append((position, frequency))

        total = len(index.ids)
        index.average_length = sum(index.lengths) / total if total else 0.0
        for term, postings in index.postings.items():
            document_frequency = len(postings)
            # Robertson/Sparck Jones idf, smoothed so common terms never score negative.
            index.idf[term] = math.log(
                1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
            )
        return index

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_tokens: Sequence[str], limit: int = 10) -> list[tuple[str, float]]:
        """Score documents containing any query term, best first."""
        if not self.ids or not query_tokens:
            return []

        scores: dict[int, float] = {}
        for term in query_tokens:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for position, frequency in postings:
                length_ratio = (
                    self.lengths[position] / self.average_length if self.average_length else 1.0
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[position] = scores.get(position, 0.0) + idf * (
                    frequency * (self.k1 + 1) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.ids[item[0]]))
        return [(self.ids[position], score) for position, score in ranked[:limit]]
