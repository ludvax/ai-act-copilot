"""Retrieval metrics, computed on provisions rather than chunks.

Deterministic and free: no model in the loop, so these can run on every change and are the
first thing to look at when answers get worse. hit@k answers "did we retrieve anything
usable", recall@k "did we get all of it", MRR "how far down the list", nDCG "is the order
right".
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What retrieval returned for one question."""

    case_id: str
    expected: tuple[str, ...]
    retrieved: tuple[tuple[str, ...], ...]  # provisions per rank, best first

    @property
    def first_hit_rank(self) -> int | None:
        for rank, provisions in enumerate(self.retrieved, start=1):
            if set(provisions) & set(self.expected):
                return rank
        return None

    @property
    def found(self) -> set[str]:
        return {provision for rank in self.retrieved for provision in rank} & set(self.expected)


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Averages over a dataset, plus the per-case detail for error analysis."""

    k: int
    cases: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    misses: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self, label: str) -> list[str]:
        return [
            label,
            f"{self.hit_rate:.2f}",
            f"{self.recall:.2f}",
            f"{self.mrr:.2f}",
            f"{self.ndcg:.2f}",
        ]


def score(outcomes: Sequence[CaseOutcome], k: int) -> RetrievalScores:
    """Aggregate per-case outcomes into the usual retrieval metrics."""
    if not outcomes:
        return RetrievalScores(k=k, cases=0, hit_rate=0.0, recall=0.0, mrr=0.0, ndcg=0.0)

    hits = 0
    reciprocal = 0.0
    recalls = 0.0
    gains = 0.0
    misses: list[str] = []

    for outcome in outcomes:
        rank = outcome.first_hit_rank
        if rank is not None and rank <= k:
            hits += 1
            reciprocal += 1.0 / rank
        else:
            misses.append(outcome.case_id)
        expected = set(outcome.expected)
        if expected:
            recalls += len(outcome.found & expected) / len(expected)
        gains += _ndcg(outcome, k)

    total = len(outcomes)
    return RetrievalScores(
        k=k,
        cases=total,
        hit_rate=hits / total,
        recall=recalls / total,
        mrr=reciprocal / total,
        ndcg=gains / total,
        misses=tuple(misses),
    )


def _ndcg(outcome: CaseOutcome, k: int) -> float:
    """Binary-relevance nDCG: relevant provisions should sit at the top."""
    expected = set(outcome.expected)
    if not expected:
        return 0.0
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, provisions in enumerate(outcome.retrieved[:k], start=1)
        if set(provisions) & expected
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return gain / ideal if ideal else 0.0
