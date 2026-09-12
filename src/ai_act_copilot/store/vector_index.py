"""Brute-force cosine search over a numpy matrix.

With ~1 900 chunks of 1 024 dimensions the whole index is 8 MB and a query is one matrix
product - roughly a millisecond. A vector database would add an operational dependency to
solve a problem this corpus does not have; see the ADR. The interface is deliberately the
one a vector store would expose, so swapping it later touches one class.
"""

from collections.abc import Container

import numpy as np
import numpy.typing as npt

from ai_act_copilot.embeddings.base import Vector


class VectorIndex:
    """Cosine similarity over L2-normalised vectors."""

    def __init__(self, ids: list[str], matrix: npt.NDArray[np.float32]) -> None:
        if matrix.size and len(ids) != matrix.shape[0]:
            raise ValueError(f"{len(ids)} ids for {matrix.shape[0]} vectors")
        self.ids = ids
        self.matrix = matrix

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self, query: Vector, limit: int = 10, *, allowed: Container[str] | None = None
    ) -> list[tuple[str, float]]:
        """Chunk ids closest to the query, best first."""
        if not self.ids or self.matrix.size == 0:
            return []

        scores = self.matrix @ query.astype(np.float32)
        candidates = min(limit if allowed is None else len(self.ids), len(self.ids))
        # argpartition finds the top-k without sorting the whole array.
        top = np.argpartition(-scores, candidates - 1)[:candidates]
        ordered = top[np.argsort(-scores[top])]

        results: list[tuple[str, float]] = []
        for position in ordered:
            identifier = self.ids[int(position)]
            if allowed is not None and identifier not in allowed:
                continue
            results.append((identifier, float(scores[int(position)])))
            if len(results) == limit:
                break
        return results
