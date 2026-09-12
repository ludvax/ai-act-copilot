"""Vector storage: an embedding cache plus the chunk-to-vector mapping.

The cache is keyed by the hash of the embedded text, not by chunk id, so re-chunking or
switching strategy only embeds passages that are genuinely new. Vectors live in the same
SQLite file as the corpus and are loaded into one numpy matrix at search time.
"""

import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt

from ai_act_copilot.embeddings.base import Vector
from ai_act_copilot.models import ChunkStrategy, Language

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_sha256 TEXT NOT NULL,
    model       TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (text_sha256, model)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE INDEX IF NOT EXISTS chunk_embeddings_by_model ON chunk_embeddings (model);
"""

_SQL_VARIABLE_WINDOW = 500


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorStore:
    """Reads and writes embeddings next to the corpus tables."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Shared across threads: the HTTP API serves sync endpoints from a pool.
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def cached_vectors(self, hashes: Sequence[str], model: str) -> dict[str, Vector]:
        """Vectors already computed for these texts, keyed by text hash."""
        found: dict[str, Vector] = {}
        for start in range(0, len(hashes), _SQL_VARIABLE_WINDOW):
            window = list(hashes[start : start + _SQL_VARIABLE_WINDOW])
            placeholders = ",".join("?" * len(window))

            query = (
                "SELECT text_sha256, vector FROM embedding_cache "  # noqa: S608
                f"WHERE model = ? AND text_sha256 IN ({placeholders})"
            )
            with closing(self._connection.execute(query, (model, *window))) as cursor:
                for row in cursor:
                    found[row["text_sha256"]] = np.frombuffer(row["vector"], dtype=np.float32)
        return found

    def store_vectors(self, model: str, items: Iterable[tuple[str, Vector]]) -> None:
        """Persist newly embedded texts, keyed by their hash."""
        with self._connection as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?)",
                [
                    (digest, model, int(vector.shape[0]), vector.astype(np.float32).tobytes())
                    for digest, vector in items
                ],
            )

    def link_chunks(self, model: str, items: Iterable[tuple[str, str]]) -> None:
        """Point chunks at the embedding of their text."""
        with self._connection as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings VALUES (?, ?, ?)",
                [(chunk_id, model, digest) for chunk_id, digest in items],
            )

    def matrix(
        self,
        *,
        model: str,
        strategy: ChunkStrategy | None = None,
        language: Language | None = None,
    ) -> tuple[list[str], npt.NDArray[np.float32]]:
        """Chunk ids and their vectors as one matrix, ready for a dot product."""
        clauses = [
            "SELECT ce.chunk_id AS chunk_id, e.vector AS vector",
            "FROM chunk_embeddings ce",
            "JOIN embedding_cache e ON e.text_sha256 = ce.text_sha256 AND e.model = ce.model",
            "JOIN chunks c ON c.chunk_id = ce.chunk_id",
            "WHERE ce.model = ?",
        ]
        parameters: list[object] = [model]
        if strategy is not None:
            clauses.append("AND c.strategy = ?")
            parameters.append(strategy.value)
        if language is not None:
            clauses.append("AND c.language = ?")
            parameters.append(language.value)
        clauses.append("ORDER BY ce.chunk_id")

        ids: list[str] = []
        vectors: list[npt.NDArray[np.float32]] = []
        with closing(self._connection.execute(" ".join(clauses), parameters)) as cursor:
            for row in cursor:
                ids.append(row["chunk_id"])
                vectors.append(np.frombuffer(row["vector"], dtype=np.float32))
        if not vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.vstack(vectors)

    def count(self, model: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?", (model,)
        ).fetchone()
        return int(row[0])
