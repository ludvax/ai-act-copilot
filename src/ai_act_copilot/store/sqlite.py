"""SQLite store for documents, provisions and chunks.

SQLite is in the standard library, the corpus is small (a few thousand chunks), and the
file can be opened with any SQL client to inspect what the retriever actually sees.
Embedding vectors are added alongside it in M2; see docs/adr/ for why no vector database.
"""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from ai_act_copilot.models import (
    Chunk,
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    language    TEXT NOT NULL,
    title       TEXT NOT NULL,
    url         TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provisions (
    provision_id TEXT NOT NULL,
    document_id  TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_id    TEXT NOT NULL,
    language     TEXT NOT NULL,
    kind         TEXT NOT NULL,
    number       TEXT NOT NULL,
    title        TEXT,
    breadcrumb   TEXT NOT NULL,
    paragraphs   TEXT NOT NULL,
    ord          INTEGER NOT NULL,
    PRIMARY KEY (provision_id, language)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL,
    language      TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    ord           INTEGER NOT NULL,
    header        TEXT NOT NULL,
    text          TEXT NOT NULL,
    token_count   INTEGER NOT NULL,
    provision_ids TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks (document_id, strategy);
CREATE INDEX IF NOT EXISTS chunks_by_strategy ON chunks (strategy, language);
CREATE INDEX IF NOT EXISTS provisions_by_document ON provisions (document_id);
"""


class CorpusStore:
    """Read/write access to the parsed corpus. Use as a context manager."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
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

    def replace_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Write a document, its provisions and its chunks, replacing any previous run."""
        with self._connection as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
            connection.execute(
                "DELETE FROM provisions WHERE document_id = ?", (document.document_id,)
            )
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document.document_id,)
            )
            connection.execute(
                "INSERT INTO documents VALUES (:document_id, :source_id, :language, :title,"
                " :url, :sha256, :fetched_at)",
                {
                    "document_id": document.document_id,
                    "source_id": document.source_id,
                    "language": document.language.value,
                    "title": document.title,
                    "url": document.url,
                    "sha256": document.sha256,
                    "fetched_at": document.fetched_at.isoformat(),
                },
            )
            connection.executemany(
                "INSERT INTO provisions VALUES (:provision_id, :document_id, :source_id,"
                " :language, :kind, :number, :title, :breadcrumb, :paragraphs, :ord)",
                [
                    {
                        "provision_id": provision.provision_id,
                        "document_id": provision.document_id,
                        "source_id": provision.source_id,
                        "language": provision.language.value,
                        "kind": provision.kind.value,
                        "number": provision.number,
                        "title": provision.title,
                        "breadcrumb": json.dumps(list(provision.breadcrumb)),
                        "paragraphs": json.dumps(
                            [paragraph.model_dump() for paragraph in provision.paragraphs]
                        ),
                        "ord": provision.order,
                    }
                    for provision in document.provisions
                ],
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (:chunk_id, :document_id, :source_id, :language,"
                " :strategy, :ord, :header, :text, :token_count, :provision_ids)",
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "source_id": chunk.source_id,
                        "language": chunk.language.value,
                        "strategy": chunk.strategy.value,
                        "ord": chunk.order,
                        "header": chunk.header,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "provision_ids": json.dumps(list(chunk.provision_ids)),
                    }
                    for chunk in chunks
                ],
            )

    def chunks(
        self, *, strategy: ChunkStrategy | None = None, language: Language | None = None
    ) -> Iterator[Chunk]:
        query = "SELECT * FROM chunks"
        filters, parameters = [], []
        if strategy is not None:
            filters.append("strategy = ?")
            parameters.append(strategy.value)
        if language is not None:
            filters.append("language = ?")
            parameters.append(language.value)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY document_id, ord"
        with closing(self._connection.execute(query, parameters)) as cursor:
            for row in cursor:
                yield _chunk_from_row(row)

    def provision(self, provision_id: str, language: Language) -> Provision | None:
        row = self._connection.execute(
            "SELECT * FROM provisions WHERE provision_id = ? AND language = ?",
            (provision_id, language.value),
        ).fetchone()
        return _provision_from_row(row) if row else None

    def document_summaries(self) -> list[dict[str, object]]:
        """One row per ingested document, with its provision and chunk counts."""
        query = """
            SELECT d.document_id, d.source_id, d.language, d.title, d.fetched_at,
                   (SELECT COUNT(*) FROM provisions p WHERE p.document_id = d.document_id)
                       AS provisions,
                   (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.document_id)
                       AS chunks,
                   (SELECT COALESCE(AVG(c.token_count), 0) FROM chunks c
                        WHERE c.document_id = d.document_id) AS avg_tokens
            FROM documents d ORDER BY d.document_id
        """
        with closing(self._connection.execute(query)) as cursor:
            return [dict(row) for row in cursor]


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        source_id=row["source_id"],
        language=Language(row["language"]),
        strategy=ChunkStrategy(row["strategy"]),
        order=row["ord"],
        header=row["header"],
        text=row["text"],
        token_count=row["token_count"],
        provision_ids=tuple(json.loads(row["provision_ids"])),
    )


def _provision_from_row(row: sqlite3.Row) -> Provision:
    return Provision(
        provision_id=row["provision_id"],
        document_id=row["document_id"],
        source_id=row["source_id"],
        language=Language(row["language"]),
        kind=ProvisionKind(row["kind"]),
        number=row["number"],
        title=row["title"],
        breadcrumb=tuple(json.loads(row["breadcrumb"])),
        paragraphs=tuple(Paragraph.model_validate(item) for item in json.loads(row["paragraphs"])),
        order=row["ord"],
    )


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)
