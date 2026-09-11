"""The ingestion pipeline: download → parse → chunk → store.

Every stage is traced, so a slow or empty ingest is diagnosable from Langfuse alone.
Document texts are never sent as span inputs/outputs — only counts.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ai_act_copilot.chunking.base import Chunker
from ai_act_copilot.chunking.fixed import FixedSizeChunker
from ai_act_copilot.chunking.structural import StructuralChunker
from ai_act_copilot.chunking.tokenizer import HuggingFaceTokenCounter, TokenCounter
from ai_act_copilot.config import Settings
from ai_act_copilot.ingestion.download import Downloader
from ai_act_copilot.ingestion.parsers.eurlex import parse_eurlex_xhtml
from ai_act_copilot.ingestion.parsers.pdf import parse_pdf
from ai_act_copilot.ingestion.sources import Source, SourceFormat, load_manifest
from ai_act_copilot.models import ChunkStrategy, Document, Language
from ai_act_copilot.observability.tracing import observe
from ai_act_copilot.store.sqlite import CorpusStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one document contributed to the corpus."""

    document_id: str
    title: str
    language: Language
    provisions: int
    chunks: int
    tokens: int
    downloaded: bool

    @property
    def average_tokens(self) -> float:
        return self.tokens / self.chunks if self.chunks else 0.0


def make_chunker(strategy: ChunkStrategy, counter: TokenCounter, settings: Settings) -> Chunker:
    if strategy is ChunkStrategy.FIXED:
        return FixedSizeChunker(
            counter,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    return StructuralChunker(
        counter, max_tokens=settings.chunk_max_tokens, min_tokens=settings.chunk_min_tokens
    )


@observe(name="ingest", capture_input=False, capture_output=False)
def ingest(
    settings: Settings,
    *,
    download: bool = False,
    force: bool = False,
    source_ids: Sequence[str] | None = None,
    strategy: ChunkStrategy | None = None,
    counter: TokenCounter | None = None,
    downloader: Downloader | None = None,
) -> list[IngestReport]:
    """Ingest the selected sources into the corpus store."""
    manifest = load_manifest(settings.sources_file)
    sources = (
        [manifest.get(source_id) for source_id in source_ids] if source_ids else manifest.sources
    )
    chunk_strategy = strategy or settings.chunk_strategy
    token_counter = counter or HuggingFaceTokenCounter()
    chunker = make_chunker(chunk_strategy, token_counter, settings)
    fetcher = downloader or Downloader(settings.raw_dir)

    reports: list[IngestReport] = []
    with CorpusStore(settings.database_path) as store:
        for source in sources:
            for language in source.languages:
                reports.append(
                    _ingest_one(
                        source, language, fetcher, chunker, store, download=download, force=force
                    )
                )
    return reports


def _ingest_one(
    source: Source,
    language: Language,
    fetcher: Downloader,
    chunker: Chunker,
    store: CorpusStore,
    *,
    download: bool,
    force: bool,
) -> IngestReport:
    downloaded = False
    if download:
        result = fetcher.fetch(source, language, force=force)
        downloaded = not result.from_cache

    record = fetcher.record_for(source, language)
    if record is None:
        raise FileNotFoundError(
            f"{source.id} ({language}) has never been downloaded; run `aiact ingest --download`"
        )

    document = _parse(
        source,
        language,
        fetcher.read(source, language),
        record.url,
        record.sha256,
        record.fetched_at,
    )
    chunks = chunker.chunk(document)
    store.replace_document(document, chunks)
    logger.info(
        "ingested %s: %d provisions, %d chunks",
        document.document_id,
        len(document.provisions),
        len(chunks),
    )
    return IngestReport(
        document_id=document.document_id,
        title=source.short_title,
        language=language,
        provisions=len(document.provisions),
        chunks=len(chunks),
        tokens=sum(chunk.token_count for chunk in chunks),
        downloaded=downloaded,
    )


@observe(name="parse", capture_input=False, capture_output=False)
def _parse(
    source: Source,
    language: Language,
    raw: bytes,
    url: str,
    sha256: str,
    fetched_at: datetime,
) -> Document:
    parse = parse_eurlex_xhtml if source.format is SourceFormat.EURLEX_XHTML else parse_pdf
    return parse(
        raw,
        source_id=source.id,
        short_title=source.short_title,
        title=source.title,
        language=language,
        url=url,
        sha256=sha256,
        fetched_at=fetched_at,
    )
