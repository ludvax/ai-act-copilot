"""Fixed-size chunking: the baseline the structural strategy must beat.

It ignores structure entirely — the document is one long string cut into overlapping
token windows, the way most RAG tutorials do it. Chunks still record which provisions
they overlap, so retrieval metrics compare like for like.
"""

from ai_act_copilot.chunking.base import chunk_id, split_by_tokens
from ai_act_copilot.chunking.tokenizer import TokenCounter
from ai_act_copilot.models import Chunk, ChunkStrategy, Document

_SEPARATOR = "\n\n"


class FixedSizeChunker:
    """Sliding token windows over the concatenated document text."""

    strategy = ChunkStrategy.FIXED

    def __init__(self, counter: TokenCounter, max_tokens: int = 512, overlap_tokens: int = 64):
        self.counter = counter
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        text, spans = _render(document)
        if not text:
            return []

        chunks: list[Chunk] = []
        cursor = 0
        for order, window in enumerate(
            split_by_tokens(text, self.counter, self.max_tokens, self.overlap_tokens)
        ):
            start = text.find(window, max(0, cursor - len(window)))
            start = start if start >= 0 else cursor
            end = start + len(window)
            cursor = start + max(1, len(window) - 1)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(document.document_id, self.strategy, order),
                    document_id=document.document_id,
                    source_id=document.source_id,
                    language=document.language,
                    strategy=self.strategy,
                    order=order,
                    header="",
                    text=window,
                    token_count=self.counter.count(window),
                    provision_ids=_overlapping(spans, start, end),
                )
            )
        return chunks


def _render(document: Document) -> tuple[str, list[tuple[int, int, str]]]:
    """Concatenate provision texts, remembering where each provision lands."""
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for provision in document.provisions:
        text = provision.text
        if not text:
            continue
        spans.append((cursor, cursor + len(text), provision.provision_id))
        parts.append(text)
        cursor += len(text) + len(_SEPARATOR)
    return _SEPARATOR.join(parts), spans


def _overlapping(spans: list[tuple[int, int, str]], start: int, end: int) -> tuple[str, ...]:
    return tuple(
        provision_id
        for span_start, span_end, provision_id in spans
        if span_start < end and start < span_end
    )
