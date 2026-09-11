"""Shared chunking machinery.

Both strategies produce :class:`Chunk` objects carrying the provisions they cover, so
retrieval quality can be compared between strategies on the same golden dataset.
"""

from typing import Protocol

from ai_act_copilot.chunking.tokenizer import TokenCounter
from ai_act_copilot.models import Chunk, ChunkStrategy, Document

BREADCRUMB_SEPARATOR = " > "


class Chunker(Protocol):
    strategy: ChunkStrategy

    def chunk(self, document: Document) -> list[Chunk]: ...


def chunk_id(document_id: str, strategy: ChunkStrategy, order: int) -> str:
    return f"{document_id}:{strategy}:{order:04d}"


def split_by_tokens(
    text: str, counter: TokenCounter, max_tokens: int, overlap_tokens: int = 0
) -> list[str]:
    """Cut ``text`` into windows of at most ``max_tokens``, overlapping by ``overlap_tokens``."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens")

    offsets = counter.offsets(text)
    if not offsets:
        return []
    if len(offsets) <= max_tokens:
        return [text[offsets[0][0] : offsets[-1][1]]]

    step = max_tokens - overlap_tokens
    windows: list[str] = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + max_tokens]
        if not window:
            break
        windows.append(text[window[0][0] : window[-1][1]])
        if start + max_tokens >= len(offsets):
            break
    return windows
