"""Structure-aware chunking.

Legal texts already come with the boundaries a reader would use, so chunks follow them:
one provision per chunk when it fits, otherwise packed paragraphs. Every chunk carries a
breadcrumb header ("AI Act > Chapter III > Article 6 — ...") so an isolated passage still
says which article it belongs to — both for the embedding and for the answer's citation.

A chunk never spans two provisions, which keeps citations exact.
"""

from ai_act_copilot.chunking.base import BREADCRUMB_SEPARATOR, chunk_id, split_by_tokens
from ai_act_copilot.chunking.tokenizer import TokenCounter
from ai_act_copilot.models import Chunk, ChunkStrategy, Document, Provision


class StructuralChunker:
    """One chunk per provision, split on paragraph boundaries when too long."""

    strategy = ChunkStrategy.STRUCTURAL

    def __init__(self, counter: TokenCounter, max_tokens: int = 512, min_tokens: int = 48):
        self.counter = counter
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for provision in document.provisions:
            header = BREADCRUMB_SEPARATOR.join(provision.breadcrumb)
            budget = max(1, self.max_tokens - self.counter.count(header))
            for body in self._bodies(provision, budget):
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(document.document_id, self.strategy, len(chunks)),
                        document_id=document.document_id,
                        source_id=document.source_id,
                        language=document.language,
                        strategy=self.strategy,
                        order=len(chunks),
                        header=header,
                        text=body,
                        token_count=self.counter.count(body),
                        provision_ids=(provision.provision_id,),
                    )
                )
        return chunks

    def _bodies(self, provision: Provision, budget: int) -> list[str]:
        text = provision.text
        if not text:
            return []
        if self.counter.count(text) <= budget:
            return [text]

        bodies: list[str] = []
        current: list[str] = []
        for paragraph in provision.paragraphs:
            piece = f"{paragraph.label} {paragraph.text}" if paragraph.label else paragraph.text
            if not piece.strip():
                continue
            if self.counter.count(piece) > budget:
                # A single paragraph too long for one chunk: fall back to token windows.
                bodies.extend(filter(None, ["\n".join(current)]))
                current = []
                bodies.extend(split_by_tokens(piece, self.counter, budget))
                continue
            candidate = [*current, piece]
            if self.counter.count("\n".join(candidate)) > budget:
                bodies.append("\n".join(current))
                current = [piece]
            else:
                current = candidate
        if current:
            bodies.append("\n".join(current))
        return self._merge_short_tail(bodies, budget)

    def _merge_short_tail(self, bodies: list[str], budget: int) -> list[str]:
        """Fold a stray fragment back into the previous chunk.

        The packer only starts a new chunk when a paragraph does not fit, so a short tail
        never fits back within budget. ``max_tokens`` is a target rather than a hard limit
        (bge-m3 accepts 8192), so a small overflow is preferable to a two-line chunk that
        carries no usable signal for retrieval.
        """
        if len(bodies) < 2 or self.counter.count(bodies[-1]) >= self.min_tokens:
            return bodies
        merged = "\n".join(bodies[-2:])
        if self.counter.count(merged) > budget + self.min_tokens:
            return bodies
        return [*bodies[:-2], merged]
