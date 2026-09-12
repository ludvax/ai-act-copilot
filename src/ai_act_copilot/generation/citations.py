"""Citation formatting and validation.

The model is asked to cite provision ids copied from the context. Whether it actually did
is checked in code rather than trusted: a citation that is not in the retrieved set is
dropped and counted. That check is also the metric the evaluation reports, so hallucinated
citations show up as a number instead of a vague impression.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ai_act_copilot.retrieval.base import RetrievedChunk

CONTEXT_TEMPLATE = "[{provision}] {breadcrumb}\n{text}"


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Which of the model's citations were real, and which were not."""

    valid: tuple[str, ...]
    invalid: tuple[str, ...]

    @property
    def precision(self) -> float:
        total = len(self.valid) + len(self.invalid)
        return len(self.valid) / total if total else 0.0


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved passages with the ids the model must cite."""
    blocks = []
    for hit in chunks:
        provision = ", ".join(hit.provision_ids) or hit.chunk.chunk_id
        blocks.append(
            CONTEXT_TEMPLATE.format(
                provision=provision,
                breadcrumb=hit.chunk.header or hit.chunk.source_id,
                text=hit.chunk.text,
            )
        )
    return "\n\n".join(blocks)


def allowed_provisions(chunks: Iterable[RetrievedChunk]) -> set[str]:
    return {provision for hit in chunks for provision in hit.provision_ids}


def check_citations(citations: Iterable[str], allowed: set[str]) -> CitationCheck:
    """Split the model's citations into those backed by a retrieved passage and the rest."""
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in citations:
        citation = raw.strip().strip("[]")
        if citation in seen:
            continue
        seen.add(citation)
        (valid if citation in allowed else invalid).append(citation)
    return CitationCheck(tuple(valid), tuple(invalid))
