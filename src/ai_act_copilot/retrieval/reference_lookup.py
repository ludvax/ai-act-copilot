"""Detect explicit citations in a question.

"What does Article 6(2) say?" is a lookup, not a similarity search: embeddings routinely
return the semantically similar Article 7 instead, and BM25 matches the word "article" in
a thousand places. Parsing the citation and fetching that exact provision is the cheapest
accuracy win in the whole pipeline.
"""

import re
from dataclasses import dataclass

from ai_act_copilot.models import ProvisionKind

# Which act the question is about, when it says so.
_SOURCE_HINTS: tuple[tuple[str, str], ...] = (
    ("ai act", "ai_act"),
    ("ai-act", "ai_act"),
    ("aia", "ai_act"),
    ("2024/1689", "ai_act"),
    ("reglement ia", "ai_act"),
    ("règlement ia", "ai_act"),
    ("gdpr", "gdpr"),
    ("rgpd", "gdpr"),
    ("2016/679", "gdpr"),
)

_ARTICLE = re.compile(
    r"\b(?:articles?|arts?\.?)\s*(?:premier|(\d{1,3}))"
    r"(?:\s*(?:\(\s*(\d{1,2})\s*\)|,?\s*(?:§|paragraphe|paragraph|par\.)\s*(\d{1,2})))?",
    re.IGNORECASE,
)
_ANNEX = re.compile(r"\b(?:annexe?s?)\s*([IVXL]{1,6}|\d{1,2})\b", re.IGNORECASE)
_RECITAL = re.compile(
    r"\b(?:recitals?|considerants?|considérants?)\s*\(?(\d{1,3})\)?", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class ProvisionReference:
    """A citation found in the question."""

    kind: ProvisionKind
    number: str
    paragraph: str | None = None
    source_id: str | None = None

    def provision_id(self, source_id: str | None = None) -> str | None:
        """Full provision id, once the act is known."""
        target = self.source_id or source_id
        return f"{target}:{self.kind}:{self.number}" if target else None


def find_references(query: str) -> list[ProvisionReference]:
    """Every explicit provision cited in the query, in order of appearance."""
    source_id = _detect_source(query)
    references: list[ProvisionReference] = []

    for match in _ARTICLE.finditer(query):
        # "Article premier" is how the French text names Article 1.
        number = match.group(1) or "1"
        paragraph = match.group(2) or match.group(3)
        references.append(ProvisionReference(ProvisionKind.ARTICLE, number, paragraph, source_id))
    for match in _ANNEX.finditer(query):
        references.append(
            ProvisionReference(ProvisionKind.ANNEX, match.group(1).upper(), None, source_id)
        )
    for match in _RECITAL.finditer(query):
        references.append(
            ProvisionReference(ProvisionKind.RECITAL, match.group(1), None, source_id)
        )
    return references


def _detect_source(query: str) -> str | None:
    lowered = query.lower()
    for needle, source_id in _SOURCE_HINTS:
        if needle in lowered:
            return source_id
    return None
