"""Domain models shared by ingestion, retrieval and evaluation.

A **provision** is the citable unit of a legal text (a recital, an article, an annex).
Evaluation labels reference provisions, never chunks, so the golden dataset stays valid
when the chunking strategy changes.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Language(StrEnum):
    EN = "en"
    FR = "fr"


class ProvisionKind(StrEnum):
    RECITAL = "rct"
    ARTICLE = "art"
    ANNEX = "anx"
    SECTION = "sec"  # guidance documents (PDF/HTML) have headed sections, not articles


class Paragraph(BaseModel):
    """A numbered paragraph, or an unlabelled block for texts without numbering."""

    model_config = ConfigDict(frozen=True)

    label: str | None = None
    text: str


class Provision(BaseModel):
    """One citable unit, e.g. ``ai_act:art:6`` in a given language."""

    model_config = ConfigDict(frozen=True)

    provision_id: str  # language-neutral, e.g. "ai_act:art:6"
    document_id: str  # e.g. "ai_act:en"
    source_id: str  # e.g. "ai_act"
    language: Language
    kind: ProvisionKind
    number: str  # "6", "III", "1"
    title: str | None = None
    breadcrumb: tuple[str, ...] = ()
    paragraphs: tuple[Paragraph, ...] = ()
    order: int = 0

    @property
    def text(self) -> str:
        return "\n".join(
            f"{p.label} {p.text}" if p.label else p.text for p in self.paragraphs
        ).strip()

    @property
    def citation(self) -> str:
        """Human-readable reference, e.g. ``Article 6`` or ``Recital (12)``."""
        match self.kind:
            case ProvisionKind.ARTICLE:
                return f"Article {self.number}"
            case ProvisionKind.RECITAL:
                return f"Recital ({self.number})"
            case ProvisionKind.ANNEX:
                return f"Annex {self.number}"
            case ProvisionKind.SECTION:
                return self.title or f"Section {self.number}"


class Document(BaseModel):
    """One source text in one language, as fetched and parsed."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_id: str
    language: Language
    title: str
    url: str
    sha256: str
    fetched_at: datetime
    provisions: tuple[Provision, ...] = ()


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    STRUCTURAL = "structural"


class Chunk(BaseModel):
    """An embeddable passage, traceable back to the provisions it covers."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_id: str
    language: Language
    strategy: ChunkStrategy
    order: int
    header: str = ""  # breadcrumb prepended at embedding time, e.g. "AI Act > Art. 6 > 2"
    text: str
    token_count: int
    provision_ids: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded: the passage in its structural context."""
        return f"{self.header}\n{self.text}" if self.header else self.text
