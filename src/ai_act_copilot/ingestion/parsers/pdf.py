"""Parser for guidance documents published as PDF (Commission, CNIL).

Guidance has no articles, so the citable unit is a numbered section: headings like
"3.2. Conditions" start a new provision. Documents without numbered headings degrade to a
single section rather than failing — they are still retrievable, just less precisely cited.

Real guidance PDFs reuse heading numbers (annexes restart at "1.", and list items are
easily mistaken for headings), so section numbers are de-duplicated here: provision ids
must be unique within a document.
"""

import io
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from pypdf import PdfReader

from ai_act_copilot.models import Document, Language, Paragraph, Provision, ProvisionKind

# A heading is "3." / "3.2." / "3.2.1." (trailing dot) or a multi-level "3.2" — a bare
# number is not enough, or every footnote marker ("207 https://...") becomes a section.
_HEADING = re.compile(r"^(?:(\d+(?:\.\d+)*)\.|(\d+(?:\.\d+)+))\s+(\S.{0,110})$")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
# Dotted leaders mark table-of-contents entries: "3.1. Rationale ..... 12".
_TOC_LEADER = re.compile(r"\.{4,}")
# Sections shorter than this carry no retrievable signal (stray fragments, page furniture).
_MIN_SECTION_WORDS = 5


@dataclass(frozen=True, slots=True)
class Section:
    """A numbered section; ``number`` is unique within the document."""

    number: str
    heading: str
    lines: list[str] = field(default_factory=list)


def parse_pdf(
    raw: bytes,
    *,
    source_id: str,
    short_title: str,
    title: str,
    language: Language,
    url: str,
    sha256: str,
    fetched_at: datetime,
) -> Document:
    """Turn a guidance PDF into a document of numbered sections."""
    reader = PdfReader(io.BytesIO(raw))
    lines = _lines(page.extract_text() or "" for page in reader.pages)
    sections = split_sections(lines, fallback_heading=title)

    provisions = [
        Provision(
            provision_id=f"{source_id}:{ProvisionKind.SECTION}:{section.number}",
            document_id=f"{source_id}:{language}",
            source_id=source_id,
            language=language,
            kind=ProvisionKind.SECTION,
            number=section.number,
            title=section.heading,
            breadcrumb=(short_title, f"Section {section.number} — {section.heading}"),
            paragraphs=tuple(
                Paragraph(label=None, text=block) for block in _paragraphs(section.lines) if block
            ),
            order=order,
        )
        for order, section in enumerate(sections)
    ]

    return Document(
        document_id=f"{source_id}:{language}",
        source_id=source_id,
        language=language,
        title=title,
        url=url,
        sha256=sha256,
        fetched_at=fetched_at,
        provisions=tuple(
            provision
            for provision in provisions
            if len(provision.text.split()) >= _MIN_SECTION_WORDS
        ),
    )


def split_sections(lines: list[str], *, fallback_heading: str) -> list[Section]:
    """Group lines under their heading, giving every section a unique number."""
    sections: list[Section] = []
    seen: Counter[str] = Counter()
    for line in lines:
        heading = _HEADING.match(line)
        if heading and not _looks_like_body(heading.group(3)):
            raw_number = heading.group(1) or heading.group(2)
            seen[raw_number] += 1
            occurrence = seen[raw_number]
            number = raw_number if occurrence == 1 else f"{raw_number}-{occurrence}"
            sections.append(Section(number, heading.group(3).strip()))
        elif sections:
            sections[-1].lines.append(line)
        else:
            seen["1"] += 1
            sections.append(Section("1", fallback_heading, [line]))
    return sections


def _lines(pages: Iterable[str]) -> list[str]:
    collected: list[str] = []
    for page_text in pages:
        for raw_line in page_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line.replace("\xa0", " ")).strip()
            if line and not _PAGE_NUMBER.match(line) and not _TOC_LEADER.search(line):
                collected.append(line)
    return collected


def _looks_like_body(text: str) -> bool:
    """A numbered list item inside a sentence is not a section heading."""
    return text.endswith((".", ";", ",")) and len(text.split()) > 12


def _paragraphs(lines: list[str]) -> list[str]:
    """Re-join wrapped lines: a new paragraph starts after a line ending a sentence."""
    blocks: list[list[str]] = [[]]
    for line in lines:
        blocks[-1].append(line)
        if line.endswith((".", ":", ";")) and len(" ".join(blocks[-1])) > 200:
            blocks.append([])
    return [" ".join(block).strip() for block in blocks if block]
