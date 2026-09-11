"""Parser for the Official Journal XHTML served by Cellar.

The format is stable across acts (the GDPR, published in 2016, is served in the same
markup as the 2024 AI Act), which is why one parser covers the whole legal corpus:

    div.eli-subdivision#rct_1      a recital
    div#cpt_III / div#cpt_III.sct_1   chapter and section, carrying the titles
    div.eli-subdivision#art_6      an article
      p.oj-ti-art                  "Article 6"
      div.eli-title > p.oj-sti-art article title
      div#006.001                  numbered paragraph (absent in some articles)
    div.eli-container#anx_III      an annex

Enumerations ("(a) ...", "(1) ...") are two-column tables: the left cell holds the label,
the right cell the text. Footnotes (``p.oj-note``) are nested *inside* articles and are
dropped, along with the superscript markers that reference them.
"""

import re
from datetime import datetime

from lxml import etree

from ai_act_copilot.models import Document, Language, Paragraph, Provision, ProvisionKind

_NS = {"x": "http://www.w3.org/1999/xhtml"}
_PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")
_LEADING_NUMBER = re.compile(r"^\d+\.\s+")
_KIND_BY_ID_PREFIX = {
    "rct": ProvisionKind.RECITAL,
    "art": ProvisionKind.ARTICLE,
    "anx": ProvisionKind.ANNEX,
}

# Untrusted XML hardening: no DTD, no entity resolution, no network access.
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)

type _Block = tuple[str | None, str]


def parse_eurlex_xhtml(
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
    """Turn one Official Journal XHTML file into a document of provisions."""
    root = etree.fromstring(raw, parser=_PARSER)
    _drop_footnotes(root)

    provisions: list[Provision] = []
    for element in root.iter(f"{{{_NS['x']}}}div"):
        identifier = element.get("id", "")
        prefix, _, number = identifier.partition("_")
        kind = _KIND_BY_ID_PREFIX.get(prefix)
        if kind is None or "." in identifier or not number:
            continue  # not a provision container (e.g. "art_6.tit_1", "cpt_III.sct_1")
        provisions.append(
            _build_provision(
                element,
                kind=kind,
                number=number,
                source_id=source_id,
                short_title=short_title,
                language=language,
                order=len(provisions),
            )
        )

    return Document(
        document_id=f"{source_id}:{language}",
        source_id=source_id,
        language=language,
        title=title,
        url=url,
        sha256=sha256,
        fetched_at=fetched_at,
        provisions=tuple(provisions),
    )


def _build_provision(
    element: etree._Element,
    *,
    kind: ProvisionKind,
    number: str,
    source_id: str,
    short_title: str,
    language: Language,
    order: int,
) -> Provision:
    heading = _heading(element, kind)
    return Provision(
        provision_id=f"{source_id}:{kind}:{number}",
        document_id=f"{source_id}:{language}",
        source_id=source_id,
        language=language,
        kind=kind,
        number=number,
        title=heading,
        breadcrumb=_breadcrumb(element, kind, number, short_title, heading),
        paragraphs=_paragraphs(element, kind),
        order=order,
    )


def _heading(element: etree._Element, kind: ProvisionKind) -> str | None:
    if kind is ProvisionKind.ARTICLE:
        return _first_text(element, ".//x:p[@class='oj-sti-art']")
    if kind is ProvisionKind.ANNEX:
        titles = [
            _text(node) for node in element.xpath("./x:p[@class='oj-doc-ti']", namespaces=_NS)
        ]
        return titles[1] if len(titles) > 1 else None
    return None


def _breadcrumb(
    element: etree._Element,
    kind: ProvisionKind,
    number: str,
    short_title: str,
    heading: str | None,
) -> tuple[str, ...]:
    """Structural context prepended to every chunk, e.g. ``AI Act > Chapter III > ...``."""
    trail = [short_title]
    for ancestor in reversed(list(element.iterancestors(f"{{{_NS['x']}}}div"))):
        identifier = ancestor.get("id", "")
        if not identifier.startswith("cpt_"):
            continue
        label = _first_text(ancestor, "./x:p[@class='oj-ti-section-1']")
        name = _first_text(ancestor, "./x:div[@class='eli-title']/x:p[@class='oj-ti-section-2']")
        if part := " — ".join(filter(None, (label, name))):
            trail.append(part)

    own = {
        ProvisionKind.ARTICLE: f"Article {number}",
        ProvisionKind.RECITAL: f"Recital ({number})",
        ProvisionKind.ANNEX: f"Annex {number}",
        ProvisionKind.SECTION: f"Section {number}",
    }[kind]
    trail.append(" — ".join(filter(None, (own, heading))))
    return tuple(trail)


def _paragraphs(element: etree._Element, kind: ProvisionKind) -> tuple[Paragraph, ...]:
    numbered = [
        child
        for child in element.iterchildren(f"{{{_NS['x']}}}div")
        if _PARAGRAPH_ID.match(child.get("id", ""))
    ]
    if numbered:
        paragraphs = []
        for child in numbered:
            label = str(int(child.get("id", "000.000").split(".")[1]))
            text = _join(_blocks(child))
            paragraphs.append(Paragraph(label=label, text=_LEADING_NUMBER.sub("", text)))
        return tuple(paragraphs)

    blocks = [block for block in _blocks(element, skip_headings=True) if block[1]]
    if kind is ProvisionKind.RECITAL:
        # A recital's label is its own number, already carried by the provision id.
        return tuple(Paragraph(label=None, text=text) for _, text in blocks)
    return tuple(Paragraph(label=label, text=text) for label, text in blocks)


def _blocks(element: etree._Element, *, skip_headings: bool = False) -> list[_Block]:
    """Flatten content into labelled blocks, keeping enumerations attached to their label."""
    blocks: list[_Block] = []
    for child in element.iterchildren():
        tag = etree.QName(child).localname
        css = child.get("class", "")
        if tag == "p":
            if skip_headings and css in {"oj-ti-art", "oj-doc-ti"}:
                continue
            if text := _text(child):
                blocks.append((None, text))
        elif tag == "table":
            blocks.extend(_table_blocks(child))
        elif tag == "div":
            if skip_headings and child.get("class") == "eli-title":
                continue
            blocks.extend(_blocks(child, skip_headings=skip_headings))
    return blocks


def _table_blocks(table: etree._Element) -> list[_Block]:
    blocks: list[_Block] = []
    for row in table.iter(f"{{{_NS['x']}}}tr"):
        cells = list(row.iterchildren(f"{{{_NS['x']}}}td"))
        if len(cells) == 2:
            label = _text(cells[0]) or None
            inner = _blocks(cells[1])
            if inner:
                first_label, first_text = inner[0]
                blocks.append((label or first_label, first_text))
                blocks.extend(inner[1:])
        else:
            for cell in cells:
                blocks.extend(_blocks(cell))
    return blocks


def _join(blocks: list[_Block]) -> str:
    return "\n".join(f"{label} {text}" if label else text for label, text in blocks).strip()


def _first_text(element: etree._Element, xpath: str) -> str | None:
    found = element.xpath(xpath, namespaces=_NS)
    return _text(found[0]) if found else None


def _text(element: etree._Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext()).replace("\xa0", " ")).strip()


def _drop_footnotes(root: etree._Element) -> None:
    """Remove footnote bodies and the superscript markers that reference them."""
    notes = root.xpath("//x:p[@class='oj-note']", namespaces=_NS)
    markers = root.xpath("//x:a[.//x:span[contains(@class, 'oj-note-tag')]]", namespaces=_NS)
    for element in [*notes, *markers]:
        _remove_keeping_tail(element)


def _remove_keeping_tail(element: etree._Element) -> None:
    parent = element.getparent()
    if parent is None:
        return
    if tail := element.tail:
        previous = element.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)
