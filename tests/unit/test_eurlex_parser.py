from datetime import UTC, datetime

import pytest

from ai_act_copilot.ingestion.parsers.eurlex import parse_eurlex_xhtml
from ai_act_copilot.models import Document, Language, ProvisionKind
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def document() -> Document:
    return parse_eurlex_xhtml(
        (FIXTURES / "sample_act.xhtml").read_bytes(),
        source_id="sample",
        short_title="Sample Act",
        title="Sample Act of 2024",
        language=Language.EN,
        url="http://example.invalid/sample",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_extracts_every_kind_of_provision(document: Document) -> None:
    kinds = [provision.kind for provision in document.provisions]

    assert kinds.count(ProvisionKind.RECITAL) == 2
    assert kinds.count(ProvisionKind.ARTICLE) == 2
    assert kinds.count(ProvisionKind.ANNEX) == 1
    assert document.document_id == "sample:en"


def _by_id(document: Document, provision_id: str):  # type: ignore[no-untyped-def]
    return next(p for p in document.provisions if p.provision_id == provision_id)


def test_recital_text_excludes_its_own_number(document: Document) -> None:
    recital = _by_id(document, "sample:rct:1")

    assert recital.text.startswith("The purpose of this Regulation")
    assert recital.citation == "Recital (1)"


def test_article_breadcrumb_carries_chapter_and_section(document: Document) -> None:
    article = _by_id(document, "sample:art:1")

    assert article.breadcrumb == (
        "Sample Act",
        "CHAPTER I — GENERAL PROVISIONS",
        "SECTION 1 — Scope",
        "Article 1 — Subject matter",
    )
    assert article.title == "Subject matter"


def test_numbered_paragraphs_keep_their_label_without_repeating_it(document: Document) -> None:
    article = _by_id(document, "sample:art:1")

    assert [paragraph.label for paragraph in article.paragraphs] == ["1", "2"]
    assert article.paragraphs[0].text.startswith("This Regulation lays down harmonised rules")


def test_enumerations_stay_attached_to_their_label(document: Document) -> None:
    article = _by_id(document, "sample:art:1")

    assert "(a) rules for placing AI systems on the market;" in article.paragraphs[0].text
    assert "(b) prohibitions of certain practices." in article.paragraphs[0].text


def test_footnotes_and_their_markers_are_removed(document: Document) -> None:
    article = _by_id(document, "sample:art:1")

    assert "OJ L 1" not in article.text
    assert "(1)" not in article.paragraphs[0].text


def test_article_without_numbered_paragraphs_keeps_definition_labels(document: Document) -> None:
    article = _by_id(document, "sample:art:2")

    assert article.paragraphs[0].label is None
    assert article.paragraphs[1].label == "(1)"
    assert "'AI system' means" in article.paragraphs[1].text


def test_annex_title_and_items(document: Document) -> None:
    annex = _by_id(document, "sample:anx:I")

    assert annex.title == "List of high-risk areas"
    assert annex.citation == "Annex I"
    assert annex.paragraphs[-1].label == "1."
