from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_act_copilot.config import Settings
from ai_act_copilot.embeddings.indexer import build_index
from ai_act_copilot.models import (
    Chunk,
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)
from ai_act_copilot.retrieval.hybrid import HybridRetriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore
from tests.doubles import FakeEmbedder

PASSAGES: list[tuple[str, str, Language, str]] = [
    (
        "ai_act:art:5",
        "5",
        Language.EN,
        "Prohibited practices include social scoring by public authorities.",
    ),
    (
        "ai_act:art:6",
        "6",
        Language.EN,
        "Classification rules for high-risk AI systems and their conditions.",
    ),
    (
        "ai_act:art:6",
        "6",
        Language.EN,
        "An AI system referred to in Annex III shall be considered high-risk.",
    ),
    ("gdpr:art:6", "6", Language.EN, "Processing shall be lawful only if consent has been given."),
    (
        "ai_act:art:5",
        "5",
        Language.FR,
        "Les pratiques interdites comprennent la notation sociale par les autorites.",
    ),
]


def _build_corpus(settings: Settings) -> None:
    with CorpusStore(settings.database_path) as store:
        for language in (Language.EN, Language.FR):
            for source_id in ("ai_act", "gdpr"):
                rows = [
                    row
                    for row in PASSAGES
                    if row[2] is language and row[0].startswith(f"{source_id}:")
                ]
                if not rows:
                    continue
                provisions = tuple(
                    Provision(
                        provision_id=provision_id,
                        document_id=f"{source_id}:{language}",
                        source_id=source_id,
                        language=language,
                        kind=ProvisionKind.ARTICLE,
                        number=number,
                        title=f"Article {number}",
                        breadcrumb=(source_id, f"Article {number}"),
                        paragraphs=(Paragraph(label="1", text=text),),
                        order=order,
                    )
                    for order, (provision_id, number, _, text) in enumerate(
                        {row[0]: row for row in rows}.values()
                    )
                )
                document = Document(
                    document_id=f"{source_id}:{language}",
                    source_id=source_id,
                    language=language,
                    title=source_id,
                    url="http://example.invalid",
                    sha256="0" * 64,
                    fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
                    provisions=provisions,
                )
                chunks = [
                    Chunk(
                        chunk_id=f"{source_id}:{language}:structural:{order:04d}",
                        document_id=f"{source_id}:{language}",
                        source_id=source_id,
                        language=language,
                        strategy=ChunkStrategy.STRUCTURAL,
                        order=order,
                        header=f"{source_id} > Article {number}",
                        text=text,
                        token_count=len(text.split()),
                        provision_ids=(provision_id,),
                    )
                    for order, (provision_id, number, _, text) in enumerate(rows)
                ]
                store.replace_document(document, chunks)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    configured = Settings(data_dir=data_dir, retrieval_top_k=3, retrieval_candidates=10)
    _build_corpus(configured)
    return configured


def _retriever(settings: Settings, embedder: FakeEmbedder) -> HybridRetriever:
    build_index(settings, embedder=embedder)
    store = CorpusStore(settings.database_path)
    vectors = VectorStore(settings.database_path)
    return HybridRetriever(store, vectors, embedder, settings=settings)


def test_finds_passages_by_keywords(settings: Settings) -> None:
    hits = _retriever(settings, FakeEmbedder()).search("social scoring", language=Language.EN)

    assert hits
    assert "ai_act:art:5" in hits[0].provision_ids
    assert set(hits[0].components) & {"bm25", "dense"}


def test_explicit_citation_is_retrieved_and_flagged(settings: Settings) -> None:
    hits = _retriever(settings, FakeEmbedder()).search(
        "What does Article 6 of the AI Act say?", language=Language.EN
    )

    assert hits[0].matched_reference
    assert "ai_act:art:6" in hits[0].provision_ids
    assert "reference" in hits[0].components


def test_one_provision_never_fills_several_slots(settings: Settings) -> None:
    hits = _retriever(settings, FakeEmbedder()).search("high-risk AI systems", language=Language.EN)

    provisions = [hit.provision_ids for hit in hits]
    assert len(provisions) == len(set(provisions))


def test_french_question_searches_the_french_corpus(settings: Settings) -> None:
    hits = _retriever(settings, FakeEmbedder()).search("Quelles sont les pratiques interdites ?")

    assert hits
    assert all(hit.chunk.language is Language.FR for hit in hits)


def test_sources_filter_restricts_results(settings: Settings) -> None:
    hits = _retriever(settings, FakeEmbedder()).search(
        "lawful processing", language=Language.EN, sources=["gdpr"]
    )

    assert all(hit.chunk.source_id == "gdpr" for hit in hits)


def test_indexing_reuses_cached_vectors_on_a_second_run(settings: Settings) -> None:
    embedder = FakeEmbedder()
    first = build_index(settings, embedder=embedder)
    embedded_first = len(embedder.embedded)

    second = build_index(settings, embedder=embedder)

    assert first.embedded == first.chunks
    assert second.embedded == 0
    assert second.reused == second.chunks
    assert second.cache_hit_rate == 1.0
    assert len(embedder.embedded) == embedded_first  # nothing was re-embedded


def test_indexing_without_a_corpus_explains_what_to_run(tmp_path: Path) -> None:
    empty = Settings(data_dir=tmp_path / "empty")

    with pytest.raises(RuntimeError, match="aiact ingest"):
        build_index(empty, embedder=FakeEmbedder())
