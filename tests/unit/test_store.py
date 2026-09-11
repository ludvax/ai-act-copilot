from datetime import UTC, datetime
from pathlib import Path

from ai_act_copilot.models import (
    Chunk,
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)
from ai_act_copilot.store.sqlite import CorpusStore


def _document() -> Document:
    provision = Provision(
        provision_id="sample:art:6",
        document_id="sample:en",
        source_id="sample",
        language=Language.EN,
        kind=ProvisionKind.ARTICLE,
        number="6",
        title="Classification rules",
        breadcrumb=("Sample Act", "CHAPTER III", "Article 6"),
        paragraphs=(Paragraph(label="1", text="First paragraph."),),
        order=0,
    )
    return Document(
        document_id="sample:en",
        source_id="sample",
        language=Language.EN,
        title="Sample Act",
        url="http://example.invalid",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        provisions=(provision,),
    )


def _chunk(order: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"sample:en:structural:{order:04d}",
        document_id="sample:en",
        source_id="sample",
        language=Language.EN,
        strategy=ChunkStrategy.STRUCTURAL,
        order=order,
        header="Sample Act > Article 6",
        text="First paragraph.",
        token_count=3,
        provision_ids=("sample:art:6",),
    )


def test_round_trips_documents_provisions_and_chunks(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as store:
        store.replace_document(_document(), [_chunk()])

        chunks = list(store.chunks(strategy=ChunkStrategy.STRUCTURAL, language=Language.EN))
        provision = store.provision("sample:art:6", Language.EN)

    assert [chunk.chunk_id for chunk in chunks] == ["sample:en:structural:0000"]
    assert chunks[0].provision_ids == ("sample:art:6",)
    assert provision is not None
    assert provision.paragraphs[0].label == "1"
    assert provision.breadcrumb == ("Sample Act", "CHAPTER III", "Article 6")


def test_reingesting_replaces_instead_of_duplicating(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as store:
        store.replace_document(_document(), [_chunk(0), _chunk(1)])
        store.replace_document(_document(), [_chunk(0)])

        summaries = store.document_summaries()
        chunks = list(store.chunks())

    assert len(chunks) == 1
    assert summaries[0]["provisions"] == 1
    assert summaries[0]["chunks"] == 1


def test_unknown_provision_returns_none(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as store:
        assert store.provision("sample:art:999", Language.EN) is None
