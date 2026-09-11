from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from ai_act_copilot.chunking.base import split_by_tokens
from ai_act_copilot.chunking.fixed import FixedSizeChunker
from ai_act_copilot.chunking.structural import StructuralChunker
from ai_act_copilot.chunking.tokenizer import WordTokenCounter
from ai_act_copilot.models import (
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)

COUNTER = WordTokenCounter()


def _provision(number: str, paragraphs: list[Paragraph], order: int = 0) -> Provision:
    return Provision(
        provision_id=f"sample:art:{number}",
        document_id="sample:en",
        source_id="sample",
        language=Language.EN,
        kind=ProvisionKind.ARTICLE,
        number=number,
        title=f"Article {number} title",
        breadcrumb=("Sample Act", "CHAPTER I", f"Article {number}"),
        paragraphs=tuple(paragraphs),
        order=order,
    )


def _document(*provisions: Provision) -> Document:
    return Document(
        document_id="sample:en",
        source_id="sample",
        language=Language.EN,
        title="Sample Act",
        url="http://example.invalid",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        provisions=provisions,
    )


@given(
    words=st.lists(st.from_regex(r"[a-z]{1,8}", fullmatch=True), min_size=1, max_size=200),
    max_tokens=st.integers(min_value=1, max_value=20),
    overlap=st.integers(min_value=0, max_value=10),
)
@hypothesis_settings(max_examples=50)
def test_split_by_tokens_respects_budget_and_loses_nothing(
    words: list[str], max_tokens: int, overlap: int
) -> None:
    overlap = min(overlap, max_tokens - 1)
    text = " ".join(words)

    windows = split_by_tokens(text, COUNTER, max_tokens, overlap)

    assert all(COUNTER.count(window) <= max_tokens for window in windows)
    covered = " ".join(windows).split()
    for word in words:
        assert word in covered


def test_split_by_tokens_overlaps_consecutive_windows() -> None:
    text = " ".join(f"w{index}" for index in range(10))

    windows = split_by_tokens(text, COUNTER, max_tokens=4, overlap_tokens=2)

    assert windows[0] == "w0 w1 w2 w3"
    assert windows[1].startswith("w2 w3")


def test_split_by_tokens_rejects_impossible_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        split_by_tokens("a b c", COUNTER, max_tokens=2, overlap_tokens=2)


def test_structural_chunker_keeps_one_provision_per_chunk() -> None:
    document = _document(
        _provision("1", [Paragraph(label="1", text="short text")], order=0),
        _provision("2", [Paragraph(label="1", text="another short text")], order=1),
    )

    chunks = StructuralChunker(COUNTER, max_tokens=64).chunk(document)

    assert len(chunks) == 2
    assert [chunk.provision_ids for chunk in chunks] == [("sample:art:1",), ("sample:art:2",)]
    assert chunks[0].strategy is ChunkStrategy.STRUCTURAL
    assert chunks[0].header == "Sample Act > CHAPTER I > Article 1"
    assert chunks[0].embedding_text.startswith("Sample Act > CHAPTER I > Article 1\n")


def test_structural_chunker_splits_long_provisions_on_paragraph_boundaries() -> None:
    paragraphs = [Paragraph(label=str(i), text=" ".join(["word"] * 30)) for i in range(1, 5)]
    document = _document(_provision("1", paragraphs))

    chunks = StructuralChunker(COUNTER, max_tokens=80, min_tokens=5).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all(chunk.provision_ids == ("sample:art:1",) for chunk in chunks)
    assert all(chunk.header for chunk in chunks)


def test_structural_chunker_merges_a_short_trailing_fragment() -> None:
    paragraphs = [
        Paragraph(label="1", text=" ".join(["word"] * 40)),
        Paragraph(label="2", text=" ".join(["word"] * 40)),
        Paragraph(label="3", text="tiny tail"),
    ]
    document = _document(_provision("1", paragraphs))

    # Budget forces one paragraph per chunk, leaving the 3-token tail on its own.
    chunks = StructuralChunker(COUNTER, max_tokens=50, min_tokens=20).chunk(document)

    assert "tiny tail" in chunks[-1].text
    assert len(chunks) == 2
    assert chunks[-1].text.startswith("2 ")  # the tail was folded into the previous chunk


def test_fixed_chunker_ignores_structure_but_tracks_provisions() -> None:
    document = _document(
        _provision("1", [Paragraph(label="1", text=" ".join(["alpha"] * 30))], order=0),
        _provision("2", [Paragraph(label="1", text=" ".join(["beta"] * 30))], order=1),
    )

    chunks = FixedSizeChunker(COUNTER, max_tokens=40, overlap_tokens=10).chunk(document)

    assert all(chunk.header == "" for chunk in chunks)
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert all(chunk.provision_ids for chunk in chunks)
    covered = {pid for chunk in chunks for pid in chunk.provision_ids}
    assert covered == {"sample:art:1", "sample:art:2"}


def test_chunkers_return_nothing_for_an_empty_document() -> None:
    document = _document()

    assert FixedSizeChunker(COUNTER).chunk(document) == []
    assert StructuralChunker(COUNTER).chunk(document) == []
