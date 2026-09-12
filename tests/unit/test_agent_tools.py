from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_act_copilot.agent.tools import (
    GetDefinition,
    GetProvision,
    SearchRegulations,
    ToolContext,
    default_tools,
    run_tool,
    strict_schema,
)
from ai_act_copilot.models import (
    Chunk,
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)
from ai_act_copilot.retrieval.base import RetrievedChunk
from ai_act_copilot.store.sqlite import CorpusStore


def _provision(provision_id: str, number: str, paragraphs: list[Paragraph]) -> Provision:
    return Provision(
        provision_id=provision_id,
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        kind=ProvisionKind.ARTICLE,
        number=number,
        title=f"Article {number}",
        breadcrumb=("AI Act", f"Article {number}"),
        paragraphs=tuple(paragraphs),
        order=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    corpus = CorpusStore(tmp_path / "corpus.db")
    document = Document(
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        title="AI Act",
        url="http://example.invalid",
        sha256="0" * 64,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        provisions=(
            _provision("ai_act:art:6", "6", [Paragraph(label="1", text="Classification rules.")]),
            _provision(
                "ai_act:art:3",
                "3",
                [
                    Paragraph(label="(1)", text="'AI system' means a machine-based system."),
                    Paragraph(label="(2)", text="'provider' means a natural or legal person."),
                ],
            ),
        ),
    )
    corpus.replace_document(document, [])
    return corpus


class StubRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    def search(
        self,
        query: str,
        *,
        language: Language | None = None,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        return self.hits


def _hit(provision_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        strategy=ChunkStrategy.STRUCTURAL,
        order=0,
        header="AI Act > Article 5",
        text=text,
        token_count=5,
        provision_ids=(provision_id,),
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1)


def _context(store: CorpusStore, hits: list[RetrievedChunk] | None = None) -> ToolContext:
    return ToolContext(retriever=StubRetriever(hits or []), store=store, language=Language.EN)


def _tool(name: str):  # type: ignore[no-untyped-def]
    return next(tool for tool in default_tools() if tool.name == name)


def test_strict_schema_makes_every_field_required() -> None:
    schema = strict_schema(SearchRegulations)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"query", "sources", "limit"}
    assert all("default" not in prop for prop in schema["properties"].values())


def test_tool_definitions_are_what_the_api_expects() -> None:
    definition = _tool("get_provision").definition()

    assert definition["name"] == "get_provision"
    assert definition["strict"] is True
    assert definition["input_schema"]["required"] == ["provision_id"]
    assert definition["description"]


def test_invalid_arguments_come_back_as_a_tool_error(store: CorpusStore) -> None:
    text, failed = run_tool(_tool("get_provision"), {"wrong": "field"}, _context(store))

    assert failed
    assert "Invalid arguments" in text


def test_search_records_the_provisions_it_returned(store: CorpusStore) -> None:
    context = _context(store, [_hit("ai_act:art:5", "Prohibited practices.")])

    text, failed = run_tool(_tool("search_regulations"), {"query": "prohibited"}, context)

    assert not failed
    assert "[ai_act:art:5]" in text
    assert context.provisions == ["ai_act:art:5"]


def test_search_without_results_suggests_what_to_do(store: CorpusStore) -> None:
    text, failed = run_tool(_tool("search_regulations"), {"query": "blockchain"}, _context(store))

    assert not failed
    assert "No passage found" in text


def test_get_provision_returns_the_full_text(store: CorpusStore) -> None:
    context = _context(store)

    text, failed = run_tool(_tool("get_provision"), {"provision_id": "ai_act:art:6"}, context)

    assert not failed
    assert "Classification rules." in text
    assert context.provisions == ["ai_act:art:6"]


def test_unknown_provision_explains_the_id_format(store: CorpusStore) -> None:
    text, failed = run_tool(
        _tool("get_provision"), {"provision_id": "ai_act:art:999"}, _context(store)
    )

    assert not failed
    assert "ai_act:art:6" in text  # shows the expected shape


def test_get_definition_finds_a_defined_term(store: CorpusStore) -> None:
    context = _context(store)

    text, failed = run_tool(_tool("get_definition"), {"term": "AI system"}, context)

    assert not failed
    assert "machine-based system" in text
    assert "ai_act:art:3" in context.provisions


def test_get_definition_without_a_match_points_to_search(store: CorpusStore) -> None:
    text, _ = run_tool(_tool("get_definition"), {"term": "blockchain"}, _context(store))

    assert "search_regulations" in text


def test_submit_answer_is_acknowledged(store: CorpusStore) -> None:
    text, failed = run_tool(
        _tool("submit_answer"), {"answer": "Article 5 applies.", "citations": []}, _context(store)
    )

    assert not failed
    assert "recorded" in text.lower()


def test_models_validate_their_inputs() -> None:
    assert SearchRegulations(query="q").limit is None
    assert GetProvision(provision_id="ai_act:art:6").provision_id.endswith("6")
    assert GetDefinition(term="provider").term == "provider"
