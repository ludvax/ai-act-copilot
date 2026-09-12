from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_act_copilot.agent.graph import build_graph, open_checkpointer, run_agent
from ai_act_copilot.agent.guardrails import Budget
from ai_act_copilot.agent.nodes import AgentDeps
from ai_act_copilot.agent.state import Route
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
from tests.conftest import REPO_ROOT
from tests.doubles import ScriptedLLM, calls, decides

PROMPTS = REPO_ROOT / "prompts"


def _hit(provision_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"chunk-{provision_id}",
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        strategy=ChunkStrategy.STRUCTURAL,
        order=0,
        header="AI Act > Article 5",
        text=text,
        token_count=6,
        provision_ids=(provision_id,),
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1)


class StubRetriever:
    def search(
        self,
        query: str,
        *,
        language: Language | None = None,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        return [_hit("ai_act:art:5", "The following AI practices are prohibited.")]


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    corpus = CorpusStore(tmp_path / "corpus.db")
    provision = Provision(
        provision_id="ai_act:art:5",
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        kind=ProvisionKind.ARTICLE,
        number="5",
        title="Prohibited practices",
        breadcrumb=("AI Act", "Article 5"),
        paragraphs=(Paragraph(label="1", text="The following practices are prohibited."),),
        order=0,
    )
    corpus.replace_document(
        Document(
            document_id="ai_act:en",
            source_id="ai_act",
            language=Language.EN,
            title="AI Act",
            url="http://example.invalid",
            sha256="0" * 64,
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            provisions=(provision,),
        ),
        [],
    )
    return corpus


def _deps(store: CorpusStore, llm: ScriptedLLM, budget: Budget | None = None) -> AgentDeps:
    return AgentDeps(
        llm=llm,
        retriever=StubRetriever(),
        store=store,
        budget=budget or Budget(max_steps=4, max_cost_usd=1.0),
        prompts_dir=PROMPTS,
    )


def test_simple_question_takes_the_cheap_rag_path(store: CorpusStore) -> None:
    llm = ScriptedLLM(
        decides(route=Route.LOOKUP, reason="one article"),
        decides(answer="Article 5 lists them.", citations=["ai_act:art:5"], abstained=False),
    )

    answer = run_agent("Which practices are prohibited?", _deps(store, llm))

    assert answer.route is Route.LOOKUP
    assert answer.citations == ("ai_act:art:5",)
    assert not answer.abstained
    assert len(llm.calls) == 2  # router + one grounded answer, no tool loop


def test_out_of_scope_question_abstains_without_a_second_call(store: CorpusStore) -> None:
    llm = ScriptedLLM(decides(route=Route.OUT_OF_SCOPE, reason="not about EU AI rules"))

    answer = run_agent("What is the capital of France?", _deps(store, llm))

    assert answer.abstained
    assert answer.citations == ()
    assert len(llm.calls) == 1


def test_complex_question_runs_the_tool_loop(store: CorpusStore) -> None:
    llm = ScriptedLLM(
        decides(route=Route.COMPLEX, reason="needs several steps"),
        calls("search_regulations", {"query": "prohibited practices"}),
        calls(
            "submit_answer",
            {"answer": "Social scoring is prohibited.", "citations": ["ai_act:art:5"]},
            "call_2",
        ),
    )

    answer = run_agent("Is social scoring allowed for a public body?", _deps(store, llm))

    assert answer.route is Route.COMPLEX
    assert answer.text == "Social scoring is prohibited."
    assert answer.citations == ("ai_act:art:5",)
    assert "ai_act:art:5" in answer.provisions_seen
    assert answer.steps == 2


def test_citation_never_retrieved_triggers_one_correction(store: CorpusStore) -> None:
    llm = ScriptedLLM(
        decides(route=Route.COMPLEX, reason="scenario"),
        calls("search_regulations", {"query": "prohibited"}),
        calls("submit_answer", {"answer": "See Article 99.", "citations": ["ai_act:art:99"]}, "c2"),
        calls(
            "submit_answer",
            {"answer": "Social scoring is prohibited.", "citations": ["ai_act:art:5"]},
            "c3",
        ),
    )

    answer = run_agent("Is social scoring allowed?", _deps(store, llm))

    assert answer.citations == ("ai_act:art:5",)
    correction = str(llm.calls[-1]["messages"][-1]["content"])
    assert "not in anything you retrieved" in correction


def test_unverifiable_citations_are_dropped_after_one_correction(store: CorpusStore) -> None:
    llm = ScriptedLLM(
        decides(route=Route.COMPLEX, reason="scenario"),
        calls("search_regulations", {"query": "prohibited"}),
        calls("submit_answer", {"answer": "See Article 99.", "citations": ["ai_act:art:99"]}, "c2"),
        calls("submit_answer", {"answer": "Still 99.", "citations": ["ai_act:art:99"]}, "c3"),
    )

    answer = run_agent("Is social scoring allowed?", _deps(store, llm))

    assert answer.citations == ()


def test_step_budget_halts_a_runaway_loop(store: CorpusStore) -> None:
    attempts = [calls("search_regulations", {"query": f"try {n}"}, f"c{n}") for n in range(6)]
    llm = ScriptedLLM(decides(route=Route.COMPLEX, reason="scenario"), *attempts)

    answer = run_agent(
        "Endless question?", _deps(store, llm, Budget(max_steps=2, max_cost_usd=1.0))
    )

    assert answer.halted_by is not None
    assert "limit 2" in answer.halted_by
    assert answer.abstained


def test_repeating_the_same_tool_call_is_refused(store: CorpusStore) -> None:
    llm = ScriptedLLM(
        decides(route=Route.COMPLEX, reason="scenario"),
        calls("search_regulations", {"query": "same"}, "c1"),
        calls("search_regulations", {"query": "same"}, "c2"),
        calls("submit_answer", {"answer": "Done.", "citations": ["ai_act:art:5"]}, "c3"),
    )

    run_agent("Repeated question?", _deps(store, llm))

    results = [
        block
        for call in llm.calls
        for message in call["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert any(item["is_error"] and "already" in str(item["content"]) for item in results)


def test_a_thread_id_continues_the_conversation(store: CorpusStore, tmp_path: Path) -> None:
    checkpointer = open_checkpointer(tmp_path / "threads.db")
    first = ScriptedLLM(
        decides(route=Route.LOOKUP, reason="one article"),
        decides(answer="Article 5 lists them.", citations=["ai_act:art:5"], abstained=False),
    )
    run_agent(
        "Which practices are prohibited?",
        _deps(store, first),
        thread_id="thread-1",
        checkpointer=checkpointer,
    )

    second = ScriptedLLM(
        decides(route=Route.LOOKUP, reason="follow-up"),
        decides(answer="For providers, Article 16 applies.", citations=[], abstained=False),
    )
    answer = run_agent(
        "And for providers?", _deps(store, second), thread_id="thread-1", checkpointer=checkpointer
    )

    assert answer.thread_id == "thread-1"
    assert any("prohibited" in str(message) for message in second.calls[-1]["messages"])


def test_the_graph_can_be_drawn(store: CorpusStore) -> None:
    diagram = build_graph(_deps(store, ScriptedLLM())).get_graph().draw_mermaid()

    assert "route" in diagram
    assert "verify" in diagram
    assert "finalize" in diagram
