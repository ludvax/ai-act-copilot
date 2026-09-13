"""What a trace actually looks like when this system answers a question.

Instrumentation is the one part of a codebase that can be completely broken while every
other test passes: the answer is still right, the spans are just missing, misnamed or
empty. These tests run the real retriever and the real Anthropic client - the doubles stop
at the network - and assert the properties Langfuse's guidance asks for: a readable tree,
correct observation types, meaningful input and output, model and cost on generations,
sessions on conversations, and no personal data on the wire.

The Langfuse SDK is OpenTelemetry underneath, so pointing a real client at an in-memory
exporter produces exactly what a live project would receive. No account, no keys, no
network: the checks run in CI on every commit.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pytest
from langfuse._client.attributes import LangfuseOtelSpanAttributes as Attr

from ai_act_copilot.agent.graph import run_agent
from ai_act_copilot.agent.guardrails import Budget
from ai_act_copilot.agent.nodes import AgentDeps, RouteDecision
from ai_act_copilot.agent.state import Route
from ai_act_copilot.config import Settings
from ai_act_copilot.embeddings.indexer import build_index
from ai_act_copilot.generation.answer import AnswerSchema, answer_question
from ai_act_copilot.llm.anthropic_client import AnthropicLLM
from ai_act_copilot.llm.base import LLMError
from ai_act_copilot.models import (
    Chunk,
    ChunkStrategy,
    Document,
    Language,
    Paragraph,
    Provision,
    ProvisionKind,
)
from ai_act_copilot.observability.tracing import trace_context
from ai_act_copilot.retrieval.hybrid import HybridRetriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore
from tests.conftest import REPO_ROOT, ExportedTraces
from tests.doubles import FakeAnthropic, FakeEmbedder, anthropic_response, anthropic_usage

PROMPTS = REPO_ROOT / "prompts"

PASSAGES = [
    ("ai_act:art:5", "5", "Prohibited practices include social scoring by public authorities."),
    ("ai_act:art:6", "6", "An AI system referred to in Annex III shall be considered high-risk."),
]


def _build_corpus(settings: Settings) -> None:
    with CorpusStore(settings.database_path) as store:
        provisions = tuple(
            Provision(
                provision_id=provision_id,
                document_id="ai_act:en",
                source_id="ai_act",
                language=Language.EN,
                kind=ProvisionKind.ARTICLE,
                number=number,
                title=f"Article {number}",
                breadcrumb=("AI Act", f"Article {number}"),
                paragraphs=(Paragraph(label="1", text=text),),
                order=order,
            )
            for order, (provision_id, number, text) in enumerate(PASSAGES)
        )
        chunks = [
            Chunk(
                chunk_id=f"ai_act:en:structural:{order:04d}",
                document_id="ai_act:en",
                source_id="ai_act",
                language=Language.EN,
                strategy=ChunkStrategy.STRUCTURAL,
                order=order,
                header=f"AI Act > Article {number}",
                text=text,
                token_count=len(text.split()),
                provision_ids=(provision_id,),
            )
            for order, (provision_id, number, text) in enumerate(PASSAGES)
        ]
        store.replace_document(
            Document(
                document_id="ai_act:en",
                source_id="ai_act",
                language=Language.EN,
                title="AI Act",
                url="http://example.invalid",
                sha256="0" * 64,
                fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
                provisions=provisions,
            ),
            chunks,
        )


@pytest.fixture
def retriever(tmp_path: Path) -> Iterator[HybridRetriever]:
    """The real hybrid retriever over a two-article corpus, embedded by a fake."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(data_dir=data_dir, retrieval_top_k=2, retrieval_candidates=10)
    _build_corpus(settings)
    embedder = FakeEmbedder()
    build_index(settings, embedder=embedder)
    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        yield HybridRetriever(store, vectors, embedder, settings=settings)


def _llm(fake: FakeAnthropic) -> AnthropicLLM:
    return AnthropicLLM(client=cast(anthropic.Anthropic, fake))


def _answered(text: str = "Article 5 lists them.", citations: list[str] | None = None) -> Any:
    """A structured-output response, as ``messages.parse`` returns it."""
    return anthropic_response(
        parsed_output=AnswerSchema(
            answer=text, citations=citations or ["ai_act:art:5"], abstained=False
        )
    )


def _routed(route: Route) -> Any:
    return anthropic_response(parsed_output=RouteDecision(route=route, reason="scripted"))


def _tool_use(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> SimpleNamespace:
    """A tool_use content block that survives ``_as_dicts`` like the real SDK's does."""
    block = {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
    return SimpleNamespace(type="tool_use", model_dump=lambda **_: dict(block))


def _asks_for(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> Any:
    return anthropic_response(
        stop_reason="tool_use",
        content=[_tool_use(name, arguments, call_id)],
        usage=anthropic_usage(input_tokens=100, output_tokens=20),
    )


# --- the single-retrieval path -------------------------------------------------------


def _ask(retriever: HybridRetriever, fake: FakeAnthropic, question: str) -> None:
    with trace_context(name="ask", tags=["cli", "ask"]):
        answer_question(question, retriever=retriever, llm=_llm(fake), prompts_dir=PROMPTS)


def test_ask_produces_one_readable_tree(retriever: HybridRetriever, traces: ExportedTraces) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "Which practices are prohibited?")

    assert traces.tree("answer-question") == [
        ("answer-question", None),
        ("generate-answer", "answer-question"),
        ("retrieve-passages", "answer-question"),
    ]


def test_each_step_declares_what_kind_of_step_it_is(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "Which practices are prohibited?")

    assert traces.observation_type("answer-question") == "chain"
    assert traces.observation_type("retrieve-passages") == "retriever"
    assert traces.observation_type("generate-answer") == "generation"


def test_the_trace_shows_the_question_and_the_answer(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(
        retriever,
        FakeAnthropic(response=_answered("Article 5 lists them.")),
        "Which practices are prohibited?",
    )

    root = traces.named("answer-question")
    assert root.attributes is not None
    assert "Which practices are prohibited?" in str(root.attributes[Attr.OBSERVATION_INPUT])
    assert "Article 5 lists them." in str(root.attributes[Attr.OBSERVATION_OUTPUT])
    assert root.attributes[Attr.TRACE_NAME] == "ask"
    assert root.attributes[Attr.TRACE_TAGS] == ("cli", "ask")


def test_generations_carry_model_usage_and_cost(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "Which practices are prohibited?")

    generation = traces.named("generate-answer")
    assert generation.attributes is not None
    assert generation.attributes[Attr.OBSERVATION_MODEL] == "claude-opus-5"
    usage = str(generation.attributes[Attr.OBSERVATION_USAGE_DETAILS])
    assert '"input": 1000' in usage
    assert '"cache_read_input_tokens": 500' in usage  # cache hits are worth 90% off
    assert Attr.OBSERVATION_COST_DETAILS in generation.attributes
    assert "effort" in str(generation.attributes[Attr.OBSERVATION_MODEL_PARAMETERS])


def test_a_generation_records_the_conversation_the_model_saw(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "Which practices are prohibited?")

    recorded = str(traces.named("generate-answer").attributes[Attr.OBSERVATION_INPUT])  # type: ignore[index]
    assert '"role": "system"' in recorded  # the prompt, not only the user turn
    assert "Which practices are prohibited?" in recorded
    assert "social scoring" in recorded  # the passages it was told to answer from


def test_retrieval_says_which_signal_found_each_passage(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "Which practices are prohibited?")

    retrieval = traces.named("retrieve-passages")
    assert retrieval.attributes is not None
    assert "Which practices are prohibited?" in str(retrieval.attributes[Attr.OBSERVATION_INPUT])
    output = str(retrieval.attributes[Attr.OBSERVATION_OUTPUT])
    assert "ai_act:art:5" in output
    assert "dense" in output
    assert traces.metadata("retrieve-passages", "embedding_model") == "fake-embed"
    assert traces.metadata("retrieve-passages", "language") == "en"


# --- the agent path ------------------------------------------------------------------


def _agent_deps(retriever: HybridRetriever, fake: FakeAnthropic) -> AgentDeps:
    return AgentDeps(
        llm=_llm(fake),
        retriever=retriever,
        store=retriever.store,
        budget=Budget(max_steps=4, max_cost_usd=1.0),
        prompts_dir=PROMPTS,
    )


def _tool_loop() -> FakeAnthropic:
    """Route to the agent, search once, then submit an answer."""
    return FakeAnthropic(
        responses=[
            _routed(Route.COMPLEX),
            _asks_for("search_regulations", {"query": "prohibited practices"}),
            _asks_for(
                "submit_answer",
                {"answer": "Social scoring is prohibited.", "citations": ["ai_act:art:5"]},
                "call_2",
            ),
        ]
    )


def test_an_agent_run_reads_as_the_graph_it_walked(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    run_agent("Is social scoring allowed?", _agent_deps(retriever, _tool_loop()))

    # Siblings are listed by name, so this asserts the shape of the tree, not the order
    # the steps ran in: two turns of the loop, each with its model call, its tools, and
    # the retrieval those tools performed nested underneath them.
    assert traces.tree("agent-run") == [
        ("agent-run", None),
        ("agent-step", "agent-run"),
        ("agent-turn", "agent-step"),
        ("agent-step", "agent-run"),
        ("agent-turn", "agent-step"),
        ("route-question", "agent-run"),
        ("route-question", "route-question"),  # the model call inside the routing step
        ("run-tools", "agent-run"),
        ("search_regulations", "run-tools"),
        ("retrieve-passages", "search_regulations"),
        ("run-tools", "agent-run"),
        ("submit_answer", "run-tools"),
        ("verify-citations", "agent-run"),
    ]


def test_the_run_and_its_steps_are_typed(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    run_agent("Is social scoring allowed?", _agent_deps(retriever, _tool_loop()))

    assert traces.observation_type("agent-run") == "agent"
    assert traces.observation_type("search_regulations") == "tool"
    assert traces.observation_type("retrieve-passages") == "retriever"
    assert traces.observation_type("agent-turn") == "generation"
    # Not a guardrail in the Langfuse sense: it judges whether the output is supported.
    assert traces.observation_type("verify-citations") == "evaluator"


def test_a_tool_call_shows_its_arguments_and_what_came_back(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    run_agent("Is social scoring allowed?", _agent_deps(retriever, _tool_loop()))

    tool = traces.named("search_regulations")
    assert tool.attributes is not None
    assert "prohibited practices" in str(tool.attributes[Attr.OBSERVATION_INPUT])
    assert "social scoring" in str(tool.attributes[Attr.OBSERVATION_OUTPUT])
    assert traces.metadata("search_regulations", "failed") is False


def test_the_run_records_the_route_it_chose_and_what_it_read(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    run_agent("Is social scoring allowed?", _agent_deps(retriever, _tool_loop()))

    root = traces.named("agent-run")
    assert root.attributes is not None
    assert "Is social scoring allowed?" in str(root.attributes[Attr.OBSERVATION_INPUT])
    assert "Social scoring is prohibited." in str(root.attributes[Attr.OBSERVATION_OUTPUT])
    assert traces.metadata("agent-run", "route") == "complex"
    assert traces.metadata("agent-run", "steps") == 2
    assert "ai_act:art:5" in str(traces.metadata("agent-run", "provisions_seen"))


def test_the_turns_of_a_conversation_share_a_session(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    first = run_agent("Is social scoring allowed?", _agent_deps(retriever, _tool_loop()))
    run_agent(
        "And for a private company?",
        _agent_deps(retriever, _tool_loop()),
        thread_id=first.thread_id,
    )

    sessions = {
        span.attributes.get(Attr.TRACE_SESSION_ID) for span in traces.spans() if span.attributes
    }
    assert sessions == {first.thread_id}
    # Two questions, two traces - grouped by the session, not merged into one.
    assert len({span.context.trace_id for span in traces.spans() if span.context}) == 2


# --- what must never leave the machine -----------------------------------------------


def test_personal_data_in_a_question_never_reaches_the_exporter(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    _ask(
        retriever,
        FakeAnthropic(response=_answered()),
        "Can I keep jean.dupont@acme.fr and 06 12 34 56 78 in a training set?",
    )

    exported = traces.all_attribute_text()
    assert "jean.dupont@acme.fr" not in exported
    assert "06 12 34 56 78" not in exported
    assert "[EMAIL]" in exported
    assert "[PHONE]" in exported
    # The question is still recognisable: redaction must not empty the trace.
    assert "training set" in exported


def test_the_corpus_survives_redaction(retriever: HybridRetriever, traces: ExportedTraces) -> None:
    _ask(retriever, FakeAnthropic(response=_answered()), "What does Annex III cover?")

    exported = traces.all_attribute_text()
    assert "ai_act:art:5" in exported
    assert "Annex III" in exported


# --- failures ------------------------------------------------------------------------


def test_a_failed_call_still_says_what_it_was_and_what_it_sent(
    retriever: HybridRetriever, traces: ExportedTraces
) -> None:
    """The observations worth finding are the broken ones; they must not be anonymous."""
    # The Anthropic SDK vendors its own httpx fork, so the response must come from it.
    refused = anthropic.APIStatusError(
        "credit balance is too low",
        response=httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com")),
        body=None,
    )
    with pytest.raises(LLMError):
        _ask(retriever, FakeAnthropic(error=refused), "Which practices are prohibited?")

    generation = traces.named("generate-answer")  # not the placeholder name
    assert generation.attributes is not None
    assert "Which practices are prohibited?" in str(generation.attributes[Attr.OBSERVATION_INPUT])
    assert generation.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
    assert "credit balance" in str(generation.attributes[Attr.OBSERVATION_STATUS_MESSAGE])
    # The failure surfaces on the step that owns the answer, not only on the model call.
    assert traces.attribute("answer-question", Attr.OBSERVATION_LEVEL) == "ERROR"
