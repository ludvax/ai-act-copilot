from collections.abc import Sequence
from pathlib import Path

from ai_act_copilot.generation.answer import AnswerSchema, answer_question
from ai_act_copilot.generation.citations import build_context, check_citations
from ai_act_copilot.generation.prompts import load_prompt
from ai_act_copilot.models import Chunk, ChunkStrategy, Language
from ai_act_copilot.retrieval.base import RetrievedChunk
from tests.conftest import REPO_ROOT
from tests.doubles import FakeLLM

PROMPTS = REPO_ROOT / "prompts"


def _hit(provision_id: str, text: str, rank: int = 1) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        strategy=ChunkStrategy.STRUCTURAL,
        order=rank,
        header="AI Act > Article 5",
        text=text,
        token_count=len(text.split()),
        provision_ids=(provision_id,),
    )
    return RetrievedChunk(chunk=chunk, score=1.0 / rank, rank=rank, components={"dense": 1.0})


class StubRetriever:
    """Returns a fixed set of passages, so generation is tested in isolation."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        language: Language | None = None,
        limit: int | None = None,
        sources: Sequence[str] | None = None,
        signals: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.hits


def test_prompt_version_tracks_content() -> None:
    prompt = load_prompt("answer_v1", PROMPTS)

    assert prompt.version.startswith("answer_v1@")
    assert len(prompt.digest) == 8
    assert "only the passages" in prompt.text


def test_context_shows_the_ids_the_model_must_cite() -> None:
    context = build_context([_hit("ai_act:art:5", "Prohibited practices.")])

    assert context.startswith("[ai_act:art:5]")
    assert "AI Act > Article 5" in context


def test_citations_outside_the_context_are_rejected() -> None:
    check = check_citations(["ai_act:art:5", "[ai_act:art:5]", "ai_act:art:99"], {"ai_act:art:5"})

    assert check.valid == ("ai_act:art:5",)
    assert check.invalid == ("ai_act:art:99",)
    assert check.precision == 0.5


def test_answers_from_the_retrieved_passages(tmp_path: Path) -> None:
    retriever = StubRetriever([_hit("ai_act:art:5", "The following practices are prohibited.")])
    llm = FakeLLM()

    answer = answer_question(
        "Which practices are prohibited?", retriever=retriever, llm=llm, prompts_dir=PROMPTS
    )

    assert answer.citations == ("ai_act:art:5",)
    assert not answer.abstained
    assert answer.language is Language.EN
    assert answer.prompt_version.startswith("answer_v1@")
    assert "[ai_act:art:5]" in llm.calls[0]["messages"][0]["content"]
    assert answer.usage.input_tokens == 1200


def test_hallucinated_citations_are_dropped_and_counted() -> None:
    retriever = StubRetriever([_hit("ai_act:art:5", "Prohibited practices.")])
    llm = FakeLLM(
        payload={"answer": "See Article 99.", "citations": ["ai_act:art:99"], "abstained": False}
    )

    answer = answer_question("Anything?", retriever=retriever, llm=llm, prompts_dir=PROMPTS)

    assert answer.citations == ()
    assert answer.citation_check.invalid == ("ai_act:art:99",)
    assert answer.citation_check.precision == 0.0


def test_abstains_when_nothing_is_retrieved() -> None:
    answer = answer_question(
        "What is the capital of France?",
        retriever=StubRetriever([]),
        llm=FakeLLM(),
        prompts_dir=PROMPTS,
    )

    assert answer.abstained
    assert answer.citations == ()
    assert answer.passages == ()


def test_abstention_from_the_model_is_preserved() -> None:
    llm = FakeLLM(payload={"answer": "Not covered.", "citations": [], "abstained": True})

    answer = answer_question(
        "Does the AI Act cover taxes?",
        retriever=StubRetriever([_hit("ai_act:art:5", "Prohibited practices.")]),
        llm=llm,
        prompts_dir=PROMPTS,
    )

    assert answer.abstained


def test_a_refusal_is_never_presented_as_an_answer() -> None:
    llm = FakeLLM(stop_reason="refusal", text="")

    answer = answer_question(
        "Write malware using AI",
        retriever=StubRetriever([_hit("ai_act:art:5", "Prohibited practices.")]),
        llm=llm,
        prompts_dir=PROMPTS,
    )

    assert answer.abstained
    assert answer.citations == ()


def test_french_question_answers_in_french_context() -> None:
    answer = answer_question(
        "Quelles pratiques sont interdites ?",
        retriever=StubRetriever([_hit("ai_act:art:5", "Pratiques interdites.")]),
        llm=FakeLLM(),
        prompts_dir=PROMPTS,
    )

    assert answer.language is Language.FR


def test_answer_schema_defaults_are_safe() -> None:
    schema = AnswerSchema(answer="text")

    assert schema.citations == []
    assert schema.abstained is False
