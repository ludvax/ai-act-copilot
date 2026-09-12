from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ai_act_copilot.evaluation.answer_runner import Mode, run_answers
from ai_act_copilot.evaluation.dataset import Category, GoldenCase
from ai_act_copilot.evaluation.judges import LLMJudge
from ai_act_copilot.evaluation.report import to_markdown, write_report
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
from tests.doubles import ScriptedLLM, decides

PROMPTS = REPO_ROOT / "prompts"


def _hit(provision_id: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        document_id="ai_act:en",
        source_id="ai_act",
        language=Language.EN,
        strategy=ChunkStrategy.STRUCTURAL,
        order=0,
        header="AI Act > Article 5",
        text="The following AI practices are prohibited.",
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
        return [_hit("ai_act:art:5")]


def _store(tmp_path: Path) -> CorpusStore:
    store = CorpusStore(tmp_path / "corpus.db")
    store.replace_document(
        Document(
            document_id="ai_act:en",
            source_id="ai_act",
            language=Language.EN,
            title="AI Act",
            url="http://example.invalid",
            sha256="0" * 64,
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            provisions=(
                Provision(
                    provision_id="ai_act:art:5",
                    document_id="ai_act:en",
                    source_id="ai_act",
                    language=Language.EN,
                    kind=ProvisionKind.ARTICLE,
                    number="5",
                    title="Prohibited practices",
                    breadcrumb=("AI Act", "Article 5"),
                    paragraphs=(Paragraph(label="1", text="Prohibited."),),
                    order=0,
                ),
            ),
        ),
        [],
    )
    return store


CASES = (
    GoldenCase(
        id="g01",
        question="Which practices are prohibited?",
        language=Language.EN,
        category=Category.LOOKUP,
        expected_provisions=("ai_act:art:5",),
        reference_answer="Article 5 lists the prohibited practices.",
        reviewed=True,
    ),
    GoldenCase(
        id="g02",
        question="What is the capital of France?",
        language=Language.EN,
        category=Category.OUT_OF_SCOPE,
        reference_answer="Not covered by the corpus.",
        expect_abstention=True,
    ),
)


def _script() -> ScriptedLLM:
    return ScriptedLLM(
        decides(answer="Article 5 lists them.", citations=["ai_act:art:5"], abstained=False),
        decides(score=2, unsupported=[], reason="grounded"),
        decides(score=2, reason="matches"),
        decides(answer="Not covered.", citations=[], abstained=True),
        decides(score=2, unsupported=[], reason="grounded"),
        decides(score=2, reason="correct abstention"),
    )


def test_rag_run_scores_every_case(tmp_path: Path) -> None:
    llm = _script()
    store = _store(tmp_path)

    run = run_answers(
        CASES,
        mode=Mode.RAG,
        llm=llm,
        retriever=StubRetriever(),
        store=store,
        prompts_dir=PROMPTS,
        judge=LLMJudge(llm, PROMPTS),
        label="rag-test",
    )

    assert run.summary.cases == 2
    assert run.summary.faithfulness == 1.0  # 2/2 on the 0-1 scale
    assert run.summary.correctness == 1.0
    assert run.summary.citation_precision == 1.0
    assert run.summary.abstention_accuracy == 1.0
    assert run.summary.reviewed_rate == 0.5  # only g01 is reviewed
    assert run.summary.cost_usd > 0


def test_abstention_is_graded_against_the_label(tmp_path: Path) -> None:
    llm = _script()

    run = run_answers(
        CASES,
        mode=Mode.RAG,
        llm=llm,
        retriever=StubRetriever(),
        store=_store(tmp_path),
        prompts_dir=PROMPTS,
        judge=LLMJudge(llm, PROMPTS),
    )

    out_of_scope = next(result for result in run.results if result.case_id == "g02")
    assert out_of_scope.abstained
    assert out_of_scope.checks.abstention_correct
    assert out_of_scope.checks.citation_precision == 1.0  # citing nothing was right


def test_running_without_a_judge_leaves_scores_empty(tmp_path: Path) -> None:
    run = run_answers(
        CASES[:1],
        mode=Mode.RAG,
        llm=ScriptedLLM(decides(answer="Article 5.", citations=["ai_act:art:5"], abstained=False)),
        retriever=StubRetriever(),
        store=_store(tmp_path),
        prompts_dir=PROMPTS,
    )

    assert run.summary.faithfulness is None
    assert run.results[0].correctness is None
    assert run.summary.citation_precision == 1.0  # deterministic checks still run


def test_markdown_report_states_its_caveats(tmp_path: Path) -> None:
    llm = _script()
    run = run_answers(
        CASES,
        mode=Mode.RAG,
        llm=llm,
        retriever=StubRetriever(),
        store=_store(tmp_path),
        prompts_dir=PROMPTS,
        judge=LLMJudge(llm, PROMPTS),
        label="rag-test",
    )

    markdown = to_markdown(run, judge_model="claude-opus-5", dataset="golden_v1.jsonl")

    assert "50% human-reviewed" in markdown
    assert "claude-opus-5" in markdown
    assert "## By category" in markdown


def test_write_report_creates_both_files(tmp_path: Path) -> None:
    llm = _script()
    run = run_answers(
        CASES,
        mode=Mode.RAG,
        llm=llm,
        retriever=StubRetriever(),
        store=_store(tmp_path),
        prompts_dir=PROMPTS,
        judge=LLMJudge(llm, PROMPTS),
        label="rag-test",
    )

    json_path, markdown_path = write_report(run, tmp_path / "results", dataset="golden_v1.jsonl")

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert '"citation_precision"' in json_path.read_text(encoding="utf-8")
