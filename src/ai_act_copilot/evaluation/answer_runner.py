"""Run the golden set end to end and score what comes out.

Both answer paths are measured the same way, so "is the agent worth it?" becomes a table
rather than an argument: quality, cost, latency and steps side by side.

Faithfulness needs the context the model actually saw. On the RAG path that is exactly the
retrieved passages; on the agent path the tools decide what it read, so the provisions the
run retrieved are reconstructed from the store - an approximation, and labelled as one.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ai_act_copilot.agent.graph import run_agent
from ai_act_copilot.agent.nodes import AgentDeps
from ai_act_copilot.evaluation.dataset import Category, GoldenCase
from ai_act_copilot.evaluation.judges import DeterministicChecks, LLMJudge, check
from ai_act_copilot.generation.answer import answer_question
from ai_act_copilot.generation.citations import build_context
from ai_act_copilot.generation.prompts import DEFAULT_PROMPTS_DIR
from ai_act_copilot.llm.base import LLMClient
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import observe
from ai_act_copilot.retrieval.base import Retriever
from ai_act_copilot.store.sqlite import CorpusStore

logger = logging.getLogger(__name__)


class Mode(StrEnum):
    RAG = "rag"  # single retrieval, one grounded answer
    AGENT = "agent"  # the LangGraph loop, with routing and tools


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One graded answer."""

    case_id: str
    category: Category
    language: Language
    answer: str
    citations: tuple[str, ...]
    abstained: bool
    checks: DeterministicChecks
    faithfulness: int | None = None
    correctness: int | None = None
    unsupported: tuple[str, ...] = ()
    route: str | None = None
    steps: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Averages over a run, in the units a reader cares about."""

    label: str
    mode: Mode
    cases: int
    faithfulness: float | None
    correctness: float | None
    citation_precision: float
    citation_recall: float
    abstention_accuracy: float
    route_accuracy: float | None
    cost_usd: float
    cost_per_case: float
    latency_median: float
    steps_mean: float
    reviewed_rate: float


@dataclass(frozen=True, slots=True)
class EvalRun:
    summary: RunSummary
    results: list[CaseResult] = field(default_factory=list)


@observe(name="eval-answers", capture_input=False, capture_output=False)
def run_answers(
    cases: Sequence[GoldenCase],
    *,
    mode: Mode,
    llm: LLMClient,
    retriever: Retriever,
    store: CorpusStore,
    judge: LLMJudge | None = None,
    agent_deps: AgentDeps | None = None,
    label: str | None = None,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> EvalRun:
    """Answer every case, then grade it."""
    results = [
        _run_case(
            case,
            mode=mode,
            llm=llm,
            retriever=retriever,
            store=store,
            judge=judge,
            agent_deps=agent_deps,
            prompts_dir=prompts_dir,
        )
        for case in cases
    ]
    reviewed = sum(case.reviewed for case in cases) / len(cases) if cases else 0.0
    summary = summarise(label or mode.value, mode, results, reviewed)
    logger.info("%s: %d cases, $%.3f", summary.label, summary.cases, summary.cost_usd)
    return EvalRun(summary=summary, results=results)


def _run_case(
    case: GoldenCase,
    *,
    mode: Mode,
    llm: LLMClient,
    retriever: Retriever,
    store: CorpusStore,
    judge: LLMJudge | None,
    agent_deps: AgentDeps | None,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> CaseResult:
    started = time.perf_counter()
    if mode is Mode.AGENT:
        if agent_deps is None:
            raise ValueError("agent mode needs agent_deps")
        answer = run_agent(case.question, agent_deps, language=case.language)
        text, citations, abstained = answer.text, tuple(answer.citations), answer.abstained
        route, steps, cost = str(answer.route), answer.steps, answer.cost_usd
        context = _context_from_provisions(store, answer.provisions_seen, case.language)
    else:
        grounded = answer_question(
            case.question,
            retriever=retriever,
            llm=llm,
            language=case.language,
            prompts_dir=prompts_dir,
        )
        text, citations, abstained = grounded.text, grounded.citations, grounded.abstained
        route, steps, cost = None, 1, grounded.cost_usd
        context = build_context(grounded.passages)
    latency = time.perf_counter() - started

    faithfulness: int | None = None
    correctness: int | None = None
    unsupported: tuple[str, ...] = ()
    if judge is not None:
        verdict = judge.faithfulness(case.question, text, context)
        faithfulness, unsupported = verdict.score, tuple(verdict.unsupported)
        correctness = judge.correctness(case.question, text, case.reference_answer).score

    return CaseResult(
        case_id=case.id,
        category=case.category,
        language=case.language,
        answer=text,
        citations=citations,
        abstained=abstained,
        checks=check(
            cited=citations,
            expected=case.expected_provisions,
            abstained=abstained,
            expect_abstention=case.expect_abstention,
            route=route,
            expected_route=str(case.expected_route) if case.expected_route else None,
        ),
        faithfulness=faithfulness,
        correctness=correctness,
        unsupported=unsupported,
        route=route,
        steps=steps,
        cost_usd=cost,
        latency_seconds=latency,
    )


def _context_from_provisions(
    store: CorpusStore, provision_ids: Sequence[str], language: Language
) -> str:
    blocks: list[str] = []
    for provision_id in provision_ids:
        provision = store.provision(provision_id, language)
        if provision is not None:
            blocks.append(f"[{provision.provision_id}] {provision.text}")
    return "\n\n".join(blocks)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(label: str, mode: Mode, results: Sequence[CaseResult], reviewed: float) -> RunSummary:
    """Aggregate case results; judge scores are reported on a 0-1 scale."""
    graded = [result for result in results if result.faithfulness is not None]
    routed = [result for result in results if result.checks.route_correct is not None]
    latencies = sorted(result.latency_seconds for result in results)
    total_cost = sum(result.cost_usd for result in results)
    return RunSummary(
        label=label,
        mode=mode,
        cases=len(results),
        faithfulness=mean([result.faithfulness or 0 for result in graded]) / 2 if graded else None,
        correctness=mean([result.correctness or 0 for result in graded]) / 2 if graded else None,
        citation_precision=mean([result.checks.citation_precision for result in results]),
        citation_recall=mean([result.checks.citation_recall for result in results]),
        abstention_accuracy=mean([float(result.checks.abstention_correct) for result in results]),
        route_accuracy=(
            mean([float(bool(result.checks.route_correct)) for result in routed])
            if routed
            else None
        ),
        cost_usd=total_cost,
        cost_per_case=total_cost / len(results) if results else 0.0,
        latency_median=latencies[len(latencies) // 2] if latencies else 0.0,
        steps_mean=mean([float(result.steps) for result in results]),
        reviewed_rate=reviewed,
    )
