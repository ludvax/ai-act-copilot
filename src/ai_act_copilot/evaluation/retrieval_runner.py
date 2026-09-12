"""Measure retrieval configurations against a golden dataset.

The point is to make "hybrid is better than dense" a measurement rather than an opinion:
the same questions run through each combination of signals, and the numbers go in the
README. This runs entirely offline apart from embedding the queries.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ai_act_copilot.config import Settings
from ai_act_copilot.embeddings.base import Embedder
from ai_act_copilot.embeddings.indexer import make_embedder
from ai_act_copilot.evaluation.dataset import RetrievalCase
from ai_act_copilot.evaluation.retrieval_metrics import CaseOutcome, RetrievalScores, score
from ai_act_copilot.models import ChunkStrategy
from ai_act_copilot.observability.tracing import observe
from ai_act_copilot.retrieval.hybrid import HybridRetriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """One way of combining the signals."""

    signals: tuple[str, ...]
    weights: dict[str, float] | None = None  # None means equal weights


# The configurations worth comparing. Weights are always explicit here: passing None would
# fall back to the production defaults and quietly compare a configuration with itself.
CONFIGURATIONS: dict[str, RetrievalConfig] = {
    "bm25 only": RetrievalConfig(("bm25",), {"bm25": 1.0}),
    "dense only": RetrievalConfig(("dense",), {"dense": 1.0}),
    "dense + bm25 (equal)": RetrievalConfig(("dense", "bm25"), {"dense": 1.0, "bm25": 1.0}),
    "dense + bm25 (bm25 x0.4)": RetrievalConfig(("dense", "bm25"), {"dense": 1.0, "bm25": 0.4}),
    "dense + references": RetrievalConfig(("reference", "dense"), {"reference": 2.0, "dense": 1.0}),
    "all three (weighted)": RetrievalConfig(
        ("reference", "dense", "bm25"), {"reference": 2.0, "dense": 1.0, "bm25": 0.4}
    ),
}


@dataclass(frozen=True, slots=True)
class ConfigurationScore:
    label: str
    signals: tuple[str, ...]
    scores: RetrievalScores


@observe(name="eval-retrieval", capture_input=False, capture_output=False)
def compare_configurations(
    settings: Settings,
    cases: Sequence[RetrievalCase],
    *,
    k: int = 5,
    strategy: ChunkStrategy | None = None,
    embedder: Embedder | None = None,
    configurations: dict[str, RetrievalConfig] | None = None,
) -> list[ConfigurationScore]:
    """Run every configuration over the dataset and return their scores."""
    backend = embedder or make_embedder(settings)
    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        retriever = HybridRetriever(store, vectors, backend, settings=settings, strategy=strategy)
        results: list[ConfigurationScore] = []
        for label, configuration in (configurations or CONFIGURATIONS).items():
            outcomes = [_run_case(retriever, case, k=k, config=configuration) for case in cases]
            results.append(ConfigurationScore(label, configuration.signals, score(outcomes, k)))
            logger.info("%s: hit@%d=%.2f", label, k, results[-1].scores.hit_rate)
    return results


def _run_case(
    retriever: HybridRetriever,
    case: RetrievalCase,
    *,
    k: int,
    config: RetrievalConfig,
) -> CaseOutcome:
    hits = retriever.search(
        case.question,
        language=case.language,
        limit=k,
        signals=config.signals,
        weights=config.weights,
    )
    return CaseOutcome(
        case_id=case.id,
        expected=case.expected_provisions,
        retrieved=tuple(hit.provision_ids for hit in hits),
    )
