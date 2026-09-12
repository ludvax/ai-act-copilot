from pathlib import Path

import pytest

from ai_act_copilot.evaluation.dataset import Category, load_cases
from ai_act_copilot.evaluation.retrieval_metrics import CaseOutcome, score
from ai_act_copilot.models import Language
from tests.conftest import REPO_ROOT

DATASET = REPO_ROOT / "evals" / "datasets" / "retrieval_mini.jsonl"


def test_loads_the_shipped_mini_dataset() -> None:
    cases = load_cases(DATASET)

    assert len(cases) == 15
    assert {case.language for case in cases} == {Language.EN, Language.FR}
    assert all(case.expected_provisions for case in cases)
    assert any(case.category is Category.CITATION for case in cases)


def test_loads_a_dataset_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "c1", "question": "Q?", "language": "en", '
        '"expected_provisions": ["ai_act:art:5"], "category": "lookup"}\n\n',
        encoding="utf-8",
    )

    (case,) = load_cases(path)

    assert case.id == "c1"
    assert case.expected_provisions == ("ai_act:art:5",)


def _outcome(case_id: str, ranks: list[list[str]], expected: list[str]) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        expected=tuple(expected),
        retrieved=tuple(tuple(rank) for rank in ranks),
    )


def test_perfect_retrieval_scores_one() -> None:
    outcomes = [_outcome("a", [["ai_act:art:5"], ["ai_act:art:6"]], ["ai_act:art:5"])]

    scores = score(outcomes, k=5)

    assert scores.hit_rate == 1.0
    assert scores.recall == 1.0
    assert scores.mrr == 1.0
    assert scores.ndcg == pytest.approx(1.0)
    assert scores.misses == ()


def test_rank_two_halves_the_reciprocal_rank() -> None:
    outcomes = [_outcome("a", [["gdpr:art:6"], ["ai_act:art:5"]], ["ai_act:art:5"])]

    scores = score(outcomes, k=5)

    assert scores.hit_rate == 1.0
    assert scores.mrr == pytest.approx(0.5)
    assert scores.ndcg < 1.0


def test_a_hit_beyond_k_counts_as_a_miss() -> None:
    outcomes = [_outcome("a", [["x"], ["y"], ["ai_act:art:5"]], ["ai_act:art:5"])]

    scores = score(outcomes, k=2)

    assert scores.hit_rate == 0.0
    assert scores.misses == ("a",)


def test_partial_recall_when_several_provisions_are_expected() -> None:
    outcomes = [_outcome("a", [["ai_act:art:5"]], ["ai_act:art:5", "ai_act:anx:III"])]

    scores = score(outcomes, k=5)

    assert scores.hit_rate == 1.0
    assert scores.recall == pytest.approx(0.5)


def test_averages_across_cases() -> None:
    outcomes = [
        _outcome("a", [["ai_act:art:5"]], ["ai_act:art:5"]),
        _outcome("b", [["x"]], ["ai_act:art:6"]),
    ]

    scores = score(outcomes, k=5)

    assert scores.cases == 2
    assert scores.hit_rate == pytest.approx(0.5)
    assert scores.misses == ("b",)


def test_empty_dataset_scores_zero() -> None:
    scores = score([], k=5)

    assert (scores.cases, scores.hit_rate, scores.mrr) == (0, 0.0, 0.0)
