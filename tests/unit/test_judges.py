from typing import Any

import pytest

from ai_act_copilot.evaluation.calibration import (
    Agreement,
    HumanLabel,
    cohen_kappa,
    compare,
    load_labels,
    to_markdown,
)
from ai_act_copilot.evaluation.judges import (
    CorrectnessVerdict,
    FaithfulnessVerdict,
    LLMJudge,
    check,
    citation_scores,
)
from ai_act_copilot.llm.base import LLMResult
from ai_act_copilot.llm.pricing import Usage
from tests.conftest import REPO_ROOT
from tests.doubles import ScriptedLLM, decides

PROMPTS = REPO_ROOT / "prompts"


def test_citation_precision_and_recall() -> None:
    precision, recall = citation_scores(
        ["ai_act:art:5", "ai_act:art:99"], ["ai_act:art:5", "ai_act:anx:III"]
    )

    assert precision == 0.5  # one of two citations is expected
    assert recall == 0.5  # one of two expected provisions is cited


def test_citing_nothing_when_nothing_is_expected_is_perfect() -> None:
    assert citation_scores([], []) == (1.0, 1.0)


def test_citing_something_when_nothing_is_expected_is_wrong() -> None:
    precision, _ = citation_scores(["ai_act:art:5"], [])

    assert precision == 0.0


def test_missing_every_citation_scores_zero() -> None:
    assert citation_scores([], ["ai_act:art:5"]) == (0.0, 0.0)


def test_abstention_and_route_checks() -> None:
    checks = check(
        cited=[],
        expected=[],
        abstained=True,
        expect_abstention=True,
        route="out_of_scope",
        expected_route="out_of_scope",
    )

    assert checks.abstention_correct
    assert checks.route_correct


def test_route_is_not_scored_when_the_case_does_not_label_one() -> None:
    checks = check(cited=[], expected=[], abstained=False, expect_abstention=False)

    assert checks.route_correct is None


def test_faithfulness_judge_returns_its_verdict() -> None:
    llm = ScriptedLLM(decides(score=1, unsupported=["Article 99 applies"], reason="extra claim"))

    verdict = LLMJudge(llm, PROMPTS).faithfulness("q", "answer", "context")

    assert verdict.score == 1
    assert verdict.unsupported == ["Article 99 applies"]
    assert "Passages:" in str(llm.calls[0]["messages"][0]["content"])


def test_correctness_judge_sees_the_reference_but_not_the_passages() -> None:
    llm = ScriptedLLM(decides(score=2, reason="matches"))

    verdict = LLMJudge(llm, PROMPTS).correctness("q", "answer", "reference answer")

    content = str(llm.calls[0]["messages"][0]["content"])
    assert verdict.score == 2
    assert "reference answer" in content
    assert "Passages:" not in content


class SilentLLM:
    """A model that answers without producing the structured verdict."""

    model = "silent"

    def complete(self, **kwargs: Any) -> LLMResult:
        return LLMResult(
            text="",
            model=self.model,
            stop_reason="end_turn",
            usage=Usage(),
            cost_usd=0.0,
            content=[],
            parsed=None,
        )


def test_a_judge_that_returns_nothing_usable_scores_zero() -> None:
    judge = LLMJudge(SilentLLM(), PROMPTS)

    assert judge.faithfulness("q", "a", "c").score == 0
    assert judge.correctness("q", "a", "r").score == 0


def test_verdict_scores_are_bounded() -> None:
    with pytest.raises(ValueError, match="less than or equal to 2"):
        FaithfulnessVerdict(score=3)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        CorrectnessVerdict(score=-1)


def test_kappa_is_one_for_perfect_agreement() -> None:
    assert cohen_kappa([2, 2, 1, 0], [2, 2, 1, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_agreement() -> None:
    # Half agreement, but chance alone would also produce half: no real signal.
    assert cohen_kappa([2, 1, 2, 1], [2, 1, 1, 2]) == pytest.approx(0.0)


def test_kappa_of_total_disagreement() -> None:
    assert cohen_kappa([2, 2, 2, 2], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same cases"):
        cohen_kappa([1, 2], [1])


class _Graded:
    def __init__(self, case_id: str, faithfulness: int | None, correctness: int | None) -> None:
        self.case_id = case_id
        self.faithfulness = faithfulness
        self.correctness = correctness


def test_compare_reports_agreement_per_metric() -> None:
    graded = [_Graded("g01", 2, 2), _Graded("g02", 2, 1), _Graded("g03", 1, 0)]
    labels = {
        "g01": HumanLabel("g01", faithfulness=2, correctness=2),
        "g02": HumanLabel("g02", faithfulness=2, correctness=2),
        "g03": HumanLabel("g03", faithfulness=1, correctness=0),
    }

    agreements = compare(graded, labels)

    faithfulness = next(a for a in agreements if a.metric == "faithfulness")
    correctness = next(a for a in agreements if a.metric == "correctness")
    assert faithfulness.pairs == 3
    assert faithfulness.exact == pytest.approx(1.0)
    assert correctness.exact == pytest.approx(2 / 3)
    assert correctness.within_one == pytest.approx(1.0)


def test_labels_load_from_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "labels.jsonl"
    path.write_text('{"case_id": "g01", "faithfulness": 2, "correctness": 1}\n\n', encoding="utf-8")

    labels = load_labels(path)

    assert labels["g01"].correctness == 1


def test_agreement_table_flags_an_unusable_judge() -> None:
    table = to_markdown([Agreement("correctness", 20, 0.5, 0.9, 0.2)])

    assert "no" in table.splitlines()[-1]
