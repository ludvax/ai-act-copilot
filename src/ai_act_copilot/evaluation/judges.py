"""Judging answers: two LLM judges, and the checks that need no model at all.

The deterministic checks come first on purpose. Whether the answer cited provisions it
actually retrieved, and whether it abstained when it should have, are facts - computing
them with a model would be slower, dearer and less reliable. The judges are reserved for
what genuinely needs reading: grounding and substantive correctness.

Both judges score 0-2 and are calibrated against human labels (see calibration.py); an
uncalibrated judge is an opinion, not a metric.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ai_act_copilot.generation.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from ai_act_copilot.llm.base import LLMClient
from ai_act_copilot.observability.tracing import observe

FAITHFULNESS_PROMPT = "judge_faithfulness_v1"
CORRECTNESS_PROMPT = "judge_correctness_v1"
MAX_SCORE = 2


class FaithfulnessVerdict(BaseModel):
    """Is every claim backed by the passages the answer was given?"""

    score: int = Field(ge=0, le=MAX_SCORE, description="2 grounded, 1 partly, 0 unsupported.")
    unsupported: list[str] = Field(
        default_factory=list, description="Short verbatim quotes of unsupported claims."
    )
    reason: str = Field(default="", description="One sentence.")


class CorrectnessVerdict(BaseModel):
    """Does the answer say what the reference says?"""

    score: int = Field(ge=0, le=MAX_SCORE, description="2 correct, 1 partial, 0 wrong.")
    reason: str = Field(default="", description="One sentence.")


@dataclass(frozen=True, slots=True)
class DeterministicChecks:
    """Facts about an answer, computed without a model."""

    citation_precision: float
    citation_recall: float
    abstention_correct: bool
    route_correct: bool | None


def citation_scores(cited: Sequence[str], expected: Sequence[str]) -> tuple[float, float]:
    """Precision and recall of the citations against the labelled provisions."""
    cited_set, expected_set = set(cited), set(expected)
    if not expected_set:
        # Nothing to cite: citing nothing is perfect, citing anything is not.
        return (1.0 if not cited_set else 0.0), 1.0
    if not cited_set:
        return 0.0, 0.0
    hits = len(cited_set & expected_set)
    return hits / len(cited_set), hits / len(expected_set)


def check(
    *,
    cited: Sequence[str],
    expected: Sequence[str],
    abstained: bool,
    expect_abstention: bool,
    route: str | None = None,
    expected_route: str | None = None,
) -> DeterministicChecks:
    precision, recall = citation_scores(cited, expected)
    return DeterministicChecks(
        citation_precision=precision,
        citation_recall=recall,
        abstention_correct=abstained == expect_abstention,
        route_correct=None if expected_route is None else route == expected_route,
    )


class LLMJudge:
    """Scores grounding and correctness with a model."""

    def __init__(self, llm: LLMClient, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> None:
        self.llm = llm
        self.prompts_dir = prompts_dir

    @observe(name="judge-faithfulness", capture_input=False, capture_output=False)
    def faithfulness(self, question: str, answer: str, context: str) -> FaithfulnessVerdict:
        prompt = load_prompt(FAITHFULNESS_PROMPT, self.prompts_dir)
        result = self.llm.complete(
            system=prompt.text,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nPassages:\n{context}\n\nAnswer:\n{answer}"
                    ),
                }
            ],
            max_tokens=1024,
            output_format=FaithfulnessVerdict,
        )
        parsed = result.parsed
        return parsed if isinstance(parsed, FaithfulnessVerdict) else FaithfulnessVerdict(score=0)

    @observe(name="judge-correctness", capture_input=False, capture_output=False)
    def correctness(self, question: str, answer: str, reference: str) -> CorrectnessVerdict:
        prompt = load_prompt(CORRECTNESS_PROMPT, self.prompts_dir)
        result = self.llm.complete(
            system=prompt.text,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
                        f"Answer to grade:\n{answer}"
                    ),
                }
            ],
            max_tokens=1024,
            output_format=CorrectnessVerdict,
        )
        parsed = result.parsed
        return parsed if isinstance(parsed, CorrectnessVerdict) else CorrectnessVerdict(score=0)
