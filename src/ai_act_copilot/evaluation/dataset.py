"""Golden datasets.

Cases are labelled with the *provisions* a correct answer must rest on, never with chunk
ids: re-chunking or switching strategy leaves the dataset valid, which is what makes two
configurations comparable.

A case can also carry a reference answer, whether abstention is the correct behaviour, and
which route the question should take - so routing, abstention and answer quality are all
measured from the same file.

``reviewed`` records whether a human has checked the case. Unreviewed cases still run, but
the report says how many of them there are: a golden set nobody has read is a liability,
not a benchmark.
"""

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ai_act_copilot.agent.state import Route
from ai_act_copilot.models import Language


class Category(StrEnum):
    LOOKUP = "lookup"  # one provision answers it
    CITATION = "citation"  # the question names the provision explicitly
    DEFINITION = "definition"
    CONCEPT = "concept"  # needs the right article without naming it
    TEMPORAL = "temporal"  # dates of application
    CROSS_TEXT = "cross-text"  # spans two regulations
    SCENARIO = "scenario"  # a described situation to classify
    OUT_OF_SCOPE = "out-of-scope"  # the corpus cannot answer; abstention is correct
    FALSE_PREMISE = "false-premise"  # the question asserts something untrue


class GoldenCase(BaseModel):
    """One labelled question."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    language: Language
    category: Category = Category.CONCEPT
    expected_provisions: tuple[str, ...] = ()
    reference_answer: str = ""
    expect_abstention: bool = False
    expected_route: Route | None = None
    reviewed: bool = False
    notes: str = ""


def load_cases(path: Path) -> tuple[GoldenCase, ...]:
    """Read a JSONL dataset."""
    return tuple(GoldenCase.model_validate(record) for record in _records(path))


def review_rate(cases: tuple[GoldenCase, ...]) -> float:
    """Share of the dataset a human has checked."""
    return sum(case.reviewed for case in cases) / len(cases) if cases else 0.0


def _records(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
