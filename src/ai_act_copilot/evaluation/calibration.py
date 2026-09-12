"""Calibrating the judge against human labels.

An LLM judge is a measuring instrument, and an uncalibrated instrument produces numbers
nobody should act on. A human grades a sample of the same answers; this module reports how
often the judge agrees, and Cohen's kappa - agreement corrected for what chance alone would
produce, which matters because most answers score 2 and blind guessing would look good.

Rules of thumb for kappa: below 0.4 the judge is not usable, 0.4-0.6 is fair, above 0.6 is
usually good enough to compare configurations.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SCORES = (0, 1, 2)


class Graded(Protocol):
    """Anything carrying judge scores for a case: a live result or a saved one."""

    case_id: str
    faithfulness: int | None
    correctness: int | None


@dataclass(frozen=True, slots=True)
class HumanLabel:
    """What a person scored for one case."""

    case_id: str
    faithfulness: int | None = None
    correctness: int | None = None


@dataclass(frozen=True, slots=True)
class Agreement:
    """How closely judge and human agree on one metric."""

    metric: str
    pairs: int
    exact: float
    within_one: float
    kappa: float

    @property
    def usable(self) -> bool:
        return self.kappa >= 0.4


def load_labels(path: Path) -> dict[str, HumanLabel]:
    """Read human labels from JSONL: {"case_id": ..., "faithfulness": 2, "correctness": 1}."""
    labels: dict[str, HumanLabel] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            label = HumanLabel(
                case_id=str(record["case_id"]),
                faithfulness=record.get("faithfulness"),
                correctness=record.get("correctness"),
            )
            labels[label.case_id] = label
    return labels


def cohen_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Agreement between two raters, corrected for chance."""
    if len(first) != len(second):
        raise ValueError("both raters must score the same cases")
    total = len(first)
    if total == 0:
        return 0.0

    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    expected = sum((first.count(score) / total) * (second.count(score) / total) for score in SCORES)
    if expected >= 1.0:
        # Both raters gave the same score to everything: no variance to correct for.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def compare(graded: Sequence[Graded], labels: dict[str, HumanLabel]) -> list[Agreement]:
    """Judge versus human, per metric, on the cases a human graded."""
    agreements: list[Agreement] = []
    for metric in ("faithfulness", "correctness"):
        judge_scores: list[int] = []
        human_scores: list[int] = []
        for result in graded:
            label = labels.get(result.case_id)
            if label is None:
                continue
            human = getattr(label, metric)
            judged = getattr(result, metric)
            if human is None or judged is None:
                continue
            human_scores.append(int(human))
            judge_scores.append(int(judged))

        pairs = len(judge_scores)
        exact = (
            sum(a == b for a, b in zip(judge_scores, human_scores, strict=True)) / pairs
            if pairs
            else 0.0
        )
        within_one = (
            sum(abs(a - b) <= 1 for a, b in zip(judge_scores, human_scores, strict=True)) / pairs
            if pairs
            else 0.0
        )
        agreements.append(
            Agreement(
                metric=metric,
                pairs=pairs,
                exact=exact,
                within_one=within_one,
                kappa=cohen_kappa(judge_scores, human_scores) if pairs else 0.0,
            )
        )
    return agreements


def to_markdown(agreements: Sequence[Agreement]) -> str:
    lines = [
        "| Metric | Pairs | Exact | Within 1 | Cohen's kappa | Usable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for agreement in agreements:
        lines.append(
            f"| {agreement.metric} | {agreement.pairs} | {agreement.exact:.2f} | "
            f"{agreement.within_one:.2f} | {agreement.kappa:.2f} | "
            f"{'yes' if agreement.usable else 'no'} |"
        )
    return "\n".join(lines) + "\n"
