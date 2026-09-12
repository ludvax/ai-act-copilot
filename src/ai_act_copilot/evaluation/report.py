"""Reports: JSON for machines, Markdown for humans.

Every run writes both, timestamped, under evals/results/. The Markdown always states how
much of the dataset a human has reviewed and which model judged it - a score without that
context invites more confidence than it deserves.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_act_copilot.evaluation.answer_runner import CaseResult, EvalRun, RunSummary


def to_json(run: EvalRun) -> dict[str, Any]:
    """The full run, case by case."""
    summary = run.summary
    return {
        "summary": {
            "label": summary.label,
            "mode": summary.mode.value,
            "cases": summary.cases,
            "faithfulness": summary.faithfulness,
            "correctness": summary.correctness,
            "citation_precision": summary.citation_precision,
            "citation_recall": summary.citation_recall,
            "abstention_accuracy": summary.abstention_accuracy,
            "route_accuracy": summary.route_accuracy,
            "cost_usd": summary.cost_usd,
            "cost_per_case": summary.cost_per_case,
            "latency_median": summary.latency_median,
            "steps_mean": summary.steps_mean,
            "reviewed_rate": summary.reviewed_rate,
        },
        "cases": [_case_json(result) for result in run.results],
    }


def _case_json(result: CaseResult) -> dict[str, Any]:
    return {
        "id": result.case_id,
        "category": result.category.value,
        "language": result.language.value,
        "abstained": result.abstained,
        "citations": list(result.citations),
        "faithfulness": result.faithfulness,
        "correctness": result.correctness,
        "unsupported": list(result.unsupported),
        "citation_precision": result.checks.citation_precision,
        "citation_recall": result.checks.citation_recall,
        "abstention_correct": result.checks.abstention_correct,
        "route": result.route,
        "route_correct": result.checks.route_correct,
        "steps": result.steps,
        "cost_usd": result.cost_usd,
        "latency_seconds": result.latency_seconds,
        "answer": result.answer,
    }


def summary_row(summary: RunSummary) -> str:
    """One line of the comparison table."""
    return (
        f"| {summary.label} | {_score(summary.faithfulness)} | {_score(summary.correctness)} | "
        f"{summary.citation_precision:.2f} | {summary.citation_recall:.2f} | "
        f"{summary.abstention_accuracy:.2f} | {_score(summary.route_accuracy)} | "
        f"${summary.cost_per_case:.3f} | {summary.latency_median:.1f}s | "
        f"{summary.steps_mean:.1f} |"
    )


TABLE_HEADER = (
    "| Run | Faithfulness | Correctness | Cite prec. | Cite recall | Abstention | Route | "
    "$/case | Latency | Steps |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def to_markdown(run: EvalRun, *, judge_model: str = "unknown", dataset: str = "") -> str:
    """A report someone can read without opening the JSON."""
    summary = run.summary
    lines = [
        f"# Evaluation — {summary.label}",
        "",
        f"- Dataset: `{dataset}` ({summary.cases} cases, "
        f"{summary.reviewed_rate:.0%} human-reviewed)",
        f"- Mode: {summary.mode.value} · Judge: {judge_model}",
        f"- Run at {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        TABLE_HEADER,
        summary_row(summary),
        "",
        "Judge scores are 0-1 (from a 0-2 scale). Citation precision counts citations that "
        "point at a provision the run actually retrieved.",
        "",
        "## By category",
        "",
        "| Category | Cases | Correctness | Cite prec. | Abstention |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category in sorted({result.category for result in run.results}):
        rows = [result for result in run.results if result.category is category]
        lines.append(
            f"| {category.value} | {len(rows)} | "
            f"{_mean_score([row.correctness for row in rows])} | "
            f"{_mean([row.checks.citation_precision for row in rows]):.2f} | "
            f"{_mean([float(row.checks.abstention_correct) for row in rows]):.2f} |"
        )

    failures = [result for result in run.results if (result.correctness or 2) < 2]
    if failures:
        lines += ["", "## Cases to look at", ""]
        for result in failures:
            unsupported = (
                f" · unsupported: {'; '.join(result.unsupported)}" if result.unsupported else ""
            )
            lines.append(
                f"- **{result.case_id}** ({result.category.value}): correctness "
                f"{result.correctness}, citations {', '.join(result.citations) or 'none'}"
                f"{unsupported}"
            )
    return "\n".join(lines) + "\n"


def write_report(
    run: EvalRun, directory: Path, *, judge_model: str = "unknown", dataset: str = ""
) -> tuple[Path, Path]:
    """Write both files and return their paths."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = directory / f"{stamp}-{run.summary.label.replace(' ', '_')}"
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "run.json"
    json_path.write_text(json.dumps(to_json(run), indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = target / "report.md"
    markdown_path.write_text(
        to_markdown(run, judge_model=judge_model, dataset=dataset), encoding="utf-8"
    )
    return json_path, markdown_path


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_score(values: Sequence[int | None]) -> str:
    graded = [value for value in values if value is not None]
    return "-" if not graded else f"{sum(graded) / len(graded) / 2:.2f}"
