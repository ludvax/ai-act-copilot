"""Command-line interface, installed as ``aiact``."""

import json
import logging
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_act_copilot import __version__
from ai_act_copilot.agent.graph import open_checkpointer, run_agent
from ai_act_copilot.agent.guardrails import Budget
from ai_act_copilot.agent.nodes import AgentDeps
from ai_act_copilot.config import get_settings
from ai_act_copilot.embeddings.indexer import build_index, make_embedder
from ai_act_copilot.evaluation.answer_runner import Mode, run_answers
from ai_act_copilot.evaluation.calibration import compare, load_labels
from ai_act_copilot.evaluation.calibration import to_markdown as agreement_table
from ai_act_copilot.evaluation.dataset import load_cases, review_rate
from ai_act_copilot.evaluation.judges import LLMJudge
from ai_act_copilot.evaluation.report import TABLE_HEADER, summary_row, write_report
from ai_act_copilot.evaluation.retrieval_runner import compare_configurations
from ai_act_copilot.generation.answer import answer_question
from ai_act_copilot.ingestion.pipeline import ingest as run_ingest
from ai_act_copilot.llm.anthropic_client import AnthropicLLM
from ai_act_copilot.llm.base import LLMError
from ai_act_copilot.models import ChunkStrategy, Language
from ai_act_copilot.observability.tracing import TracingStatus, flush_tracing, init_tracing
from ai_act_copilot.retrieval.hybrid import HybridRetriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore


def _force_utf8_output() -> None:
    """Windows consoles default to a legacy code page and crash on corpus characters.

    The corpus is European legal text plus guidance PDFs: it contains typographic dashes,
    accents and the occasional emoji. Printing must never be what fails.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", "") or ""
        if encoding.lower().replace("-", "") != "utf8":
            with suppress(AttributeError, ValueError):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


_force_utf8_output()

app = typer.Typer(
    help="Bilingual assistant for EU AI regulation (AI Act, GDPR).",
    no_args_is_help=True,
)
# Emoji substitution off: provision ids contain :art:, which Rich would turn into an
# emoji - and then fail to encode it on a legacy Windows console.
console = Console(emoji=False)


def _print_version(value: bool) -> None:
    if value:
        console.print(f"ai-act-copilot {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Configure logging and tracing once for every command."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    ctx.obj = init_tracing(settings)
    ctx.call_on_close(flush_tracing)


@app.command()
def ingest(
    download: Annotated[
        bool, typer.Option("--download", help="Fetch missing or changed source files first.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even when the cached file is current.")
    ] = False,
    source: Annotated[
        list[str] | None, typer.Option("--source", help="Limit to these source ids.")
    ] = None,
    strategy: Annotated[
        ChunkStrategy | None, typer.Option("--strategy", help="Override the chunking strategy.")
    ] = None,
) -> None:
    """Download, parse and chunk the corpus into the local store."""
    settings = get_settings()
    reports = run_ingest(
        settings, download=download, force=force, source_ids=source or None, strategy=strategy
    )

    table = Table(title="Ingested corpus")
    for column in ("Document", "Provisions", "Chunks", "Avg tokens", "Fetched"):
        table.add_column(column, justify="right" if column != "Document" else "left")
    for report in reports:
        table.add_row(
            report.document_id,
            str(report.provisions),
            str(report.chunks),
            f"{report.average_tokens:.0f}",
            "downloaded" if report.downloaded else "cached",
        )
    console.print(table)
    console.print(f"Store: {settings.database_path}")


@app.command()
def index(
    strategy: Annotated[
        ChunkStrategy | None, typer.Option("--strategy", help="Which chunking to embed.")
    ] = None,
) -> None:
    """Embed the corpus into the vector store, reusing cached vectors."""
    settings = get_settings()
    report = build_index(settings, strategy=strategy)
    console.print(
        f"[bold]{report.model}[/bold] · {report.strategy} · {report.chunks} chunks · "
        f"{report.embedded} embedded · {report.reused} reused "
        f"({report.cache_hit_rate:.0%} cache hits)"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Question or keywords.")],
    k: Annotated[int, typer.Option("-k", help="How many passages to show.")] = 8,
    language: Annotated[
        Language | None, typer.Option("--language", help="Force the corpus language.")
    ] = None,
    strategy: Annotated[ChunkStrategy | None, typer.Option("--strategy")] = None,
) -> None:
    """Show the passages a question retrieves, and which signal found them."""
    settings = get_settings()
    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        retriever = HybridRetriever(
            store, vectors, make_embedder(settings), settings=settings, strategy=strategy
        )
        hits = retriever.search(query, language=language, limit=k)

    if not hits:
        console.print("[yellow]No results. Have you run `aiact ingest` and `aiact index`?[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Top {len(hits)} passages")
    table.add_column("#", justify="right")
    table.add_column("Signals")
    table.add_column("Provision")
    table.add_column("Passage")
    for hit in hits:
        signals = "+".join(sorted(hit.components))
        table.add_row(
            str(hit.rank),
            f"{signals}{' *' if hit.matched_reference else ''}",
            ", ".join(hit.provision_ids) or "-",
            " ".join(hit.chunk.text[:140].split()) + "...",
        )
    console.print(table)


@app.command(name="eval-retrieval")
def eval_retrieval(
    dataset: Annotated[Path, typer.Option("--dataset", help="JSONL golden dataset.")] = Path(
        "evals/datasets/retrieval_mini.jsonl"
    ),
    k: Annotated[int, typer.Option("-k", help="Cut-off for the metrics.")] = 5,
    strategy: Annotated[ChunkStrategy | None, typer.Option("--strategy")] = None,
) -> None:
    """Compare retrieval configurations on a golden dataset."""
    settings = get_settings()
    cases = load_cases(dataset)
    results = compare_configurations(settings, cases, k=k, strategy=strategy)

    table = Table(title=f"Retrieval on {dataset.name} ({len(cases)} questions, k={k})")
    table.add_column("Configuration")
    for column in (f"hit@{k}", f"recall@{k}", "MRR", "nDCG"):
        table.add_column(column, justify="right")
    table.add_column("Missed")
    for result in results:
        table.add_row(*result.scores.as_row(result.label), ", ".join(result.scores.misses) or "-")
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in English or French.")],
    k: Annotated[int | None, typer.Option("-k", help="Passages to ground the answer on.")] = None,
    language: Annotated[Language | None, typer.Option("--language")] = None,
    strategy: Annotated[ChunkStrategy | None, typer.Option("--strategy")] = None,
) -> None:
    """Answer a question from the corpus, with citations."""
    settings = get_settings()
    if settings.anthropic_api_key is None:
        console.print("[red]ANTHROPIC_API_KEY is not set. Add it to .env first.[/red]")
        raise typer.Exit(code=1)

    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        retriever = HybridRetriever(
            store, vectors, make_embedder(settings), settings=settings, strategy=strategy
        )
        llm = AnthropicLLM(
            settings.anthropic_api_key.get_secret_value(),
            model=settings.llm_model,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
        )
        try:
            answer = answer_question(
                question, retriever=retriever, llm=llm, language=language, limit=k
            )
        except LLMError as error:
            # An API failure is an operational problem, not a stack trace for the user.
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from None

    console.print(
        Panel(
            answer.text, title="Answer", border_style="green" if not answer.abstained else "yellow"
        )
    )
    if answer.citations:
        console.print("Sources: " + ", ".join(answer.citations))
    if answer.citation_check.invalid:
        dropped = ", ".join(answer.citation_check.invalid)
        console.print(f"[yellow]Dropped unsupported citations: {dropped}[/yellow]")
    console.print(
        f"[dim]{answer.model} · {answer.usage.input_tokens} in / {answer.usage.output_tokens} out"
        f" · {answer.usage.cache_read_tokens} cached · ${answer.cost_usd:.4f}"
        f" · prompt {answer.prompt_version}[/dim]"
    )
    console.print("[dim]Not legal advice.[/dim]")


@app.command()
def agent(
    question: Annotated[str, typer.Argument(help="Your question, in English or French.")],
    thread: Annotated[
        str | None, typer.Option("--thread", help="Continue an earlier conversation.")
    ] = None,
    language: Annotated[Language | None, typer.Option("--language")] = None,
) -> None:
    """Answer with the agent: it routes, calls tools and verifies its own citations."""
    settings = get_settings()
    if settings.anthropic_api_key is None:
        console.print("[red]ANTHROPIC_API_KEY is not set. Add it to .env first.[/red]")
        raise typer.Exit(code=1)

    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        deps = AgentDeps(
            llm=AnthropicLLM(
                settings.anthropic_api_key.get_secret_value(),
                model=settings.llm_model,
                effort=settings.llm_effort,
                max_tokens=settings.llm_max_tokens,
            ),
            retriever=HybridRetriever(store, vectors, make_embedder(settings), settings=settings),
            store=store,
            budget=Budget(
                max_steps=settings.agent_max_steps, max_cost_usd=settings.agent_max_cost_usd
            ),
        )
        checkpointer = open_checkpointer(settings.data_dir / "index" / "threads.db")
        try:
            answer = run_agent(
                question,
                deps,
                thread_id=thread,
                checkpointer=checkpointer,
                language=language,
            )
        except LLMError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from None

    console.print(
        Panel(
            answer.text,
            title=f"Agent · {answer.route} · {answer.steps} steps",
            border_style="yellow" if answer.abstained else "green",
        )
    )
    if answer.citations:
        console.print("Sources: " + ", ".join(answer.citations))
    if answer.halted_by:
        console.print(f"[yellow]Guardrail: {answer.halted_by}[/yellow]")
    console.print(
        f"[dim]{answer.usage.input_tokens} in / {answer.usage.output_tokens} out"
        f" · ${answer.cost_usd:.4f} · thread {answer.thread_id}[/dim]"
    )
    console.print("[dim]Not legal advice.[/dim]")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    """Serve the HTTP API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ai_act_copilot.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


@app.command(name="eval-answers")
def eval_answers(
    dataset: Annotated[Path, typer.Option("--dataset")] = Path("evals/datasets/golden_v1.jsonl"),
    mode: Annotated[Mode, typer.Option("--mode", help="Which answer path to measure.")] = Mode.RAG,
    judge: Annotated[bool, typer.Option("--judge/--no-judge", help="Run the LLM judges.")] = True,
    limit: Annotated[int | None, typer.Option("--limit", help="First N cases only.")] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("evals/results"),
) -> None:
    """Answer the golden set and grade it. Writes a JSON and a Markdown report."""
    settings = get_settings()
    if settings.anthropic_api_key is None:
        console.print("[red]ANTHROPIC_API_KEY is not set. Add it to .env first.[/red]")
        raise typer.Exit(code=1)

    cases = load_cases(dataset)[:limit] if limit else load_cases(dataset)
    if review_rate(cases) < 1.0:
        console.print(
            f"[yellow]{review_rate(cases):.0%} of this dataset is human-reviewed; "
            "treat the numbers as provisional.[/yellow]"
        )

    key = settings.anthropic_api_key.get_secret_value()
    with (
        CorpusStore(settings.database_path) as store,
        VectorStore(settings.database_path) as vectors,
    ):
        retriever = HybridRetriever(store, vectors, make_embedder(settings), settings=settings)
        llm = AnthropicLLM(
            key,
            model=settings.llm_model,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
        )
        deps = AgentDeps(
            llm=llm,
            retriever=retriever,
            store=store,
            budget=Budget(
                max_steps=settings.agent_max_steps, max_cost_usd=settings.agent_max_cost_usd
            ),
        )
        try:
            run = run_answers(
                cases,
                mode=mode,
                llm=llm,
                retriever=retriever,
                store=store,
                judge=LLMJudge(AnthropicLLM(key, model=settings.judge_model)) if judge else None,
                agent_deps=deps,
                label=f"{mode.value}-{settings.llm_model}",
            )
        except LLMError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from None

    json_path, markdown_path = write_report(
        run, output, judge_model=settings.judge_model if judge else "none", dataset=dataset.name
    )
    console.print(TABLE_HEADER)
    console.print(summary_row(run.summary))
    console.print("")
    console.print(f"Reports: {markdown_path} - {json_path}")


@app.command(name="eval-calibrate")
def eval_calibrate(
    run_json: Annotated[Path, typer.Argument(help="A run.json written by eval-answers.")],
    labels: Annotated[Path, typer.Option("--labels", help="Your scores, as JSONL.")] = Path(
        "evals/human_labels.jsonl"
    ),
) -> None:
    """Check the judges against human scores before trusting their numbers."""
    if not labels.is_file():
        console.print(
            f"[red]{labels} not found.[/red] Copy evals/human_labels.example.jsonl and score "
            "about twenty cases by hand."
        )
        raise typer.Exit(code=1)

    cases = json.loads(run_json.read_text(encoding="utf-8"))["cases"]
    graded = [
        SimpleNamespace(
            case_id=case["id"],
            faithfulness=case.get("faithfulness"),
            correctness=case.get("correctness"),
        )
        for case in cases
    ]
    agreements = compare(graded, load_labels(labels))
    console.print(agreement_table(agreements))
    if any(not agreement.usable for agreement in agreements):
        console.print(
            "[yellow]Cohen's kappa below 0.4 on at least one metric: the judge does not yet "
            "agree with you enough to be quoted.[/yellow]"
        )


@app.command()
def info(ctx: typer.Context) -> None:
    """Show the effective configuration. Secrets are never printed."""
    settings = get_settings()
    tracing: TracingStatus = ctx.obj

    table = Table(show_header=False, box=None)
    table.add_row("Version", __version__)
    table.add_row("LLM model", settings.llm_model)
    table.add_row("Anthropic API key", "set" if settings.anthropic_api_key else "missing")
    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Tracing", tracing.describe())
    console.print(table)
