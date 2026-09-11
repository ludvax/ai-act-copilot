"""Command-line interface, installed as ``aiact``."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_act_copilot import __version__
from ai_act_copilot.config import get_settings
from ai_act_copilot.ingestion.pipeline import ingest as run_ingest
from ai_act_copilot.models import ChunkStrategy
from ai_act_copilot.observability.tracing import TracingStatus, flush_tracing, init_tracing

app = typer.Typer(
    help="Bilingual assistant for EU AI regulation (AI Act, GDPR).",
    no_args_is_help=True,
)
console = Console()


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
