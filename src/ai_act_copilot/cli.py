"""Command-line interface, installed as ``aiact``."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_act_copilot import __version__
from ai_act_copilot.config import get_settings
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
