"""Tracing bootstrap (Langfuse, OpenTelemetry-based).

Entry points (CLI, API, eval runner) call :func:`init_tracing` once per process.
Instrumented code imports :func:`observe` from this module rather than from ``langfuse``,
so the tracing vendor is referenced in a single place.

Without Langfuse keys the client is created disabled: ``@observe`` becomes a transparent
no-op and nothing leaves the machine. This is how tests and CI run.
"""

from dataclasses import dataclass

from langfuse import Langfuse, get_client, observe

from ai_act_copilot import __version__
from ai_act_copilot.config import Settings

__all__ = ["TracingStatus", "flush_tracing", "init_tracing", "observe"]

# The SDK logs authentication warnings when constructed without keys, even with tracing
# disabled. Placeholder keys keep a disabled client silent (the SDK does the same internally).
_DISABLED_KEY = "tracing-disabled"


@dataclass(frozen=True, slots=True)
class TracingStatus:
    enabled: bool
    detail: str

    def describe(self) -> str:
        return f"{'enabled' if self.enabled else 'disabled'} ({self.detail})"


def init_tracing(settings: Settings) -> TracingStatus:
    """Create the process-wide Langfuse client and report whether traces are exported."""
    public_key, secret_key = settings.langfuse_public_key, settings.langfuse_secret_key
    if not settings.tracing_enabled:
        return _init_disabled("AIACT_TRACING_ENABLED is false")
    if public_key is None or secret_key is None:
        return _init_disabled("no Langfuse keys configured")

    Langfuse(
        public_key=public_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        release=__version__,
    )
    return TracingStatus(enabled=True, detail=settings.langfuse_base_url)


def _init_disabled(reason: str) -> TracingStatus:
    Langfuse(public_key=_DISABLED_KEY, secret_key=_DISABLED_KEY, tracing_enabled=False)
    return TracingStatus(enabled=False, detail=reason)


def flush_tracing() -> None:
    """Export buffered spans. Short-lived processes must call this before exiting."""
    get_client().flush()
