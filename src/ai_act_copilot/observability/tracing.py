"""Tracing bootstrap and the vocabulary the rest of the code uses (Langfuse, OpenTelemetry).

Entry points (CLI, API, eval runner) call :func:`init_tracing` once per process, then wrap
each unit of work in :func:`trace_context`. Instrumented code imports :func:`observe` from
this module rather than from ``langfuse``, so the tracing vendor is named in one file.

Three decisions worth knowing about:

- **Nothing leaves the machine without keys.** With no Langfuse credentials the client is
  created disabled: ``@observe`` becomes a transparent no-op and every helper below does
  nothing. That is how the tests and CI run, and why no call site branches on it.
- **Personal data is removed at the export boundary**, not at each call site - see
  :mod:`ai_act_copilot.observability.redaction`.
- **One trace is one question.** A follow-up on the same ``thread_id`` is its own trace,
  tied to the first by a session, which is what makes a multi-turn exchange readable.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Literal

from langfuse import Langfuse, get_client, observe, propagate_attributes

from ai_act_copilot import __version__
from ai_act_copilot.config import Settings
from ai_act_copilot.observability.redaction import mask_otel_spans

__all__ = [
    "ObservationType",
    "TracingStatus",
    "current_trace_id",
    "flush_tracing",
    "in_current_trace",
    "init_tracing",
    "observe",
    "observed",
    "record_generation",
    "record_score",
    "record_span",
    "trace_context",
    "trace_url",
]

# The subset of Langfuse observation types this system produces. Typing them keeps a
# misspelling from silently degrading an observation to a plain span.
type ObservationType = Literal[
    "span", "generation", "agent", "tool", "chain", "retriever", "evaluator", "embedding"
]

# The SDK logs authentication warnings when constructed without keys, even with tracing
# disabled. Placeholder keys keep a disabled client silent (the SDK does the same internally).
_DISABLED_KEY = "tracing-disabled"


@dataclass(frozen=True, slots=True)
class TracingStatus:
    enabled: bool
    detail: str
    environment: str = "development"

    def describe(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"{state} ({self.detail})"


def init_tracing(settings: Settings) -> TracingStatus:
    """Create the process-wide Langfuse client and report whether traces are exported."""
    public_key, secret_key = settings.langfuse_public_key, settings.langfuse_secret_key
    if not settings.tracing_enabled:
        return _init_disabled("AIACT_TRACING_ENABLED is false", settings.environment)
    if public_key is None or secret_key is None:
        return _init_disabled("no Langfuse keys configured", settings.environment)

    Langfuse(
        public_key=public_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        # Keeps test and local traces out of the dashboards that describe production.
        environment=settings.environment,
        release=__version__,
        sample_rate=settings.trace_sample_rate,
        mask_otel_spans=mask_otel_spans if settings.mask_traces else None,
    )
    detail = settings.langfuse_base_url
    if settings.trace_sample_rate < 1.0:
        detail += f", {settings.trace_sample_rate:.0%} sampled"
    if not settings.mask_traces:
        detail += ", UNMASKED"
    return TracingStatus(enabled=True, detail=detail, environment=settings.environment)


def _init_disabled(reason: str, environment: str) -> TracingStatus:
    Langfuse(public_key=_DISABLED_KEY, secret_key=_DISABLED_KEY, tracing_enabled=False)
    return TracingStatus(enabled=False, detail=reason, environment=environment)


def flush_tracing() -> None:
    """Export buffered spans. Short-lived processes must call this before exiting."""
    get_client().flush()


def trace_context(
    *,
    name: str,
    session_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Attach trace-level attributes to everything observed inside the block.

    Opened at the entry point, before the first observation, so that the whole trace
    carries them. ``tags`` are fixed at creation time and therefore describe what is known
    up front - which interface, which command; anything discovered during the run (the
    route the agent took, whether it abstained) belongs in ``metadata`` or in a score.
    """
    return propagate_attributes(
        trace_name=name,
        session_id=session_id,
        tags=list(tags) or None,
        metadata=dict(metadata) if metadata else None,
        version=__version__,
    )


def observed(
    *,
    name: str,
    as_type: ObservationType,
    input: Any = None,
) -> AbstractContextManager[Any]:
    """Open an observation around a block, where a decorator cannot reach.

    Used for steps whose name is only known at run time - the agent's tool calls, where
    the observation must be named after the tool the model actually chose.
    """
    return get_client().start_as_current_observation(name=name, as_type=as_type, input=input)


def record_span(
    *,
    input: Any = None,
    output: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Set input, output and metadata on the active (non-generation) observation.

    Decorators capture no arguments by default in this codebase: a function's signature is
    a poor description of a step, and it drags dependencies and credentials into the trace.
    Every observation instead states what it received and produced.
    """
    get_client().update_current_span(
        input=input, output=output, metadata=dict(metadata) if metadata else None
    )


def record_generation(
    *,
    name: str | None = None,
    model: str,
    input: Any = None,
    output: Any = None,
    usage_details: Mapping[str, int] | None = None,
    cost_usd: float | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Attach everything a generation needs to be readable and costed.

    ``name`` renames the observation to what the call is *for* (``generate-answer``,
    ``route-question``) rather than which model served it: dashboards and judges target
    observations by name, and a name that encodes the model breaks the moment it changes.

    Costs come from the usage the API reported rather than from an estimate, which is what
    makes the Langfuse cost view worth reading.
    """
    get_client().update_current_generation(
        name=name,
        model=model,
        input=input,
        output=output,
        usage_details=dict(usage_details) if usage_details else None,
        cost_details={"total": cost_usd} if cost_usd is not None else None,
        model_parameters=dict(model_parameters) if model_parameters else None,
        metadata=dict(metadata) if metadata else None,
    )


def record_score(
    *,
    name: str,
    value: float | str,
    trace_id: str | None = None,
    comment: str | None = None,
    data_type: Any = None,
) -> None:
    """Attach an evaluation result to a trace.

    Quality is the one dimension a trace cannot show on its own. Scores are what turn the
    eval harness from a Markdown report into something filterable next to latency and cost.
    """
    get_client().create_score(
        name=name, value=value, trace_id=trace_id, comment=comment, data_type=data_type
    )


def in_current_trace[T](work: Callable[..., T]) -> Callable[..., T]:
    """Run ``work`` on another thread while keeping it inside the current trace.

    The active observation lives in a context variable, which a worker thread does not
    inherit: without this, tools executed in parallel would each start a trace of their
    own and the agent's tree would come apart exactly where it is most interesting.

    The snapshot is taken here, so this must be called on the submitting thread - and once
    per task, because a context cannot be entered twice at the same time.
    """
    snapshot = copy_context()

    def submit(*args: Any, **kwargs: Any) -> T:
        return snapshot.run(work, *args, **kwargs)

    return submit


def current_trace_id() -> str | None:
    """Id of the trace being recorded, or None when tracing is disabled."""
    return get_client().get_current_trace_id()


def trace_url(trace_id: str | None = None) -> str | None:
    """Link to a trace in Langfuse, for printing next to the answer it produced."""
    return get_client().get_trace_url(trace_id=trace_id)
