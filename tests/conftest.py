from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_act_copilot.config import Settings, get_settings
from ai_act_copilot.observability.redaction import mask_otel_spans
from ai_act_copilot.observability.tracing import init_tracing

_ENV_PREFIXES = ("AIACT_", "ANTHROPIC_", "LANGFUSE_")

FIXTURES = Path(__file__).parent / "fixtures"
# Tests run in an isolated temporary directory, so repo files need absolute paths.
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _tracing_disabled() -> None:
    """Instrumented code must never reach a real Langfuse project during tests."""
    init_tracing(Settings(tracing_enabled=False))


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep tests independent from the developer's shell environment and `.env` file."""
    import os

    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass(frozen=True, slots=True)
class ExportedTraces:
    """The spans a piece of code actually emitted, captured without a network call.

    The Langfuse SDK is OpenTelemetry underneath, so a real client pointed at an in-memory
    exporter produces exactly the spans a live project would receive. That makes the shape
    of a trace - names, types, nesting, redaction - testable in CI, with no account and no
    keys, and stops instrumentation from rotting silently between releases.
    """

    client: Langfuse
    exporter: InMemorySpanExporter

    def spans(self) -> list[ReadableSpan]:
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def names(self) -> list[str]:
        return [span.name for span in self.spans()]

    def named(self, name: str) -> ReadableSpan:
        matches = [span for span in self.spans() if span.name == name]
        assert matches, f"no span named {name!r}; got {self.names()}"
        return matches[0]

    def attribute(self, name: str, key: str) -> Any:
        attributes = self.named(name).attributes
        return attributes.get(key) if attributes else None

    def metadata(self, name: str, key: str) -> Any:
        """Metadata is flattened into one attribute per key when it reaches OpenTelemetry."""
        return self.attribute(name, f"{LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.{key}")

    def trace_metadata(self, name: str, key: str) -> Any:
        """Metadata set on the trace (via propagate_attributes), not on one observation."""
        return self.attribute(name, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}")

    def observation_type(self, name: str) -> Any:
        return self.attribute(name, LangfuseOtelSpanAttributes.OBSERVATION_TYPE)

    def tree(self, root: str) -> list[tuple[str, str | None]]:
        """(observation, parent) pairs for one trace, parents before their children.

        Walked depth-first with siblings ordered by name, which asserts the *structure*
        of a trace and deliberately not its chronology: the Windows timer is coarse enough
        that steps a few milliseconds apart share a start time, and a test that depended on
        that would pass on Linux and fail here. Scoped to the trace containing ``root``, so
        unrelated work in a fixture cannot change the result. Repeated steps - two turns of
        the same agent loop - stay in the list twice, which is the point.
        """
        anchor = self.named(root).context
        assert anchor is not None
        members = [
            span
            for span in self.spans()
            if span.context is not None and span.context.trace_id == anchor.trace_id
        ]
        names = {span.context.span_id: span.name for span in members if span.context}
        children: dict[int | None, list[ReadableSpan]] = {}
        for span in members:
            children.setdefault(span.parent.span_id if span.parent else None, []).append(span)
        for bucket in children.values():
            bucket.sort(key=lambda span: span.name)

        ordered: list[tuple[str, str | None]] = []

        def walk(parent_id: int | None) -> None:
            for span in children.get(parent_id, []):
                ordered.append((span.name, names.get(parent_id) if parent_id else None))
                if span.context is not None:
                    walk(span.context.span_id)

        walk(None)
        return ordered

    def parent_of(self, name: str) -> str | None:
        """Name of the parent span, so nesting can be asserted without span ids."""
        span = self.named(name)
        if span.parent is None:
            return None
        by_id = {
            other.context.span_id: other.name for other in self.spans() if other.context is not None
        }
        return by_id.get(span.parent.span_id)

    def all_attribute_text(self) -> str:
        """Every string that would reach Langfuse, for asserting what must never appear."""
        parts: list[str] = []
        for span in self.spans():
            parts.append(span.name)
            for value in (span.attributes or {}).values():
                parts.append(str(value))
        return "\n".join(parts)


@pytest.fixture
def traces() -> Iterator[ExportedTraces]:
    """A live Langfuse client that exports into memory instead of over the network.

    Only one client may exist per process - the SDK refuses to trace when several are
    registered, to avoid leaking spans across projects - so the session-wide disabled
    client is torn down for the duration of the test and put back afterwards.
    """
    LangfuseResourceManager.reset()
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        span_exporter=exporter,
        environment="test",
        mask_otel_spans=mask_otel_spans,
        tracing_enabled=True,
    )
    try:
        yield ExportedTraces(client=client, exporter=exporter)
    finally:
        LangfuseResourceManager.reset()
        init_tracing(Settings(tracing_enabled=False))
