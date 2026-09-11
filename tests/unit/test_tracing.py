import logging
from typing import ClassVar

import pytest

from ai_act_copilot.config import Settings
from ai_act_copilot.observability import tracing
from ai_act_copilot.observability.tracing import (
    TracingStatus,
    flush_tracing,
    init_tracing,
    observe,
)


class RecordingLangfuse:
    """Stands in for the Langfuse client so tests never open a network connection."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[RecordingLangfuse]:
    RecordingLangfuse.calls = []
    monkeypatch.setattr(tracing, "Langfuse", RecordingLangfuse)
    return RecordingLangfuse


def _set_langfuse_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.com")


def test_disabled_without_keys(recorder: type[RecordingLangfuse]) -> None:
    status = init_tracing(Settings())

    assert status == TracingStatus(enabled=False, detail="no Langfuse keys configured")
    assert recorder.calls[-1]["tracing_enabled"] is False


def test_disabled_by_flag_even_with_keys(
    recorder: type[RecordingLangfuse], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_langfuse_keys(monkeypatch)
    monkeypatch.setenv("AIACT_TRACING_ENABLED", "false")

    status = init_tracing(Settings())

    assert not status.enabled
    assert "AIACT_TRACING_ENABLED" in status.detail
    assert recorder.calls[-1]["public_key"] != "pk-lf-test"


def test_enabled_with_keys_passes_credentials(
    recorder: type[RecordingLangfuse], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_langfuse_keys(monkeypatch)

    status = init_tracing(Settings())

    assert status.describe() == "enabled (https://langfuse.example.com)"
    call = recorder.calls[-1]
    assert call["public_key"] == "pk-lf-test"
    assert call["secret_key"] == "sk-lf-test"
    assert call["base_url"] == "https://langfuse.example.com"


def test_disabled_client_is_silent_and_observe_is_transparent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    init_tracing(Settings())  # real SDK client, tracing disabled

    @observe(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    flush_tracing()
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
