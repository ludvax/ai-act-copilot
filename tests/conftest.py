from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_act_copilot.config import Settings, get_settings
from ai_act_copilot.observability.tracing import init_tracing

_ENV_PREFIXES = ("AIACT_", "ANTHROPIC_", "LANGFUSE_")

FIXTURES = Path(__file__).parent / "fixtures"


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
