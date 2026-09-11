from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_act_copilot.config import get_settings

_ENV_PREFIXES = ("AIACT_", "ANTHROPIC_", "LANGFUSE_")


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
