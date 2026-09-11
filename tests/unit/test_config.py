from pathlib import Path

import pytest

from ai_act_copilot.config import Settings, get_settings


def test_defaults_without_environment() -> None:
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.llm_model == "claude-opus-5"
    assert settings.anthropic_api_key is None
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert not settings.langfuse_configured


def test_reads_prefixed_and_conventional_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIACT_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("AIACT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.com")

    settings = Settings()

    assert settings.llm_model == "claude-sonnet-5"
    assert settings.log_level == "DEBUG"
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert settings.langfuse_configured
    assert settings.langfuse_base_url == "https://langfuse.example.com"


def test_empty_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A freshly copied `.env.example` has empty keys: they must not count as configured.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    assert not Settings().langfuse_configured


def test_loads_dotenv_file_from_working_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("AIACT_DATA_DIR=corpus\nANTHROPIC_API_KEY=sk-ant-dotenv\n")

    settings = Settings()

    assert settings.data_dir == Path("corpus")
    assert settings.anthropic_api_key is not None


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")

    assert "sk-ant-very-secret" not in repr(Settings())


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
