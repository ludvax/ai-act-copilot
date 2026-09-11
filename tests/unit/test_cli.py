import pytest
from typer.testing import CliRunner

from ai_act_copilot import __version__
from ai_act_copilot.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "info" in result.output


def test_info_reports_configuration_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert "claude-opus-5" in result.output
    assert "sk-ant-very-secret" not in result.output
    assert "disabled" in result.output  # no Langfuse keys in the test environment
