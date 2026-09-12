from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ai_act_copilot.api.app import create_app
from ai_act_copilot.config import Settings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(data_dir=tmp_path / "data", anthropic_api_key=SecretStr("sk-ant-test"))
    return TestClient(create_app(settings))


def test_health_reports_the_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_ask_validates_its_input(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "hi"})  # shorter than the minimum

    assert response.status_code == 422


def test_agent_rejects_an_unknown_language(client: TestClient) -> None:
    response = client.post(
        "/v1/agent", json={"question": "Which practices are prohibited?", "language": "de"}
    )

    assert response.status_code == 422


def test_startup_fails_loudly_without_a_key(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"), TestClient(create_app(settings)):
        pass  # pragma: no cover - the context manager raises on entry
