from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_chat_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask My Docs" in response.text


def test_ingest_is_stubbed() -> None:
    response = client.post("/api/ingest")
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_ask_is_stubbed() -> None:
    response = client.post("/api/ask", json={"question": "What is Ask My Docs?"})
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_ask_rejects_empty_question() -> None:
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 422
