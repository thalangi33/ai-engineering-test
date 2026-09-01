import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.rag.pipeline as pipeline
from app.config import settings
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
    assert 'id="embedding-model"' in response.text
    assert "/api/search" in response.text


def test_embedding_models_lists_options() -> None:
    response = client.get("/api/embedding-models")
    assert response.status_code == 200
    body = response.json()
    ids = [model["id"] for model in body["models"]]
    assert ids == [
        "text-embedding-3-small",
        "gemini-embedding-001",
        "all-MiniLM-L6-v2",
    ]
    assert body["selected"] in ids
    assert all(model["label"] for model in body["models"])


def test_ingest_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    response = client.post("/api/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["document_count"] == 6
    assert body["chunk_count"] >= 1
    assert (tmp_path / "index.json").is_file()


def test_ingest_accepts_embedding_model_in_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    response = client.post(
        "/api/ingest", json={"embedding_model": "gemini-embedding-001"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_model"] == "gemini-embedding-001"
    stored = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert stored["embedding_model"] == "gemini-embedding-001"


def test_ingest_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(settings, "llm_api_key", "  ")
    response = client.post("/api/ingest")
    assert response.status_code == 400
    assert "llm_api_key" in response.json()["detail"].lower()


def test_ingest_reports_missing_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    response = client.post("/api/ingest")
    assert response.status_code == 400
    assert "gemini_api_key" in response.json()["detail"].lower()


def test_ingest_reports_unknown_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "not-a-real-model")
    response = client.post("/api/ingest")
    assert response.status_code == 400
    assert "unsupported embedding model" in response.json()["detail"].lower()


def test_ingest_reports_embed_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "sentence-transformers is required for all-MiniLM-L6-v2. "
            "Install it with: pip install sentence-transformers"
        )

    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    monkeypatch.setattr(pipeline, "_embed_texts", _boom)
    response = client.post(
        "/api/ingest", json={"embedding_model": "all-MiniLM-L6-v2"}
    )
    assert response.status_code == 400
    assert "sentence-transformers" in response.json()["detail"].lower()


def test_ingest_missing_docs_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "docs_dir", tmp_path / "missing")
    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    response = client.post("/api/ingest")
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_search_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "embedding_model": "text-embedding-3-small",
                "chunks": [
                    {
                        "text": "Ask My Docs answers from local documents.",
                        "source": "docs/what-ask-my-docs-is.md",
                        "chunk_index": 0,
                        "heading": "What Ask My Docs is",
                        "embedding": [1.0, 0.0],
                    },
                    {
                        "text": "Put notes in the docs folder.",
                        "source": "docs/folder-conventions.md",
                        "chunk_index": 0,
                        "heading": "Document folder conventions",
                        "embedding": [0.0, 1.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0]],
    )

    response = client.post("/api/search", json={"question": "What is Ask My Docs?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is Ask My Docs?"
    assert body["chunks"][0]["source"] == "docs/what-ask-my-docs-is.md"
    assert "embedding" not in body["chunks"][0]
    assert body["chunks"][0]["score"] > body["chunks"][1]["score"]


def test_search_accepts_top_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "embedding_model": "all-MiniLM-L6-v2",
                "chunks": [
                    {
                        "text": "one",
                        "source": "docs/a.md",
                        "chunk_index": 0,
                        "heading": None,
                        "embedding": [1.0, 0.0],
                    },
                    {
                        "text": "two",
                        "source": "docs/b.md",
                        "chunk_index": 0,
                        "heading": None,
                        "embedding": [0.9, 0.1],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0]],
    )

    response = client.post(
        "/api/search", json={"question": "Ask My Docs", "top_k": 1}
    )
    assert response.status_code == 200
    assert len(response.json()["chunks"]) == 1


def test_search_missing_index_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "index_path", tmp_path / "missing.json")
    response = client.post("/api/search", json={"question": "What is Ask My Docs?"})
    assert response.status_code == 400
    assert "ingest" in response.json()["detail"].lower()


def test_search_rejects_empty_question() -> None:
    response = client.post("/api/search", json={"question": ""})
    assert response.status_code == 422


def test_ask_is_stubbed() -> None:
    response = client.post("/api/ask", json={"question": "What is Ask My Docs?"})
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_ask_rejects_empty_question() -> None:
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 422
