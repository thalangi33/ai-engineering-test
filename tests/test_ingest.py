import json
from pathlib import Path

import pytest

import app.rag.pipeline as pipeline
from app.config import settings
from app.rag.pipeline import chunk_text, ingest, load_documents


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(i + 1), 0.5] for i, _ in enumerate(texts)]


def test_ingest_sample_docs_writes_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(pipeline, "_embed_texts", _fake_embed)

    result = ingest()
    expected_chunks = chunk_text(load_documents(settings.docs_dir))

    assert result.status == "ok"
    assert result.document_count == 3
    assert result.chunk_count == len(expected_chunks)
    assert result.chunk_count
    assert "chunks" in (result.message or "")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["embedding_model"] == settings.embedding_model
    assert len(payload["chunks"]) == len(expected_chunks)
    sources = {chunk["source"] for chunk in payload["chunks"]}
    assert "docs/what-ask-my-docs-is.md" in sources
    assert "docs/folder-conventions.md" in sources
    assert "docs/build-order.md" in sources
    for stored, original in zip(payload["chunks"], expected_chunks, strict=True):
        assert stored["text"] == original["text"]
        assert stored["source"] == original["source"]
        assert stored["chunk_index"] == original["chunk_index"]
        assert stored["heading"] == original["heading"]
        assert stored["embedding"]
        assert all(isinstance(value, float) for value in stored["embedding"])


def test_ingest_empty_docs_writes_empty_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(settings, "docs_dir", docs_dir)
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(pipeline, "_embed_texts", _fake_embed)

    result = ingest()

    assert result.status == "ok"
    assert result.document_count == 0
    assert result.chunk_count == 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["chunks"] == []


def test_ingest_overwrites_existing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# Hello\n\nFirst.", encoding="utf-8")
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"chunks": [{"stale": True}]}), encoding="utf-8")
    monkeypatch.setattr(settings, "docs_dir", docs_dir)
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(pipeline, "_embed_texts", _fake_embed)

    ingest()
    (docs_dir / "b.md").write_text("# Two\n\nSecond.", encoding="utf-8")
    result = ingest()

    assert result.document_count == 2
    assert result.chunk_count == 2
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sources = {chunk["source"] for chunk in payload["chunks"]}
    assert any(source.endswith("a.md") for source in sources)
    assert any(source.endswith("b.md") for source in sources)
    assert all("stale" not in chunk for chunk in payload["chunks"])


def test_ingest_missing_docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "docs_dir", tmp_path / "missing")
    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    with pytest.raises(FileNotFoundError):
        ingest()


def test_embed_texts_empty_returns_empty() -> None:
    assert pipeline._embed_texts([]) == []


def test_embed_texts_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        pipeline._embed_texts(["hello"])


def test_embed_texts_gemini_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        pipeline._embed_texts(["hello"])


def test_embed_texts_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "not-a-real-model")
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        pipeline._embed_texts(["hello"])


class _FakeEmbeddingsResponse:
    status_code = 200
    text = ""
    reason_phrase = "OK"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}


class _FakeEmbeddingsClient:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> "_FakeEmbeddingsClient":
        return self

    def __exit__(self, *args) -> None:
        return None

    def post(self, url, headers=None, json=None):
        assert url == pipeline._OPENAI_EMBEDDINGS_URL
        assert json["input"] == ["hello"]
        assert json["model"] == settings.embedding_model
        assert headers["Authorization"] == "Bearer sk-test"
        return _FakeEmbeddingsResponse()


def test_embed_texts_uses_openai_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(settings, "llm_api_key", "sk-test")
    monkeypatch.setattr(pipeline.httpx, "Client", _FakeEmbeddingsClient)
    assert pipeline._embed_texts(["hello"]) == [[0.1, 0.2]]


class _FakeGeminiResponse:
    status_code = 200
    text = ""
    reason_phrase = "OK"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"embeddings": [{"values": [0.3, 0.4]}]}


class _FakeGeminiClient:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> "_FakeGeminiClient":
        return self

    def __exit__(self, *args) -> None:
        return None

    def post(self, url, headers=None, json=None):
        assert url.endswith("models/gemini-embedding-001:batchEmbedContents")
        assert headers["x-goog-api-key"] == "gemini-test"
        assert json["requests"][0]["content"]["parts"][0]["text"] == "hello"
        assert json["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
        return _FakeGeminiResponse()


def test_embed_texts_uses_gemini_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    monkeypatch.setattr(pipeline.httpx, "Client", _FakeGeminiClient)
    assert pipeline._embed_texts(["hello"]) == [[0.3, 0.4]]


def test_embed_texts_gemini_query_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _Client(_FakeGeminiClient):
        def post(self, url, headers=None, json=None):
            captured["taskType"] = json["requests"][0]["taskType"]
            return _FakeGeminiResponse()

    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    monkeypatch.setattr(pipeline.httpx, "Client", _Client)
    pipeline._embed_texts(["hello"], for_query=True)
    assert captured["taskType"] == "RETRIEVAL_QUERY"


class _FakeMiniLM:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        assert normalize_embeddings is True
        return [[float(i + 1), 0.5] for i, _ in enumerate(texts)]


def test_embed_texts_minilm_does_not_need_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embedding_model", "all-MiniLM-L6-v2")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(pipeline, "_load_minilm", lambda: _FakeMiniLM())
    assert pipeline._embed_texts(["hello", "world"]) == [[1.0, 0.5], [2.0, 0.5]]


def test_embed_texts_minilm_accepts_hf_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    monkeypatch.setattr(pipeline, "_load_minilm", lambda: _FakeMiniLM())
    assert pipeline._embed_texts(["hello"]) == [[1.0, 0.5]]
