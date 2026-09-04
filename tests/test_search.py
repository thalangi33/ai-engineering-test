import json
from pathlib import Path

import pytest

import app.rag.pipeline as pipeline
from app.config import settings
from app.rag.pipeline import search


def _write_index(path: Path, chunks: list[dict], model: str = "text-embedding-3-small") -> None:
    path.write_text(
        json.dumps({"embedding_model": model, "chunks": chunks}),
        encoding="utf-8",
    )


def _sample_chunks() -> list[dict]:
    return [
        {
            "text": "LeBron James is a forward for the Los Angeles Lakers.",
            "source": "docs/nba/lebron-james.md",
            "chunk_index": 0,
            "heading": "LeBron James",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "Stephen Curry is a point guard for the Golden State Warriors.",
            "source": "docs/nba/stephen-curry.md",
            "chunk_index": 0,
            "heading": "Stephen Curry",
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "text": "The weather in Tokyo is not in these notes.",
            "source": "docs/unrelated.md",
            "chunk_index": 0,
            "heading": None,
            "embedding": [0.0, 0.0, 1.0],
        },
    ]


def test_cosine_similarity_known_values() -> None:
    assert pipeline._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert pipeline._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert pipeline._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert pipeline._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="dimension mismatch"):
        pipeline._cosine_similarity([1.0], [1.0, 0.0])


def test_search_ranks_closest_chunk_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[0.9, 0.1, 0.0]],
    )

    results = search("Who is LeBron James?")

    assert [chunk["source"] for chunk in results] == [
        "docs/nba/lebron-james.md",
        "docs/nba/stephen-curry.md",
    ]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["heading"] == "LeBron James"
    assert "embedding" not in results[0]
    logged = capsys.readouterr().out
    assert "[search]" in logged
    assert "docs/nba/lebron-james.md" in logged


def test_search_respects_top_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )

    results = search("LeBron Curry", top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "docs/nba/lebron-james.md"
    assert results[1]["source"] == "docs/nba/stephen-curry.md"


def test_search_uses_index_embedding_model_and_query_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks(), model="gemini-embedding-001")
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")
    captured: dict = {}

    def fake_embed(texts: list[str], *, for_query: bool = False) -> list[list[float]]:
        captured["model"] = settings.embedding_model
        captured["for_query"] = for_query
        captured["texts"] = texts
        return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr(pipeline, "_embed_texts", fake_embed)

    results = search("Who is LeBron James?")

    assert captured["model"] == "gemini-embedding-001"
    assert captured["for_query"] is True
    assert captured["texts"] == ["Who is LeBron James?"]
    assert settings.embedding_model == "text-embedding-3-small"
    assert results[0]["source"] == "docs/nba/lebron-james.md"


def test_search_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "index_path", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Run ingest"):
        search("What is Ask My Docs?")


def test_search_empty_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, [])
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0]],
    )

    assert search("What is Ask My Docs?") == []


def test_search_rejects_empty_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "index_path", tmp_path / "index.json")
    with pytest.raises(ValueError, match="must not be empty"):
        search("   ")


def test_keyword_score_matches_heading_and_filename() -> None:
    tokens = pipeline._tokens("who is better lebron or curry?")
    assert "lebron" in tokens
    assert "curry" in tokens
    assert "better" not in tokens
    lebron = {
        "text": "A forward for the Lakers.",
        "source": "docs/nba/lebron-james.md",
        "heading": "LeBron James",
    }
    curry = {
        "text": "A point guard for the Warriors.",
        "source": "docs/nba/stephen-curry.md",
        "heading": "Stephen Curry",
    }
    assert pipeline._keyword_score(tokens, lebron) == pytest.approx(0.5)
    assert pipeline._keyword_score(tokens, curry) == pytest.approx(0.5)


def test_search_drops_non_positive_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )

    results = search("zzzz-no-overlap")

    assert [chunk["source"] for chunk in results] == ["docs/nba/lebron-james.md"]
    assert results[0]["score"] > 0


def test_search_caps_chunks_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(
        index_path,
        [
            {
                "text": "LeBron bio.",
                "source": "docs/nba/lebron-james.md",
                "chunk_index": 0,
                "heading": "LeBron James",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "text": "LeBron titles.",
                "source": "docs/nba/lebron-james.md",
                "chunk_index": 1,
                "heading": "Championships",
                "embedding": [0.99, 0.0, 0.0],
            },
            {
                "text": "LeBron style.",
                "source": "docs/nba/lebron-james.md",
                "chunk_index": 2,
                "heading": "Playing style",
                "embedding": [0.98, 0.0, 0.0],
            },
            {
                "text": "Stephen Curry bio.",
                "source": "docs/nba/stephen-curry.md",
                "chunk_index": 0,
                "heading": "Stephen Curry",
                "embedding": [0.5, 0.5, 0.0],
            },
        ],
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 3)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )

    results = search("LeBron James versus Curry")

    sources = [chunk["source"] for chunk in results]
    assert sources.count("docs/nba/lebron-james.md") == 2
    assert "docs/nba/stephen-curry.md" in sources


def test_search_expand_notes_includes_sibling_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(
        index_path,
        [
            {
                "text": "LeBron James is a forward.",
                "source": "docs/nba/lebron-james.md",
                "chunk_index": 0,
                "heading": "LeBron James",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "text": "James has four NBA titles.",
                "source": "docs/nba/lebron-james.md",
                "chunk_index": 1,
                "heading": "Championships",
                "embedding": [0.1, 0.9, 0.0],
            },
        ],
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 1)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )

    hits = search("Who is LeBron James?")
    assert len(hits) == 1
    assert hits[0]["chunk_index"] == 0

    expanded = search("Who is LeBron James?", expand_notes=True)
    assert [chunk["chunk_index"] for chunk in expanded] == [0, 1]
    assert all(chunk["source"] == "docs/nba/lebron-james.md" for chunk in expanded)


def test_search_invalid_index_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(settings, "index_path", index_path)
    with pytest.raises(ValueError, match="not valid JSON"):
        search("What is Ask My Docs?")
