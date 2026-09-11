from pathlib import Path

import pytest

import app.rag.pipeline as pipeline
from app.config import settings
from app.rag.pipeline import (
    _comparison_queries,
    _merge_comparison_chunks,
    build_prompt,
    search,
)


def _chunk(
    source: str,
    chunk_index: int,
    score: float,
    text: str = "text",
    heading: str | None = None,
) -> dict:
    return {
        "text": text,
        "source": source,
        "chunk_index": chunk_index,
        "heading": heading,
        "score": score,
    }


def test_comparison_queries_splits_who_has_more() -> None:
    assert _comparison_queries("Who has more titles, LeBron or Curry?") == [
        "LeBron titles",
        "Curry titles",
    ]


def test_comparison_queries_splits_compare_and_vs() -> None:
    assert _comparison_queries("Compare LeBron and Curry") == ["LeBron", "Curry"]
    assert _comparison_queries("LeBron vs Curry") == ["LeBron", "Curry"]
    assert _comparison_queries("Does LeBron have more titles than Curry?") == [
        "LeBron titles",
        "Curry titles",
    ]


def test_comparison_queries_skips_lookups() -> None:
    assert _comparison_queries("What is Ask My Docs?") is None
    assert _comparison_queries("Who is LeBron James?") is None
    assert _comparison_queries("Where should I put notes to query?") is None


def test_merge_comparison_chunks_caps_per_source_and_keeps_both_sides() -> None:
    lebron = "docs/nba/lebron-james.md"
    curry = "docs/nba/stephen-curry.md"
    left = [
        _chunk(lebron, 0, 0.99),
        _chunk(lebron, 1, 0.95),
        _chunk(lebron, 2, 0.90),
    ]
    right = [
        _chunk(curry, 0, 0.40),
        _chunk(curry, 1, 0.30),
    ]

    merged = _merge_comparison_chunks([left, right])

    sources = [chunk["source"] for chunk in merged]
    assert sources.count(lebron) == 2
    assert sources.count(curry) == 2
    assert sources[:2] == [lebron, curry]


def test_search_comparison_returns_both_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        '{"embedding_model": "text-embedding-3-small", "chunks": ['
        '{"text": "LeBron has four titles.", "source": "docs/nba/lebron-james.md",'
        ' "chunk_index": 0, "heading": "LeBron James", "embedding": [1.0, 0.0, 0.0]},'
        '{"text": "LeBron also won MVPs.", "source": "docs/nba/lebron-james.md",'
        ' "chunk_index": 1, "heading": "LeBron James", "embedding": [0.95, 0.05, 0.0]},'
        '{"text": "Curry has four titles.", "source": "docs/nba/stephen-curry.md",'
        ' "chunk_index": 0, "heading": "Stephen Curry", "embedding": [0.0, 1.0, 0.0]}'
        "]}",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 2)
    captured: list[str] = []

    def fake_embed(texts: list[str], *, for_query: bool = False) -> list[list[float]]:
        text = texts[0].lower()
        captured.append(texts[0])
        if "lebron" in text and "curry" not in text:
            return [[1.0, 0.0, 0.0]]
        if "curry" in text and "lebron" not in text:
            return [[0.0, 1.0, 0.0]]
        return [[0.9, 0.1, 0.0]]

    monkeypatch.setattr(pipeline, "_embed_texts", fake_embed)

    results = search("Who has more titles, LeBron or Curry?")

    sources = {chunk["source"] for chunk in results}
    assert sources == {
        "docs/nba/lebron-james.md",
        "docs/nba/stephen-curry.md",
    }
    assert any("lebron" in query.lower() and "curry" not in query.lower() for query in captured)
    assert any("curry" in query.lower() and "lebron" not in query.lower() for query in captured)
    assert captured != ["Who has more titles, LeBron or Curry?"]


def test_build_prompt_compare_allows_synthesis() -> None:
    chunks = [
        {
            "text": "James has four NBA titles.",
            "source": "docs/nba/lebron-james.md",
            "heading": "Championships and records",
        },
        {
            "text": "He has four NBA titles with Golden State.",
            "source": "docs/nba/stephen-curry.md",
            "heading": "Championships and awards",
        },
    ]
    messages = build_prompt(
        "Who has more titles, LeBron or Curry?", chunks, compare=True
    )
    system = messages[0]["content"].lower()
    user = messages[1]["content"]
    assert "count" in system
    assert "compare" in system
    assert "i don't know" in system
    assert "each side" in user.lower()
    assert "docs/nba/lebron-james.md" in user
    assert "docs/nba/stephen-curry.md" in user


def test_ask_compare_uses_compare_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.rag.pipeline import ask

    index_path = tmp_path / "index.json"
    index_path.write_text(
        '{"embedding_model": "text-embedding-3-small", "chunks": ['
        '{"text": "LeBron has four titles.", "source": "docs/nba/lebron-james.md",'
        ' "chunk_index": 0, "heading": "LeBron James", "embedding": [1.0, 0.0]},'
        '{"text": "Curry has four titles.", "source": "docs/nba/stephen-curry.md",'
        ' "chunk_index": 0, "heading": "Stephen Curry", "embedding": [0.0, 1.0]}'
        "]}",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: (
            [[1.0, 0.0]]
            if "lebron" in texts[0].lower()
            else [[0.0, 1.0]]
        ),
    )
    seen: dict = {}

    def fake_llm(messages: list[dict]) -> tuple[str, None]:
        seen["messages"] = messages
        return (
            "LeBron has four NBA titles and Curry has four. They are tied.",
            None,
        )

    monkeypatch.setattr(pipeline, "_ask_llm", fake_llm)

    result = ask("Who has more titles, LeBron or Curry?")

    user = seen["messages"][1]["content"]
    assert "each side" in user.lower()
    assert [citation.source for citation in result.citations] == [
        "docs/nba/lebron-james.md",
        "docs/nba/stephen-curry.md",
    ]
    assert "four" in result.answer.lower()
