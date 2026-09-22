import json

import pytest

from app.models import ExtractedInfo, TimeScope
from app.rag.extract import (
    chunk_extraction_text,
    chunk_matches,
    extract_info,
)


def test_extract_parses_question_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.extract.ask_llm",
        lambda messages: json.dumps(
            {
                "intent": "stats",
                "entities": ["LeBron James", "Miami Heat"],
                "metadata_filters": {"topic": "championships"},
                "time_scope": {"start": 2010, "end": 2014},
            }
        ),
    )
    info = extract_info("How many titles did LeBron win in Miami?", kind="question")
    assert info.intent == "stats"
    assert info.entities == ["LeBron James", "Miami Heat"]
    assert info.metadata_filters == {"topic": "championships"}
    assert info.time_scope == TimeScope(start=2010, end=2014)
    # The call used the shared extractor, not a live model.


def test_extract_strips_fences_and_clears_chunk_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.extract.ask_llm",
        lambda messages: (
            "```json\n"
            '{"intent": "fact", "entities": ["Stephen Curry"],'
            ' "metadata_filters": {"doc_type": "player"}, "time_scope": null}\n'
            "```"
        ),
    )
    info = extract_info("Stephen Curry is a point guard.", kind="chunk")
    assert info.intent is None
    assert info.entities == ["Stephen Curry"]
    assert info.metadata_filters == {"doc_type": "player"}
    assert info.time_scope is None


def test_extract_orders_reversed_years(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.extract.ask_llm",
        lambda messages: json.dumps(
            {
                "intent": "timeline",
                "entities": ["Chicago Bulls"],
                "metadata_filters": {},
                "time_scope": {"start": 1999, "end": 1990},
            }
        ),
    )
    info = extract_info("What did the Bulls do in the 1990s?", kind="question")
    assert info.time_scope == TimeScope(start=1990, end=1999)


def test_extract_bad_json_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.rag.extract.ask_llm", lambda messages: "nope")
    assert extract_info("x", kind="chunk") == ExtractedInfo()


def test_extract_invalid_intent_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.extract.ask_llm",
        lambda messages: json.dumps({"intent": "banter", "entities": ["x"]}),
    )
    assert extract_info("hello", kind="question") == ExtractedInfo()


def test_extract_empty_text_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"llm": False}

    def boom(messages: list[dict]) -> str:
        called["llm"] = True
        return "{}"

    monkeypatch.setattr("app.rag.extract.ask_llm", boom)
    assert extract_info("   ", kind="question") == ExtractedInfo()
    assert called["llm"] is False


def test_chunk_extraction_text_includes_heading() -> None:
    assert chunk_extraction_text(
        {"heading": "Miami Heat (2010–2014)", "text": "James joined Miami."}
    ) == "Miami Heat (2010–2014)\n\nJames joined Miami."
    assert chunk_extraction_text({"heading": None, "text": "Only body."}) == "Only body."


def test_chunk_matches_entity_time_and_metadata() -> None:
    lebron = {
        "entities": ["LeBron James", "Miami Heat"],
        "metadata_filters": {"doc_type": "player", "topic": "championships"},
        "time_scope": {"start": 2010, "end": 2014},
    }
    curry = {
        "entities": ["Stephen Curry"],
        "metadata_filters": {"doc_type": "player", "topic": "awards"},
        "time_scope": {"start": 2015, "end": 2025},
    }
    query = ExtractedInfo(
        intent="stats",
        entities=["LeBron James"],
        metadata_filters={"topic": "championships"},
        time_scope=TimeScope(start=2010, end=2014),
    )
    assert chunk_matches(query, lebron) is True
    assert chunk_matches(query, curry) is False


def test_chunk_matches_ignores_unset_filters() -> None:
    chunk = {
        "entities": ["Ask My Docs"],
        "metadata_filters": {},
        "time_scope": None,
    }
    assert chunk_matches(ExtractedInfo(), chunk) is True
    assert chunk_matches(
        ExtractedInfo(entities=["Ask My Docs"], time_scope=TimeScope(start=2020, end=2021)),
        chunk,
    ) is True
