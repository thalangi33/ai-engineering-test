import json

import pytest

from app.models import ExtractedInfo, TimeScope
from app.rag.extract import (
    chunk_extraction_text,
    chunk_matches,
    extract_chunks,
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


def _item(chunk_id: int, entity: str, *, topic: str | None = None) -> dict:
    filters = {"doc_type": "player"}
    if topic:
        filters["topic"] = topic
    return {
        "id": chunk_id,
        "entities": [entity],
        "metadata_filters": filters,
        "time_scope": {"start": 2010, "end": 2014},
    }


def test_extract_chunks_one_call_for_several_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_llm(messages: list[dict], **kwargs) -> str:
        calls.append({"messages": messages, **kwargs})
        return json.dumps({"items": [_item(0, "LeBron James"), _item(2, "Stephen Curry"), _item(1, "Miami Heat", topic="championships")]})

    monkeypatch.setattr("app.rag.extract.ask_llm", fake_llm)
    infos = extract_chunks(
        ["LeBron joined Miami.", "The Heat won titles.", "Curry plays for Golden State."],
        batch_size=8,
    )
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 150 * 3
    user = calls[0]["messages"][1]["content"]
    assert user.startswith("chunks:")
    assert "0:\nLeBron joined Miami." in user
    assert "1:\nThe Heat won titles." in user
    assert infos[0].entities == ["LeBron James"]
    assert infos[1].entities == ["Miami Heat"]
    assert infos[1].metadata_filters == {"doc_type": "player", "topic": "championships"}
    assert infos[1].time_scope == TimeScope(start=2010, end=2014)
    assert infos[2].entities == ["Stephen Curry"]
    assert all(info.intent is None for info in infos)


def test_extract_chunks_splits_on_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_llm(messages: list[dict], **kwargs) -> str:
        user = messages[1]["content"]
        calls.append(user)
        count = user.count("\n\n")
        # ids restart at 0 inside each batch
        return json.dumps({"items": [_item(index, f"Entity {len(calls)}-{index}") for index in range(count)]})

    monkeypatch.setattr("app.rag.extract.ask_llm", fake_llm)
    infos = extract_chunks(["a", "b", "c"], batch_size=2)
    assert len(calls) == 2
    assert "0:\na" in calls[0]
    assert "1:\nb" in calls[0]
    assert "0:\nc" in calls[1]
    assert [info.entities[0] for info in infos] == ["Entity 1-0", "Entity 1-1", "Entity 2-0"]


def test_extract_chunks_bad_item_keeps_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.extract.ask_llm",
        lambda messages, **kwargs: json.dumps(
            {
                "items": [
                    _item(0, "LeBron James"),
                    {"id": 1, "intent": "banter", "entities": ["nope"]},
                    _item(2, "Stephen Curry"),
                ]
            }
        ),
    )
    infos = extract_chunks(["one", "two", "three"], batch_size=8)
    assert infos[0].entities == ["LeBron James"]
    assert infos[1] == ExtractedInfo()
    assert infos[2].entities == ["Stephen Curry"]


def test_extract_chunks_bad_json_falls_back_to_single_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_llm(messages: list[dict], **kwargs) -> str:
        user = messages[1]["content"]
        calls.append(user)
        if user.startswith("chunks:"):
            return "nope"
        entity = "LeBron James" if "Miami" in user else "Stephen Curry"
        return json.dumps(_item(0, entity) | {"intent": "fact"})

    monkeypatch.setattr("app.rag.extract.ask_llm", fake_llm)
    infos = extract_chunks(["James in Miami.", "Curry in Oakland."], batch_size=8)
    assert len(calls) == 3
    assert calls[0].startswith("chunks:")
    assert infos[0].entities == ["LeBron James"]
    assert infos[1].entities == ["Stephen Curry"]
    assert all(info.intent is None for info in infos)


def test_extract_chunks_skips_blank_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_llm(messages: list[dict], **kwargs) -> str:
        calls.append(messages[1]["content"])
        return json.dumps({"items": [_item(0, "LeBron James")]})

    monkeypatch.setattr("app.rag.extract.ask_llm", fake_llm)
    infos = extract_chunks(["  ", "James in Miami."], batch_size=8)
    assert len(calls) == 1
    assert "0:\nJames in Miami." in calls[0]
    assert "1:" not in calls[0]
    assert infos[0] == ExtractedInfo()
    assert infos[1].entities == ["LeBron James"]


def test_extract_chunks_empty_input_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"llm": False}

    def boom(messages: list[dict], **kwargs) -> str:
        called["llm"] = True
        return "{}"

    monkeypatch.setattr("app.rag.extract.ask_llm", boom)
    assert extract_chunks([]) == []
    assert called["llm"] is False


def test_extract_chunks_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch size"):
        extract_chunks(["hello"], batch_size=0)


def test_chunk_matches_entity_alias() -> None:
    curry = {"entities": ["Stephen Curry"], "metadata_filters": {}, "time_scope": None}
    lebron = {"entities": ["LeBron James"], "metadata_filters": {}, "time_scope": None}
    assert chunk_matches(ExtractedInfo(entities=["Curry"]), curry) is True
    assert chunk_matches(ExtractedInfo(entities=["LeBron"]), lebron) is True
    assert chunk_matches(ExtractedInfo(entities=["Curry"]), lebron) is False
    assert chunk_matches(ExtractedInfo(entities=["Jokic"]), curry) is False
