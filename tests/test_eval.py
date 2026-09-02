import json
from pathlib import Path

import pytest

from app.models import AskResponse, Citation
from evals.score import (
    evaluate_all,
    evaluate_case,
    format_report,
    load_cases,
    missing_phrases,
    score_answer,
    score_retrieval,
    source_in,
    tally,
)


def _case(**overrides):
    base = {
        "id": "easy-1",
        "question": "What is Ask My Docs?",
        "expected_source": "docs/what-ask-my-docs-is.md",
        "should_refuse": False,
        "must_contain": ["local folder", "document"],
    }
    base.update(overrides)
    return base


def test_load_cases_from_repo() -> None:
    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert ids == ["easy-1", "easy-2", "easy-3", "paraphrase-1", "refuse-1"]
    refuse = next(case for case in cases if case["id"] == "refuse-1")
    assert refuse["should_refuse"] is True
    assert refuse["expected_source"] is None
    for case in cases:
        if not case["should_refuse"]:
            assert case["expected_source"]
            assert case["must_contain"]


def test_source_in_normalizes_paths() -> None:
    assert source_in("docs/what-ask-my-docs-is.md", ["docs/what-ask-my-docs-is.md"])
    assert source_in(
        "docs/what-ask-my-docs-is.md",
        ["/workspace/docs/what-ask-my-docs-is.md"],
    )
    assert not source_in("docs/what-ask-my-docs-is.md", ["docs/nba/lebron-james.md"])


def test_score_retrieval_pass_and_fail() -> None:
    case = _case()
    passed, detail = score_retrieval(
        case,
        [{"source": "docs/what-ask-my-docs-is.md"}, {"source": "docs/nba/lebron-james.md"}],
    )
    assert passed is True
    assert "found" in detail

    failed, detail = score_retrieval(case, [{"source": "docs/nba/lebron-james.md"}])
    assert failed is False
    assert "not in top-" in detail


def test_score_retrieval_skipped_for_refuse() -> None:
    case = _case(
        id="refuse-1",
        question="What is the weather in Tokyo tomorrow?",
        expected_source=None,
        should_refuse=True,
        must_contain=[],
    )
    skipped, detail = score_retrieval(case, [{"source": "docs/nba/lebron-james.md"}])
    assert skipped is None
    assert "n/a" in detail


def test_score_answer_requires_phrases_and_citation() -> None:
    case = _case()
    citations = [Citation(source="docs/what-ask-my-docs-is.md", snippet="local folder")]
    passed, _ = score_answer(
        case,
        "Ask My Docs answers from a local folder of documents, with citations.",
        citations,
    )
    assert passed is True

    fluent_wrong, detail = score_answer(
        case,
        "Ask My Docs is a helpful chatbot that uses AI.",
        citations,
    )
    assert fluent_wrong is False
    assert "fluent-but-wrong" in detail
    assert "local folder" in detail


def test_score_answer_refuse_rejects_citations() -> None:
    case = _case(
        id="refuse-1",
        question="What is the weather in Tokyo tomorrow?",
        expected_source=None,
        should_refuse=True,
        must_contain=[],
    )
    passed, _ = score_answer(case, "I don't know", [])
    assert passed is True

    answered, detail = score_answer(case, "Sunny and 72 degrees in Tokyo.", [])
    assert answered is False
    assert "should refuse" in detail

    fake_cite, detail = score_answer(
        case,
        "I don't know",
        [Citation(source="docs/nba/lebron-james.md", snippet="LeBron")],
    )
    assert fake_cite is False
    assert "citations" in detail


def test_score_answer_fails_when_model_refuses_known_question() -> None:
    passed, detail = score_answer(_case(), "I don't know", [])
    assert passed is False
    assert "refused" in detail


def test_missing_phrases_is_case_insensitive() -> None:
    assert missing_phrases("Re-ingest after edits.", ["ingest"]) == []
    assert missing_phrases("Nothing to see.", ["ingest"]) == ["ingest"]


def test_evaluate_case_scores_retrieval_and_answer_separately() -> None:
    case = _case()

    def fake_search(question: str):
        return [{"source": "docs/nba/lebron-james.md", "text": "LeBron plays for the Lakers."}]

    def fake_ask(question: str):
        return AskResponse(
            answer="Ask My Docs answers from a local folder of documents.",
            citations=[Citation(source="docs/what-ask-my-docs-is.md", snippet="local folder")],
        )

    score = evaluate_case(case, search_fn=fake_search, ask_fn=fake_ask)
    assert score.retrieval_pass is False
    assert score.answer_pass is True
    assert score.failed is True


def test_evaluate_all_retrieval_only_skips_ask() -> None:
    called = {"ask": False}

    def fake_search(question: str):
        if "weather" in question.lower():
            return [{"source": "docs/nba/lebron-james.md"}]
        return [{"source": "docs/what-ask-my-docs-is.md"}]

    def fake_ask(question: str):
        called["ask"] = True
        raise AssertionError("ask should not run")

    scores = evaluate_all(
        [_case(), _case(id="refuse-1", question="weather in Tokyo?", expected_source=None, should_refuse=True, must_contain=[])],
        retrieval_only=True,
        search_fn=fake_search,
        ask_fn=fake_ask,
    )
    assert called["ask"] is False
    assert scores[0].retrieval_pass is True
    assert scores[0].answer_pass is None
    assert scores[1].retrieval_pass is None
    counts = tally(scores)
    assert counts["failed"] == 0
    assert counts["answer_scored"] == 0
    report = format_report(scores)
    assert "PASS" in report
    assert "n/a" in report


def test_evaluate_case_records_search_errors() -> None:
    def boom(question: str):
        raise FileNotFoundError("Run ingest before search.")

    score = evaluate_case(_case(), retrieval_only=True, search_fn=boom)
    assert score.retrieval_pass is False
    assert "search error" in score.retrieval_detail


def test_load_cases_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="No questions"):
        load_cases(path)


def test_main_exits_1_when_a_case_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals.__main__ import main
    from evals.score import CaseScore

    monkeypatch.setattr("evals.__main__.load_cases", lambda: [_case()])
    monkeypatch.setattr(
        "evals.__main__.evaluate_all",
        lambda *args, **kwargs: [
            CaseScore(
                id="easy-1",
                question="What is Ask My Docs?",
                retrieval_pass=False,
                answer_pass=None,
                retrieval_detail="docs/what-ask-my-docs-is.md not in top-5",
                answer_detail="skipped (--retrieval-only)",
            )
        ],
    )

    assert main(["--retrieval-only"]) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "easy-1" in output


def test_main_ingest_then_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals.__main__ import main
    from evals.score import CaseScore

    called: dict = {}

    def fake_ingest(embedding_model=None):
        called["model"] = embedding_model

        class Result:
            message = "ingested"

        return Result()

    monkeypatch.setattr("evals.__main__.ingest", fake_ingest)
    monkeypatch.setattr("evals.__main__.load_cases", lambda: [_case()])
    monkeypatch.setattr(
        "evals.__main__.evaluate_all",
        lambda *args, **kwargs: [
            CaseScore(
                id="easy-1",
                question="What is Ask My Docs?",
                retrieval_pass=True,
                answer_pass=None,
                retrieval_detail="found docs/what-ask-my-docs-is.md in top-5",
                answer_detail="skipped (--retrieval-only)",
            )
        ],
    )

    assert main(["--ingest", "--embedding-model", "all-MiniLM-L6-v2", "--json"]) == 0
    assert called["model"] == "all-MiniLM-L6-v2"
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["failed"] == 0
    assert payload["cases"][0]["id"] == "easy-1"
