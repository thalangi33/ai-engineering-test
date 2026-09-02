"""Score retrieval and answers separately for evals/questions.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import PROJECT_ROOT
from app.rag.pipeline import _looks_like_refuse

DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "questions.json"


@dataclass
class CaseScore:
    id: str
    question: str
    retrieval_pass: bool | None
    answer_pass: bool | None
    retrieval_detail: str
    answer_detail: str
    retrieved_sources: list[str] = field(default_factory=list)
    answer: str | None = None
    citations: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.retrieval_pass is False or self.answer_pass is False


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    cases_path = Path(path) if path is not None else DEFAULT_CASES_PATH
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {cases_path}")
    return questions


def _normalize_source(source: str) -> str:
    return source.replace("\\", "/").strip().lstrip("./")


def source_in(expected: str | None, sources: list[str]) -> bool:
    if not expected:
        return False
    wanted = _normalize_source(expected)
    if not wanted:
        return False
    for source in sources:
        got = _normalize_source(source or "")
        if not got:
            continue
        if got == wanted or got.endswith("/" + wanted) or wanted.endswith("/" + got):
            return True
    return False


def _citation_sources(citations: Any) -> list[str]:
    sources: list[str] = []
    for citation in citations or []:
        if isinstance(citation, str):
            sources.append(citation)
        else:
            sources.append(getattr(citation, "source", None) or "")
    return sources


def missing_phrases(text: str, phrases: list[str] | None) -> list[str]:
    haystack = (text or "").lower()
    missing: list[str] = []
    for phrase in phrases or []:
        needle = (phrase or "").strip().lower()
        if needle and needle not in haystack:
            missing.append(phrase)
    return missing


def score_retrieval(
    case: dict[str, Any], chunks: list[dict[str, Any]]
) -> tuple[bool | None, str]:
    expected = case.get("expected_source")
    sources = [str(chunk.get("source") or "") for chunk in chunks]
    if case.get("should_refuse") or not expected:
        return None, "n/a (no expected source)"
    if source_in(expected, sources):
        return True, f"found {expected} in top-{len(chunks)}"
    listed = ", ".join(sources) if sources else "(none)"
    return False, f"{expected} not in top-{len(chunks)}: {listed}"


def score_answer(
    case: dict[str, Any],
    answer: str,
    citations: Any,
) -> tuple[bool | None, str]:
    sources = _citation_sources(citations)
    refused = _looks_like_refuse(answer)
    if case.get("should_refuse"):
        if refused and not sources:
            return True, "refused with no citations"
        if not refused:
            return False, "should refuse / say I don't know"
        return False, "refuse attached citations: " + ", ".join(sources)

    if refused:
        return False, "refused but the answer is in the docs"
    if not (answer or "").strip():
        return False, "empty answer"
    expected = case.get("expected_source")
    if expected and not source_in(expected, sources):
        listed = ", ".join(sources) if sources else "(none)"
        return False, f"missing citation {expected}: {listed}"
    missing = missing_phrases(answer, case.get("must_contain") or [])
    if missing:
        return False, "fluent-but-wrong; missing: " + ", ".join(missing)
    return True, "grounded answer with required phrases"


def evaluate_case(
    case: dict[str, Any],
    *,
    retrieval_only: bool = False,
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    ask_fn: Callable[..., Any] | None = None,
) -> CaseScore:
    from app.rag.pipeline import ask, search

    search_fn = search_fn or search
    ask_fn = ask_fn or ask
    question = case["question"]
    retrieved_sources: list[str] = []
    chunks: list[dict[str, Any]] = []
    try:
        chunks = search_fn(question)
        retrieved_sources = [str(chunk.get("source") or "") for chunk in chunks]
        retrieval_pass, retrieval_detail = score_retrieval(case, chunks)
    except Exception as exc:  # noqa: BLE001 — per-case isolation for the suite
        retrieval_pass, retrieval_detail = False, f"search error: {exc}"

    if retrieval_only:
        return CaseScore(
            id=case["id"],
            question=question,
            retrieval_pass=retrieval_pass,
            answer_pass=None,
            retrieval_detail=retrieval_detail,
            answer_detail="skipped (--retrieval-only)",
            retrieved_sources=retrieved_sources,
        )

    try:
        result = ask_fn(question)
        answer = getattr(result, "answer", None) or ""
        citations = getattr(result, "citations", None) or []
        answer_pass, answer_detail = score_answer(case, answer, citations)
        citation_sources = _citation_sources(citations)
    except Exception as exc:  # noqa: BLE001 — per-case isolation for the suite
        answer = None
        citation_sources = []
        answer_pass, answer_detail = False, f"ask error: {exc}"

    return CaseScore(
        id=case["id"],
        question=question,
        retrieval_pass=retrieval_pass,
        answer_pass=answer_pass,
        retrieval_detail=retrieval_detail,
        answer_detail=answer_detail,
        retrieved_sources=retrieved_sources,
        answer=answer,
        citations=citation_sources,
    )


def evaluate_all(
    cases: list[dict[str, Any]] | None = None,
    *,
    retrieval_only: bool = False,
    search_fn: Callable[..., list[dict[str, Any]]] | None = None,
    ask_fn: Callable[..., Any] | None = None,
) -> list[CaseScore]:
    if cases is None:
        cases = load_cases()
    return [
        evaluate_case(
            case,
            retrieval_only=retrieval_only,
            search_fn=search_fn,
            ask_fn=ask_fn,
        )
        for case in cases
    ]


def tally(scores: list[CaseScore]) -> dict[str, int]:
    retrieval_scored = sum(score.retrieval_pass is not None for score in scores)
    retrieval_passed = sum(score.retrieval_pass is True for score in scores)
    answer_scored = sum(score.answer_pass is not None for score in scores)
    answer_passed = sum(score.answer_pass is True for score in scores)
    failed = sum(score.failed for score in scores)
    return {
        "total": len(scores),
        "failed": failed,
        "retrieval_scored": retrieval_scored,
        "retrieval_passed": retrieval_passed,
        "answer_scored": answer_scored,
        "answer_passed": answer_passed,
    }


def _mark(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def format_report(scores: list[CaseScore]) -> str:
    counts = tally(scores)
    rows = [
        f"{'id':<14} {'retrieval':<10} {'answer':<10} detail",
        f"{'-' * 14} {'-' * 10} {'-' * 10} {'-' * 40}",
    ]
    for score in scores:
        detail = score.retrieval_detail if score.retrieval_pass is False else score.answer_detail
        if score.retrieval_pass is False and score.answer_pass is False:
            detail = f"{score.retrieval_detail}; {score.answer_detail}"
        elif score.retrieval_pass is False:
            detail = score.retrieval_detail
        rows.append(
            f"{score.id:<14} {_mark(score.retrieval_pass):<10} "
            f"{_mark(score.answer_pass):<10} {detail}"
        )
    rows.append("")
    rows.append(
        f"retrieval {counts['retrieval_passed']}/{counts['retrieval_scored']} passed   "
        f"answer {counts['answer_passed']}/{counts['answer_scored']} passed"
    )
    if counts["failed"]:
        rows.append(f"{counts['failed']} case(s) failed")
    else:
        rows.append("all scored checks passed")
    return "\n".join(rows)
