"""Prompt construction, refuse detection, and citations from chunk metadata."""

import re

from app.models import Citation

_FACT_SYSTEM = (
    "You are Ask My Docs. Answer using only the document excerpts in the user "
    "message. If they do not contain the answer, reply with exactly: I don't know. "
    "Do not use outside knowledge. Do not invent filenames, sources, or facts."
)
_STATS_SYSTEM = _FACT_SYSTEM + " Answer with the numbers in the excerpts."
_TIMELINE_SYSTEM = _FACT_SYSTEM + " Order events by years written in the excerpts."
_COMPARISON_SYSTEM = (
    "You are Ask My Docs. The user is comparing things named in the question. "
    "Use only the document excerpts in the user message. "
    "Line up facts that the excerpts state for each side (teams, titles, awards, years). "
    "If both sides are present, cover both. "
    "If one side has no excerpt, say so and answer only the side you have. "
    "Do not pick a winner unless an excerpt states one. "
    "Do not use outside knowledge. Do not invent filenames, sources, or facts. "
    "If neither side is supported, reply with exactly: I don't know."
)
_REFUSE_RE = re.compile(r"^\s*i (don't|do not) know\.?\s*$", re.IGNORECASE)
_SNIPPET_CHARS = 240


def system_prompt_for_intent(intent: str | None) -> str:
    """Return the system text for a question intent. Unknown intents use the fact prompt."""
    if intent == "comparison":
        return _COMPARISON_SYSTEM
    if intent == "stats":
        return _STATS_SYSTEM
    if intent == "timeline":
        return _TIMELINE_SYSTEM
    return _FACT_SYSTEM


def build_prompt(
    question: str, chunks: list[dict], intent: str | None = None
) -> list[dict]:
    """Return chat messages. The system text follows the question intent."""
    question = (question or "").strip()
    parts = ["Question:", question or "(empty)", "", "Document excerpts:"]
    if not chunks:
        parts.append("(none)")
    else:
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.get("source") or "unknown"
            heading = chunk.get("heading")
            header = f"[{index}] {source}"
            if heading:
                header += f" — {heading}"
            parts.append(header)
            parts.append((chunk.get("text") or "").strip())
            parts.append("")
    return [
        {"role": "system", "content": system_prompt_for_intent(intent)},
        {"role": "user", "content": "\n".join(parts).strip()},
    ]


def citation_snippet(text: str) -> str | None:
    snippet = " ".join((text or "").split())
    if not snippet:
        return None
    if len(snippet) > _SNIPPET_CHARS:
        return snippet[: _SNIPPET_CHARS - 3] + "..."
    return snippet


def citations_from_chunks(chunks: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        source = chunk.get("source")
        if not source or source in seen:
            continue
        seen.add(source)
        citations.append(
            Citation(source=source, snippet=citation_snippet(chunk.get("text") or ""))
        )
    return citations


def looks_like_refuse(answer: str) -> bool:
    return bool(_REFUSE_RE.match(answer or ""))
