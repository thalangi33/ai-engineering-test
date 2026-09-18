"""Prompt construction, refuse detection, and citations from chunk metadata."""

import re

from app.models import Citation

_SYSTEM_PROMPT = (
    "You are Ask My Docs. Answer using only the document excerpts in the user "
    "message. If they do not contain the answer, reply with exactly: I don't know. "
    "Do not use outside knowledge. Do not invent filenames, sources, or facts."
)
_REFUSE_RE = re.compile(r"^\s*i (don't|do not) know\.?\s*$", re.IGNORECASE)
_SNIPPET_CHARS = 240


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """Return chat messages. Answer only from context; otherwise say you don't know."""
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
        {"role": "system", "content": _SYSTEM_PROMPT},
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
