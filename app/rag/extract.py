"""LLM extraction of entities, metadata filters, and time scope."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.config import settings
from app.models import ExtractedInfo
from app.rag.chat import ask_llm

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_SYSTEM = (
    "Extract structured info. Reply with JSON only, no markdown.\n"
    "{\n"
    '  "intent": "fact" | "comparison" | "stats" | "timeline" | "other" | null,\n'
    '  "entities": ["..."],\n'
    '  "metadata_filters": {"doc_type": "player|team|app", "topic": "..."},\n'
    '  "time_scope": {"start": 2010, "end": 2014} | null\n'
    "}\n"
    "Rules:\n"
    "- intent: set only for questions; use null for chunks\n"
    "- entities: people, teams, product names; skip generics\n"
    "- metadata_filters: omit keys you are not sure about\n"
    "- time_scope: years the text is about; if a single year, set start and end to it; "
    "null if none\n"
    "- do not invent facts"
)

_CHUNK_BATCH_SYSTEM = (
    "Extract structured info for each chunk. Reply with JSON only, no markdown.\n"
    "{\n"
    '  "items": [\n'
    "    {\n"
    '      "id": 0,\n'
    '      "entities": ["..."],\n'
    '      "metadata_filters": {"doc_type": "player|team|app", "topic": "..."},\n'
    '      "time_scope": {"start": 2010, "end": 2014}\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- one item per chunk id; ids must match the input\n"
    "- entities: people, teams, product names; skip generics\n"
    "- metadata_filters: omit keys you are not sure about\n"
    "- time_scope: years the text is about; if a single year, set start and end to it; "
    "null if none\n"
    "- do not invent facts"
)
# Room for one chunk's entities, filters, and year range. Caps the batch reply.
_TOKENS_PER_CHUNK = 150


def chunk_extraction_text(chunk: dict) -> str:
    heading = (chunk.get("heading") or "").strip()
    body = (chunk.get("text") or "").strip()
    if heading:
        return f"{heading}\n\n{body}"
    return body


def _parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def extract_info(text: str, *, kind: str) -> ExtractedInfo:
    """Return structured info for a question or chunk. Empty on bad model output."""
    source = (text or "").strip()
    if not source:
        return ExtractedInfo()
    raw = ask_llm(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{kind}:\n{source}"},
        ]
    )
    try:
        info = ExtractedInfo.model_validate(_parse_json_object(raw))
    except (json.JSONDecodeError, ValueError, ValidationError, TypeError):
        return ExtractedInfo()
    if kind == "chunk":
        info.intent = None
    return info


def _batch_user_message(texts: list[str]) -> str:
    blocks = ["chunks:"]
    for index, text in enumerate(texts):
        blocks.append(f"{index}:\n{text}")
    return "\n\n".join(blocks)


def _chunk_info(payload: dict) -> ExtractedInfo:
    info = ExtractedInfo.model_validate(payload)
    info.intent = None
    return info


def _parse_chunk_batch(raw: str, count: int) -> list[ExtractedInfo] | None:
    """Align items by id. None when the reply is not a usable items array."""
    try:
        payload = _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    results = [ExtractedInfo() for _ in range(count)]
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0 or index >= count or index in seen:
            continue
        seen.add(index)
        try:
            results[index] = _chunk_info(item)
        except (ValidationError, TypeError, ValueError):
            results[index] = ExtractedInfo()
    if count > 0 and not seen:
        return None
    return results


def _extract_chunk_group(texts: list[str]) -> list[ExtractedInfo]:
    raw = ask_llm(
        [
            {"role": "system", "content": _CHUNK_BATCH_SYSTEM},
            {"role": "user", "content": _batch_user_message(texts)},
        ],
        max_tokens=_TOKENS_PER_CHUNK * len(texts),
    )
    parsed = _parse_chunk_batch(raw, len(texts))
    if parsed is None:
        return [extract_info(text, kind="chunk") for text in texts]
    return parsed


def extract_chunks(
    texts: list[str], *, batch_size: int | None = None
) -> list[ExtractedInfo]:
    """Extract chunk metadata in batches. One result per input, empty on failure."""
    size = settings.extract_batch_size if batch_size is None else batch_size
    if size < 1:
        raise ValueError("extract batch size must be at least 1")
    results = [ExtractedInfo() for _ in texts]
    pending: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        stripped = (text or "").strip()
        if stripped:
            pending.append((index, stripped))
    for start in range(0, len(pending), size):
        group = pending[start : start + size]
        extracted = _extract_chunk_group([text for _, text in group])
        for (index, _), info in zip(group, extracted, strict=True):
            results[index] = info
    return results


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def entities_alias(left: str, right: str) -> bool:
    """True when two entity names are the same or one is a whole-word form of the other."""
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 3:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", long) is not None


def distinct_entities(entities: list[str]) -> list[str]:
    """Drop blanks and names that alias an earlier entry, preserving order."""
    kept: list[str] = []
    for entity in entities:
        text = (entity or "").strip()
        if not text:
            continue
        if any(entities_alias(text, existing) for existing in kept):
            continue
        kept.append(text)
    return kept


def _entities_overlap(query_entities: list[str], chunk_entities: list[str]) -> bool:
    query_ents = [entity for entity in query_entities if entity and str(entity).strip()]
    chunk_ents = [entity for entity in chunk_entities if entity and str(entity).strip()]
    if not query_ents or not chunk_ents:
        return False
    return any(entities_alias(query, chunk) for query in query_ents for chunk in chunk_ents)


def chunk_matches(query: ExtractedInfo, chunk: dict) -> bool:
    """True when the chunk is compatible with the question extraction."""
    if query.entities and not _entities_overlap(query.entities, chunk.get("entities") or []):
        return False

    q_time = query.time_scope
    raw_time = chunk.get("time_scope") or {}
    if q_time and isinstance(raw_time, dict) and "start" in raw_time and "end" in raw_time:
        try:
            c_start = int(raw_time["start"])
            c_end = int(raw_time["end"])
        except (TypeError, ValueError):
            c_start = c_end = None
        if c_start is not None and (q_time.end < c_start or q_time.start > c_end):
            return False

    q_meta = query.metadata_filters
    c_meta = chunk.get("metadata_filters") or {}
    if not isinstance(c_meta, dict):
        c_meta = {}
    for key, value in q_meta.items():
        if key in c_meta and _norm(str(c_meta[key])) != _norm(str(value)):
            return False
    return True
