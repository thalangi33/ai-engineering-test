"""LLM extraction of entities, metadata filters, and time scope."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

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


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def chunk_matches(query: ExtractedInfo, chunk: dict) -> bool:
    """True when the chunk is compatible with the question extraction."""
    if query.entities:
        chunk_ents = {_norm(entity) for entity in chunk.get("entities") or [] if entity}
        query_ents = {_norm(entity) for entity in query.entities if entity}
        if not chunk_ents or chunk_ents.isdisjoint(query_ents):
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
