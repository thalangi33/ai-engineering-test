"""Detect comparison questions and merge per-side retrieval hits."""

import re

_PER_SOURCE_CAP = 2
_COMPARE_CUE_RE = re.compile(
    r"\b(?:vs\.?|versus|compar(?:e|ed|ing|ison)|difference|both|"
    r"more|most|less|least|fewer|than)\b",
    re.IGNORECASE,
)
_WHO_MORE_RE = re.compile(
    r"(?:who|which(?:\s+\w+)?)\s+has\s+(?:the\s+)?"
    r"(?:more|most|less|least|fewer)\s+"
    r"(?P<metric>.+?)[,\s]+(?P<a>.+?)\s+or\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_MORE_THAN_RE = re.compile(
    r"does\s+(?P<a>.+?)\s+have\s+(?:the\s+)?"
    r"(?:more|most|less|least|fewer)\s+"
    r"(?P<metric>.+?)\s+than\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"^(?:please\s+)?compar(?:e|ing)\s+(?P<a>.+?)\s+"
    r"(?:and|to|with|vs\.?|versus)\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_VS_RE = re.compile(
    r"^(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_DIFF_RE = re.compile(
    r"difference\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s*\??\s*$",
    re.IGNORECASE,
)


def _clean_span(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.strip("?.!,;:\"'()[]")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _valid_side(text: str) -> bool:
    return bool(text) and len(text) <= 80 and "?" not in text


def _parse_comparison(question: str) -> tuple[list[str], str] | None:
    """Return (sides, metric) for a comparison question, or None."""
    if not _COMPARE_CUE_RE.search(question):
        return None
    for pattern in (
        _WHO_MORE_RE,
        _MORE_THAN_RE,
        _COMPARE_RE,
        _DIFF_RE,
        _VS_RE,
    ):
        match = pattern.search(question)
        if not match:
            continue
        groups = match.groupdict()
        left = _clean_span(groups.get("a") or "")
        right = _clean_span(groups.get("b") or "")
        metric = _clean_span(groups.get("metric") or "")
        if not _valid_side(left) or not _valid_side(right):
            continue
        if left.lower() == right.lower():
            continue
        return [left, right], metric
    return None


def _comparison_queries(question: str) -> list[str] | None:
    """Return per-side search queries, or None when this is not a comparison."""
    parsed = _parse_comparison(question)
    if parsed is None:
        return None
    sides, metric = parsed
    if metric:
        return [f"{side} {metric}" for side in sides]
    return list(sides)


def _merge_comparison_chunks(groups: list[list[dict]]) -> list[dict]:
    """Interleave per-side hits and cap how many chunks each source may contribute."""
    seen: set[tuple[str, int]] = set()
    source_counts: dict[str, int] = {}
    merged: list[dict] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index >= len(group):
                continue
            chunk = group[index]
            source = str(chunk.get("source") or "")
            raw_index = chunk.get("chunk_index")
            chunk_index = int(raw_index) if raw_index is not None else index
            key = (source, chunk_index)
            if key in seen:
                continue
            if source_counts.get(source, 0) >= _PER_SOURCE_CAP:
                continue
            seen.add(key)
            source_counts[source] = source_counts.get(source, 0) + 1
            merged.append(chunk)
    return merged
