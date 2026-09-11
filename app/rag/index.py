"""Local JSON vector index: read, write, cosine similarity."""

import json
import math
from pathlib import Path

from app.config import PROJECT_ROOT, settings


def _resolved_index_path() -> Path:
    path = Path(settings.index_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _write_index(payload: dict) -> Path:
    path = _resolved_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)
    return path


def _read_index() -> dict:
    path = _resolved_index_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Vector index not found: {path}. Run ingest before search."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vector index is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or "chunks" not in payload:
        raise ValueError("Vector index is missing a chunks list. Re-run ingest.")
    chunks = payload["chunks"]
    if not isinstance(chunks, list):
        raise ValueError("Vector index is missing a chunks list. Re-run ingest.")
    return payload


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"query has {len(left)} dims, chunk has {len(right)}. Re-run ingest."
        )
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for x, y in zip(left, right, strict=True):
        dot += x * y
        norm_left += x * x
        norm_right += y * y
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))
