"""RAG pipeline stubs.

Implement these yourself, in this order:

1. load_documents  — read text from settings.docs_dir (done)
2. chunk_text      — split documents into overlapping chunks with metadata (done)
3. ingest          — embed chunks and store them locally (done)
4. search          — embed the question, return top_k chunks
5. build_prompt    — system instructions + retrieved context + question
6. ask_llm         — call the provider; temperature 0
7. ask             — search → prompt → LLM → citations from chunk metadata

Do not skip search debugging: print retrieved chunks before wiring the LLM.
Citations must come from retrieved chunk metadata, not from the model inventing filenames.
"""

import json
import re
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT, settings
from app.models import AskResponse, IngestResponse

_ALLOWED_SUFFIXES = {".md", ".txt"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Rough char stand-in for tokens (~4 chars/token): ~500 target, ~800 max, ~80 overlap.
_TARGET_CHARS = 2000
_MAX_CHARS = 3200
_OVERLAP_CHARS = 320
_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_EMBED_BATCH_SIZE = 64


def load_documents(docs_dir: Path) -> list[dict]:
    """Return a list of {path, text} dicts for markdown/text files under docs_dir."""
    docs_dir = Path(docs_dir).expanduser().resolve()
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            stored_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            stored_path = path.as_posix()
        documents.append({"path": stored_path, "text": text})
    return documents


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown into (heading, section_text) pairs. Heading may be None."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal heading, buf
        body = "".join(buf).strip()
        if body:
            sections.append((heading, body))
        buf = []

    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line.rstrip("\n"))
        if match:
            flush()
            heading = match.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return sections


def _overlap_tail(text: str) -> str:
    if len(text) <= _OVERLAP_CHARS:
        return text
    tail = text[-_OVERLAP_CHARS:]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def _split_long_paragraph(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + _TARGET_CHARS, length)
        if end < length:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        start = end
        while start < length and text[start] == " ":
            start += 1
    return parts or [text]


def _pack_paragraphs(paragraphs: list[str]) -> list[str]:
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > _MAX_CHARS:
            units.extend(_split_long_paragraph(paragraph))
        else:
            units.append(paragraph)

    packed: list[str] = []
    buf = ""
    for unit in units:
        candidate = f"{buf}\n\n{unit}" if buf else unit
        if buf and len(candidate) > _TARGET_CHARS:
            packed.append(buf)
            tail = _overlap_tail(buf)
            buf = f"{tail}\n\n{unit}" if tail else unit
        else:
            buf = candidate
    if buf:
        packed.append(buf)
    return packed


def chunk_text(documents: list[dict]) -> list[dict]:
    """Return chunks with text plus metadata (source, chunk_index, heading)."""
    chunks: list[dict] = []
    for document in documents:
        text = (document.get("text") or "").strip()
        if not text:
            continue
        source = document["path"]
        chunk_index = 0
        for heading, section in _split_into_sections(text):
            paragraphs = [
                part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()
            ]
            if not paragraphs:
                continue
            for part in _pack_paragraphs(paragraphs):
                chunks.append(
                    {
                        "text": part,
                        "source": source,
                        "chunk_index": chunk_index,
                        "heading": heading,
                    }
                )
                chunk_index += 1
    return chunks


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


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text using the configured model."""
    if not texts:
        return []
    api_key = (settings.llm_api_key or "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is required to embed documents.")

    vectors: list[list[float] | None] = [None] * len(texts)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), _EMBED_BATCH_SIZE):
                batch = texts[start : start + _EMBED_BATCH_SIZE]
                response = client.post(
                    _EMBEDDINGS_URL,
                    headers=headers,
                    json={"model": settings.embedding_model, "input": batch},
                )
                response.raise_for_status()
                for item in response.json()["data"]:
                    vectors[start + item["index"]] = item["embedding"]
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or exc.response.reason_phrase
        raise RuntimeError(
            f"Embedding request failed ({exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding response was missing one or more vectors.")
    return [vector for vector in vectors if vector is not None]


def ingest() -> IngestResponse:
    """Load, chunk, embed, and persist the vector index."""
    documents = load_documents(settings.docs_dir)
    chunks = chunk_text(documents)
    embeddings = _embed_texts([chunk["text"] for chunk in chunks])
    stored = [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "heading": chunk["heading"],
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    _write_index(
        {
            "embedding_model": settings.embedding_model,
            "chunks": stored,
        }
    )
    return IngestResponse(
        status="ok",
        message=(
            f"Ingested {len(documents)} documents into {len(stored)} chunks."
            if stored
            else "No chunks to ingest."
        ),
        document_count=len(documents),
        chunk_count=len(stored),
    )


def search(question: str) -> list[dict]:
    """Return the top_k most similar chunks for the question."""
    raise NotImplementedError("Implement retrieval.")


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """Return chat messages. Answer only from context; otherwise say you don't know."""
    raise NotImplementedError("Implement prompt assembly.")


def ask_llm(messages: list[dict]) -> str:
    """Send messages to the LLM and return the assistant text."""
    raise NotImplementedError("Implement the LLM call.")


def ask(question: str) -> AskResponse:
    """Retrieve, prompt, call the LLM, and attach citations from chunk metadata."""
    raise NotImplementedError("Implement ask orchestration.")
