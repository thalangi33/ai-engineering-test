"""RAG pipeline.

1. load_documents  — read text from settings.docs_dir (done)
2. chunk_text      — split documents into overlapping chunks with metadata (done)
3. ingest          — embed chunks and store them locally (done)
4. search          — embed the question, return top_k chunks (done)
5. build_prompt    — system instructions + retrieved context + question (done)
6. ask_llm         — call the provider; temperature 0 (done)
7. ask             — search → prompt → LLM → citations from chunk metadata (done)

Search still prints retrieved chunks so retrieval can be checked before trusting answers.
Citations come from retrieved chunk metadata, not from the model inventing filenames.
"""

import json
import math
import re
import time
from pathlib import Path
from typing import NoReturn

import httpx

from app.config import PROJECT_ROOT, settings
from app.models import Citation, EmbeddingModelsResponse, AskResponse, IngestResponse

_ALLOWED_SUFFIXES = {".md", ".txt"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Rough char stand-in for tokens (~4 chars/token): ~500 target, ~800 max, ~80 overlap.
_TARGET_CHARS = 2000
_MAX_CHARS = 3200
_OVERLAP_CHARS = 320
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_SYSTEM_PROMPT = (
    "You are Ask My Docs. Answer using only the document excerpts in the user "
    "message. If they do not contain the answer, reply with exactly: I don't know. "
    "Do not use outside knowledge. Do not invent filenames, sources, or facts."
)
_REFUSE_RE = re.compile(r"^\s*i (don't|do not) know\.?\s*$", re.IGNORECASE)
_SNIPPET_CHARS = 240
_OPENAI_EMBED_BATCH_SIZE = 64
_GEMINI_EMBED_BATCH_SIZE = 100
_OPENAI_EMBED_MODELS = {"text-embedding-3-small"}
_GEMINI_EMBED_MODELS = {"gemini-embedding-001"}
_LOCAL_EMBED_MODELS = {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}
_MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_MODEL_CHOICES = (
    ("text-embedding-3-small", "OpenAI · text-embedding-3-small"),
    ("gemini-embedding-001", "Gemini · gemini-embedding-001"),
    ("all-MiniLM-L6-v2", "Local · all-MiniLM-L6-v2"),
)

_minilm_model = None


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


def _print_search_results(question: str, results: list[dict]) -> None:
    print(f"[search] {question!r} → {len(results)} chunk(s)")
    for index, chunk in enumerate(results, start=1):
        heading = f" / {chunk['heading']}" if chunk.get("heading") else ""
        snippet = " ".join(chunk["text"].split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(
            f"  {index}. {chunk['score']:.3f}  {chunk['source']}{heading}\n"
            f"     {snippet}"
        )


def _embedding_backend(model: str) -> str:
    if model in _OPENAI_EMBED_MODELS:
        return "openai"
    if model in _GEMINI_EMBED_MODELS:
        return "gemini"
    if model in _LOCAL_EMBED_MODELS:
        return "local"
    raise ValueError(
        f"Unsupported embedding model {model!r}. Choose one of: "
        "text-embedding-3-small, gemini-embedding-001, all-MiniLM-L6-v2."
    )


def list_embedding_models() -> EmbeddingModelsResponse:
    selected = settings.embedding_model
    if selected in _LOCAL_EMBED_MODELS:
        selected = "all-MiniLM-L6-v2"
    return EmbeddingModelsResponse(
        models=[
            {"id": model_id, "label": label}
            for model_id, label in _EMBEDDING_MODEL_CHOICES
        ],
        selected=selected,
    )


def _raise_http_error(exc: httpx.HTTPError, what: str) -> NoReturn:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text.strip() or exc.response.reason_phrase
        raise RuntimeError(
            f"{what} failed ({exc.response.status_code}): {detail}"
        ) from exc
    raise RuntimeError(f"{what} failed: {exc}") from exc


def _raise_embed_http_error(exc: httpx.HTTPError) -> NoReturn:
    _raise_http_error(exc, "Embedding request")


def _embed_openai(texts: list[str]) -> list[list[float]]:
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
            for start in range(0, len(texts), _OPENAI_EMBED_BATCH_SIZE):
                batch = texts[start : start + _OPENAI_EMBED_BATCH_SIZE]
                response = client.post(
                    _OPENAI_EMBEDDINGS_URL,
                    headers=headers,
                    json={"model": settings.embedding_model, "input": batch},
                )
                response.raise_for_status()
                for item in response.json()["data"]:
                    vectors[start + item["index"]] = item["embedding"]
    except httpx.HTTPError as exc:
        _raise_embed_http_error(exc)

    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding response was missing one or more vectors.")
    return [vector for vector in vectors if vector is not None]


def _embed_gemini(texts: list[str], *, for_query: bool = False) -> list[list[float]]:
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to embed documents.")

    model = settings.embedding_model
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:batchEmbedContents"
    )
    task_type = "RETRIEVAL_QUERY" if for_query else "RETRIEVAL_DOCUMENT"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    vectors: list[list[float]] = []
    try:
        with httpx.Client(timeout=60.0) as client:
            for start in range(0, len(texts), _GEMINI_EMBED_BATCH_SIZE):
                batch = texts[start : start + _GEMINI_EMBED_BATCH_SIZE]
                response = client.post(
                    url,
                    headers=headers,
                    json={
                        "requests": [
                            {
                                "model": f"models/{model}",
                                "taskType": task_type,
                                "content": {"parts": [{"text": text}]},
                            }
                            for text in batch
                        ]
                    },
                )
                response.raise_for_status()
                items = response.json().get("embeddings") or []
                if len(items) != len(batch):
                    raise RuntimeError(
                        "Embedding response was missing one or more vectors."
                    )
                for item in items:
                    values = item.get("values")
                    if not values:
                        raise RuntimeError(
                            "Embedding response was missing one or more vectors."
                        )
                    vectors.append(values)
    except httpx.HTTPError as exc:
        _raise_embed_http_error(exc)
    return vectors


def _load_minilm():
    global _minilm_model
    if _minilm_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for all-MiniLM-L6-v2. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _minilm_model = SentenceTransformer(_MINILM_MODEL_ID)
    return _minilm_model


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _load_minilm()
    encoded = model.encode(texts, normalize_embeddings=True)
    return [[float(value) for value in row] for row in encoded]


def _embed_texts(
    texts: list[str], *, for_query: bool = False
) -> list[list[float]]:
    """Return one embedding vector per text using the configured model."""
    if not texts:
        return []
    backend = _embedding_backend(settings.embedding_model)
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "gemini":
        return _embed_gemini(texts, for_query=for_query)
    return _embed_local(texts)


def ingest(embedding_model: str | None = None) -> IngestResponse:
    """Load, chunk, embed, and persist the vector index."""
    model = settings.embedding_model
    if embedding_model is not None:
        embedding_model = embedding_model.strip()
        if embedding_model:
            _embedding_backend(embedding_model)
            model = embedding_model
    documents = load_documents(settings.docs_dir)
    chunks = chunk_text(documents)
    previous_model = settings.embedding_model
    settings.embedding_model = model
    try:
        embeddings = _embed_texts([chunk["text"] for chunk in chunks])
    finally:
        settings.embedding_model = previous_model
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
            "embedding_model": model,
            "chunks": stored,
        }
    )
    return IngestResponse(
        status="ok",
        message=(
            f"Ingested {len(documents)} documents into {len(stored)} chunks "
            f"with {model}."
            if stored
            else "No chunks to ingest."
        ),
        document_count=len(documents),
        chunk_count=len(stored),
        embedding_model=model,
    )


def search(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k most similar chunks for the question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")
    k = settings.top_k if top_k is None else top_k
    if k < 1:
        raise ValueError("top_k must be at least 1.")

    payload = _read_index()
    stored_chunks = payload["chunks"]
    if not stored_chunks:
        _print_search_results(question, [])
        return []

    model = payload.get("embedding_model") or settings.embedding_model
    previous_model = settings.embedding_model
    settings.embedding_model = model
    try:
        query_vectors = _embed_texts([question], for_query=True)
    finally:
        settings.embedding_model = previous_model
    query_vector = query_vectors[0]

    scored: list[tuple[float, dict]] = []
    for chunk in stored_chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        score = _cosine_similarity(query_vector, embedding)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "heading": chunk.get("heading"),
            "score": score,
        }
        for score, chunk in scored[:k]
    ]
    _print_search_results(question, results)
    return results


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


def _citation_snippet(text: str) -> str | None:
    snippet = " ".join((text or "").split())
    if not snippet:
        return None
    if len(snippet) > _SNIPPET_CHARS:
        return snippet[: _SNIPPET_CHARS - 3] + "..."
    return snippet


def _citations_from_chunks(chunks: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        source = chunk.get("source")
        if not source or source in seen:
            continue
        seen.add(source)
        citations.append(
            Citation(source=source, snippet=_citation_snippet(chunk.get("text") or ""))
        )
    return citations


def _looks_like_refuse(answer: str) -> bool:
    return bool(_REFUSE_RE.match(answer or ""))


def _print_ask_result(
    question: str,
    answer: str,
    citations: list[Citation],
    elapsed_ms: float,
    usage: dict | None = None,
) -> None:
    sources = ", ".join(citation.source for citation in citations) or "(none)"
    usage_part = ""
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            usage_part = f" tokens={prompt_tokens}+{completion_tokens}"
    print(
        f"[ask] {question!r} → {elapsed_ms:.0f}ms{usage_part}\n"
        f"  citations: {sources}\n"
        f"  answer: {answer}"
    )


def ask_llm(messages: list[dict]) -> str:
    """Send messages to the LLM and return the assistant text."""
    return _ask_llm(messages)[0]


def _ask_llm(messages: list[dict]) -> tuple[str, dict | None]:
    api_key = (settings.llm_api_key or "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is required to ask the model.")
    if not messages:
        raise ValueError("messages must not be empty.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "temperature": settings.temperature,
        "messages": messages,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(_OPENAI_CHAT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        _raise_http_error(exc, "LLM request")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response was missing choices.")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("LLM response was missing text.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return content, usage


def ask(question: str) -> AskResponse:
    """Retrieve, prompt, call the LLM, and attach citations from chunk metadata."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    started = time.perf_counter()
    chunks = search(question)
    if not chunks:
        answer = "I don't know"
        elapsed_ms = (time.perf_counter() - started) * 1000
        _print_ask_result(question, answer, [], elapsed_ms)
        return AskResponse(answer=answer, citations=[])

    messages = build_prompt(question, chunks)
    answer, usage = _ask_llm(messages)
    citations = [] if _looks_like_refuse(answer) else _citations_from_chunks(chunks)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _print_ask_result(question, answer, citations, elapsed_ms, usage)
    return AskResponse(answer=answer, citations=citations)
