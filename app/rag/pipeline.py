"""RAG pipeline orchestration.

1. load_documents  — read text from settings.docs_dir (app.rag.documents)
2. chunk_text      — split documents into overlapping chunks with metadata
3. ingest          — embed chunks and store them locally
4. search          — embed the question, return top_k chunks
5. build_prompt    — system instructions + retrieved context + question (app.rag.prompt)
6. ask_llm         — call the selected provider; temperature 0 (app.rag.chat)
7. ask             — search → prompt → LLM → citations from chunk metadata

Search still prints retrieved chunks so retrieval can be checked before trusting answers.
Citations come from retrieved chunk metadata, not from the model inventing filenames.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import httpx  # noqa: F401 — tests patch pipeline.httpx.Client

from app.config import settings
from app.models import AskResponse, Citation, IngestResponse
from app.rag.chat import ask_llm, ask_llm_with_usage, chat_backend, list_chat_models
from app.rag.documents import chunk_text, load_documents
from app.rag.embeddings import (
    _OPENAI_EMBEDDINGS_URL,
    embed_texts,
    embedding_backend,
    list_embedding_models,
)
from app.rag.index import cosine_similarity, read_index, write_index
from app.rag.prompt import build_prompt, citations_from_chunks, looks_like_refuse

# Compatibility aliases: tests and evals import these from pipeline.
_ask_llm = ask_llm_with_usage
_embed_texts = embed_texts
_embedding_backend = embedding_backend
_chat_backend = chat_backend
_read_index = read_index
_write_index = write_index
_cosine_similarity = cosine_similarity
_looks_like_refuse = looks_like_refuse
_citations_from_chunks = citations_from_chunks

__all__ = [
    "ask",
    "ask_llm",
    "build_prompt",
    "chunk_text",
    "ingest",
    "list_chat_models",
    "list_embedding_models",
    "load_documents",
    "search",
]


@contextmanager
def _using_setting(name: str, value: str) -> Iterator[None]:
    previous = getattr(settings, name)
    setattr(settings, name, value)
    try:
        yield
    finally:
        setattr(settings, name, previous)


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
    with _using_setting("embedding_model", model):
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
    with _using_setting("embedding_model", model):
        query_vectors = _embed_texts([question], for_query=True)
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


def ask(question: str, llm_model: str | None = None) -> AskResponse:
    """Retrieve, prompt, call the LLM, and attach citations from chunk metadata."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    model = settings.llm_model
    if llm_model is not None:
        llm_model = llm_model.strip()
        if llm_model:
            _chat_backend(llm_model)
            model = llm_model

    with _using_setting("llm_model", model):
        started = time.perf_counter()
        chunks = search(question)
        if not chunks:
            answer = "I don't know"
            elapsed_ms = (time.perf_counter() - started) * 1000
            _print_ask_result(question, answer, [], elapsed_ms)
            return AskResponse(answer=answer, citations=[], llm_model=model)

        messages = build_prompt(question, chunks)
        answer, usage = _ask_llm(messages)
        citations = [] if _looks_like_refuse(answer) else _citations_from_chunks(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _print_ask_result(question, answer, citations, elapsed_ms, usage)
        return AskResponse(answer=answer, citations=citations, llm_model=model)
