"""RAG pipeline orchestrator.

Sibling modules own the steps:

- documents  — load_documents, chunk_text
- embeddings — embed chunks and queries
- index      — local JSON store and cosine similarity
- compare    — split comparison questions and merge hits
- prompt     — grounded messages, refuse, citations
- chat       — Gemini, Ollama, Groq, DeepSeek

ingest / search / ask stay here so routes and tests keep importing this module.
Search still prints retrieved chunks. Citations come from chunk metadata.
"""

import time

import httpx

from app.config import settings
from app.models import AskResponse, Citation, IngestResponse
from app.rag.chat import _ask_llm, _chat_backend, ask_llm, list_chat_models
from app.rag.compare import _comparison_queries, _merge_comparison_chunks
from app.rag.documents import chunk_text, load_documents
from app.rag.embeddings import (
    _OPENAI_EMBEDDINGS_URL,
    _embed_texts,
    _embedding_backend,
    list_embedding_models,
)
from app.rag.index import _cosine_similarity, _read_index, _write_index
from app.rag.prompt import _citations_from_chunks, _looks_like_refuse, build_prompt

# Re-exported for tests and evals that import these names from pipeline.
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


def _search_similar(question: str, top_k: int) -> list[dict]:
    payload = _read_index()
    stored_chunks = payload["chunks"]
    if not stored_chunks:
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

    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "heading": chunk.get("heading"),
            "score": score,
        }
        for score, chunk in scored[:top_k]
    ]


def search(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k most similar chunks for the question.

    Comparison questions search each named side separately, then merge hits
    with a per-source cap so one document cannot fill the window.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")
    k = settings.top_k if top_k is None else top_k
    if k < 1:
        raise ValueError("top_k must be at least 1.")

    queries = _comparison_queries(question)
    if queries:
        results = _merge_comparison_chunks(
            [_search_similar(query, k) for query in queries]
        )
    else:
        results = _search_similar(question, k)
    _print_search_results(question, results)
    return results


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

    previous_model = settings.llm_model
    settings.llm_model = model
    try:
        started = time.perf_counter()
        chunks = search(question)
        if not chunks:
            answer = "I don't know"
            elapsed_ms = (time.perf_counter() - started) * 1000
            _print_ask_result(question, answer, [], elapsed_ms)
            return AskResponse(answer=answer, citations=[], llm_model=model)

        compare = _comparison_queries(question) is not None
        messages = build_prompt(question, chunks, compare=compare)
        answer, usage = _ask_llm(messages)
        citations = [] if _looks_like_refuse(answer) else _citations_from_chunks(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _print_ask_result(question, answer, citations, elapsed_ms, usage)
        return AskResponse(answer=answer, citations=citations, llm_model=model)
    finally:
        settings.llm_model = previous_model
