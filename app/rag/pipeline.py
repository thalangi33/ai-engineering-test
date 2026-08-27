"""RAG pipeline stubs.

Implement these yourself, in this order:

1. load_documents  — read text from settings.docs_dir
2. chunk_text      — split documents into overlapping chunks with metadata
3. ingest          — embed chunks and store them locally
4. search          — embed the question, return top_k chunks
5. build_prompt    — system instructions + retrieved context + question
6. ask_llm         — call the provider; temperature 0
7. ask             — search → prompt → LLM → citations from chunk metadata

Do not skip search debugging: print retrieved chunks before wiring the LLM.
Citations must come from retrieved chunk metadata, not from the model inventing filenames.
"""

from pathlib import Path

from app.models import AskResponse, IngestResponse


def load_documents(docs_dir: Path) -> list[dict]:
    """Return a list of {path, text} dicts for files under docs_dir."""
    raise NotImplementedError("Implement document loading.")


def chunk_text(documents: list[dict]) -> list[dict]:
    """Return chunks with text plus metadata (source, chunk_index, heading)."""
    raise NotImplementedError("Implement chunking.")


def ingest() -> IngestResponse:
    """Load, chunk, embed, and persist the vector index."""
    raise NotImplementedError("Implement ingest.")


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
