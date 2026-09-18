"""Embedding providers: OpenAI, Gemini, and local MiniLM."""

import httpx

from app.config import settings
from app.models import EmbeddingModelsResponse
from app.rag.http import raise_http_error

_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
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


def embedding_backend(model: str) -> str:
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
        raise_http_error(exc, "Embedding request")

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
        raise_http_error(exc, "Embedding request")
    return vectors


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _load_minilm()
    encoded = model.encode(texts, normalize_embeddings=True)
    return [[float(value) for value in row] for row in encoded]


def embed_texts(texts: list[str], *, for_query: bool = False) -> list[list[float]]:
    """Return one embedding vector per text using the configured model."""
    if not texts:
        return []
    backend = embedding_backend(settings.embedding_model)
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "gemini":
        return _embed_gemini(texts, for_query=for_query)
    return _embed_local(texts)
