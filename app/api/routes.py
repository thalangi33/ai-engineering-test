from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from app.models import (
    AskRequest,
    AskResponse,
    EmbeddingModelsResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag import pipeline

router = APIRouter()

_NOT_IMPLEMENTED = (
    "RAG is not implemented yet. Fill in the stubs in app/rag/pipeline.py."
)


def _not_implemented() -> HTTPException:
    return HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/embedding-models", response_model=EmbeddingModelsResponse)
def embedding_models() -> EmbeddingModelsResponse:
    return pipeline.list_embedding_models()


@router.post("/ingest", response_model=IngestResponse)
def ingest(body: Annotated[IngestRequest | None, Body()] = None) -> IngestResponse:
    try:
        embedding_model = body.embedding_model if body else None
        return pipeline.ingest(embedding_model=embedding_model)
    except NotImplementedError as exc:
        raise _not_implemented() from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    try:
        chunks = pipeline.search(body.question, top_k=body.top_k)
        return SearchResponse(question=body.question, chunks=chunks)
    except NotImplementedError as exc:
        raise _not_implemented() from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        return pipeline.ask(body.question)
    except NotImplementedError as exc:
        raise _not_implemented() from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
