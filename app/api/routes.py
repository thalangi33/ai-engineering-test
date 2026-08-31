from fastapi import APIRouter, HTTPException

from app.models import AskRequest, AskResponse, IngestResponse
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


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        return pipeline.ingest()
    except NotImplementedError as exc:
        raise _not_implemented() from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        return pipeline.ask(body.question)
    except NotImplementedError as exc:
        raise _not_implemented() from exc
