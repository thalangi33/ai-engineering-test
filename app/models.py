from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    embedding_model: str | None = None


class IngestResponse(BaseModel):
    status: str
    message: str
    document_count: int | None = None
    chunk_count: int | None = None
    embedding_model: str | None = None


class EmbeddingModelOption(BaseModel):
    id: str
    label: str


class EmbeddingModelsResponse(BaseModel):
    models: list[EmbeddingModelOption]
    selected: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    source: str
    snippet: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
