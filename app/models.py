from typing import Literal

from pydantic import BaseModel, Field, model_validator

Intent = Literal["fact", "comparison", "stats", "timeline", "other"]


class TimeScope(BaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def order_years(self) -> "TimeScope":
        if self.start <= self.end:
            return self
        return self.model_copy(update={"start": self.end, "end": self.start})


class ExtractedInfo(BaseModel):
    intent: Intent | None = None
    entities: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, str] = Field(default_factory=dict)
    time_scope: TimeScope | None = None


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


class ChatModelOption(BaseModel):
    id: str
    label: str


class ChatModelsResponse(BaseModel):
    models: list[ChatModelOption]
    selected: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    llm_model: str | None = None


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class SearchChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    heading: str | None = None
    score: float


class SearchResponse(BaseModel):
    question: str
    chunks: list[SearchChunk] = Field(default_factory=list)


class Citation(BaseModel):
    source: str
    snippet: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    llm_model: str | None = None
