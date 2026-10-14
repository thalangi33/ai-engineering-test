import pytest

from app.models import ExtractedInfo
import app.rag.pipeline as pipeline


@pytest.fixture(autouse=True)
def skip_llm_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ingest/search/ask tests offline; extractor unit tests patch ask_llm instead."""
    monkeypatch.setattr(
        pipeline,
        "extract_info",
        lambda text, *, kind: ExtractedInfo(),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_chunks",
        lambda texts, **kwargs: [ExtractedInfo() for _ in texts],
    )
