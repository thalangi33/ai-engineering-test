from app.config import settings
from app.rag.pipeline import chunk_text, load_documents


def test_sample_docs_keep_source_and_index() -> None:
    chunks = chunk_text(load_documents(settings.docs_dir))
    assert chunks
    sources = {chunk["source"] for chunk in chunks}
    assert "docs/what-ask-my-docs-is.md" in sources
    assert "docs/folder-conventions.md" in sources
    assert "docs/build-order.md" in sources
    for chunk in chunks:
        assert chunk["text"].strip()
        assert isinstance(chunk["chunk_index"], int)
        assert chunk["chunk_index"] >= 0
        assert "heading" in chunk


def test_short_doc_is_single_chunk() -> None:
    chunks = chunk_text([{"path": "docs/a.md", "text": "# Hello\n\nShort note."}])
    assert len(chunks) == 1
    assert chunks[0]["source"] == "docs/a.md"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["heading"] == "Hello"
    assert "Short note." in chunks[0]["text"]


def test_headings_become_sections() -> None:
    text = "# One\n\n" + ("alpha " * 40) + "\n\n## Two\n\n" + ("bravo " * 40)
    chunks = chunk_text([{"path": "h.md", "text": text}])
    headings = [chunk["heading"] for chunk in chunks]
    assert "One" in headings
    assert "Two" in headings


def test_long_doc_splits_covers_text_and_overlaps() -> None:
    paragraphs = [f"Paragraph-{i} " + ("word " * 80) for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text([{"path": "long.md", "text": text}])

    assert len(chunks) > 1
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    for paragraph in paragraphs:
        marker = paragraph.split()[0]
        assert any(marker in chunk["text"] for chunk in chunks)

    first_tail = chunks[0]["text"][-80:]
    assert any(token in chunks[1]["text"] for token in first_tail.split() if token)


def test_skips_empty_documents() -> None:
    assert chunk_text([{"path": "empty.md", "text": "   "}]) == []
