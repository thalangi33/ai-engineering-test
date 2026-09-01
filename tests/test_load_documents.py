from pathlib import Path

import pytest

from app.config import settings
from app.rag.pipeline import load_documents


def test_load_sample_docs() -> None:
    documents = load_documents(settings.docs_dir)
    by_path = {doc["path"]: doc["text"] for doc in documents}

    assert "docs/what-ask-my-docs-is.md" in by_path
    assert "docs/folder-conventions.md" in by_path
    assert "docs/build-order.md" in by_path
    assert "docs/nba/lebron-james.md" in by_path
    assert "docs/nba/stephen-curry.md" in by_path
    assert "docs/nba/nikola-jokic.md" in by_path
    assert "Ask My Docs is a small app" in by_path["docs/what-ask-my-docs-is.md"]
    assert "all-time leading scorer" in by_path["docs/nba/lebron-james.md"]
    assert "unanimous MVP" in by_path["docs/nba/stephen-curry.md"]
    assert "Sombor, Serbia" in by_path["docs/nba/nikola-jokic.md"]


def test_load_skips_unsupported_files_and_reads_nested(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    (tmp_path / "script.py").write_text("x = 1", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert {doc["text"] for doc in documents} == {"hello", "nested"}


def test_load_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
