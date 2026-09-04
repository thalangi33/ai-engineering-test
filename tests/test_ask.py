import json
from pathlib import Path

import httpx
import pytest

import app.rag.pipeline as pipeline
from app.config import settings
from app.rag.pipeline import ask, ask_llm, build_prompt


def _write_index(path: Path, chunks: list[dict], model: str = "text-embedding-3-small") -> None:
    path.write_text(
        json.dumps({"embedding_model": model, "chunks": chunks}),
        encoding="utf-8",
    )


def _sample_chunks() -> list[dict]:
    return [
        {
            "text": "Ask My Docs answers questions from a local folder of documents.",
            "source": "docs/what-ask-my-docs-is.md",
            "chunk_index": 0,
            "heading": "What Ask My Docs is",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "Put notes you want to query in the docs folder.",
            "source": "docs/folder-conventions.md",
            "chunk_index": 0,
            "heading": "Document folder conventions",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]


def test_build_prompt_includes_question_chunks_and_refuse_instruction() -> None:
    chunks = [
        {
            "text": "Point the app at markdown files in docs/.",
            "source": "docs/what-ask-my-docs-is.md",
            "heading": "What Ask My Docs is",
        }
    ]
    messages = build_prompt("What is Ask My Docs?", chunks)

    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"].lower()
    assert "only" in system
    assert "i don't know" in system
    user = messages[1]["content"]
    assert "What is Ask My Docs?" in user
    assert "docs/what-ask-my-docs-is.md" in user
    assert "Point the app at markdown files in docs/." in user
    assert "What Ask My Docs is" in user


def test_build_prompt_handles_empty_chunks() -> None:
    messages = build_prompt("What is the weather in Tokyo?", [])
    assert "(none)" in messages[1]["content"]
    assert "What is the weather in Tokyo?" in messages[1]["content"]


def test_ask_llm_posts_groq_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "  Ask My Docs answers from local files.  "}}
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                },
                request=request,
            )

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    monkeypatch.setattr(settings, "llm_model", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "temperature", 0.0)

    messages = [{"role": "user", "content": "What is Ask My Docs?"}]
    answer = ask_llm(messages)

    assert answer == "Ask My Docs answers from local files."
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer gsk-test"
    assert captured["json"]["model"] == "llama-3.1-8b-instant"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["messages"] == messages


def test_ask_llm_posts_deepseek_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "  Ask My Docs answers from local files.  "}}
                    ],
                    "usage": {"prompt_tokens": 18, "completion_tokens": 7},
                },
                request=request,
            )

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-deepseek-test")
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "temperature", 0.0)

    messages = [{"role": "user", "content": "What is Ask My Docs?"}]
    answer, usage = pipeline._ask_llm(messages)

    assert answer == "Ask My Docs answers from local files."
    assert usage == {"prompt_tokens": 18, "completion_tokens": 7}
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-deepseek-test"
    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["messages"] == messages


def test_ask_llm_posts_gemini_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Grounded in the docs."}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 6,
                    },
                },
                request=request,
            )

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-test")
    monkeypatch.setattr(settings, "llm_model", "gemini-2.0-flash")

    messages = [
        {"role": "system", "content": "Answer only from excerpts."},
        {"role": "user", "content": "What is Ask My Docs?"},
    ]
    answer, usage = pipeline._ask_llm(messages)

    assert answer == "Grounded in the docs."
    assert usage == {"prompt_tokens": 20, "completion_tokens": 6}
    assert captured["url"].endswith("models/gemini-2.0-flash:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "gemini-test"
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == (
        "Answer only from excerpts."
    )
    assert captured["json"]["contents"][0]["role"] == "user"


def test_ask_llm_posts_ollama_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "message": {"content": "From the local model."},
                    "prompt_eval_count": 40,
                    "eval_count": 9,
                },
                request=request,
            )

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "llm_model", "llama3.2")
    monkeypatch.setattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "temperature", 0.0)

    messages = [{"role": "user", "content": "What is Ask My Docs?"}]
    answer, usage = pipeline._ask_llm(messages)

    assert answer == "From the local model."
    assert usage == {"prompt_tokens": 40, "completion_tokens": 9}
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["json"]["model"] == "llama3.2"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["temperature"] == 0.0
    assert captured["json"]["messages"] == messages


def test_ask_llm_requires_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "gemini-2.0-flash")
    monkeypatch.setattr(settings, "gemini_api_key", "  ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        ask_llm([{"role": "user", "content": "hi"}])


def test_ask_llm_requires_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "groq_api_key", "")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        ask_llm([{"role": "user", "content": "hi"}])


def test_ask_llm_requires_deepseek_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "deepseek_api_key", "  ")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        ask_llm([{"role": "user", "content": "hi"}])


def test_ask_llm_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            request = httpx.Request("POST", url)
            return httpx.Response(401, text="invalid api key", request=request)

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "llm_model", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-bad")
    with pytest.raises(RuntimeError, match="LLM request failed \\(401\\)"):
        ask_llm([{"role": "user", "content": "hi"}])


def test_ask_llm_reports_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pipeline.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "llm_model", "llama3.2")
    with pytest.raises(RuntimeError, match="Ollama is not reachable"):
        ask_llm([{"role": "user", "content": "hi"}])


def test_ask_returns_answer_and_citations_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "top_k", 2)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )
    monkeypatch.setattr(
        pipeline,
        "_ask_llm",
        lambda messages: ("It answers from a local folder of documents.", None),
    )

    result = ask("What is Ask My Docs?")

    assert "local folder" in result.answer
    assert [citation.source for citation in result.citations] == [
        "docs/what-ask-my-docs-is.md",
        "docs/folder-conventions.md",
    ]
    assert "Ask My Docs answers questions" in (result.citations[0].snippet or "")
    logged = capsys.readouterr().out
    assert "[ask]" in logged
    assert "docs/what-ask-my-docs-is.md" in logged


def test_ask_ignores_model_invented_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )
    monkeypatch.setattr(
        pipeline,
        "_ask_llm",
        lambda messages: ("See secret.txt for the answer.", None),
    )

    result = ask("What is Ask My Docs?")

    sources = [citation.source for citation in result.citations]
    assert "secret.txt" not in sources
    assert "docs/what-ask-my-docs-is.md" in sources


def test_ask_refuses_without_llm_when_no_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, [])
    monkeypatch.setattr(settings, "index_path", index_path)
    called = {"llm": False}

    def fake_llm(messages: list[dict]) -> tuple[str, None]:
        called["llm"] = True
        return "should not run", None

    monkeypatch.setattr(pipeline, "_ask_llm", fake_llm)

    result = ask("What is the weather in Tokyo tomorrow?")

    assert result.answer == "I don't know"
    assert result.citations == []
    assert called["llm"] is False


def test_ask_omits_citations_when_model_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )
    monkeypatch.setattr(pipeline, "_ask_llm", lambda messages: ("I don't know", None))

    result = ask("What is the weather in Tokyo tomorrow?")

    assert result.answer == "I don't know"
    assert result.citations == []


def test_ask_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ask("   ")


def test_list_chat_models_includes_gemini_ollama_groq_and_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    listed = pipeline.list_chat_models()
    assert [model.id for model in listed.models] == [
        "gemini-2.0-flash",
        "llama3.2",
        "llama-3.1-8b-instant",
        "deepseek-v4-flash",
    ]
    assert listed.selected == "deepseek-v4-flash"
    assert "Ollama" in listed.models[1].label
    assert "3B" in listed.models[1].label
    assert "DeepSeek" in listed.models[3].label


def test_ask_uses_requested_chat_model_and_restores_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path, _sample_chunks())
    monkeypatch.setattr(settings, "index_path", index_path)
    monkeypatch.setattr(settings, "llm_model", "gemini-2.0-flash")
    monkeypatch.setattr(
        pipeline,
        "_embed_texts",
        lambda texts, for_query=False: [[1.0, 0.0, 0.0]],
    )
    seen: dict = {}

    def fake_llm(messages: list[dict]) -> tuple[str, None]:
        seen["model"] = settings.llm_model
        return "It answers from a local folder of documents.", None

    monkeypatch.setattr(pipeline, "_ask_llm", fake_llm)

    result = ask("What is Ask My Docs?", llm_model="llama-3.1-8b-instant")

    assert seen["model"] == "llama-3.1-8b-instant"
    assert result.llm_model == "llama-3.1-8b-instant"
    assert settings.llm_model == "gemini-2.0-flash"
