"""Chat backends: Gemini, Ollama, Groq, and DeepSeek."""

import httpx

from app.config import settings
from app.models import ChatModelsResponse
from app.rag.http_errors import raise_http_error

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
_GEMINI_CHAT_MODELS = {"gemini-2.0-flash"}
_OLLAMA_CHAT_MODELS = {"llama3.2", "llama3.2:3b"}
_GROQ_CHAT_MODELS = {"llama-3.1-8b-instant"}
_DEEPSEEK_CHAT_MODELS = {"deepseek-v4-flash"}
_CHAT_MODEL_CHOICES = (
    ("gemini-2.0-flash", "Gemini · gemini-2.0-flash"),
    ("llama3.2", "Ollama · llama3.2 (3B)"),
    ("llama-3.1-8b-instant", "Groq · llama-3.1-8b-instant"),
    ("deepseek-v4-flash", "DeepSeek · deepseek-v4-flash"),
)


def _chat_backend(model: str) -> str:
    if model in _GEMINI_CHAT_MODELS:
        return "gemini"
    if model in _OLLAMA_CHAT_MODELS:
        return "ollama"
    if model in _GROQ_CHAT_MODELS:
        return "groq"
    if model in _DEEPSEEK_CHAT_MODELS:
        return "deepseek"
    raise ValueError(
        f"Unsupported chat model {model!r}. Choose one of: "
        "gemini-2.0-flash, llama3.2, llama-3.1-8b-instant, deepseek-v4-flash."
    )


def list_chat_models() -> ChatModelsResponse:
    ids = [model_id for model_id, _label in _CHAT_MODEL_CHOICES]
    selected = settings.llm_model
    if selected in _OLLAMA_CHAT_MODELS:
        selected = "llama3.2"
    if selected not in ids:
        selected = ids[0]
    return ChatModelsResponse(
        models=[
            {"id": model_id, "label": label}
            for model_id, label in _CHAT_MODEL_CHOICES
        ],
        selected=selected,
    )


def ask_llm(messages: list[dict]) -> str:
    """Send messages to the LLM and return the assistant text."""
    return _ask_llm(messages)[0]


def _ask_llm(messages: list[dict]) -> tuple[str, dict | None]:
    if not messages:
        raise ValueError("messages must not be empty.")
    backend = _chat_backend(settings.llm_model)
    if backend == "gemini":
        return _ask_gemini(messages)
    if backend == "ollama":
        return _ask_ollama(messages)
    if backend == "groq":
        return _ask_groq(messages)
    return _ask_deepseek(messages)


def _messages_to_gemini(messages: list[dict]) -> dict:
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message.get("role")
        text = message.get("content") or ""
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    payload: dict = {
        "contents": contents,
        "generationConfig": {"temperature": settings.temperature},
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }
    return payload


def _gemini_usage(data: dict) -> dict | None:
    meta = data.get("usageMetadata")
    if not isinstance(meta, dict):
        return None
    prompt = meta.get("promptTokenCount")
    completion = meta.get("candidatesTokenCount")
    if prompt is None and completion is None:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def _ask_gemini(messages: list[dict]) -> tuple[str, dict | None]:
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to ask with Gemini.")
    model = settings.llm_model
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url, headers=headers, json=_messages_to_gemini(messages)
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise_http_error(exc, "LLM request")

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("LLM response was missing candidates.")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    content = "".join(part.get("text") or "" for part in parts).strip()
    if not content:
        raise RuntimeError("LLM response was missing text.")
    return content, _gemini_usage(data)


def _ask_groq(messages: list[dict]) -> tuple[str, dict | None]:
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is required to ask with Groq.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "temperature": settings.temperature,
        "messages": messages,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(_GROQ_CHAT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise_http_error(exc, "LLM request")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response was missing choices.")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("LLM response was missing text.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return content, usage


def _ask_deepseek(messages: list[dict]) -> tuple[str, dict | None]:
    api_key = (settings.deepseek_api_key or "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required to ask with DeepSeek.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "temperature": settings.temperature,
        "messages": messages,
        "thinking": {"type": "disabled"},
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(_DEEPSEEK_CHAT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise_http_error(exc, "LLM request")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response was missing choices.")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("LLM response was missing text.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return content, usage


def _ollama_usage(data: dict) -> dict | None:
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    if prompt is None and completion is None:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def _ask_ollama(messages: list[dict]) -> tuple[str, dict | None]:
    base = (settings.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": settings.temperature},
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. Start Ollama and pull llama3.2."
        ) from exc
    except httpx.HTTPError as exc:
        raise_http_error(exc, "LLM request")

    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("LLM response was missing text.")
    return content, _ollama_usage(data)
