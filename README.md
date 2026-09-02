# Ask My Docs

Chat over a local folder of markdown files. Load, chunk, ingest, search, and ask are implemented.

## Pipeline

```text
files → load text → chunk → embed chunks → store (text + vector + metadata)
                                                      ↑
question → embed question → search similar chunks ────┘
                                                      ↓
                         build prompt (system + question + chunks)
                                                      ↓
                         LLM → answer + citations
                                                      ↓
                         log (chunks used, tokens, latency)
```

Pipeline steps live in `app/rag/pipeline.py`. HTTP routes in `app/api/routes.py` call `ingest`, `search`, and `ask`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

API keys depend on the providers you pick. OpenAI embeddings need `LLM_API_KEY`. Gemini embeddings and `gemini-2.0-flash` need `GEMINI_API_KEY`. Groq `llama-3.1-8b-instant` needs `GROQ_API_KEY`. Local MiniLM and Ollama `llama3.2` do not need a cloud key (Ollama must be running). Do not put secrets in `docs/` — ingested text is sent to the selected LLM provider when you ask.

## Run

```bash
python -m app
```

- UI: http://127.0.0.1:8000
- Health: `GET /api/health`
- Ingest: `POST /api/ingest` (optional `{"embedding_model": "..."}`)
- Search: `POST /api/search` with `{"question": "..."}` (optional `top_k`)
- Ask: `POST /api/ask` with `{"question": "..."}` (optional `llm_model`)
- Chat models: `GET /api/chat-models`

## Pipeline status

| Function | Role |
|---|---|
| `load_documents` | Read files under `docs/` |
| `chunk_text` | Split with metadata (`source`, `chunk_index`) |
| `ingest` | Embed and persist a local index |
| `search` | Top-k chunks for a question |
| `build_prompt` | Answer only from context; otherwise "I don't know" |
| `ask_llm` | Gemini, Ollama llama3.2, or Groq; temperature 0 |
| `ask` | Search → prompt → LLM → citations from chunk metadata |

Sample notes in `docs/` are ingested into the local index. Draft eval questions in `evals/questions.json` — fill these in by hand before you trust the pipeline.

## Tests

```bash
pytest
```

Covers load, chunk, ingest, search, prompt, and ask. LLM HTTP calls are mocked.
