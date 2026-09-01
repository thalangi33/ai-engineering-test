# Ask My Docs

Chat over a local folder of markdown files. Load, chunk, ingest, and search are implemented. Prompting and the LLM call are still stubs.

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

Pipeline steps live in `app/rag/pipeline.py`. HTTP routes in `app/api/routes.py` call `ingest`, `search`, and `ask`. Prompting / `ask` still return **501**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

API keys are required for OpenAI or Gemini embeddings; local MiniLM does not need a key. Do not put secrets in `docs/` — ingested text will be sent to the provider later.

## Run

```bash
python -m app
```

- UI: http://127.0.0.1:8000
- Health: `GET /api/health`
- Ingest: `POST /api/ingest` (optional `{"embedding_model": "..."}`)
- Search: `POST /api/search` with `{"question": "..."}` (optional `top_k`)
- Ask: `POST /api/ask` with `{"question": "..."}` (501 until the LLM is wired)

## Pipeline status

| Function | Role |
|---|---|
| `load_documents` | Read files under `docs/` |
| `chunk_text` | Split with metadata (`source`, `chunk_index`) |
| `ingest` | Embed and persist a local index |
| `search` | Top-k chunks for a question |
| `build_prompt` | Answer only from context; otherwise "I don't know" (stub) |
| `ask_llm` | Provider call, temperature 0 (stub) |
| `ask` | Orchestrate search → prompt → LLM → citations from metadata (stub) |

Sample notes in `docs/` are for ingest once you write loading. Draft eval questions in `evals/questions.json` — fill these in by hand before you trust the pipeline.

## Tests

```bash
pytest
```

Covers load, chunk, ingest, and search. `/api/ask` still returns 501.
