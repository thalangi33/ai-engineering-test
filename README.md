# Ask My Docs

Chat over a local folder of markdown files. This repo is a **scaffold**: the web app and API run, but RAG (load, chunk, embed, retrieve, prompt, LLM) is left as stubs for you to implement.

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

Stubs live in `app/rag/pipeline.py`. Implement them in the order listed in that file. HTTP routes in `app/api/routes.py` already call `ingest` and `ask` and return **501** until those functions exist.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

API keys are not required until you implement the LLM call. Do not put secrets in `docs/` — ingested text will be sent to the provider later.

## Run

```bash
python -m app
```

- UI: http://127.0.0.1:8000
- Health: `GET /api/health`
- Ingest: `POST /api/ingest` (501 until you implement it)
- Ask: `POST /api/ask` with `{"question": "..."}` (501 until you implement it)

## What you implement later

| Function | Role |
|---|---|
| `load_documents` | Read files under `docs/` |
| `chunk_text` | Split with metadata (`source`, `chunk_index`) |
| `ingest` | Embed and persist a local index |
| `search` | Top-k chunks for a question |
| `build_prompt` | Answer only from context; otherwise "I don't know" |
| `ask_llm` | Provider call, temperature 0 |
| `ask` | Orchestrate search → prompt → LLM → citations from metadata |

Sample notes in `docs/` are for ingest once you write loading. Draft eval questions in `evals/questions.json` — fill these in by hand before you trust the pipeline.

## Tests

```bash
pytest
```

Health and stub (501) tests only. Add retrieval/answer evals when you own the pipeline.
