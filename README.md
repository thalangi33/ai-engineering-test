# Ask My Docs

Chat over a local folder of markdown files. Load, chunk, ingest, search, ask, and evals are implemented.

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

## Evals

Score `evals/questions.json`. Retrieval (expected source in top-k) is scored separately from answer quality. Fluent-but-wrong answers fail if they miss a `must_contain` phrase. Refuse cases must say `I don't know` with no citations.

```bash
# Retrieval only (no chat API key). MiniLM embeds locally.
python -m evals --ingest --embedding-model all-MiniLM-L6-v2 --retrieval-only

# Full suite (needs a configured chat model in .env)
python -m evals --ingest --embedding-model all-MiniLM-L6-v2
```

Exit code 1 if any scored check fails. `--json` prints machine-readable results.

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
| `python -m evals` | Score retrieval vs answer on `evals/questions.json` |

Sample notes in `docs/` are ingested into the local index. Hand-written eval questions live in `evals/questions.json`.

## Tests

```bash
pytest
```

Covers load, chunk, ingest, search, prompt, ask, and eval scoring. LLM HTTP calls are mocked.
