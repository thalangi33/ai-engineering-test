# Document folder conventions

Put notes you actually want to query in this `docs/` folder.

v1 conventions:

- Prefer markdown (`.md`) or plain text (`.txt`).
- Keep the corpus small at first: about 10–20 files.
- Re-ingest after you change files. A stale index is expected until you add watchers.
- Do not store passwords, API keys, or private customer data here. Ingested text will be sent to the LLM provider.

The app reads `DOCS_DIR` from the environment (default: `docs`).
