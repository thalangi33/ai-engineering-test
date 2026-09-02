"""Run with: python -m evals"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from app.config import settings
from app.rag.pipeline import ingest
from evals.score import evaluate_all, format_report, load_cases, tally


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score evals/questions.json. Retrieval (expected source in top-k) "
            "is scored separately from answer quality."
        )
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest docs into the local index before scoring.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model to use when --ingest is set.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score search only; do not call the LLM.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print machine-readable JSON instead of the table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.ingest:
        result = ingest(embedding_model=args.embedding_model)
        print(result.message, file=sys.stderr)

    cases = load_cases()
    scores = evaluate_all(cases, retrieval_only=args.retrieval_only)
    counts = tally(scores)
    if args.as_json:
        print(
            json.dumps(
                {
                    "index_path": str(settings.index_path),
                    "embedding_model": settings.embedding_model,
                    "llm_model": settings.llm_model,
                    "retrieval_only": args.retrieval_only,
                    "summary": counts,
                    "cases": [asdict(score) for score in scores],
                },
                indent=2,
            )
        )
    else:
        print(
            "Ask My Docs evals  "
            "(retrieval vs answer scored separately)\n"
            f"index: {settings.index_path}  "
            f"embedding: {settings.embedding_model}  "
            f"chat: {settings.llm_model}\n"
        )
        print(format_report(scores))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
