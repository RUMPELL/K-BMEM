"""Command-line interface for the synthetic K-BMEM retrieval example."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search import cosine_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense retrieval over a JSONL corpus.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--corpus", type=Path, default=Path("examples/corpus.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = SentenceTransformer(args.model, trust_remote_code=False)
    texts = [args.query] + [row["text"] for row in rows]
    vectors = model.encode(texts, normalize_embeddings=True)
    ranking = cosine_ranking(vectors[0], [(row["id"], vector) for row, vector in zip(rows, vectors[1:])], args.top_k)
    by_id = {row["id"]: row["text"] for row in rows}
    for rank, (document_id, score) in enumerate(ranking, 1):
        print(f"{rank}. {document_id} score={score:.4f}\n   {by_id[document_id]}")


if __name__ == "__main__":
    main()
