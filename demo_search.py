#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.kbmem.search import cosine_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--query", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("examples/corpus.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
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
