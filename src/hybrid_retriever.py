

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "evidence_chunks.jsonl"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "semantic_embeddings.npy"
TEST_CLAIMS_FILE = PROJECT_ROOT / "test_claim.txt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def load_test_claims():
    with open(TEST_CLAIMS_FILE, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def reciprocal_rank_fusion(bm25_scores, semantic_scores, k=60):
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    semantic_ranking = np.argsort(semantic_scores)[::-1]

    rrf_scores = np.zeros(len(bm25_scores))

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] += 1 / (k + rank)

    for rank, index in enumerate(semantic_ranking, start=1):
        rrf_scores[index] += 1 / (k + rank)

    return rrf_scores


def main():
    if not CHUNKS_FILE.exists():
        print("Evidence chunks file not found:", CHUNKS_FILE)
        return

    if not EMBEDDINGS_FILE.exists():
        print("Semantic embeddings file not found:", EMBEDDINGS_FILE)
        print("Run: python src/build_semantic_index.py")
        return

    if not TEST_CLAIMS_FILE.exists():
        print("Test claims file not found:", TEST_CLAIMS_FILE)
        return

    chunks = load_chunks()

    if not chunks:
        print("No evidence chunks found.")
        return

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(chunks) != len(embeddings):
        print("Error: chunk count does not match embedding count.")
        return

    tokenized_chunks = [
        tokenize(chunk.get("text", ""))
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_NAME)

    claims = load_test_claims()

    if not claims:
        print("No test claims found.")
        return

    print(f"\nTesting {len(claims)} claims")

    for claim_number, claim in enumerate(claims, start=1):

        bm25_scores = bm25.get_scores(tokenize(claim))

        claim_embedding = model.encode(
            [claim],
            normalize_embeddings=True
        )[0]

        semantic_scores = embeddings @ claim_embedding

        hybrid_scores = reciprocal_rank_fusion(
            bm25_scores,
            semantic_scores
        )

        ranked_indices = np.argsort(hybrid_scores)[::-1][:TOP_K]

        print("\n" + "=" * 70)
        print(f"CLAIM {claim_number}: {claim}")
        print("Top 3 evidence passages (Hybrid Retrieval - RRF)")
        print("=" * 70)

        for rank, index in enumerate(ranked_indices, start=1):
            print(f"\n{rank}. RRF score: {hybrid_scores[index]:.6f}")
            print("Evidence:", chunks[index].get("text", ""))
            print("-" * 60)


if __name__ == "__main__":
    main()
