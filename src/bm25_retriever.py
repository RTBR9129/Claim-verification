
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "evidence_chunks.jsonl"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def retrieve_evidence(claim, chunks, bm25, top_k=3):
    query_tokens = tokenize(claim)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:
        results.append({
            "score": float(scores[index]),
            "chunk": chunks[index]
        })

    return results


def main():
    chunks = load_chunks()

    if not chunks:
        print("No evidence chunks found.")
        return

    tokenized_chunks = [
        tokenize(chunk.get("text", ""))
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    claim = input("Enter a claim: ").strip()

    if not claim:
        print("Please enter a claim.")
        return

    results = retrieve_evidence(claim, chunks, bm25)

    print("\nClaim:", claim)
    print("\nTop 3 evidence passages (BM25):")

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n{rank}. BM25 score: {result['score']:.4f}")
        print("Evidence:", chunk.get("text", ""))
        print("-" * 60)


if __name__ == "__main__":
    main()
