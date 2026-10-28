
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CHUNKS_FILE = DATA_DIR / "evidence_chunks.jsonl"

TOP_K = 5


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    print("Loading evidence chunks...")

    chunks = load_chunks()

    if not chunks:
        print("No evidence chunks found.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000
    )

    document_matrix = vectorizer.fit_transform(texts)

    print("Indexed chunks:", len(chunks))

    while True:
        claim = input("\nEnter a claim (or type 'exit'): ").strip()

        if claim.lower() == "exit":
            break

        if not claim:
            print("Please enter a claim.")
            continue

        claim_vector = vectorizer.transform([claim])
        scores = cosine_similarity(
            claim_vector, document_matrix
        ).flatten()

        top_indices = scores.argsort()[-TOP_K:][::-1]

        print("\nTOP EVIDENCE RESULTS")
        print("-" * 50)

        for rank, index in enumerate(top_indices, start=1):
            result = chunks[index]

            print(f"\nResult {rank}")
            print("Page:", result["title"])
            print("Similarity:", round(float(scores[index]), 4))
            print("Evidence:", result["text"][:700])


if __name__ == "__main__":
    main()
