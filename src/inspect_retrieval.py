
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent

CLAIMS_FILE = PROJECT_DIR / "data" / "clean_train.jsonl"
CHUNKS_FILE = PROJECT_DIR / "data" / "evidence_chunks.jsonl"

NUM_CLAIMS = 10
TOP_K = 3


def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    print("Loading data...")

    records = load_jsonl(CLAIMS_FILE)
    chunks = load_jsonl(CHUNKS_FILE)

    records = records[:NUM_CLAIMS]

    evidence_texts = [chunk["text"] for chunk in chunks]

    print("Building evidence index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000
    )

    evidence_matrix = vectorizer.fit_transform(evidence_texts)
    claim_matrix = vectorizer.transform(
        [record["claim"] for record in records]
    )

    similarities = cosine_similarity(claim_matrix, evidence_matrix)

    print("\nRETRIEVAL RESULTS")
    print("=" * 70)

    for i, record in enumerate(records):
        top_indices = similarities[i].argsort()[-TOP_K:][::-1]

        print(f"\nClaim: {record['claim']}")
        print(f"Actual label: {record['label']}")

        for rank, index in enumerate(top_indices, start=1):
            print(f"\nEvidence {rank}")
            print(f"Similarity: {similarities[i][index]:.4f}")
            print(chunks[index]["text"][:700])
            print("-" * 50)


if __name__ == "__main__":
    main()
