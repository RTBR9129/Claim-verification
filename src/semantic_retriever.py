
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
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
    print("Loading claims and evidence...")
    claims = load_jsonl(CLAIMS_FILE)[:NUM_CLAIMS]
    chunks = load_jsonl(CHUNKS_FILE)

    evidence_texts = [chunk["text"] for chunk in chunks]
    claim_texts = [item["claim"] for item in claims]

    print("Loading semantic model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding evidence chunks...")
    evidence_embeddings = model.encode(
        evidence_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Encoding claims...")
    claim_embeddings = model.encode(
        claim_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    similarities = cosine_similarity(
        claim_embeddings,
        evidence_embeddings
    )

    print("\nSEMANTIC RETRIEVAL RESULTS")
    print("=" * 70)

    for i, claim in enumerate(claims):
        top_indices = similarities[i].argsort()[-TOP_K:][::-1]

        print(f"\nClaim: {claim['claim']}")
        print(f"Actual label: {claim['label']}")

        for rank, index in enumerate(top_indices, start=1):
            print(f"\nEvidence {rank}")
            print(f"Similarity: {similarities[i][index]:.4f}")
            print(chunks[index]["text"][:700])
            print("-" * 50)


if __name__ == "__main__":
    main()
