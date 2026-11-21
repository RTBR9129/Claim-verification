
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_DIR / "data" / "evidence_chunks.jsonl"
EMBEDDINGS_FILE = PROJECT_DIR / "data" / "semantic_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        print("No evidence chunks found.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating evidence embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    print("\nSemantic index saved successfully!")
    print("Evidence chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Saved location:", EMBEDDINGS_FILE)


if __name__ == "__main__":
    main()
