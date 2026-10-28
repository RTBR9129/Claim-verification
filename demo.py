
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

with open(DATA / "evidence_chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

embeddings = np.load(DATA / "semantic_embeddings.npy")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

claim = input("Enter a claim to verify: ")

claim_embedding = model.encode(
    [claim],
    normalize_embeddings=True
)

scores = embeddings @ claim_embedding[0]
top_indices = np.argsort(scores)[::-1][:3]

print("\nCLAIM:", claim)
print("\nTOP RETRIEVED EVIDENCE")

for rank, index in enumerate(top_indices, 1):
    print(f"\n{rank}. Similarity: {scores[index]:.4f}")
    print(chunks[index]["text"][:700])

print("\nRetrieval demo complete.")
