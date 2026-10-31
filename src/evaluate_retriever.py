
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load evidence chunks
chunks = []

with open("data/evidence_chunks.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

# Load test claims
claims = []

with open("evaluation_claims.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        claims.append(json.loads(line))

# Prepare BM25
tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

# Load semantic model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load semantic embeddings
embeddings = np.load("data/semantic_embeddings.npy")

hit_at_3 = 0
reciprocal_ranks = []
precision_scores = []

for item in claims:
    claim = item["claim"]
    expected_title = item["expected_title"]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(claim.lower().split())

    # Semantic retrieval
    claim_embedding = model.encode([claim])
    semantic_scores = cosine_similarity(
        claim_embedding, embeddings
    )[0]

    # Reciprocal Rank Fusion
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    semantic_ranking = np.argsort(semantic_scores)[::-1]

    rrf_scores = np.zeros(len(chunks))

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] += 1 / (60 + rank)

    for rank, index in enumerate(semantic_ranking, start=1):
        rrf_scores[index] += 1 / (60 + rank)

    top_3 = np.argsort(rrf_scores)[::-1][:3]

    # Check which retrieved chunks have the expected title
    relevant_positions = [
        rank
        for rank, index in enumerate(top_3, start=1)
        if chunks[index]["title"].casefold() == expected_title.casefold()
    ]

    if relevant_positions:
        hit_at_3 += 1
        reciprocal_ranks.append(1 / relevant_positions[0])
    else:
        reciprocal_ranks.append(0)

    precision_scores.append(len(relevant_positions) / 3)

    print("\nClaim:", claim)
    print("Expected page:", expected_title)
    print("Retrieved pages:", [chunks[i]["title"] for i in top_3])
    print("Relevant results:", len(relevant_positions))

# Calculate final metrics
total_claims = len(claims)

print("\n========== RETRIEVAL EVALUATION ==========")
print("Total claims:", total_claims)
print("Hit@3:", round(hit_at_3 / total_claims, 4))
print("MRR:", round(sum(reciprocal_ranks) / total_claims, 4))
print("Precision@3:", round(sum(precision_scores) / total_claims, 4))
