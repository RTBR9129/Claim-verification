
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


# Load evidence chunks
chunks = []

with open(DATA_DIR / "evidence_chunks.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))


# Load evaluation claims
claims = []

with open(PROJECT_ROOT / "evaluation_claims.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        claims.append(json.loads(line))


# Load original evidence lookup
with open(DATA_DIR / "evidence_lookup_full.json", "r", encoding="utf-8") as file:
    evidence_lookup = json.load(file)


# Prepare BM25
tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# Load semantic model and embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = np.load(DATA_DIR / "semantic_embeddings.npy")


# Normalize text for evidence matching
def normalize(text):
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


hit_at_3 = 0
hit_at_5 = 0
reciprocal_ranks = []
precision_scores = []

evidence_recall_at_3 = []
evidence_recall_at_5 = []


for item in claims:

    claim = item["claim"]
    evidence_groups = item["evidence_groups"]

    # Get all relevant page titles
    expected_titles = set()

    for group in evidence_groups:
        expected_titles.update(
            title.casefold() for title in group["titles"]
        )

    # BM25 retrieval
    bm25_scores = bm25.get_scores(claim.lower().split())
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # Semantic retrieval
    claim_embedding = model.encode(
        [claim],
        normalize_embeddings=True
    )

    semantic_scores = embeddings @ claim_embedding[0]
    semantic_ranking = np.argsort(semantic_scores)[::-1]

    # Reciprocal Rank Fusion
    rrf_scores = np.zeros(len(chunks))

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] += 1 / (60 + rank)

    for rank, index in enumerate(semantic_ranking, start=1):
        rrf_scores[index] += 1 / (60 + rank)

    ranking = np.argsort(rrf_scores)[::-1]

    top_3 = ranking[:3]
    top_5 = ranking[:5]

    # Page-level evaluation
    relevant_positions = [
        rank
        for rank, index in enumerate(ranking, start=1)
        if chunks[index]["title"].casefold() in expected_titles
    ]

    if any(
        chunks[index]["title"].casefold() in expected_titles
        for index in top_3
    ):
        hit_at_3 += 1

    if any(
        chunks[index]["title"].casefold() in expected_titles
        for index in top_5
    ):
        hit_at_5 += 1

    reciprocal_ranks.append(
        1 / relevant_positions[0] if relevant_positions else 0
    )

    relevant_top_3 = sum(
        chunks[index]["title"].casefold() in expected_titles
        for index in top_3
    )

    precision_scores.append(relevant_top_3 / 3)

    # Evidence-level evaluation
    def calculate_evidence_recall(retrieved_indices):

        retrieved_chunks = [
            {
                "title": chunks[index]["title"].casefold(),
                "text": normalize(chunks[index]["text"])
            }
            for index in retrieved_indices
        ]

        group_recalls = []

        for group in evidence_groups:

            found = 0
            total = 0

            for evidence_id in group["evidence_ids"]:

                evidence = evidence_lookup.get(evidence_id)

                if not evidence:
                    continue

                evidence_text = normalize(evidence.get("text", ""))
                evidence_title = evidence.get("title", "").casefold()

                if not evidence_text:
                    continue

                total += 1

                if any(
                    chunk["title"] == evidence_title
                    and evidence_text in chunk["text"]
                    for chunk in retrieved_chunks
                ):
                    found += 1

            if total > 0:
                group_recalls.append(found / total)

        return max(group_recalls) if group_recalls else 0

    recall_3 = calculate_evidence_recall(top_3)
    recall_5 = calculate_evidence_recall(top_5)

    evidence_recall_at_3.append(recall_3)
    evidence_recall_at_5.append(recall_5)

    print("\nClaim:", claim)
    print("Expected pages:", sorted(expected_titles))
    print("Top 3:", [chunks[i]["title"] for i in top_3])
    print("Top 5:", [chunks[i]["title"] for i in top_5])
    print("Evidence Recall@3:", round(recall_3, 4))
    print("Evidence Recall@5:", round(recall_5, 4))


# Final metrics
total_claims = len(claims)

if total_claims == 0:
    print("No evaluation claims found.")

else:
    print("\n========== RETRIEVAL EVALUATION ==========")
    print("Total claims:", total_claims)

    print("Hit@3:", round(hit_at_3 / total_claims, 4))
    print("Hit@5:", round(hit_at_5 / total_claims, 4))
    print("MRR:", round(np.mean(reciprocal_ranks), 4))
    print("Precision@3:", round(np.mean(precision_scores), 4))

    print(
        "Evidence Recall@3:",
        round(np.mean(evidence_recall_at_3), 4)
    )

    print(
        "Evidence Recall@5:",
        round(np.mean(evidence_recall_at_5), 4)
    )
