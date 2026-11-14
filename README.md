# Explainable Evidence-Grounded Misinformation Analysis & Automated Claim Verification

## 1. Project Overview

This project aims to develop an explainable NLP-based misinformation detection and automated claim verification system.

Unlike traditional fake-news detection systems that classify an entire article as either **Fake** or **Real**, our system focuses on identifying and verifying individual factual claims using relevant supporting or contradicting evidence.

The system retrieves evidence from external knowledge sources, evaluates the relationship between the claim and the retrieved evidence, and generates an explainable verification result.

The system focuses on three possible verification outcomes:

- **SUPPORTS:** The available evidence supports the claim.
- **REFUTES:** The available evidence contradicts the claim.
- **NOT ENOUGH INFO (NEI):** There is insufficient evidence to verify or refute the claim.

The ultimate goal is to provide reliable, transparent and evidence-grounded misinformation analysis.

## 2. Problem Statement

Traditional fake-news detection models often rely on superficial textual patterns and produce binary classifications without explaining why an article is considered fake or real.

Such systems have limitations:
- They classify entire articles rather than individual factual claims.
- They do not necessarily provide supporting evidence.
- They struggle with complex claims involving multiple entities, events and relationships.
- Their predictions can be difficult to interpret.

Our proposed system addresses these limitations by combining NLP preprocessing, hybrid information retrieval, transformer-based verification, evidence reasoning and explainable AI.

## 3. Project Objectives

1. Extract factual and verifiable claims from textual content.
2. Break complex claims into smaller atomic assertions.
3. Retrieve relevant evidence from available knowledge sources.
4. Combine lexical and semantic retrieval techniques.
5. Verify claims using Natural Language Inference (NLI).
6. Identify supporting and contradicting evidence.
7. Provide confidence-aware predictions.
8. Generate human-readable explanations grounded in retrieved evidence.

## 4. Dataset

**Dataset: FEVEROUS**

FEVEROUS is used as the primary dataset for claim verification.

It contains claims associated with three labels:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

The dataset includes evidence from Wikipedia, including textual passages and structured table information.

This makes it suitable for evaluating claims that require reasoning over multiple pieces of evidence, including table-based facts.

### Dataset Files

- `train.jsonl` – Training data.
- `dev.jsonl` – Development data.

The complete Wikipedia/SQLite snapshot is approximately 9.6 GB. Due to storage limitations, the project initially used extracted Wikipedia page information and locally reconstructed evidence resources.

## 5. Proposed System Architecture

The intended system consists of the following components:

### 5.1 NLP Preprocessing
- Tokenization
- Sentence segmentation
- Part-of-speech tagging
- Named Entity Recognition (NER)
- Temporal information extraction
- Dependency parsing

### 5.2 Claim Detection and Decomposition

Identifies factual statements that can be verified and separates compound claims into smaller atomic claims.

For example:

"Company X merged with Company Y in 2022 and laid off 40% of its staff."

Can be divided into:

- Atomic Claim 1: Company X merged with Company Y in 2022.
- Atomic Claim 2: Company X laid off 40% of its workforce.

### 5.3 Hybrid Evidence Retrieval

Uses multiple retrieval techniques to identify relevant evidence.

**BM25:** Retrieves evidence using lexical and keyword matching.

**Sentence Transformers:** Generates semantic embeddings to identify evidence based on meaning.

**FAISS:** Supports efficient similarity search over dense vector representations.

**Reciprocal Rank Fusion (RRF):** Combines lexical and semantic retrieval rankings.

### 5.4 Cross-Encoder Reranking

A transformer-based Cross-Encoder evaluates claim-evidence pairs jointly and reranks retrieved evidence based on relevance.

### 5.5 Natural Language Inference (NLI)

Compares the claim against retrieved evidence and predicts:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

### 5.6 Evidence Graph

Represents relationships between claims, entities, events, dates and evidence passages.

This component aims to support multi-hop reasoning when a claim cannot be verified using a single evidence passage.

### 5.7 Confidence Calibration

Uses confidence calibration techniques such as temperature scaling.

If the evidence is conflicting or the prediction confidence is below a predefined threshold, the system can flag the result for human review.

### 5.8 Explainable RAG

Uses retrieved evidence to generate a concise natural-language explanation.

The explanation should remain grounded in the retrieved evidence and include relevant source attribution.

**Note:** These components describe the proposed architecture. Their implementation status must be verified against the current source code.

## 6. Technologies and Algorithms

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP | NLTK / NLP preprocessing tools |
| Dataset | FEVEROUS |
| Lexical Retrieval | BM25 |
| Semantic Retrieval | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Ranking Fusion | Reciprocal Rank Fusion |
| Deep Learning | PyTorch / Transformers |
| Verification | Natural Language Inference |
| Explainability | Evidence-grounded explanations |
| Development Environment | Visual Studio Code |

Some technologies listed above belong to the proposed architecture and should not be considered fully integrated until verified in the repository.

## 7. Expected System Output

For every input claim, the intended system output should contain:

- Original claim
- Retrieved evidence
- Verification label (SUPPORTS / REFUTES / NEI)
- Confidence score
- Evidence source attribution
- Human-readable explanation

Example:

**Input Claim:**
"The company's revenue increased by 15%."

**Retrieved Evidence:**
"The company reported a 15% increase in annual revenue."

**Predicted Label:**
SUPPORTS

**Explanation:**
The retrieved evidence explicitly reports a 15% increase in the company's revenue, supporting the given claim.

---

**Project Mission:**

To develop an explainable, evidence-grounded NLP system that verifies factual claims using hybrid information retrieval and advanced reasoning techniques, providing transparent and confidence-aware misinformation assessments rather than simple Fake/Real classifications.
