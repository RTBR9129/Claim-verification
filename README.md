# Claim Verification

This project predicts `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO` and presents
retrieved FEVEROUS passages, an evidence graph summary, an NLI estimate, and a
short extractive explanation. Its retrieval pipeline combines SQLite FTS5
BM25 with MiniLM dense vectors, merges the rankings with reciprocal-rank
fusion, reranks passages with a cross-encoder, and classifies claim-matching
sentences with a three-way NLI model.

The default verdict comes from a claim-only linear SVM using word and character
TF-IDF features. On 7,129 test claims it scored **63.4% accuracy** and **0.470
macro F1**. On those same claims, the original logistic regression model
scored **55.9%** and **0.423**. See the [test report](reports/word_character_svm_test.json) for
the comparison and per-label results.

The CLI displays raw SVM decision scores. Retrieved passages and the advisory
NLI estimate help reviewers inspect the claim. The NLI temperature fit reached
its search limit, so treat its confidence scores as provisional.

`label_corrections.jsonl` contains explicit reviewed exceptions. The CLI shows
the model estimate and corrected final label when one matches. The Shulin
correction follows the project convention that “Republic of China (Taiwan)”
counts as “China”; this differs from labels on similar examples in the
original training data.

## Run the verifier

The checked-in code expects the prepared files under `data/`. Those files are
already present in this working copy. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/claim_verifier.py index
python src/claim_verifier.py verify --claim "Tower of Glass is a novel by Robert Silverberg."
```

If the default SVM artifact is missing, run the following from the project
root. The second command trains the SVM and saves its test comparison against
the original classifier.

```powershell
python src/claim_verifier.py train
python experiments/train_word_character_svm.py
```

The index command indexes `data/evidence_chunks.jsonl` with SQLite FTS5's BM25
ranker. The project already includes MiniLM vectors in
`data/semantic_embeddings.npy`; if that file is missing, regenerate it with
`python scripts/data_prep/build_semantic_index.py`. The first `verify` or `evaluate-hybrid`
builds a local FAISS inner-product index and downloads the MiniLM,
cross-encoder, and NLI model weights into the ignored `models/hf-cache/`
directory. Later runs reuse them.

`calibrate-hybrid` fits one temperature value using a stratified sample from
the training split and saves it to `models/evidence_calibration.json`.
`evaluate-hybrid` measures the evidence-grounded pipeline on a stratified
sample of the held-out validation split. The report records the expected label,
the predictions and scores from the original logistic regression model and NLI,
and how many errors switching to NLI would fix or introduce. It uses the original dataset labels and does not
apply manual corrections. The perfect selection ceiling assumes you could
choose a correct model whenever at least one of the two is right.

The sample report is saved to `models/hybrid_verifier_validation.json`. Use
`--max-claims 0` for all validation claims. That run saves a separate report
to `models/hybrid_verifier_full_validation.json`. `train` and `evaluate` run the claim-only baseline.
`train --max-claims 10000` is a quicker baseline training option.

The evidence graph records the claim, passage nodes, NLI relations, and shared
name matches. Its current implementation is a transparent lightweight graph,
not a trained GNN/GAT. The explanation quotes selected evidence sentences and
does not call a text generator, so it does not invent supporting details.
Low-confidence evidence and opposing model and evidence labels trigger a review recommendation. The primary
SVM verdict uses only claim text.

## Train the experimental FEVEROUS verifier

The experiment trains a logistic regression classifier on features from the
cached NLI encoder. It uses complete annotated evidence groups when available
for training, and retrieves evidence with SQLite FTS5 BM25 for validation and
test. Validation selects the feature set and regularization. The test report
compares the selected model with the saved claim-only model and the majority
label from training.

```powershell
.\.venv\Scripts\python.exe -B experiments\train_feverous_verifier.py --train-per-label 60 --validation-size 60 --test-size 100
```

The script saves `models/feverous_supervised_experiment.json` and
`models/feverous_supervised_head.joblib`. The regular verifier continues to use
its existing model.

### Fine-tune the evidence model

This pilot updates the last two transformer layers and the classification head
of the cached NLI model. It selects an epoch on validation claims and compares
the result with the default SVM on the same separate test sample. Validation
and test use retrieved evidence. The comparison reports how many SVM errors
the fine-tuned model fixes and how many correct SVM answers it loses.

```powershell
.\.venv\Scripts\python.exe -B experiments\fine_tune_feverous.py
```

The report is written to `models/feverous_fine_tuning_pilot.json`.

## Rebuild the evidence corpus

The large source data is local and is excluded from Git. To reproduce the
prepared evidence files from the supplied archive, run these from the project
root in order:

```powershell
python scripts/data_prep/prepare_training_data.py
python scripts/data_prep/split_dataset.py
python scripts/data_prep/extract_required_titles.py
python scripts/data_prep/extract_required_pages.py
python scripts/data_prep/build_evidence_lookup.py
python scripts/data_prep/extract_text.py
python scripts/data_prep/chunk_text.py
```

The page extraction step scans the large wiki archive. If the embeddings are
missing, regenerate them with the semantic index builder documented above.

## Project map

- [Verifier CLI](src/claim_verifier.py) and [evidence pipeline](src/evidence_pipeline.py).
- [Data preparation](scripts/data_prep) contains the dataset, evidence, and index builders.
- [Evaluation](scripts/evaluation) contains the standalone retrieval benchmark.
  Run `python scripts/evaluation/evaluate_retriever.py` with
  [evaluation_claims.jsonl](evaluation_claims.jsonl) at the project root.
- [Experiments](experiments) contains SVM training, the FEVEROUS training pilots,
  and the structured table extraction prototype.
- [Tests](tests) run with `python -B -m unittest discover -s tests`.
- [Retired scripts](archive/retired-scripts.zip) preserves the original files and paths
  for older diagnostics, demos, and early experiments.

## Data notes

The cleaned claim file excludes malformed or unlabeled source rows. The
retrieval indexes, calibration file, trained classifier, and model weights are
generated locally and should not be committed. The retrieval corpus is limited
to pages associated with the labeled claims, so claims outside that corpus can
return no matching evidence and should be reviewed by a person.

## Project proposal

### 1. Project Overview

This project aims to develop an explainable NLP-based misinformation detection and automated claim verification system.

Unlike traditional fake-news detection systems that classify an entire article as either **Fake** or **Real**, our system focuses on identifying and verifying individual factual claims using relevant supporting or contradicting evidence.

The system retrieves evidence from external knowledge sources, evaluates the relationship between the claim and the retrieved evidence, and generates an explainable verification result.

The system focuses on three possible verification outcomes:

- **SUPPORTS:** The available evidence supports the claim.
- **REFUTES:** The available evidence contradicts the claim.
- **NOT ENOUGH INFO (NEI):** There is insufficient evidence to verify or refute the claim.

The ultimate goal is to provide reliable, transparent and evidence-grounded misinformation analysis.

### 2. Problem Statement

Traditional fake-news detection models often rely on superficial textual patterns and produce binary classifications without explaining why an article is considered fake or real.

Such systems have limitations:
- They classify entire articles rather than individual factual claims.
- They do not necessarily provide supporting evidence.
- They struggle with complex claims involving multiple entities, events and relationships.
- Their predictions can be difficult to interpret.

Our proposed system addresses these limitations by combining NLP preprocessing, hybrid information retrieval, transformer-based verification, evidence reasoning and explainable AI.

### 3. Project Objectives

1. Extract factual and verifiable claims from textual content.
2. Break complex claims into smaller atomic assertions.
3. Retrieve relevant evidence from available knowledge sources.
4. Combine lexical and semantic retrieval techniques.
5. Verify claims using Natural Language Inference (NLI).
6. Identify supporting and contradicting evidence.
7. Provide confidence-aware predictions.
8. Generate human-readable explanations grounded in retrieved evidence.

### 4. Dataset

**Dataset: FEVEROUS**

FEVEROUS is used as the primary dataset for claim verification.

It contains claims associated with three labels:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

The dataset includes evidence from Wikipedia, including textual passages and structured table information.

This makes it suitable for evaluating claims that require reasoning over multiple pieces of evidence, including table-based facts.

#### Dataset Files

- `train.jsonl` – Training data.
- `dev.jsonl` – Development data.

The complete Wikipedia/SQLite snapshot is approximately 9.6 GB. Due to storage limitations, the project initially used extracted Wikipedia page information and locally reconstructed evidence resources.

### 5. Proposed System Architecture

The intended system consists of the following components:

#### 5.1 NLP Preprocessing
- Tokenization
- Sentence segmentation
- Part-of-speech tagging
- Named Entity Recognition (NER)
- Temporal information extraction
- Dependency parsing

#### 5.2 Claim Detection and Decomposition

Identifies factual statements that can be verified and separates compound claims into smaller atomic claims.

For example:

"Company X merged with Company Y in 2022 and laid off 40% of its staff."

Can be divided into:

- Atomic Claim 1: Company X merged with Company Y in 2022.
- Atomic Claim 2: Company X laid off 40% of its workforce.

#### 5.3 Hybrid Evidence Retrieval

Uses multiple retrieval techniques to identify relevant evidence.

**BM25:** Retrieves evidence using lexical and keyword matching.

**Sentence Transformers:** Generates semantic embeddings to identify evidence based on meaning.

**FAISS:** Supports efficient similarity search over dense vector representations.

**Reciprocal Rank Fusion (RRF):** Combines lexical and semantic retrieval rankings.

#### 5.4 Cross-Encoder Reranking

A transformer-based Cross-Encoder evaluates claim-evidence pairs jointly and reranks retrieved evidence based on relevance.

#### 5.5 Natural Language Inference (NLI)

Compares the claim against retrieved evidence and predicts:

- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

#### 5.6 Evidence Graph

Represents relationships between claims, entities, events, dates and evidence passages.

This component aims to support multi-hop reasoning when a claim cannot be verified using a single evidence passage.

#### 5.7 Confidence Calibration

Uses confidence calibration techniques such as temperature scaling.

If the evidence is conflicting or the prediction confidence is below a predefined threshold, the system can flag the result for human review.

#### 5.8 Explainable RAG

Uses retrieved evidence to generate a concise natural-language explanation.

The explanation should remain grounded in the retrieved evidence and include relevant source attribution.

**Note:** These components describe the proposed architecture. Their implementation status must be verified against the current source code.

### 6. Technologies and Algorithms

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

### 7. Expected System Output

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
