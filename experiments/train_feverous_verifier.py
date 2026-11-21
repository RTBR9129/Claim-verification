"""Train a FEVEROUS verifier head over the project's frozen NLI encoder."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path
from random import Random

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from claim_verifier import (
    DATA_DIR, INDEX_FILE, LABELS, MODEL_DIR, MODEL_FILE, TRAIN_FILE,
    VALIDATION_FILE, build_search_index, load_referenced_evidence,
    load_training_records, normalize_claim, retrieve_evidence,
)
from evidence_pipeline import MODEL_CACHE, NLI_MODEL


MAX_TOKENS = 256
TOP_K = 3
NEI = "NOT ENOUGH INFO"
STOP_WORDS = set("a an the is was were are of at in on and to from by has have had it with".split())


def annotated_context(record, lookup):
    """Serialize the first complete annotation, including available table context."""
    for group in record["evidence"]:
        content = group.get("content", [])
        if not content or any(not lookup.get(key, {}).get("text", "").strip() for key in content):
            continue
        parts = []
        context_ids = group.get("context", {})
        for key in content:
            item = lookup[key]
            context = []
            for context_id in context_ids.get(key, []):
                row = lookup.get(context_id)
                if row and row["text"].strip() and row["text"] != item["title"]:
                    context.append(row["text"].strip())
            details = " | ".join(dict.fromkeys([*context, item["text"].strip()]))
            parts.append(f"{item['title']}: {details}")
        return "\n".join(dict.fromkeys(parts))
    return ""


def content_terms(text):
    return set(re.findall(r"\w+", text.casefold())) - STOP_WORDS


def retrieved_context(claim, db):
    """Keep several relevant spans, with the page title for each passage."""
    terms = content_terms(claim)
    parts = []
    for item in retrieve_evidence(claim, top_k=TOP_K, db=db):
        spans = []
        for sentence in re.split(r"(?<=[.!?])\s+", item["text"]):
            words = sentence.split()
            if len(words) <= 80:
                if words:
                    spans.append(sentence.strip())
            else:
                spans.extend(" ".join(words[i:i + 80]) for i in range(0, len(words), 60))
        if not spans:
            continue
        scores = [len(terms & content_terms(span)) for span in spans]
        best = max(range(len(spans)), key=scores.__getitem__)
        parts.append(f"{item['title']}: {spans[best]}")
    return "\n".join(parts)


def distinct_records(records, excluded):
    seen = set(excluded)
    unique = []
    for record in records:
        key = normalize_claim(record["claim"])
        if key and key not in seen:
            unique.append(record)
            seen.add(key)
    return unique


def balanced_training_sample(records, per_label, seed):
    sampler = Random(seed)
    sample = []
    for label in sorted(LABELS):
        candidates = [record for record in records if record["label"] == label]
        sample.extend(sampler.sample(candidates, min(len(candidates), per_label)))
    sampler.shuffle(sample)
    return sample


def encode_pairs(encoder, contexts, claims, split, batch_size=12):
    import numpy as np
    import torch

    hidden_batches = []
    logit_batches = []
    encoder.model.eval()
    with torch.inference_mode():
        for start in range(0, len(claims), batch_size):
            stop = start + batch_size
            inputs = encoder.tokenizer(
                contexts[start:stop], claims[start:stop],
                padding=True, truncation=True, max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            sequence = encoder.model.base_model(**inputs).last_hidden_state
            hidden_batches.append(sequence[:, 0].cpu().numpy())
            logit_batches.append(encoder.model.classifier(sequence).cpu().numpy())
            if stop % 60 == 0 or stop >= len(claims):
                print(f"  Encoded {split}: {min(stop, len(claims))}/{len(claims)}", flush=True)
    return {
        "nli_scores": np.concatenate(logit_batches),
        "nli_scores_and_embedding": np.hstack([
            np.concatenate(logit_batches), np.concatenate(hidden_batches),
        ]),
    }


def score(expected, predicted):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(
            expected, predicted, average="macro", labels=sorted(LABELS), zero_division=0,
        )),
        "per_label": classification_report(
            expected, predicted, labels=sorted(LABELS), output_dict=True, zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-label", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=90)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()
    if min(args.train_per_label, args.threads) < 1 or min(args.validation_size, args.test_size) < 60:
        parser.error("Training count and threads must be positive. Validation and test need at least 60 claims.")

    import joblib
    import torch
    from sentence_transformers import CrossEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(args.threads)
    training = distinct_records(load_training_records(TRAIN_FILE), set())
    training_keys = {normalize_claim(r["claim"]) for r in training}
    validation = distinct_records(load_training_records(VALIDATION_FILE), training_keys)
    validation_keys = training_keys | {normalize_claim(r["claim"]) for r in validation}
    test = distinct_records(load_training_records(DATA_DIR / "test_claims.jsonl"), validation_keys)
    training_sample = balanced_training_sample(training, args.train_per_label, args.seed)

    def sample_split(records, size, seed):
        selected, _ = train_test_split(
            records, train_size=size, random_state=seed, stratify=[r["label"] for r in records],
        )
        return selected

    dev_sample = sample_split(validation, args.validation_size, args.seed)
    test_sample = sample_split(test, args.test_size, args.seed + 1)
    print(f"Training {len(training_sample)}, validation {len(dev_sample)}, test {len(test_sample)}.", flush=True)
    annotated = [r for r in training_sample if r["label"] != NEI]
    lookup = load_referenced_evidence(annotated)

    build_search_index()
    with closing(sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True)) as db:
        gold_count = 0
        train_contexts = []
        for record in training_sample:
            context = annotated_context(record, lookup) if record["label"] != NEI else ""
            if context:
                gold_count += 1
            else:
                context = retrieved_context(record["claim"], db)
            train_contexts.append(context)
        print(f"Training contexts: {gold_count} annotated, {len(train_contexts) - gold_count} retrieved.", flush=True)
        dev_contexts = [retrieved_context(record["claim"], db) for record in dev_sample]
        test_contexts = [retrieved_context(record["claim"], db) for record in test_sample]

    train_claims = [r["claim"] for r in training_sample]
    train_labels = [r["label"] for r in training_sample]
    dev_claims = [r["claim"] for r in dev_sample]
    dev_labels = [r["label"] for r in dev_sample]
    test_claims = [r["claim"] for r in test_sample]
    test_labels = [r["label"] for r in test_sample]

    encoder = CrossEncoder(
        NLI_MODEL, device="cpu", cache_folder=str(MODEL_CACHE), local_files_only=True,
    )
    train_features = encode_pairs(encoder, train_contexts, train_claims, "training")
    dev_features = encode_pairs(encoder, dev_contexts, dev_claims, "validation")

    candidates = []
    for feature_set, features in train_features.items():
        for strength in (0.01, 0.1, 1.0):
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(C=strength, max_iter=500, random_state=args.seed),
            )
            classifier.fit(features, train_labels)
            metrics = score(dev_labels, classifier.predict(dev_features[feature_set]))
            candidates.append((metrics, feature_set, strength, classifier))
    dev_metrics, feature_set, strength, model = max(
        candidates, key=lambda candidate: (candidate[0]["macro_f1"], candidate[0]["accuracy"]),
    )
    print(f"Validation selected {feature_set}, C={strength}, accuracy={dev_metrics['accuracy']:.1%}.", flush=True)

    test_features = encode_pairs(encoder, test_contexts, test_claims, "test")
    model_predictions = model.predict(test_features[feature_set])
    baseline = joblib.load(MODEL_FILE)
    baseline_predictions = baseline.predict(test_claims)
    majority = Counter(r["label"] for r in training).most_common(1)[0][0]
    report = {
        "seed": args.seed,
        "training_claims": len(train_labels),
        "training_label_counts": dict(Counter(train_labels)),
        "annotated_training_contexts": gold_count,
        "encoder": NLI_MODEL,
        "feature_set": feature_set,
        "classifier_c": strength,
        "max_tokens": MAX_TOKENS,
        "validation_claims": len(dev_labels),
        "validation_selection": dev_metrics,
        "test_claims": len(test_labels),
        "test_label_counts": dict(Counter(test_labels)),
        "test_evidence": "SQLite FTS5 BM25, three passages per claim",
        "test_claims_deduplicated_against_training_and_validation": True,
        "manual_corrections_applied": False,
        "test_trained_verifier": score(test_labels, model_predictions),
        "test_claim_only_baseline": score(test_labels, baseline_predictions),
        "test_majority_reference": score(test_labels, [majority] * len(test_labels)),
    }
    report_path = MODEL_DIR / "feverous_supervised_experiment.json"
    model_path = MODEL_DIR / "feverous_supervised_head.joblib"
    joblib.dump({
        "classifier": model, "encoder": NLI_MODEL, "feature_set": feature_set, "max_tokens": MAX_TOKENS,
    }, model_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("test_trained_verifier", "test_claim_only_baseline", "test_majority_reference"):
        result = report[name]
        print(f"{name}: accuracy={result['accuracy']:.1%}, macro F1={result['macro_f1']:.3f}")
    print(f"Saved {report_path} and {model_path}.")


if __name__ == "__main__":
    main()

