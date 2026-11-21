"""Train and evaluate a claim classifier with word and character features."""

from __future__ import annotations

import json
from collections import Counter
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from train_feverous_verifier import distinct_records
from claim_verifier import (
    DATA_DIR, LABELS, MODEL_DIR, MODEL_FILE, SVM_MODEL_FILE, TRAIN_FILE, VALIDATION_FILE,
    load_training_records, normalize_claim,
)


def load_splits():
    seen = set()
    for path in (TRAIN_FILE, VALIDATION_FILE, DATA_DIR / "test_claims.jsonl"):
        rows = distinct_records(load_training_records(path), seen)
        claims = [row["claim"] for row in rows]
        seen.update(normalize_claim(claim) for claim in claims)
        yield claims, [row["label"] for row in rows]


def scores(expected, predicted):
    labels = sorted(LABELS)
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(
            expected, predicted, labels=labels, average="macro", zero_division=0,
        )),
        "per_label": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0,
        ),
    }


def main():
    (train, train_labels), (dev, dev_labels), (test, test_labels) = load_splits()
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=80_000,
            sublinear_tf=True, dtype=np.float32,
        )),
        ("character", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=60_000,
            sublinear_tf=True, dtype=np.float32,
        )),
    ])
    train_vectors = features.fit_transform(train)
    dev_vectors = features.transform(dev)
    # Selected by macro F1 on validation in the comparison of C and class weights.
    classifier = LinearSVC(C=0.5, class_weight="balanced", random_state=91)
    classifier.fit(train_vectors, train_labels)
    dev_metrics = scores(dev_labels, classifier.predict(dev_vectors))
    del train_vectors, dev_vectors

    model = Pipeline([("features", features), ("classifier", classifier)])
    baseline = joblib.load(MODEL_FILE)
    majority = Counter(train_labels).most_common(1)[0][0]
    test_predictions = model.predict(test)
    report = {
        "train_claims": len(train),
        "validation_claims": len(dev),
        "test_claims": len(test),
        "test_label_counts": dict(Counter(test_labels)),
        "normalized_claim_duplicates_removed": True,
        "manual_corrections_applied": False,
        "validation_word_character_svm": dev_metrics,
        "test_word_character_svm": scores(test_labels, test_predictions),
        "test_claim_only_baseline": scores(test_labels, baseline.predict(test)),
        "test_majority_reference": scores(test_labels, [majority] * len(test)),
    }
    model_path = SVM_MODEL_FILE
    report_path = MODEL_DIR / "word_character_svm_test.json"
    joblib.dump(model, model_path, compress=3)
    restored = joblib.load(model_path)
    if not np.array_equal(restored.predict(test[:10]), test_predictions[:10]):
        raise RuntimeError("Saved model predictions changed after reloading.")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report.items():
        if isinstance(result, dict) and "accuracy" in result:
            print(f"{name}: accuracy={result['accuracy']:.2%}, macro F1={result['macro_f1']:.3f}")
    print(f"Saved {model_path} and {report_path}.")


if __name__ == "__main__":
    main()

