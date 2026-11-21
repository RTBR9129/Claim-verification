"""Fine-tune the cached NLI encoder on FEVEROUS and compare it with the SVM."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing

import joblib
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sklearn.model_selection import train_test_split

from train_feverous_verifier import (
    NEI, annotated_context, balanced_training_sample, distinct_records, retrieved_context, score,
)
from claim_verifier import (
    DATA_DIR, INDEX_FILE, LABELS, MODEL_DIR, SVM_MODEL_FILE, TRAIN_FILE, VALIDATION_FILE,
    build_search_index, load_referenced_evidence, load_training_records, normalize_claim,
)
from evidence_pipeline import MODEL_CACHE, NLI_MODEL


NLI_TO_DATASET = {"entailment": "SUPPORTS", "contradiction": "REFUTES", "neutral": NEI}
MAX_TOKENS = 160
BATCH_SIZE = 8


def nli_label_order(id2label):
    order = [NLI_TO_DATASET[str(id2label[index]).casefold()] for index in range(len(id2label))]
    if len(order) != len(LABELS) or set(order) != LABELS:
        raise ValueError("The NLI model must have three distinct FEVEROUS relations.")
    return order


def trainable_parameters(model):
    """Update the last two transformer layers and the NLI classification head."""
    model.requires_grad_(False)
    model.base_model.encoder.layer[-2:].requires_grad_(True)
    model.classifier.requires_grad_(True)
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def paired_changes(expected, baseline, candidate):
    fixed = sum(old != gold and new == gold for gold, old, new in zip(expected, baseline, candidate))
    broke = sum(old == gold and new != gold for gold, old, new in zip(expected, baseline, candidate))
    return {"fixed": fixed, "broke": broke, "net": fixed - broke}


def load_split_samples(per_label, validation_size, test_size, seed):
    train = distinct_records(load_training_records(TRAIN_FILE), set())
    used = {normalize_claim(row["claim"]) for row in train}
    validation = distinct_records(load_training_records(VALIDATION_FILE), used)
    used.update(normalize_claim(row["claim"]) for row in validation)
    test = distinct_records(load_training_records(DATA_DIR / "test_claims.jsonl"), used)

    def stratified(rows, size, rng):
        selected, _ = train_test_split(
            rows, train_size=size, random_state=rng, stratify=[row["label"] for row in rows],
        )
        return selected

    return (
        balanced_training_sample(train, per_label, seed),
        stratified(validation, validation_size, seed),
        stratified(test, test_size, seed + 1),
    )


def prepare_inputs(training, validation, test):
    lookup = load_referenced_evidence([row for row in training if row["label"] != NEI])
    build_search_index()
    with closing(sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True)) as db:
        annotated = 0
        train_pairs = []
        for row in training:
            context = annotated_context(row, lookup) if row["label"] != NEI else ""
            annotated += bool(context)
            train_pairs.append((context or retrieved_context(row["claim"], db), row["claim"]))
        print(f"Training inputs: {annotated} annotated, {len(training) - annotated} retrieved.", flush=True)
        validation_pairs = [(retrieved_context(row["claim"], db), row["claim"]) for row in validation]
        print(f"Prepared {len(validation_pairs)} validation pairs from retrieval.", flush=True)
        test_pairs = [(retrieved_context(row["claim"], db), row["claim"]) for row in test]
        print(f"Prepared {len(test_pairs)} test pairs from retrieval.", flush=True)
    return train_pairs, validation_pairs, test_pairs, annotated


def encode(tokenizer, pairs, device):
    return tokenizer(
        [context for context, _ in pairs],
        [claim for _, claim in pairs],
        padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt",
    ).to(device)


def predict(model, tokenizer, pairs, order, device):
    model.eval()
    labels = []
    with torch.inference_mode():
        for start in range(0, len(pairs), BATCH_SIZE):
            inputs = encode(tokenizer, pairs[start:start + BATCH_SIZE], device)
            indices = model(**inputs).logits.argmax(dim=1).cpu().tolist()
            labels.extend(order[index] for index in indices)
    return labels


def run_training(model, tokenizer, train, targets, validation, validation_labels, order, device, epochs, seed):
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW([param for _, param in parameters], lr=2e-5)
    best_metrics = best_state = None
    best_epoch = 0
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = list(range(len(train)))
        rng.shuffle(shuffled)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(shuffled), BATCH_SIZE):
            indices = shuffled[start:start + BATCH_SIZE]
            inputs = encode(tokenizer, [train[index] for index in indices], device)
            batch_labels = torch.tensor([targets[index] for index in indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=batch_labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for _, param in parameters], max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
            if steps % 8 == 0:
                print(f"  Epoch {epoch}: {min(start + BATCH_SIZE, len(train))}/{len(train)} pairs.", flush=True)
        estimates = predict(model, tokenizer, validation, order, device)
        metrics = score(validation_labels, estimates)
        print(
            f"Epoch {epoch}: loss={loss_sum / steps:.3f}, validation accuracy={metrics['accuracy']:.1%}, "
            f"macro F1={metrics['macro_f1']:.3f}.",
            flush=True,
        )
        if best_metrics is None or (metrics["macro_f1"], metrics["accuracy"]) > (
            best_metrics["macro_f1"], best_metrics["accuracy"],
        ):
            best_epoch = epoch
            best_metrics = metrics
            best_state = {name: param.detach().cpu().clone() for name, param in parameters}

    with torch.no_grad():
        for name, param in parameters:
            param.copy_(best_state[name])
    return best_metrics, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-label", type=int, default=40)
    parser.add_argument("--validation-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()
    if min(args.train_per_label, args.epochs) < 1 or min(args.validation_size, args.test_size) < 60:
        parser.error("Use positive training counts and epochs and at least 60 validation and test claims.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, test = load_split_samples(
        args.train_per_label, args.validation_size, args.test_size, args.seed,
    )
    print(f"Fine-tuning on {device}: {len(train)} train, {len(validation)} validation, {len(test)} test.", flush=True)
    train_pairs, validation_pairs, test_pairs, annotated = prepare_inputs(train, validation, test)
    encoder = CrossEncoder(
        NLI_MODEL, device=str(device), cache_folder=str(MODEL_CACHE), local_files_only=True,
    )
    model, tokenizer = encoder.model, encoder.tokenizer
    order = nli_label_order(model.config.id2label)
    train_targets = [order.index(row["label"]) for row in train]
    validation_labels = [row["label"] for row in validation]
    best, epoch = run_training(
        model, tokenizer, train_pairs, train_targets, validation_pairs, validation_labels,
        order, device, args.epochs, args.seed,
    )

    test_labels = [row["label"] for row in test]
    estimates = predict(model, tokenizer, test_pairs, order, device)
    svm = joblib.load(SVM_MODEL_FILE)
    baseline = svm.predict([row["claim"] for row in test])
    report = {
        "encoder": NLI_MODEL,
        "updated_weights": "last two transformer layers and classification head",
        "seed": args.seed,
        "training_claims": len(train),
        "annotated_training_contexts": annotated,
        "train_label_counts": dict(Counter(row["label"] for row in train)),
        "validation_claims": len(validation),
        "selected_epoch": epoch,
        "selected_validation_scores": best,
        "test_claims": len(test),
        "test_label_counts": dict(Counter(test_labels)),
        "test_evidence": "BM25 retrieved passages; no annotated evidence used during test inference",
        "manual_corrections_applied": False,
        "fine_tuned_test": score(test_labels, estimates),
        "svm_test": score(test_labels, baseline),
        "paired_changes_vs_svm": paired_changes(test_labels, baseline, estimates),
    }
    output = MODEL_DIR / "feverous_fine_tuning_pilot.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("fine_tuned_test", "svm_test"):
        result = report[name]
        print(f"{name}: accuracy={result['accuracy']:.1%}, macro F1={result['macro_f1']:.3f}")
    print(f"Paired comparison: {report['paired_changes_vs_svm']}")
    print(f"Saved {output}.")


if __name__ == "__main__":
    main()

