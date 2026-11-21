"""Claim verifier with a TF-IDF label model and retrieved review passages."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "evidence_chunks.jsonl"
INDEX_FILE = DATA_DIR / "evidence_search.sqlite3"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "claim_verifier.joblib"
SVM_MODEL_FILE = MODEL_DIR / "word_character_svm.joblib"
MODEL_INFO_FILE = MODEL_DIR / "claim_verifier.json"
CORRECTIONS_FILE = ROOT / "label_corrections.jsonl"
TRAIN_FILE = DATA_DIR / "train_claims.jsonl"
VALIDATION_FILE = DATA_DIR / "validation_claims.jsonl"
FALLBACK_TRAIN_FILE = DATA_DIR / "clean_train.jsonl"
EVIDENCE_LOOKUP_FILE = DATA_DIR / "evidence_lookup_full.json"
LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_TOP_K = 3
MAX_EVIDENCE_CHARS = 900
INDEX_VERSION = "2"


def require_chunk_data() -> None:
    if not CHUNKS_FILE.is_file():
        raise FileNotFoundError(
            f"Evidence chunks are missing: {CHUNKS_FILE}\n"
            "Run the data preparation steps in README.md first."
        )


def build_search_index(force: bool = False) -> int:
    """Build a small-memory FTS5 index from the existing chunk JSONL."""
    require_chunk_data()
    source_mtime = CHUNKS_FILE.stat().st_mtime_ns

    if INDEX_FILE.exists() and not force:
        try:
            with sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True) as db:
                metadata = dict(db.execute("SELECT key,value FROM metadata"))
                count = db.execute("SELECT count(*) FROM evidence").fetchone()[0]
            if (metadata.get("source_mtime_ns") == str(source_mtime)
                    and metadata.get("index_version") == INDEX_VERSION):
                print(f"Search index is current ({count:,} evidence chunks).")
                return count
        except sqlite3.Error:
            pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = INDEX_FILE.with_suffix(".building.sqlite3")
    if temp_file.exists():
        temp_file.unlink()

    print(f"Building evidence search index from {CHUNKS_FILE} ...", flush=True)
    db = sqlite3.connect(temp_file)
    count = 0
    started = time.time()
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute(
            "CREATE VIRTUAL TABLE evidence USING fts5("
            "title, text, chunk_id UNINDEXED, tokenize='unicode61')"
        )
        batch: list[tuple[str, str, str]] = []
        with CHUNKS_FILE.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                batch.append((
                    str(record.get("title", "")),
                    str(record.get("text", "")),
                    str(record.get("chunk_id", "")),
                ))
                if len(batch) >= 2000:
                    db.executemany(
                        "INSERT INTO evidence(title,text,chunk_id) VALUES(?,?,?)",
                        batch,
                    )
                    count += len(batch)
                    batch.clear()
                    if count % 20000 == 0:
                        print(f"  Indexed {count:,} chunks", flush=True)
            if batch:
                db.executemany(
                    "INSERT INTO evidence(title,text,chunk_id) VALUES(?,?,?)",
                    batch,
                )
                count += len(batch)
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute(
            "INSERT INTO metadata(key,value) VALUES('source_mtime_ns',?)",
            (str(source_mtime),),
        )
        db.execute(
            "INSERT INTO metadata(key,value) VALUES('index_version',?)",
            (INDEX_VERSION,),
        )
        db.execute("CREATE VIRTUAL TABLE evidence_vocab USING fts5vocab(evidence, 'row')")
        db.commit()
    except Exception:
        db.close()
        if temp_file.exists():
            temp_file.unlink()
        raise
    else:
        db.close()

    os.replace(temp_file, INDEX_FILE)
    print(
        f"Indexed {count:,} chunks in {time.time() - started:.1f}s. "
        f"Index: {INDEX_FILE}",
        flush=True,
    )
    return count


def _search(db: sqlite3.Connection, query: str, top_k: int) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT title, chunk_id, text, bm25(evidence, 3.0, 1.0, 0.0) AS score "
        "FROM evidence WHERE evidence MATCH ? ORDER BY score, rowid LIMIT ?",
        (query, top_k),
    ).fetchall()
    return [
        {"title": title, "chunk_id": chunk_id, "text": text, "score": float(score)}
        for title, chunk_id, text, score in rows
    ]


def retrieve_evidence(
    claim: str, top_k: int = DEFAULT_TOP_K,
    db: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    if db is None and not INDEX_FILE.is_file():
        build_search_index()
    raw_terms = list(dict.fromkeys(TOKEN_RE.findall(claim)))
    terms = list(dict.fromkeys(term.casefold() for term in raw_terms))
    if not terms:
        return []
    own_connection = db is None
    if own_connection:
        db = sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True)
    assert db is not None
    try:
        placeholders = ",".join("?" for _ in terms)
        frequencies = dict(db.execute(
            f"SELECT term,doc FROM evidence_vocab WHERE term IN ({placeholders})",
            terms,
        ).fetchall())
        informative = sorted(
            (term for term in terms if term in frequencies),
            key=lambda term: (frequencies[term], term),
        )[:8]
        if not informative:
            return []

        # Search proper-name terms in page titles first. This keeps a claim
        # about Shulin grounded on the Shulin District page instead of a
        # nearby page that merely mentions Shulin and China.
        title_queries: list[list[str]] = []
        current_entity: list[str] = []
        connectors = {"of", "the", "and", "de", "van", "von", "la", "di"}
        for term in raw_terms:
            if term[:1].isupper() and term.casefold() not in {"the", "a", "an"}:
                current_entity.append(term.casefold())
            elif term.casefold() in connectors and current_entity:
                current_entity.append(term.casefold())
            else:
                if current_entity:
                    title_queries.append(current_entity)
                    current_entity = []
        if current_entity:
            title_queries.append(current_entity)
        if len(title_queries) > 1:
            broad_place_titles = {"china", "taiwan"}
            title_queries = [
                entity for entity in title_queries
                if " ".join(entity) not in broad_place_titles
            ]
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add_results(found: list[dict[str, Any]]) -> None:
            for item in found:
                key = (item["title"].casefold(), str(item["chunk_id"]))
                if key not in seen:
                    seen.add(key)
                    results.append(item)

        for entity in title_queries[:6]:
            query = " AND ".join(f'title:"{term}"' for term in entity)
            add_results(_search(db, query, top_k))
            if len(results) >= top_k:
                break

        # Rare claim terms make AND searches both faster and more specific.
        for size in range(min(6, len(informative)), 1, -1):
            query = " AND ".join(f'"{term}"' for term in informative[:size])
            add_results(_search(db, query, top_k))
            if len(results) >= top_k:
                return results[:top_k]
        query = " OR ".join(f'"{term}"' for term in informative[:6])
        add_results(_search(db, query, top_k))
        return results[:top_k]
    finally:
        if own_connection:
            db.close()


def load_training_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            claim = record.get("claim")
            label = record.get("label")
            if isinstance(claim, str) and claim.strip() and label in LABELS:
                evidence = record.get("evidence", [])
                records.append({
                    "claim": claim.strip(),
                    "label": label,
                    "evidence": evidence if isinstance(evidence, list) else [],
                })
    if not records:
        raise ValueError(f"No usable labeled claims found in {path}")
    return records


class StreamingJSONReader:
    """Read a large top-level JSON object one key/value pair at a time."""

    def __init__(self, path: Path):
        self.stream = path.open("r", encoding="utf-8")
        self.decoder = json.JSONDecoder()
        self.buffer = self.stream.read(1 << 20)
        self.position = 0
        self.eof = not self.buffer

    def _fill(self) -> bool:
        remainder = self.buffer[self.position:]
        more = self.stream.read(1 << 20)
        self.buffer = remainder + more
        self.position = 0
        self.eof = not more
        return bool(more)

    def _skip_whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def _decode_value(self) -> Any:
        while True:
            self._skip_whitespace()
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError:
                if not self._fill():
                    raise

    def items(self):
        self._skip_whitespace()
        if self.buffer[self.position:self.position + 1] != "{":
            raise ValueError("Evidence lookup must be a JSON object")
        self.position += 1
        first = True
        while True:
            self._skip_whitespace()
            if self.buffer[self.position:self.position + 1] == "}":
                self.position += 1
                return
            if not first:
                if self.buffer[self.position:self.position + 1] != ",":
                    raise ValueError("Malformed evidence lookup: expected comma")
                self.position += 1
            key = self._decode_value()
            self._skip_whitespace()
            if self.buffer[self.position:self.position + 1] != ":":
                raise ValueError("Malformed evidence lookup: expected colon")
            self.position += 1
            yield key, self._decode_value()
            first = False


def collect_evidence_ids(records: list[dict[str, Any]]) -> set[str]:
    needed: set[str] = set()
    for record in records:
        for group in record.get("evidence", []):
            if not isinstance(group, dict):
                continue
            for evidence_id in group.get("content", []):
                if isinstance(evidence_id, str):
                    needed.add(evidence_id)
                for context_id in group.get("context", {}).get(evidence_id, []):
                    if isinstance(context_id, str):
                        needed.add(context_id)
    return needed


def load_referenced_evidence(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if not EVIDENCE_LOOKUP_FILE.is_file():
        raise FileNotFoundError(f"Evidence lookup is missing: {EVIDENCE_LOOKUP_FILE}")
    needed = collect_evidence_ids(records)
    normalized = {key: unicodedata.normalize("NFC", key) for key in needed}
    matchable = set(normalized.values())
    requested_titles = {key.removesuffix("_title") for key in matchable if key.endswith("_title")}
    exact: dict[str, dict[str, str]] = {}
    aliases: dict[str, dict[str, str] | None] = {}
    known_titles: dict[str, str] = {}

    reader = StreamingJSONReader(EVIDENCE_LOOKUP_FILE)
    try:
        for count, (key, value) in enumerate(reader.items(), 1):
            if isinstance(value, dict):
                if requested_titles and isinstance(value.get("title"), str):
                    title = value["title"]
                    canonical_title = unicodedata.normalize("NFC", title)
                    if canonical_title in requested_titles:
                        known_titles[canonical_title] = canonical_title

                canonical_key = key if key.isascii() else unicodedata.normalize("NFC", key)
                if canonical_key in matchable:
                    entry = {
                        "title": str(value.get("title", "")),
                        "text": str(value.get("text", "")),
                    }
                    if key in needed:
                        exact[key] = entry
                    if canonical_key not in aliases:
                        aliases[canonical_key] = entry
                    elif aliases[canonical_key] != entry:
                        # Canonically equivalent spellings can have conflicting values.
                        aliases[canonical_key] = None

            if count % 1_000_000 == 0:
                print(f"  Scanned {count:,} lookup records.", flush=True)
    finally:
        reader.stream.close()

    found = dict(exact)
    alias_count = title_count = 0
    for key in needed - exact.keys():
        canonical = normalized[key]
        entry = aliases.get(canonical)
        if entry is not None:
            found[key] = entry
            alias_count += 1
        elif canonical not in aliases and canonical.endswith("_title"):
            title = known_titles.get(canonical.removesuffix("_title"))
            if title is not None:
                found[key] = {"title": title, "text": title}
                title_count += 1
    print(
        f"Annotated evidence available: {len(found):,}/{len(needed):,} IDs "
        f"({len(found) / max(1, len(needed)):.1%}). "
        f"Recovered {alias_count:,} normalized IDs and {title_count:,} legacy title entries.",
        flush=True,
    )
    return found


def gold_evidence_groups(
    record: dict[str, Any], lookup: dict[str, dict[str, str]],
) -> list[str]:
    groups: list[str] = []
    for group in record.get("evidence", []):
        if not isinstance(group, dict):
            continue
        context = group.get("context", {})
        parts: list[str] = []
        seen: set[str] = set()
        for evidence_id in group.get("content", []):
            ids = [evidence_id]
            ids.extend(context.get(evidence_id, []))
            for item_id in ids:
                item = lookup.get(item_id) if isinstance(item_id, str) else None
                if not item or not item["text"].strip():
                    continue
                key = item["title"] + "\n" + item["text"].casefold()
                if key not in seen:
                    seen.add(key)
                    parts.append(f"{item['title']}: {item['text']}")
        if parts:
            groups.append("\n".join(parts)[:12_000])
    return groups[:5]


def model_input(claim: str, evidence: list[dict[str, Any]] | str) -> str:
    if isinstance(evidence, str):
        context = evidence
    else:
        context = "\n".join(
            f"{item['title']}: {item['text']}" for item in evidence
        )
    return f"Claim: {claim}\nEvidence: {context}"


def make_pair_examples(
    records: list[dict[str, Any]], lookup: dict[str, dict[str, str]],
    top_k: int, split_name: str, all_groups: bool = True,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    retrieval_count = 0
    with sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True) as db:
        for number, record in enumerate(records, 1):
            groups = gold_evidence_groups(record, lookup)
            if not all_groups:
                groups = groups[:1]
            if groups:
                for evidence_text in groups:
                    texts.append(model_input(record["claim"], evidence_text))
                    labels.append(record["label"])
            else:
                evidence = retrieve_evidence(record["claim"], top_k=top_k, db=db)
                texts.append(model_input(record["claim"], evidence))
                labels.append(record["label"])
                retrieval_count += 1
            if number % 1_000 == 0 or number == len(records):
                print(
                    f"  Prepared {split_name}: {number:,}/{len(records):,} claims",
                    flush=True,
                )
    print(
        f"  {split_name}: {len(texts):,} claim/evidence examples; "
        f"retrieved evidence used for {retrieval_count:,} claims without mapped annotations.",
        flush=True,
    )
    return texts, labels


def normalize_claim(claim: str) -> str:
    return " ".join(TOKEN_RE.findall(claim.casefold()))


def load_label_corrections() -> dict[str, dict[str, str]]:
    corrections: dict[str, dict[str, str]] = {}
    if not CORRECTIONS_FILE.is_file():
        return corrections
    with CORRECTIONS_FILE.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid correction at {CORRECTIONS_FILE}:{line_number}: {error}"
                ) from error
            claim = item.get("claim")
            label = item.get("label")
            if not isinstance(claim, str) or label not in LABELS:
                raise ValueError(
                    f"Correction at {CORRECTIONS_FILE}:{line_number} needs a claim and valid label"
                )
            corrections[normalize_claim(claim)] = {
                "label": label,
                "note": str(item.get("note", "User-provided correction")),
            }
    return corrections


def train_model(max_claims: int, seed: int) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Run: python -m pip install -r requirements.txt"
        ) from error

    data_path = TRAIN_FILE if TRAIN_FILE.is_file() else FALLBACK_TRAIN_FILE
    train_records = load_training_records(data_path)
    validation_path = (
        VALIDATION_FILE
        if data_path == TRAIN_FILE and VALIDATION_FILE.is_file()
        else None
    )
    if validation_path:
        validation_records = load_training_records(validation_path)
    else:
        train_records, validation_records = train_test_split(
            train_records,
            test_size=0.2,
            random_state=seed,
            stratify=[record["label"] for record in train_records],
        )
    if max_claims and len(train_records) > max_claims:
        # Stratified sample keeps the smaller NOT ENOUGH INFO class represented.
        train_records, _ = train_test_split(
            train_records,
            train_size=max_claims,
            random_state=seed,
            stratify=[record["label"] for record in train_records],
        )
    print(
        f"Training on {len(train_records):,} claims; "
        f"validating on {len(validation_records):,} separate claims.",
        flush=True,
    )
    train_x = [record["claim"] for record in train_records]
    train_y = [record["label"] for record in train_records]
    validation_x = [record["claim"] for record in validation_records]
    validation_y = [record["label"] for record in validation_records]

    classifier = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    print("Fitting classifier...", flush=True)
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(validation_x)
    accuracy = float(accuracy_score(validation_y, predictions))
    macro_f1 = float(f1_score(validation_y, predictions, average="macro"))
    report = classification_report(
        validation_y, predictions, labels=sorted(LABELS),
        output_dict=True, zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_FILE, compress=3)
    info = {
        "model": "TF-IDF + logistic regression",
        "training_file": str(data_path.relative_to(ROOT)),
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "validation_file": (
            str(validation_path.relative_to(ROOT))
            if validation_path else "stratified holdout"
        ),
        "prediction_uses_retrieved_evidence": False,
        "training_input": "claim text only",
        "validation_input": "claim text only",
        "evidence_display": "SQLite FTS5 BM25 over evidence_chunks.jsonl",
        "seed": seed,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "validation_report": report,
    }
    MODEL_INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Claim-only validation accuracy: {accuracy:.4f}")
    print(f"Claim-only validation macro F1: {macro_f1:.4f}")
    print(classification_report(validation_y, predictions, zero_division=0))
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved training report: {MODEL_INFO_FILE}")


def verify_claim(claim: str, top_k: int) -> None:
    import joblib
    from evidence_pipeline import verify_claim_pipeline

    if not SVM_MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Default claim model not found: {SVM_MODEL_FILE}\n"
            "Train it first with: python experiments/train_word_character_svm.py"
        )
    classifier = joblib.load(SVM_MODEL_FILE)
    model_label = str(classifier.predict([claim])[0])
    margins = classifier.decision_function([claim])[0]
    correction = load_label_corrections().get(normalize_claim(claim))
    final_label = correction["label"] if correction else model_label

    build_search_index()
    result = verify_claim_pipeline(claim, top_k, retrieve_evidence)
    evidence = result["evidence"]

    print(f"\nClaim: {claim}")
    print(f"Final label: {final_label}")
    if correction:
        print(f"Applied manual correction: {correction['note']}")
    print("Model: Word and character TF-IDF with a linear SVM")
    print(f"Claim-only model estimate: {model_label}")
    print("SVM decision scores (higher favors the label; scores are not probabilities):")
    for class_label, margin in sorted(
        zip(classifier.classes_, margins), key=lambda pair: pair[1], reverse=True,
    ):
        print(f"  {class_label}: {margin:+.3f}")
    print(f"Evidence NLI estimate (advisory): {result['label']}")
    print(f"Evidence NLI confidence estimate: {result['confidence']:.1%} ({result['calibration']})")
    directional_labels = {"SUPPORTS", "REFUTES"}
    opposing_labels = {model_label, result["label"]} == directional_labels
    review_recommended = result["human_review"] or opposing_labels
    review_reason = " (claim model and evidence estimate give opposite labels)" if opposing_labels else ""
    print(f"Review recommended: {'yes' if review_recommended else 'no'}{review_reason}")
    print("Aggregated evidence relation scores:")
    for relation, score in result.get("nli_aggregate", {}).items():
        print(f"  {relation.upper()}: {score:.1%}")
    graph = result["graph"]
    print(
        "Evidence graph counts: "
        f"supports={graph['supports']}, refutes={graph['refutes']}, "
        f"neutral={graph['neutral']}"
    )
    print("\nRetrieved evidence:")
    if not evidence:
        print("  No matching passages found.")
    for rank, item in enumerate(evidence, 1):
        print(
            f"\n{rank}. {item['title']} (chunk {item['chunk_id']}) "
            f"relation={item['relation'].upper()} "
            f"RRF={item.get('rrf_score', 0.0):.4f} "
            f"rerank={item.get('rerank_score', 0.0):.3f}"
        )
        passage = item["text"].strip()
        if len(passage) > MAX_EVIDENCE_CHARS:
            passage = passage[:MAX_EVIDENCE_CHARS].rstrip() + " ... [truncated]"
        print(f"   {passage}")
        print(
            "   NLI: " + ", ".join(
                f"{name}={score:.1%}" for name, score in item["nli"].items()
            )
        )
        if item.get("nli_basis"):
            print(f"   NLI basis: {item['nli_basis']}")
    print("\nEvidence explanation (advisory):")
    print(result["explanation"])
    print("\nReview the retrieved passages before relying on the label.")


def evaluate_end_to_end() -> None:
    try:
        import joblib
        from sklearn.metrics import accuracy_score, classification_report, f1_score
    except ImportError as error:
        raise RuntimeError(
            "Evaluation dependencies are missing. Run: python -m pip install -r requirements.txt"
        ) from error
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Train it first with: python src/claim_verifier.py train"
        )
    records = load_training_records(VALIDATION_FILE)
    classifier = joblib.load(MODEL_FILE)
    build_search_index()
    features: list[str] = []
    labels = [record["label"] for record in records]
    with sqlite3.connect(f"file:{INDEX_FILE}?mode=ro", uri=True) as db:
        for number, record in enumerate(records, 1):
            evidence = retrieve_evidence(record["claim"], top_k=DEFAULT_TOP_K, db=db)
            features.append(record["claim"])
            if number % 250 == 0 or number == len(records):
                print(f"  Evaluated retrieval: {number:,}/{len(records):,}", flush=True)
    predictions = classifier.predict(features)
    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro"))
    report = classification_report(
        labels, predictions, labels=sorted(LABELS),
        output_dict=True, zero_division=0,
    )
    metrics_path = MODEL_DIR / "claim_verifier_end_to_end.json"
    metrics_path.write_text(json.dumps({
        "validation_file": str(VALIDATION_FILE.relative_to(ROOT)),
        "validation_records": len(records),
        "retrieval_top_k": DEFAULT_TOP_K,
        "prediction_uses_retrieved_evidence": False,
        "prediction_input": "claim text only; retrieved passages are displayed for review",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
    }, indent=2), encoding="utf-8")
    print(f"End-to-end validation accuracy: {accuracy:.4f}")
    print(f"End-to-end validation macro F1: {macro_f1:.4f}")
    print(classification_report(labels, predictions, zero_division=0))
    print(f"Saved end-to-end report: {metrics_path}")


def calibrate_hybrid(max_claims: int, seed: int) -> None:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from evidence_pipeline import reset_pipeline_for_calibration

    build_search_index()
    records = load_training_records(TRAIN_FILE)
    if max_claims and len(records) > max_claims:
        records, _ = train_test_split(
            records,
            train_size=max_claims,
            random_state=seed,
            stratify=[record["label"] for record in records],
        )
    pipeline = reset_pipeline_for_calibration(retrieve_evidence)
    outputs: list[list[float]] = []
    targets: list[int] = []
    label_indices = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}
    for number, record in enumerate(records, 1):
        result = pipeline.verify(record["claim"], DEFAULT_TOP_K)
        scores = result.get("nli_aggregate", {"refutes": 0, "supports": 0, "neutral": 1})
        outputs.append([
            scores.get("refutes", 0.0),
            scores.get("supports", 0.0),
            scores.get("neutral", 1.0),
        ])
        targets.append(label_indices[record["label"]])
        if number % 50 == 0 or number == len(records):
            print(f"  Calibrated sample: {number:,}/{len(records):,}", flush=True)

    probabilities = np.clip(np.asarray(outputs, dtype=np.float64), 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(probabilities)
    target_indices = np.arange(len(targets))
    target_array = np.asarray(targets, dtype=np.int64)
    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 501))
    nlls = []
    for temperature in temperatures:
        logits = log_probabilities / temperature
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        nlls.append(float(-np.log(scaled[target_indices, target_array]).mean()))
    best = int(np.argmin(nlls))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    calibration_path = MODEL_DIR / "evidence_calibration.json"
    calibration_path.write_text(json.dumps({
        "temperature": float(temperatures[best]),
        "temperature_search_min": float(temperatures[0]),
        "temperature_search_max": float(temperatures[-1]),
        "upper_bound_hit": best == len(temperatures) - 1,
        "calibration_records": len(records),
        "calibration_file": str(TRAIN_FILE.relative_to(ROOT)),
        "seed": seed,
        "nll_before": nlls[int(np.argmin(np.abs(temperatures - 1.0)))],
        "nll_after": nlls[best],
        "label_order": ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"],
    }, indent=2), encoding="utf-8")
    print(f"Temperature: {temperatures[best]:.3f}")
    print(f"Calibration NLL: {nlls[best]:.4f} (before: {nlls[int(np.argmin(np.abs(temperatures - 1.0)))]:.4f})")
    print(f"Saved confidence calibration: {calibration_path}")


def summarize_model_disagreements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure the effect of choosing the evidence label on disagreements."""
    counts = {
        "agreements": 0,
        "disagreements": 0,
        "baseline_errors_fixed": 0,
        "baseline_correct_answers_lost": 0,
        "both_wrong_on_disagreement": 0,
    }
    by_evidence_label: dict[str, dict[str, int]] = {
        label: {"disagreements": 0, "fixed": 0, "lost": 0, "both_wrong": 0}
        for label in sorted(LABELS)
    }
    any_model_correct = 0

    for row in rows:
        gold = row["expected_label"]
        baseline = row["baseline_label"]
        evidence = row["evidence_label"]
        any_model_correct += int(gold in (baseline, evidence))
        if baseline == evidence:
            counts["agreements"] += 1
            continue

        counts["disagreements"] += 1
        label_counts = by_evidence_label[evidence]
        label_counts["disagreements"] += 1
        if evidence == gold:
            counts["baseline_errors_fixed"] += 1
            label_counts["fixed"] += 1
        elif baseline == gold:
            counts["baseline_correct_answers_lost"] += 1
            label_counts["lost"] += 1
        else:
            counts["both_wrong_on_disagreement"] += 1
            label_counts["both_wrong"] += 1

    return {
        **counts,
        "by_evidence_label": by_evidence_label,
        "oracle_correct": any_model_correct,
        "oracle_accuracy": any_model_correct / len(rows) if rows else None,
    }


def evaluate_hybrid(max_claims: int, seed: int, top_k: int) -> None:
    from tempfile import NamedTemporaryFile

    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from evidence_pipeline import EvidencePipeline

    build_search_index()
    records = load_training_records(VALIDATION_FILE)
    if max_claims and len(records) > max_claims:
        records, _ = train_test_split(
            records,
            train_size=max_claims,
            random_state=seed,
            stratify=[record["label"] for record in records],
        )

    baseline_predictions = baseline_scores = None
    baseline_classes = []
    if MODEL_FILE.is_file():
        import joblib

        baseline = joblib.load(MODEL_FILE)
        claims = [record["claim"] for record in records]
        baseline_predictions = baseline.predict(claims)
        baseline_scores = baseline.predict_proba(claims)
        baseline_classes = [str(label) for label in baseline.classes_]

    pipeline = EvidencePipeline(retrieve_evidence)
    rows = []
    for index, record in enumerate(records):
        result = pipeline.verify(record["claim"], top_k)
        row = {
            "claim": record["claim"],
            "expected_label": record["label"],
            "evidence_label": result["label"],
            "evidence_scores": result.get("nli_aggregate"),
            "evidence_retrieved": len(result["evidence"]),
            "evidence_review_recommended": result["human_review"],
            "evidence_reason": result.get("reason"),
        }
        if baseline_predictions is not None and baseline_scores is not None:
            row["baseline_label"] = str(baseline_predictions[index])
            row["baseline_scores"] = {
                label: float(score)
                for label, score in zip(baseline_classes, baseline_scores[index])
            }
        rows.append(row)
        number = index + 1
        if number % 25 == 0 or number == len(records):
            print(f"  Hybrid evaluation: {number:,}/{len(records):,}", flush=True)

    labels = [row["expected_label"] for row in rows]
    predictions = [row["evidence_label"] for row in rows]
    label_order = sorted(LABELS)
    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, labels=label_order, average="macro", zero_division=0))
    baseline_metrics = disagreements = None
    if baseline_predictions is not None:
        baseline_metrics = {
            "accuracy": float(accuracy_score(labels, baseline_predictions)),
            "macro_f1": float(f1_score(
                labels, baseline_predictions, labels=label_order, average="macro", zero_division=0,
            )),
            "classification_report": classification_report(
                labels, baseline_predictions, labels=label_order, output_dict=True, zero_division=0,
            ),
        }
        disagreements = summarize_model_disagreements(rows)
    report = {
        "validation_file": str(VALIDATION_FILE.relative_to(ROOT)),
        "validation_records": len(records),
        "seed": seed,
        "retrieval_top_k": max(3, top_k),
        "pipeline": "BM25 + MiniLM FAISS + RRF + cross-encoder + NLI",
        "primary_prediction_policy": "comparison against the original logistic regression baseline; verify uses the SVM",
        "manual_corrections_applied": False,
        "confidence_temperature": pipeline.temperature,
        "evidence_score_status": pipeline.calibration_status,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "claim_only_baseline": baseline_metrics,
        "disagreements": disagreements,
        "classification_report": classification_report(
            labels, predictions, labels=label_order, output_dict=True, zero_division=0,
        ),
        "claims": rows,
    }
    metrics_path = MODEL_DIR / "hybrid_verifier_validation.json"
    if max_claims == 0:
        metrics_path = MODEL_DIR / "hybrid_verifier_full_validation.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=MODEL_DIR,
            prefix=metrics_path.stem + "-", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(report, output, ensure_ascii=False, indent=2)
        temporary_path.replace(metrics_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Hybrid validation accuracy: {accuracy:.4f}")
    print(f"Hybrid validation macro F1: {macro_f1:.4f}")
    if baseline_metrics and disagreements is not None:
        print(
            "Claim-only baseline on same sample: "
            f"accuracy={baseline_metrics['accuracy']:.4f}, "
            f"macro F1={baseline_metrics['macro_f1']:.4f}"
        )
        print(
            f"Model disagreements: {disagreements['disagreements']:,}. "
            f"Choosing the evidence label fixes {disagreements['baseline_errors_fixed']:,} "
            f"and loses {disagreements['baseline_correct_answers_lost']:,} correct answers. "
            f"Both models are wrong on {disagreements['both_wrong_on_disagreement']:,} disagreements."
        )
        print(f"Perfect selection ceiling on this set: {disagreements['oracle_accuracy']:.1%}")
    print(classification_report(labels, predictions, labels=label_order, zero_division=0))
    print(f"Saved hybrid validation report: {metrics_path}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Retrieve evidence and predict a FEVEROUS claim label."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Build or refresh the local evidence search index")
    index_parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the index is current")
    train_parser = subparsers.add_parser("train", help="Train and validate the claim verifier")
    train_parser.add_argument("--max-claims", type=int, default=0,
                              help="Optional stratified cap on training rows; 0 uses all training claims")
    train_parser.add_argument("--seed", type=int, default=42)
    verify_parser = subparsers.add_parser("verify", help="Predict a claim label and show retrieved evidence")
    verify_parser.add_argument("--claim", required=True, help="Claim text to check")
    verify_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    subparsers.add_parser("evaluate", help="Evaluate the original logistic regression baseline on the validation split")
    calibrate_parser = subparsers.add_parser(
        "calibrate-hybrid", help="Fit temperature scaling on labeled training claims"
    )
    calibrate_parser.add_argument("--max-claims", type=int, default=250)
    calibrate_parser.add_argument("--seed", type=int, default=42)
    hybrid_eval_parser = subparsers.add_parser(
        "evaluate-hybrid", help="Evaluate hybrid retrieval and NLI on held-out claims"
    )
    hybrid_eval_parser.add_argument("--max-claims", type=int, default=100)
    hybrid_eval_parser.add_argument("--seed", type=int, default=42)
    hybrid_eval_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    try:
        if args.command == "index":
            build_search_index(force=args.rebuild)
        elif args.command == "train":
            if args.max_claims < 0:
                parser.error("--max-claims must be >= 0")
            train_model(args.max_claims, args.seed)
        elif args.command == "verify":
            if not args.claim.strip() or args.top_k < 1:
                parser.error("--claim must not be empty and --top-k must be >= 1")
            verify_claim(args.claim.strip(), args.top_k)
        elif args.command == "evaluate":
            evaluate_end_to_end()
        elif args.command == "calibrate-hybrid":
            if args.max_claims < 0:
                parser.error("--max-claims must be >= 0")
            calibrate_hybrid(args.max_claims, args.seed)
        elif args.command == "evaluate-hybrid":
            if args.max_claims < 0 or args.top_k < 1:
                parser.error("--max-claims must be >= 0 and --top-k must be >= 1")
            evaluate_hybrid(args.max_claims, args.seed, args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
