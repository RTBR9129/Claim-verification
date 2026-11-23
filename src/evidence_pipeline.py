"""Hybrid evidence retrieval and evidence-grounded claim verification."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "evidence_chunks.jsonl"
EMBEDDINGS_FILE = DATA_DIR / "semantic_embeddings.npy"
FAISS_FILE = DATA_DIR / "evidence_faiss.index"
FAISS_META_FILE = DATA_DIR / "evidence_faiss.json"
MODEL_CACHE = ROOT / "models" / "hf-cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-distilroberta-base"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_EVIDENCE_CHARS = 900


def _load_chunks() -> list[dict[str, Any]]:
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError(f"No evidence chunks found in {CHUNKS_FILE}")
    return chunks


def _load_dense_index(embeddings: np.ndarray, faiss: Any) -> Any:
    expected = {
        "chunk_mtime_ns": CHUNKS_FILE.stat().st_mtime_ns,
        "embedding_mtime_ns": EMBEDDINGS_FILE.stat().st_mtime_ns,
        "count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
    }
    if FAISS_FILE.is_file() and FAISS_META_FILE.is_file():
        try:
            metadata = json.loads(FAISS_META_FILE.read_text(encoding="utf-8"))
            if metadata == expected:
                return faiss.read_index(str(FAISS_FILE))
        except (OSError, ValueError, RuntimeError):
            pass

    vectors = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_FILE))
    FAISS_META_FILE.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return index


class EvidencePipeline:
    """BM25 + MiniLM + RRF + cross-encoder + NLI pipeline."""

    def __init__(self, lexical_retrieve: Callable[..., list[dict[str, Any]]]):
        try:
            import faiss
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "The hybrid verifier needs the optional model packages. "
                "Install them with: python -m pip install -r requirements.txt"
            ) from error

        if not CHUNKS_FILE.is_file() or not EMBEDDINGS_FILE.is_file():
            raise FileNotFoundError(
                "The hybrid verifier needs data/evidence_chunks.jsonl and "
                "data/semantic_embeddings.npy. Rebuild the evidence corpus and "
                "run: python scripts/data_prep/build_semantic_index.py"
            )

        self.chunks = _load_chunks()
        embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match "
                f"evidence chunks ({len(self.chunks)}). Rebuild the semantic index."
            )
        self.faiss = _load_dense_index(embeddings, faiss)
        self.lexical_retrieve = lexical_retrieve
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL, cache_folder=str(MODEL_CACHE), device="cpu"
        )
        self.reranker = CrossEncoder(
            RERANKER_MODEL, cache_folder=str(MODEL_CACHE), device="cpu", max_length=512
        )
        self.nli = CrossEncoder(
            NLI_MODEL, cache_folder=str(MODEL_CACHE), device="cpu", max_length=512
        )
        self.calibration_file = ROOT / "models" / "evidence_calibration.json"
        self.temperature = 1.0
        self.is_calibrated = False
        self.calibration_warning = False
        if self.calibration_file.is_file():
            try:
                calibration = json.loads(self.calibration_file.read_text(encoding="utf-8"))
                self.temperature = float(calibration["temperature"])
                self.is_calibrated = self.temperature > 0
                self.calibration_warning = bool(
                    calibration.get("upper_bound_hit", self.temperature >= 3.999)
                )
            except (OSError, ValueError, KeyError, TypeError):
                self.temperature = 1.0
                self.is_calibrated = False
        self.lexical_retrieve("index warmup", top_k=1)

    @staticmethod
    def _key(item: dict[str, Any]) -> tuple[str, str]:
        return item["title"].casefold(), str(item["chunk_id"])

    def retrieve(self, claim: str, top_k: int = 5) -> list[dict[str, Any]]:
        candidate_k = max(20, top_k * 8)
        lexical = self.lexical_retrieve(claim, top_k=candidate_k)
        query = self.embedder.encode(
            [claim], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        dense_scores, dense_ids = self.faiss.search(query, candidate_k)

        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        rrf_k = 60
        for rank, item in enumerate(lexical, start=1):
            key = self._key(item)
            record = by_key.setdefault(key, dict(item))
            record["bm25_rank"] = rank
            record["bm25_score"] = float(item.get("score", 0.0))
            record["rrf_score"] = record.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)

        for rank, (score, chunk_id) in enumerate(zip(dense_scores[0], dense_ids[0]), start=1):
            if chunk_id < 0:
                continue
            item = dict(self.chunks[int(chunk_id)])
            key = self._key(item)
            record = by_key.setdefault(key, item)
            record["dense_rank"] = rank
            record["dense_score"] = float(score)
            record["rrf_score"] = record.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)

        candidates = sorted(
            by_key.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True
        )[:candidate_k]
        pairs = [
            (claim, f"{item.get('title', '')}: {item.get('text', '')[:2400]}")
            for item in candidates
        ]
        rerank_scores = np.asarray(
            self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        ).reshape(-1)
        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidates[:top_k]

    def _nli_probabilities(self, claim: str, evidence: list[dict[str, Any]]) -> list[dict[str, float]]:
        stop_words = {
            "the", "a", "an", "is", "was", "were", "are", "be", "been",
            "by", "in", "on", "at", "to", "of", "and", "or", "with",
            "for", "from", "that", "this", "it", "as", "has", "have",
        }
        claim_terms = {
            token.casefold() for token in TOKEN_RE.findall(claim)
            if len(token) > 2 and token.casefold() not in stop_words
        }
        predicate = re.search(
            r"\b(?:is|was|are|were|becomes?|has|have|had|wrote|writes|"
            r"built|founded|created|served|located|born|won|died|published)\b",
            claim,
            flags=re.IGNORECASE,
        )
        subject_terms = set()
        if predicate:
            subject_terms = {
                token.casefold() for token in TOKEN_RE.findall(claim[:predicate.start()])
                if len(token) > 2 and token.casefold() not in stop_words
            }
        object_entities = (
            self._entity_terms(claim[predicate.end():]) if predicate else set()
        )
        owners: list[int] = []
        pair_weights: list[float] = []
        pairs: list[tuple[str, str]] = []
        pair_texts: list[str] = []
        for owner, item in enumerate(evidence):
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            candidates = []
            for sentence in sentences:
                terms = {
                    token.casefold() for token in TOKEN_RE.findall(sentence)
                    if len(token) > 2 and token.casefold() not in stop_words
                }
                title_terms = {
                    token.casefold() for token in TOKEN_RE.findall(str(item.get("title", "")))
                }
                if subject_terms and not subject_terms.issubset(terms | title_terms):
                    continue
                overlap = len(claim_terms & terms)
                if overlap:
                    ratio = overlap / max(1, len(claim_terms))
                    candidates.append((overlap, ratio, sentence[:700]))
            candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
            if not candidates and not subject_terms:
                candidates = [(0, 0.0, text[:700])]
            for overlap, ratio, sentence in candidates[:3]:
                important_terms = subject_terms | object_entities
                positions = [
                    (sentence.casefold().find(term), term)
                    for term in important_terms
                    if sentence.casefold().find(term) >= 0
                ]
                if positions:
                    subject_text = " ".join(TOKEN_RE.findall(claim[:predicate.start()])).casefold() if predicate else ""
                    subject_positions = []
                    if subject_text:
                        offset = 0
                        while True:
                            found = sentence.casefold().find(subject_text, offset)
                            if found < 0:
                                break
                            subject_positions.append(found)
                            offset = found + 1
                    subject_object_spans = []
                    for subject_start_candidate in subject_positions:
                        for term in object_entities:
                            object_start_candidate = sentence.casefold().find(
                                term, subject_start_candidate + len(subject_text)
                            )
                            if object_start_candidate >= 0:
                                subject_object_spans.append((
                                    object_start_candidate - subject_start_candidate,
                                    subject_start_candidate,
                                    object_start_candidate + len(term),
                                ))
                    if subject_object_spans:
                        _, subject_start, object_end = min(subject_object_spans)
                    else:
                        subject_start = -1
                        object_end = -1
                    start = max(
                        0,
                        subject_start if subject_start >= 0 else min(position for position, _ in positions),
                    )
                    end = min(
                        len(sentence),
                        max(
                            object_end if object_end >= 0 else 0,
                            max(position + len(term) for position, term in positions),
                        ) + 45,
                    )
                    excerpt = sentence[start:end]
                else:
                    excerpt = sentence[:700]
                premise = f"{item.get('title', '')}. {excerpt}"[:1000]
                pairs.append((premise, claim))
                owners.append(owner)
                pair_weights.append(0.5 + ratio)
                pair_texts.append(excerpt)
        # If strict matching rejects everything, use passages from a named subject page.
        if not pairs and predicate:
            prefix = " ".join(TOKEN_RE.findall(claim[:predicate.start()])).casefold()
            for owner, item in enumerate(evidence):
                title = " ".join(TOKEN_RE.findall(str(item.get("title", "")))).casefold()
                if not title or not prefix:
                    continue
                starts_with_title = prefix == title or prefix.startswith(title + " ")
                contains_multiword_title = " " in title and f" {title} " in f" {prefix} "
                if not (starts_with_title or contains_multiword_title):
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", item.get("text", "").strip())
                ranked = sorted(
                    (
                        (len(claim_terms & set(TOKEN_RE.findall(sentence.casefold()))), sentence)
                        for sentence in sentences if sentence.strip()
                    ),
                    key=lambda candidate: candidate[0],
                    reverse=True,
                )
                for overlap, sentence in ranked[:3]:
                    if not overlap:
                        continue
                    excerpt = sentence[:700]
                    pairs.append((f"{item['title']}. {excerpt}"[:1000], claim))
                    owners.append(owner)
                    pair_weights.append(0.5 + overlap / max(1, len(claim_terms)))
                    pair_texts.append(excerpt)
        probabilities, indices = self._score_pairs(pairs)
        selected: list[int | None] = [None] * len(evidence)
        selected_scores = [-1.0] * len(evidence)
        for row, owner in enumerate(owners):
            support = probabilities[row, indices["supports"]]
            refute = probabilities[row, indices["refutes"]]
            directional = float(max(support, refute) * pair_weights[row])
            if directional > selected_scores[owner]:
                selected[owner] = row
                selected_scores[owner] = directional
        results = []
        for owner, row in enumerate(selected):
            if row is None:
                results.append({"refutes": 0.0, "supports": 0.0, "neutral": 1.0})
                evidence[owner]["nli_basis"] = "No claim-matching sentence found."
                continue
            results.append({
                label: float(probabilities[row, index])
                for label, index in indices.items()
                if index < probabilities.shape[1]
            })
            evidence[owner]["nli_basis"] = pair_texts[row]
        return results

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, dict[str, int]]:
        logits = np.asarray(self.nli.predict(pairs, batch_size=16, show_progress_bar=False))
        if not pairs:
            return np.empty((0, 3)), {}
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        config = getattr(getattr(self.nli, "model", None), "config", None)
        labels = {str(name).upper(): int(index) for index, name in (
            getattr(config, "id2label", {}) or {}
        ).items()}
        indices = {
            "refutes": next((v for k, v in labels.items() if "CONTRAD" in k), 0),
            "supports": next((v for k, v in labels.items() if "ENTAIL" in k), 1),
            "neutral": next((v for k, v in labels.items() if "NEUTRAL" in k), 2),
        }
        scaled = logits[:, :3].astype(np.float64)
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities, indices

    @staticmethod
    def _compound_parts(claim: str, title: str) -> list[str]:
        # Restrict the split to a shared subject anchored by a retrieved page title.
        subject = re.match(rf"^{re.escape(title)}(?=\s|,)", claim, re.IGNORECASE)
        if not subject:
            return []
        body = claim[subject.end():].strip().rstrip(".!?")
        parts = re.split(
            r",?\s+(?:but|and)\s+(?=(?:did|does|do|was|were|is|are|has|have|had)\b)",
            body, maxsplit=2, flags=re.IGNORECASE,
        )
        if len(parts) < 2:
            return []
        role = re.match(r"^,\s*(a[n]?\s+[^,]{1,95}),\s*(.+)$", parts[0], re.IGNORECASE)
        if role:
            parts = [f"is {role[1]}", role[2], *parts[1:]]
        elif parts[0].startswith(","):
            return []
        return [f"{subject[0]} {part.strip()}." for part in parts if part.strip()]

    def check_compound_claim(
        self, claim: str, evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Check simple coordinated clauses against complete sentences on the subject page."""
        titles = {item["title"] for item in evidence if item.get("title")}
        for title in sorted(titles, key=lambda name: (-len(name), name)):
            clauses = self._compound_parts(claim, title)
            if clauses:
                break
        else:
            return []

        subject_pages = [item for item in evidence if item["title"].casefold() == title.casefold()]
        stops = set("a an the is was were are has have had do does did for in of to by not".split())

        def terms(text: str) -> set[str]:
            return set(TOKEN_RE.findall(text.casefold())) - stops

        sentences = []
        for item in subject_pages:
            text = re.sub(r"\s+", " ", item["text"]).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if sentence and len(sentence) <= 700:
                    sentences.append((item, sentence, terms(sentence)))

        results = []
        pairs = []
        checks_to_score = []
        for clause in clauses:
            keywords = terms(clause[len(title):])
            check = {"claim": clause, "label": "NOT ENOUGH INFO"}
            results.append(check)
            if not sentences or not keywords:
                continue
            item, sentence, tokens = max(sentences, key=lambda candidate: len(keywords & candidate[2]))
            overlap = len(keywords & tokens)
            if not overlap:
                continue
            check.update(title=item["title"], chunk_id=item["chunk_id"], sentence=sentence)
            # A negative clause needs enough coverage of the disputed detail.
            # For example, attendance alone should not settle academic eligibility.
            if re.search(r"\b(?:not|never|no)\b", clause[len(title):], re.IGNORECASE):
                if overlap / len(keywords) < 2 / 3:
                    check["note"] = "The passage leaves some details of this part unaddressed."
                    continue
            pairs.append((f"{item['title']}. {sentence}", clause))
            checks_to_score.append(check)

        if pairs:
            scores, indices = self._score_pairs(pairs)
            for check, row in zip(checks_to_score, scores):
                relation = max(indices, key=lambda name: row[indices[name]])
                if relation in {"supports", "refutes"} and row[indices[relation]] >= 0.7:
                    check["label"] = relation.upper()
        return results

    @staticmethod
    def _entity_terms(text: str) -> set[str]:
        return {
            match.casefold()
            for match in re.findall(r"\b([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)", text)
            if len(match) > 2
        }

    def verify(
        self, claim: str, top_k: int = 5,
    ) -> dict[str, Any]:
        evidence = self.retrieve(claim, top_k=max(3, top_k))
        if not evidence:
            return {
                "label": "NOT ENOUGH INFO", "confidence": 0.0,
                "human_review": True, "reason": "No passages were retrieved.",
                "evidence": [], "graph": {"supports": 0, "refutes": 0, "neutral": 0},
                "explanation": "The indexed corpus returned no evidence for this claim.",
                "calibration": self.calibration_status,
            }

        probabilities = self._nli_probabilities(claim, evidence)
        active_indices = [
            index for index, item in enumerate(evidence)
            if item.get("nli_basis") != "No claim-matching sentence found."
        ]
        for index, (item, scores) in enumerate(zip(evidence, probabilities)):
            item["nli"] = scores
            item["relation"] = max(scores, key=scores.get) if index in active_indices else "unrelated"
        if not active_indices:
            return {
                "label": "NOT ENOUGH INFO", "confidence": 0.0,
                "human_review": True, "reason": "No retrieved passage matched the claim subject.",
                "evidence": evidence, "graph": {"supports": 0, "refutes": 0, "neutral": 0},
                "explanation": "Retrieved passages did not contain a sentence about the claim subject.",
                "calibration": self.calibration_status,
            }
        rerank = np.array(
            [evidence[index]["rerank_score"] for index in active_indices], dtype=np.float64
        )
        weights = np.exp(rerank - rerank.max())
        weights /= weights.sum()
        aggregated = {
            label: float(sum(
                weight * probabilities[index][label]
                for weight, index in zip(weights, active_indices)
            ))
            for label in ("supports", "refutes", "neutral")
        }
        if self.is_calibrated:
            aggregate_values = np.array(
                [aggregated["refutes"], aggregated["supports"], aggregated["neutral"]],
                dtype=np.float64,
            )
            aggregate_logits = np.log(np.maximum(aggregate_values, 1e-12)) / self.temperature
            aggregate_logits -= aggregate_logits.max()
            calibrated = np.exp(aggregate_logits)
            calibrated /= calibrated.sum()
            aggregated = {
                "refutes": float(calibrated[0]),
                "supports": float(calibrated[1]),
                "neutral": float(calibrated[2]),
            }
        supporting = aggregated["supports"]
        refuting = aggregated["refutes"]
        conflict = min(supporting, refuting)
        strongest = max(supporting, refuting)
        winning_relation = max(aggregated, key=aggregated.get)
        if winning_relation == "neutral" or (
            conflict >= 0.28 and abs(supporting - refuting) < 0.18
        ):
            label = "NOT ENOUGH INFO"
        else:
            label = "SUPPORTS" if supporting > refuting else "REFUTES"
        confidence = strongest if label != "NOT ENOUGH INFO" else max(aggregated.values())
        low_confidence = confidence < 0.65
        conflicting_scores = supporting >= 0.25 and refuting >= 0.25
        human_review = label == "NOT ENOUGH INFO" or low_confidence or conflicting_scores
        review_reason = None
        if conflicting_scores:
            review_reason = "The evidence model found signals both for and against the claim."
        elif low_confidence:
            review_reason = "The leading evidence score is below the review cutoff of 65%."

        claim_entities = self._entity_terms(claim)
        graph = {
            "claim_node": claim,
            "evidence_nodes": [
                {
                    "node": item["title"], "chunk_id": item["chunk_id"],
                    "relation": item["relation"],
                    "shared_entities": sorted(
                        claim_entities & self._entity_terms(item.get("text", ""))
                    ),
                }
                for item in evidence
            ],
            "supports": sum(item["relation"] == "supports" for item in evidence),
            "refutes": sum(item["relation"] == "refutes" for item in evidence),
            "neutral": sum(item["relation"] == "neutral" for item in evidence),
        }
        explanation = self._explanation(label, evidence, aggregated, human_review)
        return {
            "label": label,
            "confidence": confidence,
            "human_review": human_review,
            "evidence": evidence,
            "nli_aggregate": aggregated,
            "review_reason": review_reason,
            "graph": graph,
            "explanation": explanation,
            "calibration": self.calibration_status,
        }

    @property
    def calibration_status(self) -> str:
        if self.is_calibrated:
            detail = f"temperature-scaled NLI confidence; T={self.temperature:.3f}"
            if self.calibration_warning:
                detail += " (calibration search hit its upper bound; use cautiously)"
            return detail
        return "uncalibrated NLI confidence; run calibrate-hybrid"

    @staticmethod
    def _explanation(
        label: str, evidence: list[dict[str, Any]],
        aggregate: dict[str, float], human_review: bool,
    ) -> str:
        lead = {
            "SUPPORTS": "The retrieved evidence favors the claim.",
            "REFUTES": "The retrieved evidence favors a contradiction of the claim.",
            "NOT ENOUGH INFO": "The retrieved evidence does not resolve the claim confidently.",
        }[label]
        relations = ", ".join(
            f"{name} {score:.2f}" for name, score in aggregate.items()
        )
        citations = []
        for rank, item in enumerate(evidence[:3], start=1):
            snippet = re.sub(r"\s+", " ", item.get("nli_basis") or item.get("text", "")).strip()
            snippet = snippet[:220].rstrip() + ("…" if len(snippet) > 220 else "")
            citations.append(f"[{rank}] {item['title']} (chunk {item['chunk_id']}): “{snippet}”")
        review = " Human review is recommended because the evidence is weak or conflicting." if human_review else ""
        return f"{lead} NLI evidence scores: {relations}.{review}\n" + "\n".join(citations)


_PIPELINE: EvidencePipeline | None = None


def verify_claim_pipeline(
    claim: str,
    top_k: int,
    lexical_retrieve: Callable[..., list[dict[str, Any]]],
) -> dict[str, Any]:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = EvidencePipeline(lexical_retrieve)
    result = _PIPELINE.verify(claim, top_k)
    clause_checks = _PIPELINE.check_compound_claim(claim, result["evidence"])
    if clause_checks:
        result["clause_checks"] = clause_checks
        result["human_review"] = result["human_review"] or any(
            check["label"] != "SUPPORTS" for check in clause_checks
        )
    return result


def reset_pipeline_for_calibration(
    lexical_retrieve: Callable[..., list[dict[str, Any]]],
) -> EvidencePipeline:
    global _PIPELINE
    _PIPELINE = EvidencePipeline(lexical_retrieve)
    _PIPELINE.temperature = 1.0
    _PIPELINE.is_calibrated = False
    return _PIPELINE
