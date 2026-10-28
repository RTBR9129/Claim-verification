
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "clean_train.jsonl"
CHUNKS_FILE = PROJECT_DIR / "data" / "evidence_chunks.jsonl"

TOP_K = 3
MAX_CLAIMS = 25000


def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    print("Loading evidence chunks...")
    chunks = load_jsonl(CHUNKS_FILE)

    print("Loading claims...")
    records = load_jsonl(DATA_FILE)

    # Keep the experiment manageable.
    if len(records) > MAX_CLAIMS:
        records, _ = train_test_split(
            records,
            train_size=MAX_CLAIMS,
            random_state=42,
            stratify=[item["label"] for item in records]
        )

    claims = [item["claim"] for item in records]
    labels = [item["label"] for item in records]

    print("Claims selected:", len(claims))
    print("Evidence chunks:", len(chunks))

    print("\nBuilding evidence index...")
    evidence_texts = [item["text"] for item in chunks]

    retriever = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000,
        min_df=1
    )

    evidence_matrix = retriever.fit_transform(evidence_texts)

    print("Retrieving evidence for claims...")
    claim_matrix = retriever.transform(claims)

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(claim_matrix, evidence_matrix)

    combined_texts = []

    for i, claim in enumerate(claims):
        top_indices = similarities[i].argsort()[-TOP_K:][::-1]

        evidence = " ".join(
            chunks[index]["text"] for index in top_indices
        )

        combined_texts.append(
            "Claim: " + claim + " Evidence: " + evidence
        )

    X_train, X_test, y_train, y_test = train_test_split(
        combined_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            min_df=2
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\nTraining evidence-grounded baseline...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nEvidence-grounded evaluation")
    print("-" * 40)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))


if __name__ == "__main__":
    main()
