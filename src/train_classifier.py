
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "clean_train.jsonl"


def load_data():
    claims = []
    labels = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            claims.append(item["claim"])
            labels.append(item["label"])

    return claims, labels


def main():
    print("Loading cleaned training data...")

    claims, labels = load_data()
    print("Total claims:", len(claims))

    # Use a smaller stratified sample for the first experiment.
    X_train, X_test, y_train, y_test = train_test_split(
        claims,
        labels,
        train_size=20000,
        test_size=5000,
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

    print("\nTraining baseline classifier...")
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out claims...")
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nEnter claims to test the model.")
    print("Type 'exit' to stop.")

    while True:
        claim = input("\nClaim: ").strip()

        if claim.lower() == "exit":
            break

        if not claim:
            continue

        prediction = model.predict([claim])[0]
        print("Predicted label:", prediction)


if __name__ == "__main__":
    main()
