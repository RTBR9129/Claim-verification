
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train.jsonl"
TITLES_FILE = DATA_DIR / "page_titles.txt"

CLAIMS_TO_CHECK = 500


def load_claims():
    claims = []

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {TRAIN_FILE}"
        )

    with open(TRAIN_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            if not item.get("claim", "").strip():
                continue

            claims.append(item)

            if len(claims) >= CLAIMS_TO_CHECK:
                break

    return claims


def extract_page_titles(claims):
    page_titles = set()

    for claim in claims:
        for evidence_group in claim.get("evidence", []):
            context = evidence_group.get("context", {})

            for evidence_id in evidence_group.get("content", []):
                items = context.get(evidence_id, [])

                for item in items:
                    if isinstance(item, str) and item.endswith("_title"):
                        title = item[:-6].replace("_", " ").strip()

                        if title:
                            page_titles.add(title)

    return sorted(page_titles)


def main():
    claims = load_claims()
    titles = extract_page_titles(claims)

    DATA_DIR.mkdir(exist_ok=True)

    with open(TITLES_FILE, "w", encoding="utf-8") as file:
        for title in titles:
            file.write(title + "\n")

    label_counts = Counter(
        str(item.get("label", "UNKNOWN"))
        for item in claims
    )

    print("WEEK 1: DATASET INSPECTION")
    print("-" * 35)
    print("Claims loaded:", len(claims))
    print("Label distribution:", dict(label_counts))
    print("Unique evidence page titles:", len(titles))
    print("Saved titles to:", TITLES_FILE)

    if claims:
        print("\nExample claim:")
        print(claims[0]["claim"])
        print("Label:", claims[0].get("label"))

    print("\nFirst 10 page titles:")
    for title in titles[:10]:
        print("-", title)


if __name__ == "__main__":
    main()
