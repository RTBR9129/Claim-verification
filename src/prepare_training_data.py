
import json
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "train.jsonl"
OUTPUT_FILE = DATA_DIR / "clean_train.jsonl"

VALID_LABELS = {
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO"
}


def main():
    label_counts = Counter()
    skipped = 0
    saved = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as source, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for line in source:
            if not line.strip():
                skipped += 1
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            claim = item.get("claim")
            label = item.get("label")

            if not isinstance(claim, str) or not claim.strip():
                skipped += 1
                continue

            if label not in VALID_LABELS:
                skipped += 1
                continue

            record = {
                "id": item.get("id"),
                "claim": claim.strip(),
                "label": label,
                "evidence": item.get("evidence", [])
            }

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            label_counts[label] += 1
            saved += 1

    print("TRAINING DATA PREPARATION COMPLETE")
    print("-" * 40)
    print("Valid records saved:", saved)
    print("Records skipped:", skipped)
    print("Label distribution:")

    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nOutput saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
