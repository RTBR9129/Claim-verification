
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "clean_train.jsonl"
LOOKUP_FILE = DATA_DIR / "evidence_lookup.json"


def main():
    with open(LOOKUP_FILE, "r", encoding="utf-8") as file:
        lookup = json.load(file)

    total = 0
    complete = 0
    partial = 0
    unavailable = 0

    for line in open(TRAIN_FILE, "r", encoding="utf-8"):
        if not line.strip():
            continue

        record = json.loads(line)
        total += 1

        evidence_ids = set()

        for group in record.get("evidence", []):
            evidence_ids.update(group.get("content", []))

        if not evidence_ids:
            unavailable += 1
            continue

        found = sum(
            1 for evidence_id in evidence_ids
            if evidence_id in lookup
        )

        if found == len(evidence_ids):
            complete += 1
        elif found > 0:
            partial += 1
        else:
            unavailable += 1

    print("\nCLAIM-LEVEL EVIDENCE COVERAGE")
    print("-" * 40)
    print("Total claims:", total)
    print("All evidence available:", complete)
    print("Some evidence available:", partial)
    print("No evidence available:", unavailable)

    print(f"\nComplete coverage: {complete / total * 100:.2f}%")
    print(f"Partial coverage: {partial / total * 100:.2f}%")


if __name__ == "__main__":
    main()
