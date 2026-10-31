
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "clean_train.jsonl"
LOOKUP_FILE = DATA_DIR / "evidence_lookup_full.json"


def main():
    print("Loading evidence lookup...")

    with open(LOOKUP_FILE, "r", encoding="utf-8") as file:
        lookup = json.load(file)

    total_claims = 0
    claims_with_evidence = 0
    total_evidence_ids = 0
    found_evidence_ids = 0

    with open(TRAIN_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            total_claims += 1

            evidence_ids = set()

            for group in record.get("evidence", []):
                evidence_ids.update(group.get("content", []))

            if evidence_ids:
                claims_with_evidence += 1

            for evidence_id in evidence_ids:
                total_evidence_ids += 1

                if evidence_id in lookup:
                    found_evidence_ids += 1

    coverage = (
        found_evidence_ids / total_evidence_ids * 100
        if total_evidence_ids else 0
    )

    print("\nEVIDENCE COVERAGE REPORT")
    print("-" * 40)
    print("Total claims:", total_claims)
    print("Claims with evidence IDs:", claims_with_evidence)
    print("Total unique evidence references counted:", total_evidence_ids)
    print("Evidence IDs found:", found_evidence_ids)
    print("Evidence IDs missing:", total_evidence_ids - found_evidence_ids)
    print(f"Evidence coverage: {coverage:.2f}%")


if __name__ == "__main__":
    main()
