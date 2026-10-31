
import json
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "clean_train.jsonl"
LOOKUP_FILE = DATA_DIR / "evidence_lookup_full.json"

with open(LOOKUP_FILE, "r", encoding="utf-8") as file:
    lookup = json.load(file)

missing = Counter()

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        record = json.loads(line)

        for group in record.get("evidence", []):
            for evidence_id in group.get("content", []):
                if evidence_id not in lookup:
                    missing[evidence_id] += 1

print("TOTAL MISSING REFERENCES:", sum(missing.values()))
print("UNIQUE MISSING IDs:", len(missing))

print("\nMISSING ID TYPES:")
for evidence_type, count in sorted(
    Counter(
        "_".join(eid.split("_")[-4:-2])
        for eid in missing
    ).items()
):
    print(evidence_type, count)

print("\nFIRST 20 MISSING IDs:")
for evidence_id, count in missing.most_common(20):
    print(evidence_id, "| Occurrences:", count)
