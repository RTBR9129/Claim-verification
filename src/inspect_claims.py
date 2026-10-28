
import json
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_DIR / "data" / "train.jsonl"

label_counts = Counter()
checked = 0

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        item = json.loads(line)
        label_counts[item.get("label", "MISSING")] += 1

        if checked < 3:
            print("\n" + "=" * 60)
            print("CLAIM:", item.get("claim"))
            print("LABEL:", item.get("label"))
            print("EVIDENCE:")
            print(json.dumps(item.get("evidence"), indent=2, ensure_ascii=False))
            checked += 1

print("\n" + "=" * 60)
print("LABEL COUNTS IN THE FULL TRAINING FILE")
print("=" * 60)

for label, count in label_counts.items():
    print(f"{label}: {count}")

print("\nTotal records:", sum(label_counts.values()))
