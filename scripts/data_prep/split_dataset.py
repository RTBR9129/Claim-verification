
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Load the dataset
data = []

with open(DATA_DIR / "clean_train.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

# Extract labels
labels = [item["label"] for item in data]

# Split into 80% training and 20% temporary data
train_data, temp_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Split temporary data equally into validation and testing
temp_labels = [item["label"] for item in temp_data]

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

# Save datasets
datasets = {
    "train": train_data,
    "validation": val_data,
    "test": test_data
}

for name, records in datasets.items():
    with open(DATA_DIR / f"{name}_claims.jsonl", "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    print(f"\n{name.upper()}: {len(records)} claims")
    print(Counter(item["label"] for item in records))

print("\nDataset splitting completed!")
