
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "clean_train.jsonl"
OUTPUT_FILE = DATA_DIR / "all_required_titles.txt"


def main():
    titles = set()

    with open(TRAIN_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            for group in record.get("evidence", []):
                for evidence_id in group.get("content", []):
                    if "_header_cell_" in evidence_id:
                        title = evidence_id.split("_header_cell_")[0]
                    elif "_cell_" in evidence_id:
                        title = evidence_id.split("_cell_")[0]
                    elif "_sentence_" in evidence_id:
                        title = evidence_id.split("_sentence_")[0]
                    elif "_section_" in evidence_id:
                        title = evidence_id.split("_section_")[0]
                    elif "_item_" in evidence_id:
                        title = evidence_id.split("_item_")[0]
                    else:
                        continue

                    titles.add(title)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for title in sorted(titles):
            file.write(title + "\n")

    print("REQUIRED PAGE TITLES")
    print("-" * 40)
    print("Unique titles:", len(titles))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
