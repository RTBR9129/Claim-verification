
import json
from pathlib import Path

DATA_DIR = Path("data")
PAGES_FILE = DATA_DIR / "selected_wiki_pages.jsonl"
TITLES_FILE = DATA_DIR / "page_titles.txt"
MISSING_FILE = DATA_DIR / "missing_page_titles.txt"


def main():
    page_count = 0
    titles = set()
    invalid_records = 0

    with open(PAGES_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                page = json.loads(line)
                title = page.get("title")

                if not title:
                    invalid_records += 1
                    continue

                titles.add(title)
                page_count += 1

            except json.JSONDecodeError:
                invalid_records += 1

    with open(TITLES_FILE, "r", encoding="utf-8") as file:
        requested_count = sum(
            1 for line in file if line.strip()
        )

    with open(MISSING_FILE, "r", encoding="utf-8") as file:
        missing_count = sum(
            1 for line in file if line.strip()
        )

    print("WEEK 1: DATA VALIDATION")
    print("-" * 35)
    print("Requested titles:", requested_count)
    print("Valid page records:", page_count)
    print("Unique page titles:", len(titles))
    print("Missing titles:", missing_count)
    print("Invalid records:", invalid_records)

    if page_count == len(titles) and invalid_records == 0:
        print("\nPASS: All saved records are valid.")
    else:
        print("\nCHECK: Some records need investigation.")


if __name__ == "__main__":
    main()
