import json
import zipfile
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

ZIP_PATH = DATA_DIR / "feverous-wiki-pages.zip"
TITLES_FILE = DATA_DIR / "all_required_titles.txt"
OUTPUT_FILE = DATA_DIR / "all_required_wiki_pages.jsonl"


def normalize_title(title):
    return unicodedata.normalize("NFC", title).strip()


def main():
    with open(TITLES_FILE, "r", encoding="utf-8") as file:
        required_titles = {
            normalize_title(line)
            for line in file
            if line.strip()
        }

    print("Required titles:", len(required_titles))
    print("Scanning ZIP archive...")

    found_titles = set()
    matched_records = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".jsonl")
        ]

        with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
            for file_number, member in enumerate(members, start=1):
                with archive.open(member) as source:
                    for raw_line in source:
                        if not raw_line.strip():
                            continue

                        record = json.loads(raw_line.decode("utf-8"))
                        title = normalize_title(record.get("title", ""))

                        if title in required_titles:
                            output.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                            found_titles.add(title)
                            matched_records += 1

                if file_number % 50 == 0:
                    print(
                        f"Scanned {file_number}/{len(members)} files | "
                        f"Matched pages: {len(found_titles)}"
                    )

    missing_titles = required_titles - found_titles

    print("\nEXTRACTION COMPLETE")
    print("-" * 40)
    print("Required titles:", len(required_titles))
    print("Matched titles:", len(found_titles))
    print("Records saved:", matched_records)
    print("Missing titles:", len(missing_titles))
    print("Saved to:", OUTPUT_FILE)

    with open(
        DATA_DIR / "missing_required_titles.txt",
        "w",
        encoding="utf-8"
    ) as file:
        for title in sorted(missing_titles):
            file.write(title + "\n")


if __name__ == "__main__":
    main()