
import json
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

ZIP_FILE = DATA_DIR / "feverous-wiki-pages.zip"
TITLES_FILE = DATA_DIR / "page_titles.txt"
OUTPUT_FILE = DATA_DIR / "selected_wiki_pages.jsonl"
MISSING_FILE = DATA_DIR / "missing_page_titles.txt"


def normalize_title(title):
    return " ".join(title.replace("_", " ").split()).casefold()


def main():
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_FILE}")

    if not TITLES_FILE.exists():
        raise FileNotFoundError(f"Titles file not found: {TITLES_FILE}")

    with open(TITLES_FILE, "r", encoding="utf-8") as file:
        targets = {
            normalize_title(line.strip())
            for line in file
            if line.strip()
        }

    found = set()
    scanned_files = 0
    saved_pages = 0

    print("Target page titles:", len(targets))
    print("Scanning ZIP without extracting it...")
    print("This may take a while because the archive is large.")

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".jsonl")
        ]

        with (
            open(OUTPUT_FILE, "w", encoding="utf-8") as output,
            open(MISSING_FILE, "w", encoding="utf-8") as missing
        ):
            for member_name in members:
                scanned_files += 1

                with archive.open(member_name, "r") as source:
                    for raw_line in source:
                        record = json.loads(raw_line)

                        title = record.get("title", "")
                        normalized = normalize_title(title)

                        if normalized in targets and normalized not in found:
                            output.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                            found.add(normalized)
                            saved_pages += 1

                print(
                    f"Files scanned: {scanned_files}/{len(members)}"
                    f" | Pages found: {saved_pages}/{len(targets)}"
                )

            for title in sorted(targets - found):
                missing.write(title + "\n")

    print("\nSCAN COMPLETE")
    print("Pages requested:", len(targets))
    print("Pages found:", saved_pages)
    print("Pages not found:", len(targets - found))
    print("Saved pages to:", OUTPUT_FILE)
    print("Missing titles saved to:", MISSING_FILE)


if __name__ == "__main__":
    main()
