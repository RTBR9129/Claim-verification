

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

PAGES_FILE = DATA_DIR / "selected_wiki_pages.jsonl"
OUTPUT_FILE = DATA_DIR / "extracted_text.jsonl"


def clean_wikipedia_text(text):
    """Convert Wikipedia link markup into readable text."""
    import re

    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text.strip()


def collect_text(value):
    """Collect text recursively from nested FEVEROUS structures."""
    parts = []

    if isinstance(value, str):
        cleaned = clean_wikipedia_text(value)
        if cleaned:
            parts.append(cleaned)

    elif isinstance(value, dict):
        # A table cell stores its readable content in "value".
        if "value" in value and isinstance(value["value"], str):
            parts.extend(collect_text(value["value"]))
        else:
            for key, item in value.items():
                if key in ("id", "is_header", "column_span", "row_span"):
                    continue
                parts.extend(collect_text(item))

    elif isinstance(value, list):
        for item in value:
            parts.extend(collect_text(item))

    return parts


def extract_page_text(page):
    parts = []

    for key, value in page.items():
        if key.startswith(("sentence_", "table_", "list_")):
            parts.extend(collect_text(value))

    # Preserve order while removing exact duplicate pieces of text.
    unique_parts = list(dict.fromkeys(parts))
    return " ".join(unique_parts)


def main():
    page_count = 0

    with open(PAGES_FILE, "r", encoding="utf-8") as source, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for line in source:
            if not line.strip():
                continue

            page = json.loads(line)
            text = extract_page_text(page)

            record = {
                "title": page.get("title", ""),
                "text": text
            }

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            page_count += 1

            if page.get("title") == "Tower of Glass":
                print("DEBUG: Tower of Glass characters:", len(text))
                print("DEBUG: Author found:", "Author" in text)
                print("DEBUG: United States found:", "United States" in text)
                print("DEBUG: English found:", "English" in text)

    print("\nWEEK 2: COMPLETE TEXT EXTRACTION")
    print("-" * 40)
    print("Pages processed:", page_count)
    print("Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
