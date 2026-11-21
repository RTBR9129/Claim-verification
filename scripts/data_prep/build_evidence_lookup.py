
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

PAGES_FILE = DATA_DIR / "all_required_wiki_pages.jsonl"
OUTPUT_FILE = DATA_DIR / "evidence_lookup_full.json"


def clean_text(text):
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text.strip()


def extract_evidence(page):
    title = page.get("title", "")
    evidence = {}

    def add_evidence(local_id, text, evidence_type):
        if not local_id or not isinstance(text, str):
            return

        text = clean_text(text)
        full_id = f"{title}_{local_id}"

        evidence[full_id] = {
            "title": title,
            "type": evidence_type,
            "text": text
        }

    if title:
        add_evidence("title", title, "title")

    def scan_nested(obj):
        if isinstance(obj, dict):

            if "id" in obj and "value" in obj:
                cell_id = obj["id"]
                value = obj["value"]

                if isinstance(cell_id, str) and isinstance(value, str):
                    evidence_type = (
                        "header_cell"
                        if cell_id.startswith("header_cell_")
                        else "table_cell"
                    )

                    # Preserve empty cells as well.
                    add_evidence(cell_id, value, evidence_type)

            for value in obj.values():
                scan_nested(value)

        elif isinstance(obj, list):
            for item in obj:
                scan_nested(item)

    for key, value in page.items():

        if key.startswith("sentence_") and isinstance(value, str):
            add_evidence(key, value, "sentence")

        elif key.startswith("section_") and isinstance(value, dict):
            add_evidence(key, value.get("value", ""), "section")

        elif key.startswith("table_") and isinstance(value, dict):

            # Extract table caption using its correct evidence ID.
            caption = value.get("caption")

            if isinstance(caption, str):
                caption_id = key.replace(
                    "table_", "table_caption_", 1
                )
                add_evidence(caption_id, caption, "table_caption")

            scan_nested(value)

        elif key.startswith("list_"):
            scan_nested(value)

    return evidence


def main():
    evidence_lookup = {}
    page_count = 0

    print("Building evidence lookup...")

    with open(PAGES_FILE, "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            page = json.loads(line)
            page_evidence = extract_evidence(page)

            evidence_lookup.update(page_evidence)
            page_count += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        json.dump(
            evidence_lookup,
            output,
            ensure_ascii=False,
            indent=2
        )

    print("\nEVIDENCE LOOKUP CREATED")
    print("-" * 40)
    print("Pages processed:", page_count)
    print("Evidence items:", len(evidence_lookup))
    print("Output:", OUTPUT_FILE)

    # Verify previously missing examples.
    test_ids = [
        "2006–07 San Jose Sharks season_table_caption_1",
        "Hunar Hali_cell_1_1_2",
        "Marrecs de Salt_header_cell_4_0_0",
        "Michael Folivi_cell_1_2_0"
    ]

    print("\nTEST RESULTS")
    print("-" * 40)

    for test_id in test_ids:
        if test_id in evidence_lookup:
            print("FOUND:", test_id)
            print("TEXT:", repr(evidence_lookup[test_id]["text"]))
        else:
            print("MISSING:", test_id)


if __name__ == "__main__":
    main()
