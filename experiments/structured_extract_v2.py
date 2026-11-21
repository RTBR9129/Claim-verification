import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PAGES_FILE = DATA_DIR / "all_required_wiki_pages.jsonl"
OUTPUT_FILE = DATA_DIR / "structured_extraction_v2_preview.jsonl"

TARGET_PAGES = {"Chemical Chords", "Mario Party", "Besa Tsintsadze"}


def clean_text(value):
    """Remove Wikipedia link markup while keeping the displayed text."""
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def get_cell_text(cell):
    if not isinstance(cell, dict):
        return clean_text(cell)
    return clean_text(cell.get("value", cell.get("text", "")))


def get_table_rows(table):
    """Find the table's row list without assuming one exact FEVEROUS schema."""
    if not isinstance(table, dict):
        return []

    for key in ("rows", "table_rows", "data"):
        candidate = table.get(key)
        if isinstance(candidate, list) and candidate:
            if all(isinstance(row, list) for row in candidate):
                return candidate

    # Some FEVEROUS exports store the rows directly in a list-valued field.
    for value in table.values():
        if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            if any(isinstance(cell, dict) and ("value" in cell or "is_header" in cell)
                   for row in value for cell in row):
                return value
    return []


def get_section_map(page):
    """Build a best-effort element-ID -> section path map from section metadata."""
    result = {}
    sections = page.get("sections", {})
    if isinstance(sections, dict):
        iterable = sections.items()
    elif isinstance(sections, list):
        iterable = enumerate(sections)
    else:
        iterable = []

    for key, section in iterable:
        if not isinstance(section, dict):
            continue
        heading = clean_text(section.get("title", section.get("name", section.get("text", ""))))
        if not heading:
            continue
        ids = section.get("contents", section.get("content", section.get("elements", [])))
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(ids, list):
            for element_id in ids:
                if isinstance(element_id, str):
                    result[element_id] = heading
        result[str(key)] = heading
    return result


def find_section_context(element_key, page, section_map):
    if element_key in section_map:
        return section_map[element_key]

    # Use order metadata when it contains section identifiers or headings.
    order = page.get("order", [])
    if isinstance(order, list):
        current = []
        for item in order:
            item_text = item if isinstance(item, str) else ""
            if item_text in section_map:
                current = [section_map[item_text]]
            elif item_text == element_key:
                return " > ".join(current)
    return ""


def make_row_records(rows, title, table_key, caption, section):
    records = []
    active_headers = {}
    for row_index, row in enumerate(rows):
        cells = []
        for col_index, cell in enumerate(row):
            if isinstance(cell, dict):
                cell_id = cell.get("id", "")
                text = get_cell_text(cell)
                is_header = bool(cell.get("is_header", False))
                row_span = cell.get("row_span", 1)
                col_span = cell.get("column_span", 1)
            else:
                cell_id = ""
                text = clean_text(cell)
                is_header = False
                row_span = 1
                col_span = 1

            cells.append({
                "id": cell_id,
                "column_index": col_index,
                "text": text,
                "is_header": is_header,
                "row_span": row_span,
                "column_span": col_span
            })

        # Header cells update the active column labels, including multi-level headers.
        for cell in cells:
            if cell["is_header"] and cell["text"]:
                active_headers[cell["column_index"]] = cell["text"]

        header_only_row = bool(cells) and all(
            cell["is_header"] or not cell["text"] for cell in cells
        )

        if header_only_row:
            continue

        mapped_values = {}
        evidence_ids = []
        for cell in cells:
            if cell["id"]:
                evidence_ids.append(cell["id"])
            if not cell["text"]:
                continue

            header = active_headers.get(cell["column_index"])
            if header and header != cell["text"]:
                mapped_values[header] = cell["text"]
            else:
                mapped_values[f"column_{cell['column_index'] + 1}"] = cell["text"]

        if mapped_values:
            readable = " | ".join(f"{key}: {value}" for key, value in mapped_values.items())
            records.append({
                "record_type": "table_row",
                "page_title": title,
                "table_key": table_key,
                "table_caption": caption,
                "section_context": section,
                "row_index": row_index,
                "values": mapped_values,
                "text": readable,
                "evidence_ids": evidence_ids,
                "cells": cells
            })

    return records


def extract_page(page):
    title = page.get("title", "UNKNOWN")
    section_map = get_section_map(page)
    output = []

    for key, value in page.items():
        if key.startswith("sentence_"):
            text = ""
            ids = []
            if isinstance(value, dict):
                text = clean_text(value.get("text", value.get("value", "")))
                if value.get("id"):
                    ids.append(value["id"])
            else:
                text = clean_text(value)

            if text:
                output.append({
                    "record_type": "sentence",
                    "page_title": title,
                    "element_key": key,
                    "section_context": find_section_context(key, page, section_map),
                    "text": text,
                    "evidence_ids": ids
                })

        elif key.startswith("table_"):
            table = value if isinstance(value, dict) else {}
            caption = clean_text(
                table.get("table_caption", table.get("caption", ""))
            )
            section = find_section_context(key, page, section_map)
            rows = get_table_rows(table)

            output.extend(
                make_row_records(rows, title, key, caption, section)
            )

    return output


def main():
    if not PAGES_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {PAGES_FILE}")

    found = set()
    output_count = 0

    with PAGES_FILE.open("r", encoding="utf-8") as source, \
            OUTPUT_FILE.open("w", encoding="utf-8") as destination:

        for line in source:
            line = line.strip()
            if not line:
                continue

            page = json.loads(line)
            title = page.get("title", "")
            if title not in TARGET_PAGES:
                continue

            found.add(title)
            for record in extract_page(page):
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_count += 1

    missing = sorted(TARGET_PAGES - found)
    print("Structured extraction v2 complete.")
    print(f"Pages found: {len(found)}/{len(TARGET_PAGES)}")
    print(f"Structured records written: {output_count}")
    print(f"Preview saved to: {OUTPUT_FILE}")
    if missing:
        print("Pages not found: " + ", ".join(missing))


if __name__ == "__main__":
    main()
