
import json
from pathlib import Path

FILE = Path("data/selected_wiki_pages.jsonl")

with open(FILE, "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)

        if page.get("title") == "Tower of Glass":
            table_data = page.get("table_0", {})
            rows = table_data.get("table", [])

            print("Page:", page["title"])
            print("Number of table rows:", len(rows))

            print("\nExtracted table values:")

            for row in rows:
                for cell in row:
                    if isinstance(cell, dict):
                        print(cell.get("value", ""))

            break
