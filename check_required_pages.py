import json
from pathlib import Path

FILE = Path("data/all_required_wiki_pages.jsonl")

count = 0

with open(FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            record = json.loads(line)
            count += 1

            if count == 1:
                print("First page title:", record.get("title"))
                print("Page keys:", list(record.keys()))

print("\nTotal records:", count)