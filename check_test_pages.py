
import json

file_path = "data/selected_wiki_pages.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)
        print(page.get("title", "Unknown"))
