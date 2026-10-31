import json

file_path = "data/selected_wiki_pages.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)

        if page.get("title") == "Michael Folivi":
            print("SENTENCE 0:")
            print(json.dumps(
                page.get("sentence_0", "NOT FOUND"),
                indent=2,
                ensure_ascii=False
            ))
            break