
import json

targets = [
    "Sydney Airport",
    "Richard Nixon",
    "Isaac Newton",
    "Margaret Atwood",
    "Fundamental theorem of calculus"
]

with open("data/extracted_text.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)

        if page.get("title") in targets:
            print("\n" + "=" * 70)
            print("PAGE:", page["title"])
            print("=" * 70)

            print(json.dumps(page, indent=2, ensure_ascii=False)[:2500])
