
import json
from pathlib import Path

FILE = Path("data/extracted_text.jsonl")

with open(FILE, "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)

        if page.get("title") == "Tower of Glass":
            text = page.get("text", "")

            print("Page title:", page["title"])
            print("Total characters:", len(text))
            print("Author found:", "Author" in text)
            print("Robert Silverberg found:", "Robert Silverberg" in text)
            print("Country found:", "Country" in text)
            print("United States found:", "United States" in text)
            print("Language found:", "Language" in text)
            print("English found:", "English" in text)

            print("\nText at the end of the extracted content:")
            print(text[-1000:])
            break
    else:
        print("Tower of Glass was not found.")
