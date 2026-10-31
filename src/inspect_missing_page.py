import json
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAGES_FILE = PROJECT_DIR / "data" / "all_required_wiki_pages.jsonl"

def normalize(text):
    return unicodedata.normalize("NFC", text).casefold()

search_term = normalize("Auk")

found = 0

with open(PAGES_FILE, "r", encoding="utf-8") as file:
    for line in file:
        page = json.loads(line)
        title = page.get("title", "")

        if search_term in normalize(title):
            print("MATCH:", repr(title))
            found += 1

print("\nTotal similar titles found:", found)