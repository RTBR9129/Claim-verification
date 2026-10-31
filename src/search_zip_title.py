import json
import zipfile
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ZIP_FILE = PROJECT_DIR / "data" / "feverous-wiki-pages.zip"

def normalize(text):
    return unicodedata.normalize("NFC", text).casefold()

target = normalize("Aukštaitian dialect")
found = False

with zipfile.ZipFile(ZIP_FILE, "r") as archive:
    files = [
        name for name in archive.namelist()
        if name.endswith(".jsonl")
    ]

    for filename in files:
        with archive.open(filename) as file:
            for line in file:
                page = json.loads(line.decode("utf-8"))
                title = page.get("title", "")

                if normalize(title) == target:
                    print("FOUND!")
                    print("Archive file:", filename)
                    print("Title:", repr(title))
                    found = True
                    break

        if found:
            break

if not found:
    print("PAGE NOT FOUND IN ORIGINAL ZIP")