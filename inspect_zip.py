import zipfile
from pathlib import Path

ZIP_PATH = Path("data/feverous-wiki-pages.zip")

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    members = [
        name for name in archive.namelist()
        if name.endswith(".jsonl")
    ]

    print("Total JSONL files:", len(members))

    print("\nFirst 20 JSONL filenames:")
    for name in members[:20]:
        print(name)

    print("\nLast 5 JSONL filenames:")
    for name in members[-5:]:
        print(name)