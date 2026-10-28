
import json
import zipfile
from pathlib import Path

ZIP_PATH = Path("data/feverous-wiki-pages.zip")

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    members = [
        name for name in archive.namelist()
        if name.endswith(".jsonl")
    ]

    print("JSONL files in archive:", len(members))

    if members:
        first_file = members[0]
        print("\nInspecting:", first_file)

        with archive.open(first_file) as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                record = json.loads(raw_line.decode("utf-8"))

                print("\nRecord keys:", list(record.keys()))
                print("\nFirst record:")
                print(json.dumps(record, indent=2)[:2500])

                break

print("\nInspection complete. No files extracted.")
