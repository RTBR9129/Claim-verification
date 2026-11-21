

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "extracted_text.jsonl"
OUTPUT_FILE = DATA_DIR / "evidence_chunks.jsonl"

CHUNK_SIZE = 500
OVERLAP = 100


def split_into_chunks(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - OVERLAP

    return chunks


def main():
    page_count = 0
    chunk_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as source, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for line in source:
            if not line.strip():
                continue

            page = json.loads(line)
            title = page.get("title", "")
            text = page.get("text", "")

            chunks = split_into_chunks(text)

            for index, chunk in enumerate(chunks):
                record = {
                    "title": title,
                    "chunk_id": index,
                    "text": chunk
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                chunk_count += 1

            page_count += 1

    print("WEEK 2: TEXT CHUNKING COMPLETE")
    print("-" * 40)
    print("Pages processed:", page_count)
    print("Total chunks created:", chunk_count)
    print("Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
