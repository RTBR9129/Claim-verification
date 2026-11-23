"""Local web interface for the existing claim verifier. No web framework needed."""

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import sys
from threading import Lock

SITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SITE.parent / "src"))

from claim_verifier import (
    DATA_DIR, DEFAULT_TOP_K, build_search_index, load_training_records, retrieve_evidence,
)
from evidence_pipeline import verify_claim_pipeline

MAX_CLAIM_LENGTH = 1000
MAX_REQUEST_BYTES = 16_384
MODEL_LOCK = Lock()
STATIC_FILES = {
    "/": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}


@lru_cache(maxsize=1)
def random_claims():
    return [row["claim"] for row in load_training_records(DATA_DIR / "test_claims.jsonl")]


def check_claim(claim):
    # Keep model loading and inference serial while static files remain responsive.
    with MODEL_LOCK:
        build_search_index()
        result = verify_claim_pipeline(claim, DEFAULT_TOP_K, retrieve_evidence)
    passages = sorted(result["evidence"], key=lambda item: item["relation"] == "unrelated")
    return {
        "claim": claim,
        "label": result["label"],
        "scores": result.get("nli_aggregate"),
        "review": result["human_review"],
        "parts": [
            {"claim": part["claim"], "label": part["label"]}
            for part in result.get("clause_checks", [])
        ],
        "passages": [
            {
                "title": item["title"],
                "text": (item["nli_basis"] if item.get("nli_basis") and item["relation"] != "unrelated"
                         else item["text"])[:900],
                "relation": item["relation"].upper(),
            }
            for item in passages
        ],
    }


class WebsiteHandler(BaseHTTPRequestHandler):
    def send_data(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, data):
        self.send_data(status, "application/json", json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path in STATIC_FILES:
            name, content_type = STATIC_FILES[self.path]
            self.send_data(200, content_type, (SITE / name).read_bytes())
        elif self.path == "/api/random":
            try:
                claim = random.choice(random_claims())
                self.send_json(200, {"claim": claim})
            except (OSError, ValueError) as exc:
                self.send_json(503, {"error": str(exc)})
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if self.path != "/api/verify":
            self.send_json(404, {"error": "Page not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "Send the claim as JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            claim = payload.get("claim") if isinstance(payload, dict) else None
        except ValueError:
            self.send_json(400, {"error": "Invalid claim request."})
            return
        if not isinstance(claim, str) or not 0 < len(claim.strip()) <= MAX_CLAIM_LENGTH:
            self.send_json(400, {"error": f"Enter a claim of 1 to {MAX_CLAIM_LENGTH} characters."})
            return
        try:
            self.send_json(200, check_claim(claim.strip()))
        except Exception as exc:
            self.log_error("Verification failed: %s", exc)
            self.send_json(503, {"error": "The verifier could not complete the check. See the PowerShell window for details."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    with ThreadingHTTPServer(("127.0.0.1", args.port), WebsiteHandler) as server:
        print(f"Open http://127.0.0.1:{server.server_port} in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
