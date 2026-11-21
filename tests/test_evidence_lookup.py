"""Check evidence IDs and legacy page-title references."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data_prep.build_evidence_lookup import extract_evidence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import claim_verifier


class EvidenceLookupTest(unittest.TestCase):
    def test_reads_normalized_ids_and_recovers_known_page_titles(self):
        content_id = "Cafe\u0301_sentence_0"
        records = [{"evidence": [{
            "content": [content_id, "Missing_title"],
            "context": {content_id: ["Cafe\u0301_title"]},
        }]}]
        payload = {"Caf\u00e9_sentence_0": {"title": "Caf\u00e9", "text": "A fact."}}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as folder:
            path = Path(folder) / "lookup.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(claim_verifier, "EVIDENCE_LOOKUP_FILE", path), patch("builtins.print"):
                result = claim_verifier.load_referenced_evidence(records)

        self.assertEqual(result[content_id]["text"], "A fact.")
        self.assertEqual(result["Cafe\u0301_title"]["text"], "Caf\u00e9")
        self.assertNotIn("Missing_title", result)

    def test_conflicting_aliases_do_not_replace_a_direct_match(self):
        first = "\u01fa_sentence_0"
        second = "A\u030a\u0301_sentence_0"
        alias = "\u00c5\u0301_sentence_0"
        records = [{"evidence": [{"content": [first, alias]}]}]
        payload = {
            first: {"title": "\u01fa", "text": "One"},
            second: {"title": "\u01fa", "text": "Two"},
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as folder:
            path = Path(folder) / "lookup.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(claim_verifier, "EVIDENCE_LOOKUP_FILE", path), patch("builtins.print"):
                result = claim_verifier.load_referenced_evidence(records)

        self.assertEqual(result[first]["text"], "One")
        self.assertNotIn(alias, result)

    def test_new_lookup_includes_page_title(self):
        records = extract_evidence({"title": "Example", "sentence_0": "A fact."})
        self.assertEqual(records["Example_title"], {
            "title": "Example", "type": "title", "text": "Example",
        })

