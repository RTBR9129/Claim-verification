"""Tests for the data sent to the local claim verification website."""

import unittest
from unittest.mock import patch

from website import server


class WebsiteTest(unittest.TestCase):
    def test_scores_and_parts_come_from_the_verifier(self):
        scores = {"supports": 0.118, "refutes": 0.347, "neutral": 0.535}
        parts = [
            {"claim": "Peter was born in Vienna.", "label": "SUPPORTS"},
            {"claim": "Peter did not qualify.", "label": "NOT ENOUGH INFO"},
        ]
        evidence = {
            "label": "NOT ENOUGH INFO", "human_review": True,
            "nli_aggregate": scores, "clause_checks": parts,
            "evidence": [
                {"title": "Other page", "text": "Background.", "relation": "unrelated"},
                {"title": "Peter", "text": "Full passage.", "nli_basis": "He attended school.", "relation": "neutral"},
            ],
        }

        with (
            patch.object(server, "build_search_index"),
            patch.object(server, "verify_claim_pipeline", return_value=evidence),
        ):
            result = server.check_claim("Peter attended school.")

        self.assertEqual(result["label"], "NOT ENOUGH INFO")
        self.assertNotIn("evidence_label", result)
        self.assertEqual(result["scores"], scores)
        self.assertEqual(result["parts"], parts)
        self.assertTrue(result["review"])
        self.assertEqual(result["passages"][0]["text"], "He attended school.")
