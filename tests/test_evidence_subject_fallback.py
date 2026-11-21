"""Regression checks for the title fallback in evidence selection."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from evidence_pipeline import EvidencePipeline


class SubjectFallbackTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = EvidencePipeline.__new__(EvidencePipeline)
        self.pipeline.nli = Mock()
        self.pipeline.nli.model.config.id2label = {
            0: "contradiction", 1: "entailment", 2: "neutral",
        }
        self.pipeline.nli.predict.side_effect = lambda pairs, **_: np.tile(
            [5.0, -2.0, -1.0], (len(pairs), 1),
        )

    def test_named_page_can_supply_a_complete_sentence_when_strict_match_fails(self):
        claim = "The Example League, which was formed in 1990, had 12 teams."
        sentence = "In its first season, there were 10 teams in the division."
        evidence = [{"title": "Example League", "text": sentence}]

        scores = self.pipeline._nli_probabilities(claim, evidence)

        self.assertEqual(evidence[0]["nli_basis"], sentence)
        self.assertGreater(scores[0]["refutes"], 0.9)
        premise, hypothesis = self.pipeline.nli.predict.call_args.args[0][0]
        self.assertIn(sentence, premise)
        self.assertEqual(hypothesis, claim)

    def test_similar_but_different_page_does_not_enter_fallback(self):
        claim = "Shulin, which was reorganized, is located in China."
        evidence = [{"title": "Shulan", "text": "Shulan is a city in China."}]

        scores = self.pipeline._nli_probabilities(claim, evidence)

        self.assertEqual(evidence[0]["nli_basis"], "No claim-matching sentence found.")
        self.assertEqual(scores[0]["neutral"], 1.0)
        self.assertEqual(self.pipeline.nli.predict.call_args.args[0], [])

