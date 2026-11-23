"""Regression checks for evidence selection and compound claims."""

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

    def test_unmatched_retrieved_passage_keeps_display_metadata(self):
        self.pipeline.is_calibrated = False
        self.pipeline.retrieve = Mock(return_value=[
            {"title": "Shulan", "text": "Shulan is a city in China.", "chunk_id": 0},
        ])

        result = self.pipeline.verify("Shulin, which was reorganized, is located in China.")

        self.assertEqual(result["label"], "NOT ENOUGH INFO")
        self.assertEqual(result["evidence"][0]["relation"], "unrelated")
        self.assertEqual(result["evidence"][0]["nli"]["neutral"], 1.0)


class CompoundClaimTest(unittest.TestCase):
    def test_checks_each_part_and_cites_the_school_sentence(self):
        pipeline = EvidencePipeline.__new__(EvidencePipeline)
        pipeline.nli = Mock()
        pipeline.nli.model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        pipeline.nli.predict.return_value = np.array([[-5.0, 5.0, -2.0], [-5.0, 5.0, -2.0]])
        claim = (
            "Peter Example, an avant-garde filmmaker, was born in Vienna, Austria "
            "but did not qualify academically for the Primary School in Mistelbach."
        )
        school = "He attended the Primary School in Mistelbach from 1965 to 1969."
        evidence = [{
            "title": "Peter Example",
            "chunk_id": 0,
            "text": "Peter Example is an avant-garde filmmaker. "
                    "Peter Example was born in Vienna, Austria. " + school,
        }]

        checks = pipeline.check_compound_claim(claim, evidence)

        self.assertEqual([check["label"] for check in checks], [
            "SUPPORTS", "SUPPORTS", "NOT ENOUGH INFO",
        ])
        self.assertIn("did not qualify academically", checks[2]["claim"])
        self.assertEqual(checks[2]["sentence"], school)
        self.assertEqual(len(pipeline.nli.predict.call_args.args[0]), 2)

    def test_does_not_split_a_simple_claim_or_a_coordination_with_a_different_subject(self):
        for claim in (
            "Peter Example was born in Vienna and raised in Graz.",
            "Peter Example was born in Vienna, but Mary Example was born in Graz.",
        ):
            self.assertEqual(EvidencePipeline._compound_parts(claim, "Peter Example"), [])

