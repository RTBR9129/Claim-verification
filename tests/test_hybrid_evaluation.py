"""Regression checks for the paired model evaluation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import claim_verifier


class HybridEvaluationTest(unittest.TestCase):
    def test_disagreement_summary_counts_wins_losses_and_oracle(self):
        rows = [
            ("SUPPORTS", "REFUTES", "SUPPORTS"),
            ("SUPPORTS", "SUPPORTS", "REFUTES"),
            ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
            ("REFUTES", "REFUTES", "REFUTES"),
            ("REFUTES", "SUPPORTS", "SUPPORTS"),
        ]
        summary = claim_verifier.summarize_model_disagreements([
            dict(zip(("expected_label", "baseline_label", "evidence_label"), row))
            for row in rows
        ])

        self.assertEqual(summary["agreements"], 2)
        self.assertEqual(summary["disagreements"], 3)
        self.assertEqual(summary["baseline_errors_fixed"], 1)
        self.assertEqual(summary["baseline_correct_answers_lost"], 1)
        self.assertEqual(summary["both_wrong_on_disagreement"], 1)
        self.assertEqual(summary["by_evidence_label"]["SUPPORTS"]["fixed"], 1)
        self.assertEqual(summary["oracle_accuracy"], 0.6)

    def test_report_keeps_original_labels_and_distinguishes_missing_nli_scores(self):
        baseline = Mock()
        baseline.classes_ = ["NOT ENOUGH INFO", "REFUTES", "SUPPORTS"]
        baseline.predict.return_value = ["REFUTES", "SUPPORTS"]
        baseline.predict_proba.return_value = [[0.1, 0.5, 0.4], [0.1, 0.1, 0.8]]
        evidence = Mock(temperature=4.0, calibration_status="Test score status")
        evidence.verify.side_effect = [
            {
                "label": "NOT ENOUGH INFO", "evidence": [], "human_review": True,
                "reason": "No passages were retrieved.",
            },
            {
                "label": "REFUTES", "evidence": [{}], "human_review": False,
                "nli_aggregate": {"refutes": 0.8, "supports": 0.1, "neutral": 0.1},
            },
        ]
        records = [
            {"claim": "One", "label": "SUPPORTS"},
            {"claim": "Two", "label": "REFUTES"},
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as directory:
            model_dir = Path(directory)
            model_path = model_dir / "baseline.joblib"
            model_path.touch()
            with (
                patch.object(claim_verifier, "MODEL_DIR", model_dir),
                patch.object(claim_verifier, "MODEL_FILE", model_path),
                patch.object(claim_verifier, "build_search_index"),
                patch.object(claim_verifier, "load_training_records", return_value=records),
                patch.object(claim_verifier, "load_label_corrections") as corrections,
                patch("joblib.load", return_value=baseline),
                patch("evidence_pipeline.EvidencePipeline", return_value=evidence),
                patch("builtins.print"),
            ):
                claim_verifier.evaluate_hybrid(max_claims=0, seed=42, top_k=3)

            path = model_dir / "hybrid_verifier_full_validation.json"
            report = json.loads(path.read_text(encoding="utf-8"))

        corrections.assert_not_called()
        self.assertFalse(report["manual_corrections_applied"])
        self.assertEqual(report["disagreements"]["baseline_errors_fixed"], 1)
        self.assertEqual(report["claims"][0]["expected_label"], "SUPPORTS")
        self.assertEqual(report["claims"][0]["baseline_scores"]["REFUTES"], 0.5)
        self.assertIsNone(report["claims"][0]["evidence_scores"])
        self.assertEqual(report["claims"][1]["evidence_scores"]["refutes"], 0.8)
