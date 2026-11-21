"""Check default SVM output and manual corrections without loading evidence models."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import claim_verifier


class VerifyDefaultModelTest(unittest.TestCase):
    def test_svm_sets_default_label_and_keeps_evidence_visible(self):
        classifier = Mock(spec=["predict", "decision_function", "classes_"])
        classifier.classes_ = ["NOT ENOUGH INFO", "REFUTES", "SUPPORTS"]
        classifier.predict.return_value = ["REFUTES"]
        classifier.decision_function.return_value = [[-0.8, 0.4, -0.3]]
        path = Mock()
        path.is_file.return_value = True
        evidence = {
            "label": "SUPPORTS",
            "confidence": 0.55,
            "calibration": "test calibration",
            "human_review": False,
            "evidence": [{
                "title": "Example article", "chunk_id": 0, "relation": "supports",
                "text": "A passage for the audience to review.", "nli": {"supports": 0.55},
            }],
            "graph": {"supports": 1, "refutes": 0, "neutral": 0},
            "explanation": "Example evidence explanation.",
        }
        for corrections, final_label in (
            ({}, "REFUTES"),
            ({"an example": {"label": "SUPPORTS", "note": "Reviewed."}}, "SUPPORTS"),
        ):
            with (
                self.subTest(final_label=final_label),
                patch.object(claim_verifier, "SVM_MODEL_FILE", path),
                patch.object(claim_verifier, "build_search_index"),
                patch.object(claim_verifier, "load_label_corrections", return_value=corrections),
                patch("joblib.load", return_value=classifier) as load_model,
                patch("evidence_pipeline.verify_claim_pipeline", return_value=evidence),
                redirect_stdout(io.StringIO()) as output,
            ):
                claim_verifier.verify_claim("An example.", 3)
                load_model.assert_called_once_with(path)
                text = output.getvalue()
                self.assertIn(f"Final label: {final_label}", text)
                self.assertIn("Claim-only model estimate: REFUTES", text)
                self.assertIn("REFUTES: +0.400", text)
                self.assertIn("scores are not probabilities", text)
                self.assertIn("Evidence NLI estimate (advisory): SUPPORTS", text)
                self.assertIn(
                    "Review recommended: yes (claim model and evidence estimate give opposite labels)",
                    text,
                )
                self.assertIn("Example article (chunk 0)", text)
                self.assertEqual("Applied manual correction: Reviewed." in text, bool(corrections))

    def test_missing_default_model_reports_training_command_before_retrieval(self):
        path = Mock()
        path.is_file.return_value = False
        with (
            patch.object(claim_verifier, "SVM_MODEL_FILE", path),
            patch.object(claim_verifier, "build_search_index") as build_index,
            self.assertRaisesRegex(FileNotFoundError, "train_word_character_svm.py"),
        ):
            claim_verifier.verify_claim("An example.", 3)
        build_index.assert_not_called()

