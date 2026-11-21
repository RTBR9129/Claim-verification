"""Check the pilot's trainable layers and paired evaluation."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from fine_tune_feverous import nli_label_order, paired_changes, trainable_parameters


class FineTuningPilotTest(unittest.TestCase):
    def test_model_label_order_comes_from_configuration(self):
        order = nli_label_order({0: "neutral", 1: "entailment", 2: "contradiction"})
        self.assertEqual(order, ["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"])

    def test_freezes_lower_layers_and_unfreezes_upper_layers_and_head(self):
        model = torch.nn.Module()
        model.base_model = torch.nn.Module()
        model.base_model.embeddings = torch.nn.Embedding(5, 2)
        model.base_model.encoder = torch.nn.Module()
        model.base_model.encoder.layer = torch.nn.ModuleList([
            torch.nn.Linear(2, 2) for _ in range(4)
        ])
        model.classifier = torch.nn.Linear(2, 3)

        trained = dict(trainable_parameters(model))

        self.assertFalse(model.base_model.embeddings.weight.requires_grad)
        self.assertFalse(model.base_model.encoder.layer[1].weight.requires_grad)
        self.assertIn("base_model.encoder.layer.2.weight", trained)
        self.assertIn("base_model.encoder.layer.3.weight", trained)
        self.assertIn("classifier.weight", trained)

    def test_paired_changes_exclude_shared_mistakes(self):
        expected = ["SUPPORTS", "REFUTES", "SUPPORTS", "REFUTES"]
        before = ["REFUTES", "REFUTES", "REFUTES", "SUPPORTS"]
        after = ["SUPPORTS", "SUPPORTS", "SUPPORTS", "NOT ENOUGH INFO"]

        self.assertEqual(paired_changes(expected, before, after), {"fixed": 2, "broke": 1, "net": 1})

