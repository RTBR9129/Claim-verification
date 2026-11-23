"""Check that the random demo forwards the claim without the dataset label."""

import subprocess
import unittest
from unittest.mock import patch

import run


class RandomClaimTest(unittest.TestCase):
    def test_seed_repeats_the_claim_without_forwarding_its_dataset_label(self):
        records = [
            {"claim": "A sentence with punctuation, a comma, and spaces.", "label": "REFUTES"},
            {"claim": "Another example.", "label": "SUPPORTS"},
        ]
        with (
            patch.object(run, "load_training_records", return_value=records) as load,
            patch.object(run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as verify,
        ):
            self.assertEqual(run.main(["--seed", "5"]), 0)
            self.assertEqual(run.main(["--seed", "5"]), 0)

        load.assert_called_with(run.DATA_DIR / "test_claims.jsonl")
        first, second = (call.args[0] for call in verify.call_args_list)
        self.assertEqual(first, second)
        self.assertIn(first[-1], [f"--claim={record['claim']}" for record in records])
        self.assertEqual(first[2], "verify")

    def test_details_flag_is_forwarded_to_verifier(self):
        with (
            patch.object(run, "load_training_records", return_value=[{"claim": "Example."}]),
            patch.object(run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as verify,
        ):
            self.assertEqual(run.main(["--details"]), 0)

        self.assertEqual(verify.call_args.args[0][-2:], ["--claim=Example.", "--details"])
