"""Checks for FEVEROUS experiment input preparation."""

import unittest

from experiments.train_feverous_verifier import annotated_context, distinct_records


class FeverousTrainingTest(unittest.TestCase):
    def test_annotation_keeps_table_header_with_value(self):
        record = {
            "evidence": [{
                "content": ["player_cell"],
                "context": {"player_cell": ["title", "season", "header"]},
            }],
        }
        lookup = {
            "title": {"title": "Player", "text": "Player"},
            "season": {"title": "Player", "text": "2012–13"},
            "header": {"title": "Player", "text": "Appearances"},
            "player_cell": {"title": "Player", "text": "22"},
        }
        self.assertEqual(annotated_context(record, lookup), "Player: 2012–13 | Appearances | 22")

    def test_incomplete_annotation_does_not_count_as_training_evidence(self):
        record = {"evidence": [{"content": ["known", "missing"]}]}
        lookup = {"known": {"title": "Page", "text": "Only one fact."}}
        self.assertEqual(annotated_context(record, lookup), "")

    def test_split_excludes_normalized_duplicates(self):
        rows = [{"claim": "A claim."}, {"claim": "Second!"}, {"claim": "SECOND"}]
        self.assertEqual(distinct_records(rows, {"a claim"}), [{"claim": "Second!"}])

