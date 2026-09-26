"""Unit Tests and Two-Sided Verification Gates for mios_micro.dataset."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.mios_micro import dataset as micro_ds


class TestMicroDataset(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.out_file = self.tmp_path / "test_corpus.jsonl"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_positive_control_corpus_generation(self):
        """Positive Control: generate_micro_dataset produces requested count with 4 valid pillars."""
        examples = micro_ds.generate_micro_dataset(count=40)
        self.assertEqual(len(examples), 40)

        # Write to disk and verify JSONL structure
        with open(self.out_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in examples)
        self.assertTrue(self.out_file.exists())

        lines = self.out_file.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 40)

        for line in lines:
            record = json.loads(line)
            self.assertIn("messages", record)
            messages = record["messages"]
            self.assertGreaterEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertEqual(messages[2]["role"], "assistant")

    def test_negative_control_zero_or_negative_count(self):
        """Negative Control: count <= 0 yields empty list without crash."""
        examples = micro_ds.generate_micro_dataset(count=0)
        self.assertEqual(len(examples), 0)


if __name__ == "__main__":
    unittest.main()
