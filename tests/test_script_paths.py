"""Check that data preparation locates its project data from another directory."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptPathTest(unittest.TestCase):
    def test_chunk_text_resolves_inputs_and_outputs_from_its_project_root(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            scripts = root / "scripts" / "data_prep"
            scripts.mkdir(parents=True)
            script = shutil.copyfile(
                PROJECT_ROOT / "scripts" / "data_prep" / "chunk_text.py",
                scripts / "chunk_text.py",
            )
            data = root / "data"
            data.mkdir()
            (data / "extracted_text.jsonl").write_text(
                json.dumps({"title": "Example", "text": "A passage to index."}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [sys.executable, "-B", str(script)],
                cwd=scripts,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )

            output = (data / "evidence_chunks.jsonl").read_text(encoding="utf-8")
            self.assertEqual(
                json.loads(output),
                {"title": "Example", "chunk_id": 0, "text": "A passage to index."},
            )

