"""The tracked synthetic fixture is constrained to smoke-testing use."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_smoke_dataset import build  # noqa: E402


class SmokeDatasetTests(unittest.TestCase):
    def test_fixture_is_balanced_and_explicitly_synthetic(self) -> None:
        data = build(20)
        self.assertEqual(len(data), 40)
        self.assertEqual(data["label"].value_counts().to_dict(), {"legitimate": 20, "phishing": 20})
        self.assertEqual(set(data["source"]), {"SYNTHETIC_SMOKE_ONLY"})


if __name__ == "__main__":
    unittest.main()
