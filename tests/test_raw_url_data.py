"""Network-free tests for raw URL preparation and domain-disjoint splitting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_url_data import domain_disjoint_split, main, prepare_raw_url_data, registered_domain  # noqa: E402


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": [
                "https://login.badone.com/a",
                "https://cdn.badone.com/b",
                "http://pay.badtwo.net/",
                "https://goodone.org/",
                "https://www.goodone.org/account",
                "https://goodtwo.co.uk/",
            ],
            "label": ["phishing", 1, 1, "legitimate", 0, 0],
            "source": ["test"] * 6,
            "collected_at": ["2026-01-01T00:00:00Z"] * 6,
        }
    )


class RawUrlDataTests(unittest.TestCase):
    def test_preparation_canonicalises_and_audits_duplicates(self) -> None:
        frame = sample_frame()
        frame.loc[len(frame)] = ["https://goodtwo.co.uk/#fragment", 0, "test", "2026-01-01T00:00:00Z"]
        prepared, audit = prepare_raw_url_data(frame)
        self.assertEqual(audit.input_rows, 7)
        self.assertEqual(audit.duplicate_rows_removed, 1)
        self.assertEqual(audit.unique_rows, 6)
        self.assertEqual(set(prepared["label"]), {0, 1})
        self.assertIn("registered_domain", prepared.columns)

    def test_conflicting_labels_are_rejected(self) -> None:
        frame = sample_frame()
        frame.loc[len(frame)] = ["https://goodtwo.co.uk/", 1, "test", "2026-01-01T00:00:00Z"]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_raw_url_data(frame)

    def test_split_has_no_registered_domain_overlap(self) -> None:
        prepared, _ = prepare_raw_url_data(sample_frame())
        train, test, manifest = domain_disjoint_split(prepared, test_size=0.34, candidates=50)
        self.assertFalse(set(train["registered_domain"]).intersection(test["registered_domain"]))
        self.assertEqual(manifest["train_rows"] + manifest["test_rows"], len(prepared))

    def test_private_suffix_tenants_are_separate_groups(self) -> None:
        self.assertNotEqual(
            registered_domain("https://one.github.io/a"),
            registered_domain("https://two.github.io/b"),
        )

    def test_cli_writes_audit_and_locked_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.csv"
            output_dir = root / "out"
            sample_frame().to_csv(input_path, index=False)
            original_argv = sys.argv
            try:
                sys.argv = [
                    "raw_url_data.py",
                    "--input", str(input_path),
                    "--output-dir", str(output_dir),
                    "--test-size", "0.34",
                ]
                main()
            finally:
                sys.argv = original_argv
            self.assertTrue((output_dir / "data_audit.json").is_file())
            self.assertTrue((output_dir / "locked_test_records.csv").is_file())


if __name__ == "__main__":
    unittest.main()
