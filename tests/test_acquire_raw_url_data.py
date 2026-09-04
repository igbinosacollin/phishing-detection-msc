"""Offline tests for official-feed parsing and provenance output."""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acquire_raw_url_data import _parse_commoncrawl_records, assemble  # noqa: E402


class AcquisitionTests(unittest.TestCase):
    def test_assemble_creates_balanced_input_and_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phish = root / "phishtank.csv"
            pd.DataFrame({
                "url": ["https://bad-one.example/a", "http://bad-two.example/login"],
                "verified": ["yes", "yes"], "online": ["yes", "yes"],
                "verification_time": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            }).to_csv(phish, index=False)
            tranco = root / "tranco.zip"
            with zipfile.ZipFile(tranco, "w") as archive:
                archive.writestr("top-1m.csv", "1,example.org\n2,example.net\n3,not a domain\n")
            output = root / "input.csv"
            manifest = root / "manifest.json"
            legitimate = pd.DataFrame({
                "url": ["https://example.org/account", "https://example.net/docs?topic=test"],
                "label": ["legitimate", "legitimate"],
                "source": ["test Common Crawl metadata"] * 2,
                "collected_at": ["2026-01-01T00:00:00Z"] * 2,
            })
            record = assemble(
                phish, tranco, output, manifest, per_class=2, legitimate=legitimate,
                legitimate_metadata={"collection_id": "CC-TEST", "cdx_api": "https://index.example"},
            )
            created = pd.read_csv(output)
            self.assertEqual(len(created), 4)
            self.assertEqual(created["label"].value_counts().to_dict(), {"phishing": 2, "legitimate": 2})
            self.assertEqual(record["rows"], 4)
            self.assertTrue(manifest.is_file())
            self.assertTrue(all(len(source["sha256"]) == 64 for source in record["sources"][:2]))
            self.assertEqual(record["sources"][2]["collection_id"], "CC-TEST")
            self.assertEqual(record["url_shape_audit"]["legitimate"]["non_root_or_query_rate"], 1.0)

    def test_cdx_parser_rejects_roots_and_non_html_without_fetching_candidate_urls(self) -> None:
        payload = "\n".join([
            '{"url": "https://example.org/", "status": "200", "mime": "text/html"}',
            '{"url": "https://example.org/account?next=home", "status": "200", "mime": "text/html", "timestamp": "20260903000000"}',
            '{"url": "https://example.org/logo.png", "status": "200", "mime": "image/png"}',
            '{"url": "https://other.org/page", "status": "200", "mime": "text/html"}',
        ])
        self.assertEqual(
            _parse_commoncrawl_records(payload, "example.org", 5),
            [{"url": "https://example.org/account?next=home", "capture_timestamp": "20260903000000", "status": "200", "mime": "text/html"}],
        )


if __name__ == "__main__":
    unittest.main()
