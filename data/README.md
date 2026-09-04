# Raw URL research data

This folder must contain only documented research snapshots. Do not add live
phishing URLs to source control unless their storage and access risks have been
approved by the supervisor.

Each supplied CSV must contain these columns:

| Column | Meaning |
|---|---|
| `url` | Original URL text, retained only when ethics and storage controls permit it |
| `label` | `phishing`/`1` or `legitimate`/`0` |
| `source` | Dataset/feed name and version or snapshot identifier |
| `collected_at` | ISO-8601 UTC collection or verification timestamp |

For every snapshot, record its checksum, licence/terms, collection command or
manual procedure, row counts, label rationale, deduplication decisions and the
date it was frozen. Never fetch or visit a suspect URL during preparation.

Use `acquire_raw_url_data.py` for the PhishTank/Tranco study input. It writes a
JSON source manifest containing retrieval metadata and SHA-256 hashes, then
creates `raw_url_input.csv` outside version control. Start with the template
when using another approved source; copy it outside this repository if source
terms or safety controls prohibit keeping raw URLs in Git history.

`raw_url_data.py` validates this schema and creates a registered-domain-disjoint
outer test partition. Keep the final test partition untouched until the model,
calibration and threshold have been locked on the development partition.
