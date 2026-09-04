"""Raw-URL dataset preparation for the PROM02 phishing research project.

This module deliberately does not download, fetch, or visit URLs.  It prepares a
locally supplied research dataset for a reproducible lexical-feature experiment.
Every record must retain its source and collection timestamp so that the study can
report its provenance and hold out domains and time periods honestly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = ("url", "label", "source", "collected_at")
LABELS = {"phishing": 1, "legitimate": 0, "1": 1, "0": 0, 1: 1, 0: 0}
# Use tldextract's bundled suffix snapshot only.  Dataset preparation must not
# make a network request or silently change as the public suffix list changes.
_TLD = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=(), include_psl_private_domains=True)


@dataclass(frozen=True)
class DataAudit:
    input_rows: int
    unique_rows: int
    duplicate_rows_removed: int
    phishing_rows: int
    legitimate_rows: int
    registered_domains: int
    earliest_collected_at: str
    latest_collected_at: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def canonicalise_url(value: object) -> str:
    """Return a stable URL key, rejecting values with no host name.

    The raw ``url`` column is preserved in the prepared frame.  This key exists
    only to find duplicates and conflicting labels; it is not a replacement for
    the feature extractor's input.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("url must be a non-empty string")
    raw = value.strip()
    candidate = raw if "://" in raw else f"http://{raw}"
    parsed = urlsplit(candidate)
    if not parsed.hostname:
        raise ValueError(f"url has no host name: {value!r}")

    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname.lower().rstrip(".")
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, host, path, parsed.query, ""))


def registered_domain(url: str) -> str:
    """Return the registrable domain used to keep related URLs out of both splits."""
    host = urlsplit(canonicalise_url(url)).hostname or ""
    ext = _TLD(host)
    domain = ".".join(part for part in (ext.domain, ext.suffix) if part)
    # IP-address hosts have no public suffix; retain the host as their group.
    return domain or host.lower()


def _normalise_labels(values: Iterable[object]) -> pd.Series:
    normalised = pd.Series(values).map(lambda value: LABELS.get(str(value).strip().lower(), None))
    if normalised.isna().any():
        invalid = sorted({str(value) for value, label in zip(values, normalised) if pd.isna(label)})
        raise ValueError(f"label must be phishing/legitimate or 1/0; invalid values: {invalid[:5]}")
    return normalised.astype(int)


def prepare_raw_url_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataAudit]:
    """Validate, deduplicate, and annotate a supplied raw-URL dataset.

    A conflicting duplicate is a data-quality error and is never silently
    resolved. Exact duplicates with the same label are removed and recorded in
    the returned audit object.
    """
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    prepared = frame.loc[:, REQUIRED_COLUMNS].copy()
    prepared["url"] = prepared["url"].map(canonicalise_url)
    prepared["label"] = _normalise_labels(prepared["label"])
    prepared["source"] = prepared["source"].astype(str).str.strip()
    if (prepared["source"] == "").any():
        raise ValueError("source must be present for every row")
    # Sources legitimately use different ISO-8601 renderings (for example a
    # PhishTank ``T`` timestamp and a pandas-generated space timestamp). Pandas
    # 2+/3+ needs explicit mixed-format parsing for that provenance-preserving
    # combined file.
    prepared["collected_at"] = pd.to_datetime(prepared["collected_at"], utc=True, errors="raise", format="mixed")
    prepared["registered_domain"] = prepared["url"].map(registered_domain)

    conflicts = prepared.groupby("url")["label"].nunique()
    conflicting_urls = conflicts[conflicts > 1]
    if not conflicting_urls.empty:
        raise ValueError(f"conflicting labels for {len(conflicting_urls)} canonical URL(s)")

    input_rows = len(prepared)
    prepared = prepared.drop_duplicates(subset=["url", "label"], keep="first").reset_index(drop=True)
    audit = DataAudit(
        input_rows=input_rows,
        unique_rows=len(prepared),
        duplicate_rows_removed=input_rows - len(prepared),
        phishing_rows=int((prepared["label"] == 1).sum()),
        legitimate_rows=int((prepared["label"] == 0).sum()),
        registered_domains=int(prepared["registered_domain"].nunique()),
        earliest_collected_at=prepared["collected_at"].min().isoformat(),
        latest_collected_at=prepared["collected_at"].max().isoformat(),
    )
    return prepared, audit


def domain_disjoint_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    candidates: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Create one registered-domain-disjoint split with near-matched prevalence.

    Candidate group splits are scored by test-size and class-prevalence distance.
    This function only creates an outer split. Model selection and threshold
    selection must occur inside the returned training partition.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if frame["label"].nunique() != 2:
        raise ValueError("both phishing and legitimate labels are required")
    if frame["registered_domain"].nunique() < 2:
        raise ValueError("at least two registered domains are required")

    target_rate = float(frame["label"].mean())
    target_rows = len(frame) * test_size
    best: tuple[float, list[int], list[int], int] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + offset)
        train_index, test_index = next(splitter.split(frame, frame["label"], frame["registered_domain"]))
        test = frame.iloc[test_index]
        if test["label"].nunique() != 2:
            continue
        score = abs(len(test) - target_rows) / max(len(frame), 1) + abs(float(test["label"].mean()) - target_rate)
        if best is None or score < best[0]:
            best = (score, list(train_index), list(test_index), random_state + offset)

    if best is None:
        raise ValueError("could not form a two-class domain-disjoint test split")
    _, train_index, test_index, selected_seed = best
    train = frame.iloc[train_index].sort_index().reset_index(drop=True)
    test = frame.iloc[test_index].sort_index().reset_index(drop=True)
    if set(train["registered_domain"]).intersection(test["registered_domain"]):
        raise AssertionError("domain overlap detected after grouped split")
    manifest: dict[str, int | float] = {
        "random_state": selected_seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_phishing_rate": float(train["label"].mean()),
        "test_phishing_rate": float(test["label"].mean()),
        "train_domains": int(train["registered_domain"].nunique()),
        "test_domains": int(test["registered_domain"].nunique()),
    }
    return train, test, manifest


def main() -> None:
    """Validate a local CSV and write a traceable domain-disjoint outer split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with url, label, source and collected_at columns")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audited CSV and JSON outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=100)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    prepared, audit = prepare_raw_url_data(frame)
    train, test, manifest = domain_disjoint_split(
        prepared,
        test_size=args.test_size,
        random_state=args.random_state,
        candidates=args.candidates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output_dir / "prepared_records.csv", index=False)
    train.to_csv(args.output_dir / "development_records.csv", index=False)
    test.to_csv(args.output_dir / "locked_test_records.csv", index=False)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit.as_dict(), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit.as_dict(), "split": manifest}, indent=2))


if __name__ == "__main__":
    main()
