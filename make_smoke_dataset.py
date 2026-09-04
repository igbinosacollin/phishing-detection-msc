"""Create a clearly labelled synthetic raw-URL smoke dataset for local tests.

It is useful for checking commands and application wiring only. It must never be
used in the dissertation's results tables or described as real phishing evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(rows_per_class: int = 80) -> pd.DataFrame:
    if rows_per_class < 20:
        raise ValueError("use at least 20 rows per class so grouped CV can run")
    records: list[dict[str, str]] = []
    for index in range(rows_per_class):
        records.append({
            "url": f"https://www.legitimate{index}.co.uk/products/account-{index % 7}",
            "label": "legitimate",
            "source": "SYNTHETIC_SMOKE_ONLY",
            "collected_at": "2026-09-03T00:00:00Z",
        })
        records.append({
            "url": f"http://secure-paypa1-verify-{index}.net/login/confirm?account={10000 + index}&token=a%2Fb",
            "label": "phishing",
            "source": "SYNTHETIC_SMOKE_ONLY",
            "collected_at": "2026-09-03T00:00:00Z",
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke_raw_urls.csv"))
    parser.add_argument("--rows-per-class", type=int, default=80)
    args = parser.parse_args()
    dataset = build(args.rows_per_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"wrote {len(dataset)} synthetic smoke records to {args.output}")


if __name__ == "__main__":
    main()
