"""Build a traceable raw-URL data set without visiting any candidate URL.

PhishTank supplies verified-online phishing URLs. Legitimate candidates are
sampled from the URL metadata in a Common Crawl index, restricted to hosts
ranked by Tranco. Unlike a top-domain list, a Common Crawl index includes real
paths and query strings. The script never requests a URL listed in either
source and never downloads a WARC or page body.

The resulting legitimate class is still a benign proxy: a Tranco-hosted URL is
not proof of safety. The manifest therefore records the exact collection,
selection rules and URL-shape audit. It deliberately rejects the historic
bare-root construction (``https://<domain>/``), which created a label leak.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pandas as pd

from raw_url_data import registered_domain


PHISHTANK_DEFAULT = "https://data.phishtank.com/data/online-valid.csv"
TRANCO_DEFAULT = "https://tranco-list.eu/top-1m.csv.zip"
COMMONCRAWL_COLLECTIONS = "https://index.commoncrawl.org/collinfo.json"
USER_AGENT = "PROM02-phishing-research/2.0 (academic URL-metadata collection)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def download(url: str, destination: Path) -> dict[str, str]:
    """Download only a declared source feed, never URLs held inside that feed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
        return {
            "url": url,
            "retrieved_at": now_utc(),
            "content_type": response.headers.get("Content-Type", ""),
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
        }


def fetch_text(url: str, *, timeout: int = 60) -> str:
    """Fetch an index response or collection manifest, not a candidate URL."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_phishtank(path: Path, limit: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda name: name in {"url", "verification_time", "verified", "online"})
    if "url" not in frame:
        raise ValueError("PhishTank CSV did not contain a url column")
    if "verified" in frame:
        frame = frame[frame["verified"].astype(str).str.lower().eq("yes")]
    if "online" in frame:
        frame = frame[frame["online"].astype(str).str.lower().eq("yes")]
    timestamp = frame.get("verification_time", pd.Series([pd.NaT] * len(frame)))
    output = pd.DataFrame({
        "url": frame["url"].astype(str),
        "label": "phishing",
        "source": "PhishTank verified online feed",
        "collected_at": pd.to_datetime(timestamp, utc=True, errors="coerce").fillna(pd.Timestamp.now(tz="UTC")),
    })
    return output.drop_duplicates("url").head(limit).reset_index(drop=True)


def _read_tranco_domains(path: Path, limit: int) -> pd.DataFrame:
    """Read ranked hosts only; they are later used to constrain index queries."""
    frame = pd.read_csv(path, header=None, names=["rank", "domain"], compression="infer", nrows=limit * 2)
    frame["domain"] = frame["domain"].astype(str).str.strip().str.lower().str.rstrip(".")
    frame = frame[frame["domain"].str.match(r"^[a-z0-9.-]+$", na=False)]
    return frame.drop_duplicates("domain").head(limit).reset_index(drop=True)


def _latest_commoncrawl_index(fetcher: Callable[[str], str] = fetch_text) -> tuple[str, str]:
    """Return the latest public Common Crawl CDX endpoint and its collection id."""
    collections = json.loads(fetcher(COMMONCRAWL_COLLECTIONS))
    if not isinstance(collections, list) or not collections:
        raise ValueError("Common Crawl collection manifest contained no indexes")
    latest = collections[0]
    endpoint = str(latest.get("cdx-api", "")).strip()
    identifier = str(latest.get("id", "")).strip()
    if not endpoint or not identifier:
        raise ValueError("Common Crawl collection manifest omitted cdx-api or id")
    return endpoint, identifier


def _is_full_http_url(value: object) -> bool:
    """Accept only real HTTP(S) paths or query strings; reject bare roots."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    return bool((parsed.path or "/") != "/" or parsed.query)


def _parse_commoncrawl_records(payload: str, allowed_domain: str, max_records: int) -> list[dict[str, str]]:
    """Parse CDX JSON-lines metadata and retain unique, full URLs on one host group."""
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in payload.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or str(record.get("status", "")) != "200":
            continue
        mime = str(record.get("mime-detected") or record.get("mime") or "").lower()
        if "html" not in mime:
            continue
        url = str(record.get("url", "")).strip()
        if not _is_full_http_url(url):
            continue
        try:
            if registered_domain(url) != allowed_domain:
                continue
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        output.append({
            "url": url,
            "capture_timestamp": str(record.get("timestamp", "")),
            "status": str(record.get("status", "")),
            "mime": mime,
        })
        if len(output) >= max_records:
            break
    return output


def _commoncrawl_query_url(endpoint: str, domain: str, response_limit: int) -> str:
    query = urllib.parse.urlencode({
        "url": domain,
        "matchType": "domain",
        "output": "json",
        "filter": ["status:200", "mime:text/html"],
        "collapse": "urlkey",
        "limit": max(response_limit, 1),
    }, doseq=True)
    return f"{endpoint}?{query}"


def collect_commoncrawl_tranco_urls(
    tranco: pd.DataFrame,
    *,
    target_rows: int,
    endpoint: str,
    collection_id: str,
    max_per_domain: int,
    delay_seconds: float,
    timeout: int,
    fetcher: Callable[..., str] = fetch_text,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect URL metadata sequentially and politely from the Common Crawl index."""
    if target_rows < 1 or max_per_domain < 1 or delay_seconds < 0:
        raise ValueError("target_rows and max_per_domain must be positive; delay_seconds cannot be negative")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    failures: list[dict[str, str]] = []
    queries = 0
    for _, candidate in tranco.iterrows():
        if len(selected) >= target_rows:
            break
        domain = str(candidate["domain"])
        url = _commoncrawl_query_url(endpoint, domain, max_per_domain * 10)
        try:
            try:
                payload = fetcher(url, timeout=timeout)
            except TypeError:  # permits a minimal test double without keyword arguments
                payload = fetcher(url)
            rows = _parse_commoncrawl_records(payload, domain, max_per_domain)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            failures.append({"domain": domain, "error": f"{type(exc).__name__}: {exc}"})
            rows = []
        queries += 1
        for row in rows:
            if row["url"] in seen:
                continue
            seen.add(row["url"])
            selected.append({
                "url": row["url"],
                "label": "legitimate",
                "source": f"Common Crawl {collection_id} URL index; Tranco-hosted benign proxy",
                "collected_at": now_utc(),
                "tranco_rank": int(candidate["rank"]),
                "tranco_domain": domain,
                "crawl_collection": collection_id,
                "capture_timestamp": row["capture_timestamp"],
                "mime": row["mime"],
            })
            if len(selected) >= target_rows:
                break
        if delay_seconds and len(selected) < target_rows:
            time.sleep(delay_seconds)
    if len(selected) < target_rows:
        raise ValueError(
            f"Common Crawl collection produced {len(selected)} full legitimate URL candidates; "
            f"needed {target_rows}. Increase --cc-domain-candidates or choose another index."
        )
    frame = pd.DataFrame(selected[:target_rows])
    metadata: dict[str, Any] = {
        "collection_id": collection_id,
        "cdx_api": endpoint,
        "queries_completed": queries,
        "query_failures": failures,
        "max_urls_per_registered_domain": max_per_domain,
        "delay_seconds": delay_seconds,
        "selection_rule": "HTTP(S) status 200 HTML URL metadata on a Tranco-ranked registered domain; a non-root path or query is required; no candidate URL was fetched.",
    }
    return frame, metadata


def url_shape_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Report class-wise lexical shape so source-construction confounds are visible."""
    rows: dict[str, Any] = {}
    for label, part in frame.groupby("label"):
        parsed = part["url"].map(urlsplit)
        lengths = part["url"].str.len()
        non_root = parsed.map(lambda value: (value.path or "/") != "/" or bool(value.query))
        queries = parsed.map(lambda value: bool(value.query))
        rows[str(label)] = {
            "rows": int(len(part)),
            "mean_url_length": round(float(lengths.mean()), 3),
            "median_url_length": float(lengths.median()),
            "non_root_or_query_rate": round(float(non_root.mean()), 6),
            "query_string_rate": round(float(queries.mean()), 6),
        }
    return rows


def _drop_cross_label_domains(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    grouped = frame["url"].map(registered_domain)
    labels_per_domain = frame.assign(_domain=grouped).groupby("_domain")["label"].nunique()
    conflicts = set(labels_per_domain[labels_per_domain > 1].index)
    if not conflicts:
        return frame.reset_index(drop=True), 0
    kept = frame.loc[~grouped.isin(conflicts)].reset_index(drop=True)
    return kept, int(len(frame) - len(kept))


def assemble(
    phish_path: Path,
    tranco_path: Path,
    output_csv: Path,
    manifest_path: Path,
    per_class: int,
    legitimate: pd.DataFrame,
    *,
    legitimate_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Make a balanced input after validating the full-path legitimate class."""
    phishing = _read_phishtank(phish_path, per_class)
    legitimate = legitimate.loc[:, ["url", "label", "source", "collected_at"]].copy()
    legitimate = legitimate[legitimate["url"].map(_is_full_http_url)].drop_duplicates("url").head(per_class)
    if len(phishing) < per_class or len(legitimate) < per_class:
        raise ValueError(f"needed {per_class} records per class; got phishing={len(phishing)}, legitimate={len(legitimate)}")
    combined = pd.concat([phishing, legitimate], ignore_index=True)
    combined, removed_cross_label_domains = _drop_cross_label_domains(combined)
    class_counts = combined["label"].value_counts().to_dict()
    if class_counts.get("phishing", 0) != per_class or class_counts.get("legitimate", 0) != per_class:
        raise ValueError(
            "cross-label registered-domain conflicts reduced the balanced data set; "
            "collect additional candidates and rerun rather than keeping mixed-label domains"
        )
    shape = url_shape_audit(combined)
    if shape["legitimate"]["non_root_or_query_rate"] != 1.0:
        raise AssertionError("legitimate class contains a bare root; refusing to recreate the known confound")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    manifest: dict[str, Any] = {
        "created_at": now_utc(),
        "rows": int(len(combined)),
        "per_class_requested": per_class,
        "phishing_rows": int(len(phishing)),
        "legitimate_rows": int(len(legitimate)),
        "cross_label_domain_rows_removed": removed_cross_label_domains,
        "legitimate_strategy": "Common Crawl URL metadata restricted to Tranco hosts; non-root path or query required",
        "url_shape_audit": shape,
        "sources": [
            {"name": "PhishTank", "path": str(phish_path), "sha256": sha256(phish_path), "license_or_terms": "Review PhishTank terms before redistribution."},
            {"name": "Tranco", "path": str(tranco_path), "sha256": sha256(tranco_path), "license_or_terms": "Review Tranco methodology and source licences before redistribution."},
            {"name": "Common Crawl URL index", **legitimate_metadata},
        ],
        "limitations": [
            "URLs are not fetched, resolved, opened or otherwise visited by this pipeline.",
            "A Tranco-hosted Common Crawl URL is a benign proxy, not a ground-truth safety label.",
            "The locked test set shares the same collection mechanism; later external evaluation remains required.",
            "Do not commit raw feeds or derived URL lists unless their terms permit it.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/restricted"))
    parser.add_argument("--per-class", type=int, default=10_000, help="Rows per class; 10,000 gives a 20k-row balanced input")
    parser.add_argument("--phishtank-url", default=PHISHTANK_DEFAULT)
    parser.add_argument("--tranco-url", default=TRANCO_DEFAULT)
    parser.add_argument("--phish-file", type=Path, help="Use an already-downloaded PhishTank CSV")
    parser.add_argument("--tranco-file", type=Path, help="Use an already-downloaded Tranco CSV/ZIP")
    parser.add_argument("--no-download", action="store_true", help="Require --phish-file and --tranco-file")
    parser.add_argument("--cc-index", help="Common Crawl CDX API endpoint; default resolves the latest public collection")
    parser.add_argument("--cc-collection-id", help="Required with --cc-index for provenance; otherwise resolved automatically")
    parser.add_argument("--cc-domain-candidates", type=int, help="Tranco hosts to query; default is sufficient for the requested class size and per-domain cap")
    parser.add_argument("--cc-max-per-domain", type=int, default=25, help="Maximum selected URLs from one registered domain")
    parser.add_argument("--cc-delay-seconds", type=float, default=0.5, help="Pause between sequential index queries; respect Common Crawl service capacity")
    parser.add_argument("--cc-timeout", type=int, default=60)
    args = parser.parse_args()
    if args.per_class < 1 or args.cc_max_per_domain < 1 or args.cc_delay_seconds < 0:
        parser.error("--per-class and --cc-max-per-domain must be positive; --cc-delay-seconds cannot be negative")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phish_path = args.phish_file or output_dir / "phishtank_online_valid.csv"
    tranco_path = args.tranco_file or output_dir / "tranco_top_1m.csv.zip"
    feed_metadata: dict[str, dict[str, str]] = {}
    if not args.no_download:
        if not args.phish_file:
            feed_metadata["phishtank"] = download(args.phishtank_url, phish_path)
        if not args.tranco_file:
            feed_metadata["tranco"] = download(args.tranco_url, tranco_path)
    if not phish_path.is_file() or not tranco_path.is_file():
        parser.error("source feed missing; provide --phish-file and --tranco-file, or omit --no-download")
    if not feed_metadata:
        feed_metadata = {
            "phishtank": {"url": args.phishtank_url, "source_file": str(phish_path), "local_file_mtime": datetime.fromtimestamp(phish_path.stat().st_mtime, tz=timezone.utc).isoformat(), "retrieval_status": "preexisting_local_file"},
            "tranco": {"url": args.tranco_url, "source_file": str(tranco_path), "local_file_mtime": datetime.fromtimestamp(tranco_path.stat().st_mtime, tz=timezone.utc).isoformat(), "retrieval_status": "preexisting_local_file"},
        }
    if args.cc_index:
        if not args.cc_collection_id:
            parser.error("--cc-collection-id is required when --cc-index is supplied")
        endpoint, collection_id = args.cc_index, args.cc_collection_id
    else:
        endpoint, collection_id = _latest_commoncrawl_index()
    candidate_count = args.cc_domain_candidates or max(1_000, (args.per_class + args.cc_max_per_domain - 1) // args.cc_max_per_domain * 3)
    tranco = _read_tranco_domains(tranco_path, candidate_count)
    legitimate, cc_metadata = collect_commoncrawl_tranco_urls(
        tranco,
        target_rows=args.per_class,
        endpoint=endpoint,
        collection_id=collection_id,
        max_per_domain=args.cc_max_per_domain,
        delay_seconds=args.cc_delay_seconds,
        timeout=args.cc_timeout,
    )
    legitimate.to_csv(output_dir / "commoncrawl_tranco_full_url_provenance.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = assemble(
        phish_path,
        tranco_path,
        output_dir / "raw_url_input.csv",
        output_dir / "source_manifest.json",
        args.per_class,
        legitimate,
        legitimate_metadata=cc_metadata,
    )
    manifest["download_metadata"] = feed_metadata
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
