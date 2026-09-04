"""
Command-line scorer.

Default model: model_deployable13.joblib, the XGBoost model trained on the 13
UCI features obtainable without loading the page. Locked-test F1 0.7805
(95% CI 0.7595 to 0.7994).

The raw-URL bundle in artifacts/raw_url/ scores F1 0.9729 on its own locked test
set and is deliberately NOT the default: its legitimate training class was built
as bare "https://<domain>/" roots, so it classifies every real legitimate URL
carrying a path as phishing at 100% confidence. It is retained as evidence for
dissertation section 4.8 and can be loaded with --artefact to reproduce that
finding.

Usage:
    python predict.py https://example.com/login
    python predict.py --no-network https://example.com   (skip WHOIS and DNS)
    python predict.py --artefact  https://www.bbc.co.uk/news/x   (reproduce 4.8)
"""
from __future__ import annotations
import argparse, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTEFACT_BUNDLE = ROOT / "artifacts" / "raw_url" / "raw_url_model.joblib"

DEFAULT_URLS = [
    "https://www.bbc.co.uk/news/technology-68351312",
    "https://canvas.sunderland.ac.uk/courses/83909/assignments/232036",
    "http://192.168.4.11@secure-paypa1-login.tk/verify?acct=1",
    "http://https-paypal-secure-login.tk/account/verify",
]


def run_deployable(urls, use_network: bool):
    from phish_core import score, MODEL_NAME, FEATURES
    print(f"model: {MODEL_NAME} on {len(FEATURES)} features "
          f"({'WHOIS and DNS enabled' if use_network else 'offline, lexical only'})\n")
    for u in urls:
        r = score(u, use_network=use_network)
        print(f"{r['probability']*100:6.2f}%  {r['verdict']:<12} {u[:66]}")
        for x in r["reasons"][:4]:
            print(f"{'':>22}- {x}")
        if r["unavailable"]:
            print(f"{'':>22}could not check: {', '.join(r['unavailable'])}")


def run_artefact(urls):
    """Reproduce section 4.8: the high-scoring model that rejects real URLs."""
    import joblib, pandas as pd
    from url_features import extract_raw_url_features
    if not ARTEFACT_BUNDLE.is_file():
        sys.exit(f"artefact bundle not found at {ARTEFACT_BUNDLE}")
    b = joblib.load(ARTEFACT_BUNDLE)
    m, feats, thr = b["model"], b["features"], b["threshold"]
    print("ARTEFACT MODEL (dissertation section 4.8) — not fit for use.")
    print(f"model: {b.get('model_name','XGBoost')} on {len(feats)} raw lexical features, "
          f"threshold {thr}\n")
    for u in urls:
        f = extract_raw_url_features(u)
        p = float(m.predict_proba(pd.DataFrame([[f[c] for c in feats]], columns=feats))[0, 1])
        print(f"{p*100:6.2f}%  {'phishing' if p >= thr else 'legitimate':<12} {u[:66]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="URLs to score (defaults to a demo set)")
    ap.add_argument("--no-network", action="store_true", help="skip WHOIS and DNS lookups")
    ap.add_argument("--artefact", action="store_true",
                    help="load the rejected raw-URL model to reproduce section 4.8")
    a = ap.parse_args()
    urls = a.urls or DEFAULT_URLS
    if a.artefact:
        run_artefact(urls)
    else:
        run_deployable(urls, use_network=not a.no_network)
