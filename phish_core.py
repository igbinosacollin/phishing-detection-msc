"""
Shared scoring for the Streamlit app, the email monitor and the CLI.

Model choice, recorded deliberately. Two candidate models exist in this project:

  * model_deployable13.joblib  -- XGBoost on the 13 UCI features obtainable from a
    URL without loading the page. Locked test F1 0.7805 (95% CI 0.7595-0.7994).
  * artifacts/raw_url/raw_url_model.joblib -- XGBoost on 26 raw lexical features,
    locked-test F1 0.9729.

The second scores far higher and is NOT used, because its training data builds
every legitimate example as "https://<domain>/" while every phishing example is a
real PhishTank URL with a path. The model therefore separates bare roots from
deep URLs rather than legitimate sites from phishing, and classifies real
legitimate URLs such as bbc.co.uk/news/technology-68351312 as phishing with
100% confidence. See section 4.8 of the dissertation.

On a hand-checked probe of 10 real URLs the deployable-13 model is correct on 9;
the raw-URL model is correct on 3 and flags every legitimate URL as phishing.
"""
from __future__ import annotations
import re, warnings
warnings.filterwarnings("ignore")
import pandas as pd, joblib

from url_features import (extract, extract_extra, whois_detail,
                          DERIVABLE, LEXICAL, HOST_BASED)

_BUNDLE = joblib.load("model_deployable13.joblib")
MODEL, FEATURES, MODEL_NAME = _BUNDLE["model"], _BUNDLE["features"], _BUNDLE["name"]

LABELS = {
    "having_ip_address": "uses a raw IP address instead of a domain name",
    "url_length": "unusually long address",
    "shortining_service": "uses a link shortener",
    "having_at_symbol": "contains an @ symbol, which hides the real destination",
    "double_slash_redirecting": "contains a redirect using //",
    "prefix_suffix": "hyphen in the domain name",
    "having_sub_domain": "unusual number of subdomains",
    "port": "non-standard port number",
    "https_token": "the word https appears inside the domain name",
    "age_of_domain": "domain registered recently",
    "domain_registration_length": "short registration period",
    "dnsrecord": "no DNS record found",
    "abnormal_url": "domain does not match its WHOIS record",
}

URL_RE = re.compile(r'(?:(?:https?|ftp)://|www\.)[^\s<>"\'\)\]]+', re.I)

# The verdict boundary. 0.5 is the default; section 4.6 of the dissertation shows
# that a lower value roughly halves missed phishing at the cost of more false
# alarms, which is the trade a deployed tool should probably make.
THRESHOLD = 0.5


def find_urls(text: str) -> list[str]:
    seen, out = set(), []
    for m in URL_RE.findall(text or ""):
        u = m.rstrip('.,;:!?)"\'')
        if u.lower() not in seen:
            seen.add(u.lower()); out.append(u)
    return out


def registration_flags(w: dict) -> list[str]:
    """Explicit warnings from the real registration dates.

    These are display-layer checks, deliberately separate from the model. The
    model's age_of_domain feature uses the UCI threshold of roughly six months
    because that is the encoding it was trained on; changing it would invalidate
    the model. A reader is better served by graded warnings at intervals that
    match how domains are actually used, so those are computed here instead.
    """
    out = []
    if not w.get("available"):
        return out
    age = w.get("age_days")
    if age is not None:
        if age < 30:
            out.append(f"domain registered {age} days ago, which is a strong warning sign")
        elif age < 90:
            out.append(f"domain registered {age} days ago, under three months old")
        elif age < 365:
            out.append(f"domain registered {age} days ago, less than a year old")
    exp = w.get("expires_in_days")
    if exp is not None:
        if exp < 0:
            out.append("registration has expired")
        elif exp < 365:
            out.append(f"registration expires in {exp} days, under a year of cover paid for")
    return out


def score(url: str, use_network: bool = True) -> dict:
    feats = extract(url, use_network=use_network)
    row = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
    p = float(MODEL.predict_proba(row)[0, 1])
    _w = (whois_detail(url) if use_network
          else {"available": False, "note": "offline mode, no lookup performed"})
    reasons = [LABELS[f] for f in FEATURES if feats[f] == -1 and f in LABELS]
    clean = [LABELS[f] for f in FEATURES if feats[f] == 1 and f in LABELS]
    return {
        "url": url, "probability": p, "threshold": THRESHOLD,
        "verdict": "Phishing" if p >= 0.5 else ("Suspicious" if p >= 0.3 else "Likely safe"),
        "reasons": reasons, "clean": clean, "features": feats,
        "unavailable": [f for f in HOST_BASED if feats[f] == 0],
        "extra": extract_extra(url),
        "whois": _w,
        "registration_flags": registration_flags(_w),
    }


def shap_contributions(url: str, use_network: bool = True, top: int = 6):
    import shap
    feats = extract(url, use_network=use_network)
    row = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
    clf = MODEL.named_steps["clf"] if hasattr(MODEL, "named_steps") else MODEL
    sv = shap.TreeExplainer(clf).shap_values(row)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
    s = pd.Series(sv[0], index=FEATURES).sort_values(key=abs, ascending=False)
    return [(f, LABELS.get(f, f), int(feats[f]), float(v)) for f, v in s.head(top).items()]
