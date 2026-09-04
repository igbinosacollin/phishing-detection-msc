"""
url_features.py — Objective O3: raw URL -> feature vector, without visiting the site.

Design constraint (from the project's ethics position): URLs are processed as TEXT
only. The page is never fetched. WHOIS and DNS are metadata lookups about the
domain, not requests to the suspect web server, so they are permitted.

Consequence: only 13 of the UCI 30 features are derivable. The other 17 need page
HTML or dead third-party services (Alexa, Google PageRank). They are returned as 0
("unknown/neutral" in the UCI {-1,0,1} coding) and listed in NOT_DERIVABLE so the
gap is explicit rather than hidden.
"""
from __future__ import annotations
import os, re, math, socket, ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse

import tldextract

# Keep extraction deterministic and network-free.  The bundled public-suffix
# snapshot is sufficient for feature parsing; refreshing it at runtime would
# make an offline/security-sensitive classifier depend on external state.
_TLD = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())

# --- UCI 30-feature schema, in the order the trained model expects -------------
UCI_FEATURES = [
    "having_ip_address", "url_length", "shortining_service", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix", "having_sub_domain",
    "sslfinal_state", "domain_registration_length", "favicon", "port",
    "https_token", "request_url", "url_of_anchor", "links_in_tags", "sfh",
    "submitting_to_email", "abnormal_url", "redirect", "on_mouseover",
    "rightclick", "popupwindow", "iframe", "age_of_domain", "dnsrecord",
    "web_traffic", "page_rank", "google_index", "links_pointing_to_page",
    "statistical_report",
]

# Derivable from the URL string alone (no network at all)
LEXICAL = [
    "having_ip_address", "url_length", "shortining_service", "having_at_symbol",
    "double_slash_redirecting", "prefix_suffix", "having_sub_domain", "port",
    "https_token",
]
# Derivable with WHOIS/DNS metadata lookups (still never contacts the web server)
HOST_BASED = ["domain_registration_length", "abnormal_url", "age_of_domain", "dnsrecord"]

DERIVABLE = LEXICAL + HOST_BASED
NOT_DERIVABLE = [f for f in UCI_FEATURES if f not in DERIVABLE]

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorte.st", "rb.gy",
    "tiny.cc", "lnkd.in", "trib.al", "shorturl.at", "s.id", "t.ly", "urlz.fr",
}


def _norm(url: str) -> str:
    url = url.strip()
    return url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url


def _is_ip(host: str) -> bool:
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        # hex-encoded IPs, e.g. 0x2E.0xA1...
        return bool(re.fullmatch(r"(0x[0-9a-fA-F]{1,2}\.){3}0x[0-9a-fA-F]{1,2}", host))


# --- WHOIS / DNS, each failure-tolerant ---------------------------------------
# WHOIS speaks on TCP port 43, which many hosting providers block outbound. Without
# a guard every lookup would stall until its timeout and the application would feel
# broken rather than degraded. After a few consecutive failures the lookups are
# abandoned for the life of the process and the four host-based features report as
# unavailable, which the interface already surfaces to the user.
_whois_cache: dict[str, object] = {}
_whois_failures = 0
_WHOIS_GIVE_UP_AFTER = int(os.environ.get("PHISH_WHOIS_GIVE_UP_AFTER", "3"))
_WHOIS_DISABLED = os.environ.get("PHISH_DISABLE_WHOIS", "").lower() in ("1", "true", "yes")


def whois_available() -> bool:
    """False once lookups have been abandoned, so callers can say why."""
    return not _WHOIS_DISABLED and _whois_failures < _WHOIS_GIVE_UP_AFTER


def _whois(domain: str, timeout: float = 5.0):
    global _whois_failures
    if domain in _whois_cache:
        return _whois_cache[domain]
    if not whois_available():
        return None
    rec = None
    try:
        import whois
        socket.setdefaulttimeout(timeout)
        rec = whois.whois(domain)
        if rec is not None:
            _whois_failures = 0
    except Exception:
        _whois_failures += 1
        if _whois_failures >= _WHOIS_GIVE_UP_AFTER:
            import logging
            logging.getLogger(__name__).warning(
                "WHOIS unreachable after %d attempts; host-based features will report as "
                "unavailable for the rest of this session", _whois_failures)
        rec = None
    _whois_cache[domain] = rec
    return rec


def _first_date(v):
    if isinstance(v, list):
        v = next((d for d in v if isinstance(d, datetime)), None)
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    return None


def _has_dns(domain: str, timeout: float = 4.0) -> bool:
    try:
        import dns.resolver
        r = dns.resolver.Resolver()
        r.lifetime = r.timeout = timeout
        r.resolve(domain, "A")
        return True
    except Exception:
        try:
            socket.setdefaulttimeout(timeout)
            socket.gethostbyname(domain)
            return True
        except Exception:
            return False


# --- main extractor -----------------------------------------------------------
def extract(url: str, use_network: bool = True) -> dict:
    """Return all 30 UCI columns. Non-derivable ones are 0 (unknown).

    UCI coding: -1 = phishing-like, 0 = suspicious/unknown, 1 = legitimate-like.
    """
    raw = _norm(url)
    p = urlparse(raw)
    host = (p.hostname or "").lower()
    ext = _TLD(raw)
    domain = ".".join([x for x in (ext.domain, ext.suffix) if x])

    f = {k: 0 for k in UCI_FEATURES}

    # 1 having_ip_address
    f["having_ip_address"] = -1 if _is_ip(host) else 1
    # 2 url_length: <54 legit, 54-75 suspicious, >75 phishing
    n = len(raw)
    f["url_length"] = 1 if n < 54 else (0 if n <= 75 else -1)
    # 3 shortining_service
    f["shortining_service"] = -1 if domain in SHORTENERS else 1
    # 4 having_at_symbol
    f["having_at_symbol"] = -1 if "@" in raw else 1
    # 5 double_slash_redirecting: '//' occurring after the scheme
    f["double_slash_redirecting"] = -1 if raw.find("//", 8) > 0 else 1
    # 6 prefix_suffix: hyphen in the registered domain
    f["prefix_suffix"] = -1 if "-" in (ext.domain or "") else 1
    # 7 having_sub_domain: count dots in subdomain part
    subs = [s for s in (ext.subdomain or "").split(".") if s and s != "www"]
    f["having_sub_domain"] = 1 if len(subs) == 0 else (0 if len(subs) == 1 else -1)
    # 11 port: explicit non-standard port
    f["port"] = -1 if (p.port and p.port not in (80, 443)) else 1
    # 12 https_token: 'https' appearing inside the host name (spoof trick)
    f["https_token"] = -1 if "https" in host.replace("https://", "") else 1

    if use_network and domain:
        # 25 dnsrecord
        alive = _has_dns(domain)
        f["dnsrecord"] = 1 if alive else -1
        rec = _whois(domain)
        created = _first_date(getattr(rec, "creation_date", None)) if rec else None
        expires = _first_date(getattr(rec, "expiration_date", None)) if rec else None
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        # 24 age_of_domain: >= 6 months legit
        if created:
            f["age_of_domain"] = 1 if (now - created).days >= 182 else -1
        # 9 domain_registration_length: expires > 1 year out
        if expires:
            f["domain_registration_length"] = 1 if (expires - now).days > 365 else -1
        # 18 abnormal_url: hostname absent from the WHOIS record
        if rec is not None:
            wd = getattr(rec, "domain_name", None)
            wd = (wd[0] if isinstance(wd, list) and wd else wd) or ""
            f["abnormal_url"] = 1 if str(wd).lower().startswith(ext.domain or "\0") else -1
    return f


def whois_detail(url: str, timeout: float = 6.0) -> dict:
    """Actual registration dates for display, not for the model.

    The model consumes age_of_domain and domain_registration_length as {-1, 0, 1}
    flags because that is the UCI encoding it was trained on. A person reading a
    verdict is far better served by "registered 4 days ago" than by a flag, so the
    real values are returned separately.
    """
    raw = _norm(url)
    ext = tldextract.extract(raw)
    domain = ".".join([x for x in (ext.domain, ext.suffix) if x])
    out = {"domain": domain, "created": None, "expires": None, "registrar": None,
           "age_days": None, "expires_in_days": None, "available": False,
           "note": "no WHOIS record returned"}
    if not domain:
        out["note"] = "could not parse a domain from that address"
        return out
    rec = _whois(domain, timeout=timeout)
    if rec is None:
        return out
    created = _first_date(getattr(rec, "creation_date", None))
    expires = _first_date(getattr(rec, "expiration_date", None))
    reg = getattr(rec, "registrar", None)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out["registrar"] = (reg[0] if isinstance(reg, list) and reg else reg) or None
    if created:
        out["created"] = created.date().isoformat()
        out["age_days"] = (now - created).days
    if expires:
        out["expires"] = expires.date().isoformat()
        out["expires_in_days"] = (expires - now).days
    if created or expires:
        out["available"] = True
        out["note"] = ""
    else:
        out["note"] = "registry returned a record but withheld the dates"
    return out


# --- extra lexical features (O3 asks for >=15 of our own) ---------------------
EXTRA_FEATURES = [
    "n_dots", "n_hyphens", "n_digits", "digit_ratio", "n_special",
    "host_length", "path_depth", "has_query", "host_entropy",
]

# The model-facing schema deliberately contains only deterministic lexical
# features. Training and production can therefore use exactly the same
# extractor without an internet connection, time-varying DNS answers, or a
# request to the URL being classified. The older UCI constants above are kept
# only for the historic benchmark scripts.
RAW_URL_FEATURES = [
    "url_length", "host_length", "path_length", "query_length", "fragment_length",
    "n_dots", "n_hyphens", "n_underscores", "n_digits", "digit_ratio",
    "n_special", "path_depth", "n_subdomains", "n_query_params", "has_https",
    "has_ip_host", "has_at_symbol", "has_punycode", "has_nonstandard_port",
    "has_shortener", "has_double_slash_path", "has_percent_encoding",
    "has_login_token", "has_brand_token", "host_entropy", "url_entropy",
]

_LOGIN_WORDS = ("login", "signin", "sign-in", "verify", "verification", "secure", "account", "update", "auth")
_BRAND_WORDS = ("paypal", "microsoft", "office", "apple", "amazon", "google", "facebook", "netflix", "bank")


def _entropy(value: str) -> float:
    """Shannon entropy for a string, with a stable zero for empty values."""
    if not value:
        return 0.0
    return -sum((value.count(char) / len(value)) * math.log2(value.count(char) / len(value)) for char in set(value))


def _safe_port(parsed) -> int | None:
    """Return a parsed port without allowing malformed input to abort scoring."""
    try:
        return parsed.port
    except ValueError:
        return None


def extract_raw_url_features(url: str) -> dict[str, float | int]:
    """Extract the production model's deterministic raw-URL features.

    This function never resolves DNS, performs WHOIS, opens a socket, or fetches
    the supplied URL. It is the only feature function used by
    ``train_raw_url_model.py``, ``predict.py``, the Streamlit interface, and the
    inbound-email webhook.
    """
    raw = _norm(url)
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    ext = _TLD(raw)
    port = _safe_port(parsed)
    path = parsed.path or ""
    query = parsed.query or ""
    lower = raw.lower()
    digits = sum(char.isdigit() for char in raw)
    subdomains = [part for part in (ext.subdomain or "").split(".") if part and part != "www"]
    query_params = [part for part in query.split("&") if part]
    nonstandard_port = int(port is not None and port not in (80, 443))

    features: dict[str, float | int] = {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(parsed.fragment or ""),
        "n_dots": raw.count("."),
        "n_hyphens": raw.count("-"),
        "n_underscores": raw.count("_"),
        "n_digits": digits,
        "digit_ratio": round(digits / max(len(raw), 1), 6),
        "n_special": sum(raw.count(char) for char in "@?%=&_~"),
        "path_depth": len([part for part in path.split("/") if part]),
        "n_subdomains": len(subdomains),
        "n_query_params": len(query_params),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip_host": int(_is_ip(host)),
        "has_at_symbol": int("@" in raw),
        "has_punycode": int("xn--" in host),
        "has_nonstandard_port": nonstandard_port,
        "has_shortener": int(".".join(part for part in (ext.domain, ext.suffix) if part) in SHORTENERS),
        "has_double_slash_path": int(raw.find("//", len(parsed.scheme) + 3) >= 0),
        "has_percent_encoding": int("%" in raw),
        "has_login_token": int(any(word in lower for word in _LOGIN_WORDS)),
        "has_brand_token": int(any(word in lower for word in _BRAND_WORDS)),
        "host_entropy": round(_entropy(host), 6),
        "url_entropy": round(_entropy(raw), 6),
    }
    # Failing loudly here prevents accidental train/inference schema drift.
    if list(features) != RAW_URL_FEATURES:
        raise AssertionError("raw URL feature schema changed without updating RAW_URL_FEATURES")
    return features


def extract_extra(url: str) -> dict:
    raw = _norm(url)
    p = urlparse(raw)
    host = (p.hostname or "").lower()
    path = p.path or ""
    digits = sum(c.isdigit() for c in raw)

    def entropy(s: str) -> float:
        if not s:
            return 0.0
        return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))

    return {
        "n_dots": raw.count("."),
        "n_hyphens": raw.count("-"),
        "n_digits": digits,
        "digit_ratio": round(digits / max(len(raw), 1), 4),
        "n_special": sum(raw.count(c) for c in "@?%=&_~"),
        "host_length": len(host),
        "path_depth": len([s for s in path.split("/") if s]),
        "has_query": int(bool(p.query)),
        "host_entropy": round(entropy(host), 4),
    }


def feature_report() -> str:
    return (f"{len(DERIVABLE)}/30 UCI features derivable without visiting the page "
            f"({len(LEXICAL)} lexical, {len(HOST_BASED)} host-based via WHOIS/DNS); "
            f"{len(NOT_DERIVABLE)} require page HTML or dead third-party services. "
            f"Plus {len(EXTRA_FEATURES)} additional lexical features -> "
            f"{len(DERIVABLE) + len(EXTRA_FEATURES)} own features total (O3 requires >=15).")


if __name__ == "__main__":
    import sys, json
    print(feature_report())
    for u in (sys.argv[1:] or ["https://www.bbc.co.uk/news",
                               "http://192.168.4.11@secure-paypa1-login.tk/verify?acct=1"]):
        print(f"\n{u}")
        d = extract(u); d.update(extract_extra(u))
        print(json.dumps({k: v for k, v in d.items() if k in DERIVABLE or k in EXTRA_FEATURES}))
