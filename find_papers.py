"""Find and verify candidate papers via the Semantic Scholar API, with backoff."""
import json, sys, time, urllib.parse, urllib.request

TOPICS = [
 "phishing website detection machine learning comparison classifiers",
 "malicious URL detection lexical features survey",
 "adversarial examples evasion phishing detection classifiers",
 "concept drift machine learning security detection model ageing",
 "visual similarity phishing detection brand logo deep learning",
 "internationalized domain name homograph attack detection",
 "phishing detection explainability SHAP feature attribution",
 "dodging pitfalls evaluation machine learning computer security",
 "phishing email detection natural language processing BERT",
 "real time phishing detection browser extension latency",
 "phishing kits cloud hosting abuse legitimate services measurement",
 "class imbalance benchmark dataset construction intrusion phishing detection",
]
API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,venue,authors,abstract,tldr,citationCount,externalIds,openAccessPdf,publicationTypes"

BAD_VENUE = ("american journal", "international journal of research", "irjet",
             "ijraset", "journal of emerging", "turkish journal", "webology")

def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PROM02-litreview/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                w = 8 * (i + 1); print(f"  429, waiting {w}s", file=sys.stderr); time.sleep(w); continue
            print(f"  HTTP {e.code}", file=sys.stderr); return None
        except Exception as e:
            time.sleep(4); continue
    return None

def keep(p):
    if not p.get("title") or not p.get("year"): return False
    if p["year"] < 2017: return False
    v = (p.get("venue") or "").lower()
    if any(b in v for b in BAD_VENUE): return False
    cites = p.get("citationCount") or 0
    # established work, or recent work in a named venue
    return cites >= 25 or (cites >= 5 and v and "arxiv" not in v)

out = []
for t in TOPICS:
    print(f"searching: {t[:52]}", file=sys.stderr)
    d = get(f"{API}?{urllib.parse.urlencode({'query': t, 'limit': 8, 'fields': FIELDS})}")
    for p in (d or {}).get("data") or []:
        if not keep(p): continue
        out.append({
            "title": p["title"], "year": p["year"], "venue": p.get("venue"),
            "authors": [a["name"] for a in (p.get("authors") or [])][:9],
            "cites": p.get("citationCount"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "arxiv": (p.get("externalIds") or {}).get("ArXiv"),
            "oa": bool(p.get("openAccessPdf")),
            "tldr": (p.get("tldr") or {}).get("text"),
            "abstract": (p.get("abstract") or "")[:900],
        })
    time.sleep(5)

seen, uniq = set(), []
for p in out:
    k = p["title"].lower()
    if k not in seen: seen.add(k); uniq.append(p)
uniq.sort(key=lambda x: -(x["cites"] or 0))
json.dump(uniq, open("candidate_papers.json", "w"), indent=1)
print(f"\n{len(uniq)} candidates passed the quality gate\n")
for p in uniq[:26]:
    a = p["authors"]
    who = (a[0].split()[-1] + (" et al." if len(a) > 2 else "")) if a else "?"
    print(f"[{p['cites']:>5}] {p['year']} {who:<16} {p['title'][:66]}")
    print(f"        {(p['venue'] or '?')[:56]}")
