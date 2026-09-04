"""Verify candidate papers against Crossref, the authoritative DOI registry."""
import json, sys, time, urllib.parse, urllib.request

CANDIDATES = [
 ("Dos and Don'ts of Machine Learning in Computer Security", "evaluation pitfalls in security ML"),
 ("TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time", "temporal bias"),
 ("Sunrise to Sunset: Analyzing the End-to-end Life Cycle and Effectiveness of Phishing Attacks at Scale", "phishing lifecycle"),
 ("PhishTime: Continuous Longitudinal Measurement of the Effectiveness of Anti-phishing Blacklists", "blacklist effectiveness"),
 ("Needle in a Haystack: Tracking Down Elite Phishing Domains in the Wild", "squatting domains"),
 ("Detecting and Characterizing Lateral Phishing at Scale", "email phishing at scale"),
 ("URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection", "deep URL representation"),
 ("Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages", "visual detection"),
 ("VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity", "visual detection"),
 ("CrawlPhish: Large-scale Analysis of Client-side Cloaking Techniques in Phishing", "client-side evasion"),
 ("Bypassing detection of URL-based phishing attacks using generative adversarial deep neural networks", "adversarial URLs"),
 ("Sok: a comprehensive reexamination of phishing research from the security perspective", "systematisation"),
 ("Detecting homograph attacks internationalized domain names", "homograph attacks"),
 ("Machine learning based phishing detection: a systematic literature review of feature engineering", "feature engineering"),
 ("Explainable machine learning for phishing detection", "explainability"),
]
API = "https://api.crossref.org/works"

def q(title):
    url = f"{API}?{urllib.parse.urlencode({'query.bibliographic': title, 'rows': 3, 'select': 'title,author,issued,container-title,DOI,type,is-referenced-by-count,event'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "PROM02-litreview/1.0 (mailto:bj06ec@student.sunderland.ac.uk)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["message"]["items"]
    except Exception as e:
        print(f"  err {e}", file=sys.stderr); return []

ok = []
for title, why in CANDIDATES:
    best = None
    for it in q(title):
        t = (it.get("title") or [""])[0].lower()
        if not t: continue
        if t[:40] == title.lower()[:40] or title.lower()[:30] in t:
            best = it; break
    if best:
        auth = best.get("author") or []
        names = [f"{a.get('family','')}, {(a.get('given','') or '')[:1]}." for a in auth[:9] if a.get("family")]
        yr = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
        venue = (best.get("container-title") or best.get("event", {}).get("name") or [""])
        venue = venue[0] if isinstance(venue, list) else venue
        ok.append({"why": why, "title": (best.get("title") or [""])[0], "year": yr,
                   "venue": venue, "authors": names, "doi": best.get("DOI"),
                   "cites": best.get("is-referenced-by-count")})
        print(f"OK   [{best.get('is-referenced-by-count',0):>5}] {yr} {(best.get('title') or [''])[0][:60]}", flush=True)
    else:
        print(f"MISS       -    -   {title[:60]}", flush=True)
    time.sleep(1)

json.dump(ok, open("verified_new_papers.json", "w"), indent=1)
print(f"\n{len(ok)} of {len(CANDIDATES)} verified against Crossref")
