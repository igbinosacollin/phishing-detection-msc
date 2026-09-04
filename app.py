"""PROM02 Streamlit application backed by the shared raw-URL model."""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd

from phish_core import score, shap_contributions, find_urls, MODEL_NAME, FEATURES
from screenshot_ocr import extract_text as extract_screenshot_text

st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣", layout="centered")

st.title("Phishing URL Detector")
st.caption(f"{MODEL_NAME} trained on {len(FEATURES)} features obtainable without loading "
           "the page: 9 read from the address text, 4 from WHOIS and DNS records about the "
           "domain. The suspect page itself is never opened. PROM02 MSc Cybersecurity, "
           "Igbinosa Collins Osakponmwen.")

with st.expander("What this tool can and cannot do"):
    st.markdown(
        "This model scores a web address **without visiting it**, because loading a "
        "suspected phishing page can run attacker-controlled code and confirm that the "
        "address is live. It is trained and evaluated with the same network-free feature "
        "extractor used here.\n\n"
        "A low-risk result is not proof that a link is safe. Treat every verdict as "
        "decision support and verify important requests through an independently known "
        "official channel.")


def verdict_block(r):
    p = r["probability"] * 100
    if r["verdict"] == "Phishing":
        st.error(f"### {r['verdict']} — {p:.1f}%")
    elif r["verdict"] == "Suspicious":
        st.warning(f"### {r['verdict']} — {p:.1f}%")
    else:
        st.success(f"### {r['verdict']} — {p:.1f}% risk")
    st.progress(min(r["probability"], 1.0))

    if r["reasons"]:
        label = ("**Why:**" if r["verdict"] != "Likely safe"
                 else "**Risk indicators present, but not enough to flag it:**")
        st.markdown(label)
        for x in r["reasons"]:
            st.markdown(f"- {x}")
    else:
        st.markdown("**No risk indicators found in the address.**")

    w = r.get("whois", {})
    if w.get("available"):
        cols = st.columns(3)
        cols[0].metric("Domain age",
                       f"{w['age_days']} days" if w.get("age_days") is not None else "unknown")
        cols[1].metric("Registered", w.get("created") or "unknown")
        cols[2].metric("Expires", w.get("expires") or "unknown")
        if w.get("registrar"):
            st.caption(f"Registrar: {w['registrar']}")
    elif w.get("note"):
        st.caption(f"Registration dates: {w['note']}.")

    for x in r.get("registration_flags", []):
        st.warning(x)

    if r["unavailable"]:
        st.caption("Could not check: " + ", ".join(r["unavailable"]) +
                   ". WHOIS is often rate limited or withheld by the registry, so absence "
                   "of a warning here is not evidence of safety.")
    st.caption(f"Verdict boundary: {r['threshold']:.2f}. The page was not opened; WHOIS and "
               "DNS records about the domain were queried.")


tab_url, tab_email, tab_shot = st.tabs(["Paste a URL", "Forwarded email", "Screenshot"])

# ---------------------------------------------------------------- URL mode ---
with tab_url:
    url = st.text_input("Web address", placeholder="https://example.com/login")
    explain = st.checkbox("Show SHAP explanation", value=True)
    if st.button("Check", type="primary", key="b_url") and url.strip():
        with st.spinner("Analysing URL text without opening the link..."):
            r = score(url.strip())
        verdict_block(r)
        if explain:
            st.markdown("**How each feature moved the decision**")
            rows = [{"Feature": lab, "Value": v,
                     "Effect": ("toward phishing" if s > 0 else "toward legitimate"),
                     "SHAP": round(s, 3)}
                    for _f, lab, v, s in shap_contributions(url.strip())]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("SHAP values are in log-odds. Positive pushes toward phishing.")

# -------------------------------------------------------------- email mode ---
with tab_email:
    st.markdown("Paste a forwarded email. Every link in it is extracted and scored "
                "separately, and the message is judged by its worst link.")
    body = st.text_area("Email text", height=200,
                        placeholder="Paste the full message including headers if you have them")
    if st.button("Scan email", type="primary", key="b_mail") and body.strip():
        urls = find_urls(body)
        if not urls:
            st.info("No links found in that message.")
        else:
            st.write(f"Found **{len(urls)}** link(s).")
            results = [score(u) for u in urls]
            worst = max(results, key=lambda r: r["probability"])
            if worst["probability"] >= worst["threshold"]:
                st.error(f"### Do not trust this email — worst link scores "
                         f"{worst['probability']*100:.1f}%")
            elif worst["probability"] >= worst["threshold"] * 0.65:
                st.warning(f"### Treat with caution — worst link scores "
                           f"{worst['probability']*100:.1f}%")
            else:
                st.success("### No link in this message looks like phishing")
            st.dataframe(pd.DataFrame([
                {"Link": r["url"][:70], "Risk %": round(r["probability"] * 100, 1),
                 "Verdict": r["verdict"], "Reasons": "; ".join(r["reasons"][:2]) or "none"}
                for r in results]), hide_index=True, use_container_width=True)

# --------------------------------------------------------- screenshot mode ---
with tab_shot:
    st.markdown("Upload a screenshot of a suspicious message. Text is read out of the "
                "image and any addresses found are scored.")
    st.caption("A screenshot only shows the link text a reader can see. The real "
               "destination of a hyperlink lives in the message source and cannot be "
               "recovered from a picture, so this mode is weaker than forwarding the "
               "email itself.")
    img = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    if img is not None:
        st.image(img, use_container_width=True)
        try:
            text = extract_screenshot_text(img)
        except Exception as e:
            st.error(f"Text recognition unavailable: {e}")
            text = ""
        if text:
            with st.expander("Text read from the image"):
                st.text(text[:2000])
            urls = find_urls(text)
            if not urls:
                st.info("No web addresses were readable in that image.")
            else:
                results = [score(u) for u in urls]
                st.dataframe(pd.DataFrame([
                    {"Link": r["url"][:70], "Risk %": round(r["probability"] * 100, 1),
                     "Verdict": r["verdict"]} for r in results]),
                    hide_index=True, use_container_width=True)
