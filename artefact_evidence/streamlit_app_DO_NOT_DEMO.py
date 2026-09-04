"""PROM02's local Streamlit interface for the frozen raw-URL model."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from predict import DEFAULT_BUNDLE, load_bundle, score


@st.cache_resource(show_spinner=False)
def _load(model_path: str):
    return load_bundle(model_path)


def main() -> None:
    st.set_page_config(page_title="PROM02 Phishing URL Checker", page_icon="🛡️", layout="centered")
    st.title("Phishing URL checker")
    st.caption("Academic prototype: it analyses URL text only. It does not open the website, resolve DNS, or guarantee safety.")
    model_path = Path(os.getenv("PROM02_MODEL_PATH", str(DEFAULT_BUNDLE)))
    try:
        bundle = _load(str(model_path))
    except Exception as exc:
        st.error(f"A trained raw-URL model is required: {exc}")
        st.code("python train_raw_url_model.py --input data/restricted/raw_url_input.csv --output-dir artifacts/raw_url")
        return

    url = st.text_input("Paste a URL", placeholder="https://example.com/account")
    if st.button("Check URL", type="primary", disabled=not bool(url.strip())):
        try:
            result = score(url, model_path)
        except Exception as exc:
            st.error(f"Could not analyse that value: {exc}")
            return
        probability = result["phishing_probability"]
        if result["verdict"] == "phishing":
            st.error(f"Potential phishing: {probability:.1%} model probability")
        else:
            st.success(f"Lower-risk result: {probability:.1%} model probability")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Model: {result['model']} • decision threshold: {result['threshold']:.2f} • network access: disabled")
        with st.expander("Feature summary"):
            st.json(result["features"])

    st.divider()
    st.caption(f"Feature set: {len(bundle['features'])} deterministic lexical features. Treat the output as decision support, not proof that a site is safe.")


if __name__ == "__main__":
    main()
