# Artefact evidence — do not demonstrate

`streamlit_app_DO_NOT_DEMO.py` runs the **rejected** raw-URL model documented in
dissertation section 4.8. That model scores F1 0.9729 on its own locked test set
and classifies every real legitimate URL carrying a path as phishing at 100%
confidence, because its legitimate training class was generated as bare
`https://<domain>/` roots. It is kept only so the negative result stays
reproducible.

**The working system is in the parent directory:**

| Use | File | Command |
|---|---|---|
| Web app, three input modes | `app.py` | `./venv/bin/streamlit run app.py --server.port 8534` |
| Command line | `predict.py` | `./venv/bin/python predict.py <url>` |
| Email, IMAP polling | `email_monitor.py` | `./venv/bin/python email_monitor.py --demo` |
| Email, HTTP webhook | `email_webhook.py` | run with uvicorn; fails closed if the model is missing |

To reproduce section 4.8 without launching the broken app:

    ./venv/bin/python predict.py --artefact https://www.bbc.co.uk/news/technology-68351312
