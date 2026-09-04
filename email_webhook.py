"""Inbound-email URL scanning webhook for the PROM02 prototype.

It accepts SendGrid Inbound Parse-compatible form fields (``from``, ``subject``,
``text``, and ``html``), extracts URL-looking text, scores it locally, and returns
the reply body. Set SMTP_* variables and ENABLE_EMAIL_REPLY=true to send the same
reply to the original sender. This service never follows extracted links.
"""
from __future__ import annotations

import html
import os
import re
import secrets
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from phish_core import score as _score_url, MODEL_NAME  # honest deployable-13 model

DEFAULT_BUNDLE = "model_deployable13.joblib"


def load_bundle(path=None):
    """Validate that the configured model file exists before the webhook accepts work.

    The webhook must fail closed: replying "no phishing found" because the model
    could not be loaded would be worse than refusing to answer.
    """
    from pathlib import Path as _P
    p = _P(path or DEFAULT_BUNDLE)
    if not p.is_absolute():
        p = _P(__file__).resolve().parent / p
    if not p.is_file():
        raise FileNotFoundError(f"model bundle not found: {p}")
    return {"model_name": MODEL_NAME, "path": str(p)}


def score(url, _model_path=None):
    """Score one URL with the deployable model.

    Previously this called the raw-URL bundle, which classifies every real
    legitimate URL carrying a path as phishing (dissertation section 4.8).
    """
    r = _score_url(url)
    return {"url": url, "probability": r["probability"], "verdict": r["verdict"],
            "reasons": r["reasons"], "unavailable": r["unavailable"]}

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"()]+")
MAX_URLS_PER_MESSAGE = 20


def extract_urls(*parts: str | None) -> list[str]:
    """Extract unique HTTP(S)-looking text safely; this does not request a URL."""
    seen: set[str] = set()
    output: list[str] = []
    for part in parts:
        if not part:
            continue
        plain = html.unescape(re.sub(r"<[^>]+>", " ", part))
        for raw in URL_RE.findall(plain):
            candidate = raw.rstrip(".,;:!?)]}")
            normalised = candidate if candidate.lower().startswith(("http://", "https://")) else f"http://{candidate}"
            if not urlparse(normalised).hostname or normalised in seen:
                continue
            seen.add(normalised)
            output.append(normalised)
            if len(output) >= MAX_URLS_PER_MESSAGE:
                return output
    return output


def build_reply(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No HTTP(S) URLs were found in the submitted email. No links were opened."
    lines = ["PROM02 phishing URL scan (academic prototype)", "", "No extracted URL was opened or fetched."]
    for result in results:
        lines.append(f"- {result['verdict'].upper()}: {result['phishing_probability']:.1%} (threshold {result['threshold']:.2f}) — {result['url']}")
    lines.extend(["", "This is decision support, not a guarantee that a link is safe. If in doubt, do not click it and verify via an independently known official channel."])
    return "\n".join(lines)


def _send_smtp_reply(recipient: str, subject: str, body: str) -> None:
    """Send an optional reply only when explicitly enabled by deployment config."""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM")
    if not all((host, user, password, sender)):
        raise RuntimeError("SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD and SMTP_FROM are required for email replies")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = f"Re: {subject or 'URL security scan'}"
    message.set_content(body)
    with smtplib.SMTP_SSL(host, int(os.getenv("SMTP_PORT", "465")), timeout=20) as client:
        client.login(user, password)
        client.send_message(message)


def create_app(model_path: str | Path | None = None, webhook_secret: str | None = None) -> FastAPI:
    app = FastAPI(title="PROM02 email URL scanner", version="1.0")
    configured_model = Path(model_path or os.getenv("PROM02_MODEL_PATH", str(DEFAULT_BUNDLE)))
    required_secret = webhook_secret if webhook_secret is not None else os.getenv("WEBHOOK_SHARED_SECRET", "")
    try:
        # Validate once when the service starts.  A request must never be the
        # first point at which a missing, legacy, or smoke-only bundle is found.
        load_bundle(configured_model)
        model_error: str | None = None
    except Exception as exc:
        model_error = str(exc)

    @app.get("/health")
    def health() -> JSONResponse:
        ready = bool(required_secret) and model_error is None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "not_ready",
                "model_path": str(configured_model),
                "model_ready": model_error is None,
                "webhook_secret_configured": bool(required_secret),
                "email_reply_enabled": os.getenv("ENABLE_EMAIL_REPLY", "false").lower() == "true",
            },
        )

    @app.post("/inbound/email")
    async def inbound_email(request: Request, x_prom02_webhook_secret: str | None = Header(default=None)) -> dict[str, Any]:
        if not required_secret:
            raise HTTPException(status_code=503, detail="WEBHOOK_SHARED_SECRET is not configured")
        if not secrets.compare_digest(required_secret, x_prom02_webhook_secret or ""):
            raise HTTPException(status_code=401, detail="invalid webhook secret")
        if model_error is not None:
            raise HTTPException(status_code=503, detail="raw-URL model is not ready")
        form = await request.form()
        sender = str(form.get("from", ""))
        subject = str(form.get("subject", ""))
        urls = extract_urls(str(form.get("text", "")), str(form.get("html", "")))
        results = [score(url, configured_model) for url in urls]
        reply = build_reply(results)
        reply_sent = False
        if os.getenv("ENABLE_EMAIL_REPLY", "false").lower() == "true" and sender:
            _send_smtp_reply(sender, subject, reply)
            reply_sent = True
        return {"urls_found": len(urls), "results": results, "reply": reply, "reply_sent": reply_sent, "network_access": False}

    return app


app = create_app()
