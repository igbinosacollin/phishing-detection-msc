"""Tests for URL extraction and request authentication in the email webhook."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from email_webhook import build_reply, create_app, extract_urls  # noqa: E402


class EmailWebhookTests(unittest.TestCase):
    def test_extract_urls_deduplicates_and_never_needs_network(self) -> None:
        urls = extract_urls("Check https://example.org/a?x=1 and www.example.net.", "<a href='https://example.org/a?x=1'>again</a>")
        self.assertEqual(urls, ["https://example.org/a?x=1", "http://www.example.net"])

    def test_reply_states_that_no_links_were_opened(self) -> None:
        reply = build_reply([])
        self.assertIn("No links were opened", reply)

    def test_webhook_rejects_bad_secret_before_scoring(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # pragma: no cover - dependency installation issue
            self.skipTest(str(exc))
        app = create_app(model_path="missing.joblib", webhook_secret="correct")
        with TestClient(app) as client, patch("email_webhook.score") as mocked_score:
            response = client.post("/inbound/email", data={"text": "https://example.org"}, headers={"x-prom02-webhook-secret": "wrong"})
        self.assertEqual(response.status_code, 401)
        mocked_score.assert_not_called()

    def test_webhook_fails_closed_when_secret_or_model_is_not_ready(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # pragma: no cover - dependency installation issue
            self.skipTest(str(exc))
        app = create_app(model_path="missing.joblib", webhook_secret="")
        with TestClient(app) as client, patch("email_webhook.score") as mocked_score:
            health = client.get("/health")
            response = client.post("/inbound/email", data={"text": "https://example.org"})
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "not_ready")
        self.assertFalse(health.json()["model_ready"])
        self.assertFalse(health.json()["webhook_secret_configured"])
        self.assertEqual(response.status_code, 503)
        mocked_score.assert_not_called()

    def test_webhook_does_not_score_when_the_model_is_invalid(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # pragma: no cover - dependency installation issue
            self.skipTest(str(exc))
        app = create_app(model_path="missing.joblib", webhook_secret="correct")
        with TestClient(app) as client, patch("email_webhook.score") as mocked_score:
            response = client.post(
                "/inbound/email",
                data={"text": "https://example.org"},
                headers={"x-prom02-webhook-secret": "correct"},
            )
        self.assertEqual(response.status_code, 503)
        mocked_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
