"""Capture artefact screenshots for the dissertation.

Defaults to the deployed application. Pass a different address as the first
argument, or set PHISH_APP_URL, to capture a local instance instead.

Streamlit Community Cloud serves the host page and then embeds the app itself
in an iframe, so the outer document contains no application markup at all. The
iframe source is resolved first and the capture is taken from that address.
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
from urllib.parse import urljoin
import os
import sys
import time

OUT = Path("screenshots"); OUT.mkdir(exist_ok=True)
LIVE = "https://phishing-detection-msc-atyt6hkvckqthybexapfab.streamlit.app"
URL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHISH_APP_URL", LIVE)).rstrip("/")
PHISH = "http://192.168.4.11@secure-paypa1-login.tk/verify?acct=1"
EMAIL = """From: security@paypa1-alerts.tk
Subject: FW: Your account will be suspended

Dear customer, your account is on hold. Confirm here:
http://192.168.4.11@secure-paypa1-login.tk/verify?acct=1
Or use our short link https://bit.ly/3xTfake
Genuine site for reference: https://www.paypal.com/uk/signin
"""

# The hosted app cold-starts and does its WHOIS and DNS lookups over the
# network, so every wait here is longer than it needs to be locally.
T = 180_000


def idle(pg, secs=2):
    """Wait until Streamlit has stopped running the script."""
    try:
        pg.wait_for_selector('[data-testid="stStatusWidget"]', state="detached",
                             timeout=T)
    except Exception:
        pass
    time.sleep(secs)


def app_url(pg, url):
    """Return the address that actually serves the Streamlit application."""
    pg.goto(url, timeout=120_000)
    try:
        pg.wait_for_selector('iframe[title="streamlitApp"]', timeout=60_000)
        src = pg.get_attribute('iframe[title="streamlitApp"]', "src")
    except Exception:
        return url
    if not src:
        return url
    if src.startswith("//"):
        src = "https:" + src
    return urljoin(url + "/", src)


print(f"capturing from {URL}")
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1080, "height": 1500}, device_scale_factor=2)

    inner = app_url(pg, URL)
    if inner != URL:
        print(f"application served at {inner}")
        pg.goto(inner, timeout=120_000)
    pg.wait_for_selector("text=Phishing URL Detector", timeout=T)
    idle(pg, 3)

    # --- 1. URL tab with a phishing verdict ---------------------------------
    pg.get_by_placeholder("https://example.com/login").fill(PHISH)
    pg.get_by_role("button", name="Check").click()
    pg.wait_for_selector("text=Phishing", timeout=T)
    pg.wait_for_selector("text=How each feature moved", timeout=T)
    pg.wait_for_selector("text=SHAP values are in log-odds", timeout=T)
    pg.wait_for_selector('[data-testid="stDataFrame"]', timeout=T)
    idle(pg, 3)
    pg.screenshot(path=str(OUT / "app_url_phishing.png"), full_page=True)
    print("1/3 url tab")

    # --- 2. Forwarded email tab ---------------------------------------------
    pg.get_by_role("tab", name="Forwarded email").click(); time.sleep(2)
    pg.get_by_placeholder("Paste the full message including headers if you have them").fill(EMAIL)
    pg.get_by_role("button", name="Scan email").click()
    pg.wait_for_selector("text=link(s)", timeout=T)
    pg.wait_for_selector('[data-testid="stDataFrame"]', timeout=T)
    idle(pg, 3)
    pg.screenshot(path=str(OUT / "app_email_scan.png"), full_page=True)
    print("2/3 email tab")

    # --- 3. Screenshot tab ---------------------------------------------------
    pg.get_by_role("tab", name="Screenshot").click(); idle(pg, 3)
    pg.screenshot(path=str(OUT / "app_screenshot_tab.png"), full_page=True)
    print("3/3 screenshot tab")
    b.close()
print("done")
