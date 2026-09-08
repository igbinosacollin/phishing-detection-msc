"""
Email forwarding mode (Objective O7).

The proposal specified a SendGrid inbound-parse webhook. That requires a domain
you own with MX records pointing at SendGrid, plus a publicly reachable endpoint.
This optional local alternative uses IMAP and SMTP: a user forwards a suspicious
message to a mailbox, the monitor reads it, extracts every URL, scores each one,
and replies with a per-URL verdict. The separate ``email_webhook.py`` implements
the proposal's SendGrid-compatible inbound-webhook route for domain deployment.

Credentials are read from the environment and are never stored in this file:

    export PHISH_IMAP_HOST=imap.gmail.com
    export PHISH_SMTP_HOST=smtp.gmail.com
    export PHISH_MAILBOX=your.address@gmail.com
    export PHISH_APP_PASSWORD=your-16-char-app-password

Use a Google app password on a throwaway account, never your main password.
Run once:      python email_monitor.py --once
Run polling:   python email_monitor.py --poll 60
Dry run:       python email_monitor.py --demo    (no network, no mailbox needed)
"""
from __future__ import annotations
import os, re, sys, time, email, imaplib, smtplib, argparse, logging
from email.header import decode_header, make_header
from email.message import EmailMessage

from phish_core import score, find_urls, MODEL_NAME

log = logging.getLogger("email_monitor")

IMAP_HOST = os.environ.get("PHISH_IMAP_HOST", "imap.gmail.com")
SMTP_HOST = os.environ.get("PHISH_SMTP_HOST", "smtp.gmail.com")
MAILBOX   = os.environ.get("PHISH_MAILBOX", "")
APP_PW    = os.environ.get("PHISH_APP_PASSWORD", "")


# Addresses and headers that must never receive an automatic reply. RFC 3834
# ("Recommendations for Automatic Responses to Electronic Mail") requires a
# responder to stay silent when the incoming message is itself automated, both to
# avoid mail loops and because the reply cannot be read by anyone.
NO_REPLY_LOCAL = ("no-reply", "noreply", "do-not-reply", "donotreply", "mailer-daemon",
                  "postmaster", "bounce", "bounces", "notification", "notifications",
                  "alerts", "automated")


def should_reply(msg: email.message.Message, sender: str) -> tuple[bool, str]:
    """Decide whether a reply is appropriate. Returns (reply, reason_if_not)."""
    if not sender or "@" not in sender:
        return False, "no usable sender address"

    local, _, domain = sender.lower().partition("@")
    if any(tok in local for tok in NO_REPLY_LOCAL):
        return False, f"sender is an unattended address ({sender})"

    # RFC 3834 and common precedence headers
    if (msg.get("Auto-Submitted") or "no").lower() != "no":
        return False, "message is Auto-Submitted"
    if (msg.get("Precedence") or "").lower() in ("bulk", "list", "junk", "auto_reply"):
        return False, "message carries a bulk or list Precedence header"
    for h in ("List-Id", "List-Unsubscribe", "X-Auto-Response-Suppress", "Auto-Response-Suppress"):
        if msg.get(h):
            return False, f"message carries {h}"

    # Never reply to ourselves; that is how loops start.
    if MAILBOX and sender.lower() == MAILBOX.lower():
        return False, "message is from this mailbox"
    return True, ""


def body_text(msg: email.message.Message, _depth: int = 0) -> str:
    """Flatten a message to searchable text.

    Three things this has to survive. HTML parts are kept whole so that link
    targets inside href and src attributes are preserved, since a "click here"
    button carries its destination in the attribute and not in the visible text.
    Forwarded mail is frequently attached as a nested message/rfc822 part rather
    than inlined, so nested messages are parsed recursively. And the subject line
    is included, because a URL sometimes appears only there.
    """
    if _depth > 6:
        return ""
    parts = []

    if _depth == 0:
        subj = msg.get("Subject", "")
        if subj:
            try:
                parts.append(str(make_header(decode_header(subj))))
            except Exception:
                parts.append(subj)

    def decode(part):
        raw = part.get_payload(decode=True)
        if raw is None:
            payload = part.get_payload()
            return payload if isinstance(payload, str) else ""
        for cs in filter(None, [part.get_content_charset(), "utf-8", "latin-1"]):
            try:
                return raw.decode(cs, errors="replace")
            except (LookupError, UnicodeDecodeError):
                continue
        return raw.decode("utf-8", errors="replace")

    for part in ([msg] if not msg.is_multipart() else msg.walk()):
        ctype = part.get_content_type()
        if ctype == "message/rfc822":
            for inner in part.get_payload():
                if isinstance(inner, email.message.Message):
                    parts.append(body_text(inner, _depth + 1))
            continue
        if part.get_content_maintype() == "multipart":
            continue
        if ctype in ("text/plain", "text/html"):
            try:
                parts.append(decode(part))
            except Exception as exc:
                log.warning("could not decode a %s part: %s", ctype, exc)

    text = "\n".join(p for p in parts if p)

    # Spam is frequently malformed on purpose: broken boundaries, wrong charsets,
    # declared encodings that do not match the payload. When structured parsing
    # yields nothing usable, fall back to scanning the raw source, which still
    # contains any href targets even if the MIME tree cannot be walked.
    if _depth == 0 and len(text.strip()) < 40:
        try:
            raw = msg.as_bytes().decode("utf-8", errors="replace")
        except Exception:
            raw = str(msg)
        decoded = _decode_embedded(raw)
        if decoded:
            log.info("structured parse returned %d chars; using raw-source fallback",
                     len(text.strip()))
            text = text + "\n" + decoded
    return text


def _decode_embedded(raw: str) -> str:
    """Pull readable text out of a raw message, decoding base64 blocks it contains."""
    import base64, binascii
    out = [raw]
    # long unbroken base64 runs are usually an encoded body part
    for block in re.findall(r"(?:[A-Za-z0-9+/=]{60,}\s*){2,}", raw):
        compact = re.sub(r"\s+", "", block)
        try:
            dec = base64.b64decode(compact + "=" * (-len(compact) % 4), validate=False)
            txt = dec.decode("utf-8", errors="replace")
            if txt.count("\ufffd") < len(txt) * 0.3:
                out.append(txt)
        except (binascii.Error, ValueError):
            continue
    return "\n".join(out)


def describe(msg: email.message.Message) -> str:
    """One-line summary of a message's MIME structure, for diagnosing misses."""
    bits = []
    for part in ([msg] if not msg.is_multipart() else msg.walk()):
        ct = part.get_content_type()
        if part.get_content_maintype() == "multipart":
            continue
        raw = part.get_payload(decode=True)
        bits.append(f"{ct}({len(raw) if raw else 0}b)")
    return ", ".join(bits) or "no parts"


def assess(text: str) -> tuple[list[dict], str]:
    """Score every URL in the text and compose the reply body."""
    urls = find_urls(text)
    if not urls:
        low = text.lower()
        if "mailto:" in low and "href" in low:
            return [], (
                "No web addresses were found in the message you forwarded.\n\n"
                "The message does contain links, but they are all mailto: addresses "
                "rather than web addresses, and this tool only assesses web links.\n\n"
                "If you forwarded this from a spam or junk folder, that is the likely "
                "cause. Mail providers commonly strip or disable web links in messages "
                "they have already classified as spam, so the copy that reaches this "
                "mailbox no longer contains them. Forwarding from the inbox, or "
                "forwarding as an attachment, usually preserves the original links.\n")
        if "href" in low or "<html" in low:
            return [], (
                "No web addresses were found in the message you forwarded.\n\n"
                "The message contains formatting but no web links this tool can read. "
                "If the message shows a button or an image you would normally click, "
                "its destination may have been removed by your mail provider, which is "
                "common for messages already marked as spam. It is also possible the "
                "content is a single image with the link drawn into the picture, in "
                "which case there is no text to extract.\n")
        return [], ("No web addresses were found in the message you forwarded, so there "
                    "was nothing to check.\n")

    results = [score(u) for u in urls]
    worst = max(results, key=lambda r: r["probability"])

    if worst["probability"] >= worst["threshold"]:
        head = (f"DO NOT TRUST THIS MESSAGE.\nAt least one link scores "
                f"{worst['probability']*100:.1f}% likely phishing.\n")
    elif worst["probability"] >= worst["threshold"] * 0.65:
        head = (f"TREAT WITH CAUTION.\nThe most suspicious link scores "
                f"{worst['probability']*100:.1f}%.\n")
    else:
        head = "No link in this message looks like phishing.\n"

    lines = [head, f"{len(results)} link(s) checked.\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['url']}")
        lines.append(f"   verdict: {r['verdict']} ({r['probability']*100:.1f}%)")
        if r["reasons"]:
            for x in r["reasons"]:
                lines.append(f"   - {x}")
        if r["unavailable"]:
            lines.append(f"   could not check: {', '.join(r['unavailable'])}")
        lines.append("")

    checked = sum(1 for r in results if not r["unavailable"])
    lines.append("How this works: each address was assessed from its own text plus the "
                 "public WHOIS and DNS records for its domain. The linked pages themselves "
                 "were never opened, because loading a suspected phishing page runs code "
                 "chosen by the attacker.")
    if checked < len(results):
        lines.append(f"Registry data was unavailable for "
                     f"{len(results) - checked} of {len(results)} link(s); where a check "
                     "could not be performed it is listed above, and the absence of a "
                     "warning is not evidence of safety.")
    lines.append(f"Model: {MODEL_NAME}, measured F1 0.78 under these conditions, so it is "
                 "wrong roughly one time in five. Treat it as decision support, not an "
                 "answer; a low score is not proof that a link is safe.")
    lines.append("Automated reply from a University of Sunderland MSc project. Do not "
                 "rely on it for real security decisions.")
    return results, "\n".join(lines)


def send_reply(to_addr: str, subject: str, body: str) -> None:
    if not (MAILBOX and APP_PW):
        raise RuntimeError("PHISH_MAILBOX and PHISH_APP_PASSWORD must be set")
    m = EmailMessage()
    m["From"], m["To"] = MAILBOX, to_addr
    m["Subject"] = f"Re: {subject}"[:120]
    m["Auto-Submitted"] = "auto-replied"
    m["X-Auto-Response-Suppress"] = "All"
    m.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, 465) as s:
        s.login(MAILBOX, APP_PW)
        s.send_message(m)
    log.info("replied to %s", to_addr)


def process_unread(send: bool = True) -> int:
    if not (MAILBOX and APP_PW):
        raise RuntimeError("PHISH_MAILBOX and PHISH_APP_PASSWORD must be set")
    n = 0
    with imaplib.IMAP4_SSL(IMAP_HOST) as im:
        im.login(MAILBOX, APP_PW)
        im.select("INBOX")
        _typ, data = im.search(None, "UNSEEN")
        for num in data[0].split():
            _typ, raw = im.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(raw[0][1])
            sender = email.utils.parseaddr(msg.get("From", ""))[1]
            subject = str(make_header(decode_header(msg.get("Subject", "(no subject)"))))
            ok, why = should_reply(msg, sender)
            if not ok:
                log.info("skipped: %s", why)
                im.store(num, "+FLAGS", "\\Seen")
                continue

            text = body_text(msg)
            results, reply = assess(text)
            log.info("from=%s links=%d chars=%d", sender, len(results), len(text))
            if not results:
                log.warning("no links found. structure: %s", describe(msg))
            if send:
                send_reply(sender, subject, reply)
            im.store(num, "+FLAGS", "\\Seen")
            n += 1
    return n


DEMO = """From: colleague@example.com
Subject: FW: Your account will be suspended

Please review urgently.

Dear customer, your account is on hold. Confirm here:
http://192.168.4.11@secure-paypa1-login.tk/verify?acct=1
Or use our short link https://bit.ly/3xTfake
Neutral reference URL: https://www.openai.com/
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process unread mail once")
    ap.add_argument("--poll", type=int, metavar="SECONDS", help="poll continuously")
    ap.add_argument("--demo", action="store_true", help="run on a built-in sample, no mailbox")
    ap.add_argument("--inspect-n", default="3", metavar="N",
                    help="how many recent messages to inspect (default 3)")
    ap.add_argument("--inspect", action="store_true",
                    help="print the MIME structure and extracted text of unread mail, send nothing")
    ap.add_argument("--no-send", action="store_true", help="read and print, do not reply")
    a = ap.parse_args()

    if a.demo or not (a.once or a.poll or a.inspect):
        _r, reply = assess(DEMO)
        print("=== reply that would be sent ===\n")
        print(reply)
        sys.exit(0)
    if a.inspect:
        # Structural diagnostics only. No sender, subject or body content is printed,
        # so the output can be shared without exposing the contents of the mailbox.
        import imaplib as _i
        with _i.IMAP4_SSL(IMAP_HOST) as im:
            im.login(MAILBOX, APP_PW); im.select("INBOX")
            _t, data = im.search(None, "ALL")
            ids = data[0].split()[-int(a.inspect_n):]
            print(f"inspecting the last {len(ids)} message(s), structure only\n")
            for k, num in enumerate(ids, 1):
                _t, raw = im.fetch(num, "(RFC822)")
                blob = raw[0][1]
                m = email.message_from_bytes(blob)
                txt = body_text(m)
                links = find_urls(txt)
                enc = {p.get("Content-Transfer-Encoding", "none")
                       for p in ([m] if not m.is_multipart() else m.walk())
                       if p.get_content_maintype() != "multipart"}
                print(f"[{k}] raw={len(blob)}b  parts=({describe(m)})")
                print(f"    encodings={sorted(e for e in enc if e)}")
                print(f"    extracted_text={len(txt)}chars  links_found={len(links)}")
                if links:
                    hosts = sorted({__import__('urllib.parse', fromlist=['x'])
                                    .urlparse(u if '://' in u else 'http://' + u).hostname or '?'
                                    for u in links})
                    print(f"    hostnames={hosts}")
                else:
                    low = txt.lower()
                    print(f"    decoded: href={low.count('href')}  http={low.count('http')}  "
                          f"src={low.count('src=')}  mailto={low.count('mailto')}")
                    schemes = sorted(set(re.findall(
                        r"href=[\"']?([a-zA-Z][a-zA-Z0-9+.-]{0,12}):", txt)))
                    print(f"    href schemes: {schemes[:8] or 'none (relative links only)'}")
                    raw_low = blob.lower()
                    print(f"    raw: href={raw_low.count(b'href')}  http={raw_low.count(b'http')}")
                print()
    elif a.once:
        print(f"processed {process_unread(send=not a.no_send)} message(s)")
    elif a.poll:
        while True:
            try:
                process_unread(send=not a.no_send)
            except Exception as e:
                log.error("%s", e)
            time.sleep(a.poll)
