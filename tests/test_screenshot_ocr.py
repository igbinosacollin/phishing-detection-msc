"""End-to-end offline proof that screenshot OCR feeds URL text to the app safely."""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phish_core import find_urls  # noqa: E402
from screenshot_ocr import extract_text  # noqa: E402


@unittest.skipUnless(shutil.which("tesseract"), "Tesseract OCR executable is not installed")
class ScreenshotOcrTests(unittest.TestCase):
    def test_safe_screenshot_url_is_recognised_without_network_access(self) -> None:
        image = Image.new("RGB", (1500, 220), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 42)
        draw.text((30, 70), "Check https://example.org/account", fill="black", font=font)
        text = extract_text(image)
        urls = find_urls(text)
        self.assertIn("https://example.org/account", urls)


if __name__ == "__main__":
    unittest.main()
