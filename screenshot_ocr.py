"""Safe OCR helpers for the optional screenshot input mode.

The component reads only pixels in an uploaded image. It never follows text
recognised as a URL; callers pass recognised text through the same raw-URL
classifier used by the application and email modes.
"""
from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from PIL import Image
import pytesseract


def extract_text(image_source: bytes | BinaryIO | Image.Image) -> str:
    """OCR a screenshot, returning plain text with no network activity."""
    if isinstance(image_source, Image.Image):
        image = image_source
    elif isinstance(image_source, bytes):
        image = Image.open(BytesIO(image_source))
    else:
        image = Image.open(image_source)
    image = image.convert("RGB")
    return pytesseract.image_to_string(image)
