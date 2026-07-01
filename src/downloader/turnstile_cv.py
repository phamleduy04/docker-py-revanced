"""Locate the Cloudflare Turnstile checkbox in a screenshot via OpenCV template matching.

The checkbox lives in a cross-origin iframe behind a closed shadow root, so DOM locators cannot reach it and an
iframe-relative click is flagged as a bot. This mirrors Kameleo's image-recognition approach
(https://github.com/kameleo-io/kameleo/tree/master/articles/click-the-cloudflare-turnstile) so the checkbox can be
located by pixels on the main frame. Cloudflare renders the same checkbox everywhere, so a single-stage match of the
checkbox template is enough here -- Kameleo's extra "whole widget frame" stage carried a site-specific background
that does not match APKMirror, whereas the checkbox template matches APKMirror at ~1.0 confidence.

NOTE: image recognition only finds *where* to click. It does not defeat fingerprint detection; the anti-detect
browser (CloakBrowser here) must still hide the automation, or Cloudflare re-issues the challenge after the click.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from loguru import logger

# Checkbox templates captured at a 1280px-wide viewport with device-scale-factor 1 (light and dark themes).
_ASSET_DIR = Path(__file__).parent / "assets" / "cloudflare_turnstile"
_CHECKBOX_TEMPLATES = ("captcha_box.png", "captcha_box_dark.png")
# Matching below this normalized correlation is treated as "not found" to avoid clicking random page pixels.
_MATCH_CONFIDENCE_THRESHOLD = 0.7


class ClickPoint(NamedTuple):
    """Full-frame pixel coordinates of the Turnstile checkbox center."""

    x: int
    y: int


def _decode(buffer: bytes) -> np.ndarray:
    """Decode a PNG/JPEG byte buffer into an OpenCV BGR image."""
    image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        msg = "Failed to decode image buffer for Turnstile template matching."
        raise ValueError(msg)
    return image


def locate_checkbox(full_screenshot: bytes) -> ClickPoint | None:
    """Locate the Turnstile checkbox center in a full-page screenshot, trying light then dark templates."""
    source = _decode(full_screenshot)

    best: tuple[float, int, int] | None = None
    for template_name in _CHECKBOX_TEMPLATES:
        template = _decode((_ASSET_DIR / template_name).read_bytes())
        # A template larger than the source can never match and makes matchTemplate raise, so skip it.
        if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
            continue
        result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < _MATCH_CONFIDENCE_THRESHOLD:
            continue
        template_h, template_w = template.shape[:2]
        center = (max_val, max_loc[0] + template_w // 2, max_loc[1] + template_h // 2)
        if best is None or center[0] > best[0]:
            best = center

    if best is None:
        logger.debug("Turnstile CV: no checkbox template matched above the confidence threshold.")
        return None

    confidence, center_x, center_y = best
    logger.info(f"Turnstile CV: located checkbox (confidence {confidence:.2f}) at ({center_x}, {center_y}).")
    return ClickPoint(x=center_x, y=center_y)
