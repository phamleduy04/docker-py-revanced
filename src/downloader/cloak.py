"""Shared CloakBrowser helpers for sites that gate downloads behind Cloudflare Turnstile.

APKMirror and Uptodown both need a real browser to clear Turnstile, so the launch options and the
"Verify you are human" checkbox click live here instead of being duplicated per downloader.
"""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from src.utils import request_timeout, resource_folder, slugify

# CloakBrowser logs through the standard library while this project logs through loguru, so without a bridge its
# first-run Chromium download (~200 MB) produces no output at all and an unattended build looks frozen.
CLOAK_LOGGER_NAME = "cloakbrowser"

# Screenshots land in the mounted resource folder so they survive the container and are reachable for debugging.
CLOAK_DEBUG_SCREENSHOT_DIR = Path(resource_folder) / "debug-screenshots"
# CloakBrowser runs inside the Docker container as root, so Chromium needs container-safe launch flags.
CLOAK_BROWSER_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]
# Playwright expects milliseconds while the rest of the downloader config stores request timeouts in seconds.
CLOAK_REQUEST_TIMEOUT_MS = request_timeout * 1000
# Cloudflare's interactive "Verify you are human" checkbox lives in a cross-origin iframe that needs a short wait.
CLOAK_CHALLENGE_CLICK_TIMEOUT_MS = 10_000
# After clicking the checkbox, Cloudflare validates and redirects; give it room before declaring the challenge unsolved.
CLOAK_CHALLENGE_SETTLE_TIMEOUT_MS = 20_000
# Cloudflare Turnstile renders its checkbox inside these iframes; we click via the iframe's on-page bounding box.
CLOAK_CHALLENGE_FRAME_SELECTOR = "iframe[src*='challenges.cloudflare.com'], iframe[src*='turnstile']"
# Selectors that may hold the on-page checkbox: the full-page interstitial exposes the challenge iframe, while
# embedded widgets also expose the `.cf-turnstile` div. The managed page renders BOTH a hidden 0x0 orchestration
# iframe and the visible widget iframe, so candidates are filtered by size rather than trusting the first match.
CLOAK_CHALLENGE_WIDGET_SELECTORS = (CLOAK_CHALLENGE_FRAME_SELECTOR, ".cf-turnstile")
# Minimum clickable footprint (CSS px) that separates the real Turnstile widget from the hidden orchestration iframe.
CLOAK_CHALLENGE_MIN_WIDGET_WIDTH = 50
CLOAK_CHALLENGE_MIN_WIDGET_HEIGHT = 30
# The checkbox sits near the left edge of the widget; offset inward so the click lands on it, not the border.
CLOAK_CHALLENGE_CHECKBOX_X_OFFSET = 30
# Move the pointer in several steps instead of teleporting so the cursor path resembles a human before clicking.
CLOAK_CHALLENGE_MOUSE_STEPS = 12


class _LoguruBridge(logging.Handler):
    """Forward CloakBrowser's standard-library log records into loguru so container logs show its progress."""

    def emit(self: "_LoguruBridge", record: logging.LogRecord) -> None:
        """Re-emit one record, falling back to INFO for levels loguru does not define."""
        level = record.levelname if record.levelname in logger._core.levels else "INFO"  # noqa: SLF001
        logger.log(level, record.getMessage())


def forward_browser_logs() -> None:
    """Route CloakBrowser's logger into loguru so the first-run binary download reports progress instead of stalling."""
    browser_logger = logging.getLogger(CLOAK_LOGGER_NAME)
    if any(isinstance(handler, _LoguruBridge) for handler in browser_logger.handlers):
        return
    browser_logger.addHandler(_LoguruBridge())
    browser_logger.setLevel(logging.INFO)
    # Records are already delivered to loguru, so propagation would only duplicate them on the root logger.
    browser_logger.propagate = False


def load_browser_dependencies() -> tuple[Any, Any]:
    """Import CloakBrowser lazily so flows that never hit a challenged site skip the browser import cost.

    :raises ImportError: When CloakBrowser or Playwright is unavailable; callers wrap this in their own error.
    """
    from cloakbrowser import launch  # noqa: PLC0415
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: PLC0415

    forward_browser_logs()
    return launch, PlaywrightTimeoutError


def save_debug_screenshot(page: Any, url: str) -> Path | None:
    """Best-effort full-page screenshot so a persisting Cloudflare challenge can be inspected after the fact."""
    try:
        CLOAK_DEBUG_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        screenshot_path = CLOAK_DEBUG_SCREENSHOT_DIR / f"{slugify(url)}-{uuid4().hex[:8]}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception as exc:  # noqa: BLE001
        # A screenshot failure must never mask the original Cloudflare/download error.
        logger.warning(f"Failed to save CloakBrowser debug screenshot for {url}: {exc}")
        return None
    logger.info(f"Saved CloakBrowser debug screenshot to {screenshot_path}")
    return screenshot_path


def locate_challenge_widget(page: Any) -> Any:
    """Return the on-page Turnstile element whose bounding box positions the "Verify you are human" checkbox.

    The managed challenge renders a hidden 0x0 orchestration iframe alongside the visible widget iframe, so
    candidates are filtered by a minimum clickable footprint instead of trusting the first DOM match.
    """
    for selector in CLOAK_CHALLENGE_WIDGET_SELECTORS:
        try:
            candidates = page.query_selector_all(selector)
        except Exception as exc:  # noqa: BLE001
            # A selector the page cannot evaluate is not fatal; the remaining selectors may still find the widget.
            logger.debug(f"Turnstile selector '{selector}' could not be queried: {exc}")
            continue
        for element in candidates:
            box = element.bounding_box()
            if (
                box
                and box["width"] >= CLOAK_CHALLENGE_MIN_WIDGET_WIDTH
                and box["height"] >= CLOAK_CHALLENGE_MIN_WIDGET_HEIGHT
            ):
                logger.debug(f"Selected Turnstile widget via '{selector}' with box {box}.")
                return element
    return None


def locate_checkbox_via_cv(page: Any) -> tuple[float, float] | None:
    """Locate the checkbox by pixels via OpenCV template matching against a full-page screenshot."""
    try:
        # Import lazily so downloaders that never see a challenge never pay the OpenCV/numpy import cost.
        from src.downloader.turnstile_cv import locate_checkbox  # noqa: PLC0415

        point = locate_checkbox(page.screenshot(full_page=True))
    except Exception as exc:  # noqa: BLE001
        # CV is a best-effort locator; any failure falls back to DOM geometry rather than aborting the click.
        logger.debug(f"Turnstile CV locate failed: {exc}")
        return None
    if point is None:
        return None
    return float(point.x), float(point.y)


def locate_checkbox_via_dom(page: Any) -> tuple[float, float] | None:
    """Locate the checkbox from the widget iframe's on-page bounding box as a fallback to CV."""
    widget = locate_challenge_widget(page)
    if widget is None:
        return None
    box = widget.bounding_box()
    if not box:
        return None
    # The checkbox sits at the left of the widget, vertically centered.
    return box["x"] + CLOAK_CHALLENGE_CHECKBOX_X_OFFSET, box["y"] + box["height"] / 2


def attempt_challenge_click(page: Any, url: str, playwright_timeout_error: Any) -> None:
    """Click Cloudflare's "Verify you are human" checkbox using real main-frame mouse coordinates.

    The checkbox lives in a cross-origin iframe behind a closed shadow root, so Playwright locators cannot reach
    it, and `frame_locator().click()` dispatches a CDP click relative to the iframe (screenX/screenY < 100) that
    Cloudflare flags as a bot. We instead resolve the checkbox's full-frame pixel position -- OpenCV template
    matching first, DOM iframe geometry as fallback -- and drive `page.mouse` so the click looks human.
    """
    try:
        # Wait for the Turnstile widget to render before screenshotting for CV or measuring its DOM box.
        page.wait_for_selector(CLOAK_CHALLENGE_FRAME_SELECTOR, timeout=CLOAK_CHALLENGE_CLICK_TIMEOUT_MS)
    except Exception:  # noqa: BLE001
        logger.debug(f"No Cloudflare challenge iframe rendered for {url}; challenge may not use a checkbox.")
        return

    coordinates = locate_checkbox_via_cv(page) or locate_checkbox_via_dom(page)
    if coordinates is None:
        logger.debug(f"No Cloudflare Turnstile checkbox found to click for {url}.")
        return

    click_x, click_y = coordinates
    try:
        # Move (in steps) then click on the main frame so screenX/screenY look human rather than iframe-relative.
        page.mouse.move(click_x, click_y, steps=CLOAK_CHALLENGE_MOUSE_STEPS)
        page.mouse.click(click_x, click_y)
        logger.info(f"Clicked Cloudflare checkbox for {url} at ({click_x:.0f}, {click_y:.0f}).")
    except Exception as exc:  # noqa: BLE001
        # A failed click must not mask the underlying challenge; let the caller decide how to proceed.
        logger.debug(f"Could not click Cloudflare challenge checkbox for {url}: {exc}")
        return

    try:
        # After the click Cloudflare validates and redirects, so wait for the real page to settle.
        page.wait_for_load_state("networkidle", timeout=CLOAK_CHALLENGE_SETTLE_TIMEOUT_MS)
    except playwright_timeout_error:
        logger.debug(f"Timed out waiting for {url} to settle after clicking the challenge.")
