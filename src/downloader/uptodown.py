"""Upto Down Downloader."""

from typing import Any, Self

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from src.app import APP
from src.downloader.cloak import (
    CLOAK_BROWSER_ARGS,
    CLOAK_REQUEST_TIMEOUT_MS,
    attempt_challenge_click,
    load_browser_dependencies,
    save_debug_screenshot,
)
from src.downloader.download import Downloader
from src.exceptions import UptoDownAPKDownloadError
from src.utils import bs4_parser, handle_request_response, request_header, request_timeout

# Uptodown's download button carries the app/file IDs its own JavaScript posts to the token endpoint.
UPTODOWN_DOWNLOAD_BUTTON_SELECTOR = "#detail-download-button"
# Path fragment of `/ajax/app/<app-id>/file/<file-id>/download-url`, the request that answers with the signed token.
UPTODOWN_DOWNLOAD_URL_ENDPOINT_MARKER = "/download-url"
# Cloudflare Turnstile runs before the token request, so allow more time than a plain navigation.
UPTODOWN_TOKEN_TIMEOUT_MS = 90_000
# Bound the click itself so a button that never becomes actionable fails fast instead of eating the token budget.
UPTODOWN_CLICK_TIMEOUT_MS = 30_000


class UptoDown(Downloader):
    """Files downloader."""

    @staticmethod
    def _is_xapk_variant_page(page: str) -> bool:
        """Detect Uptodown variant URLs that expose the real XAPK file instead of the store bridge."""
        return page.rstrip("/").endswith("-x")

    @staticmethod
    def _is_xapk_store_bridge(detail_download_button: Tag, page: str) -> bool:
        """Detect generic XAPK download pages whose direct token is missing and needs the legacy variant path."""
        button_classes = detail_download_button.get("class", [])
        # Direct variant pages already point at app bytes, so only generic pages are eligible for fallback rewriting.
        return "xapk" in button_classes and not UptoDown._is_xapk_variant_page(page)

    def _resolve_xapk_variant_page(self: Self, detail_download_button: Tag, page: str, app: str) -> str:
        """Build the direct XAPK variant URL from Uptodown's generic app-store bridge button."""
        download_version = detail_download_button.get("data-download-version")
        if not download_version:
            msg = f"Unable to resolve direct XAPK download for {app} from uptodown."
            raise UptoDownAPKDownloadError(msg, url=page)

        # Version pages already end with a file ID, so drop it before appending the variant token.
        base_page, _, last_segment = page.rstrip("/").rpartition("/")
        if not last_segment.isdigit():
            base_page = page.rstrip("/")

        # Uptodown encodes the real file endpoint as `/download/<file-id>-x` behind the variants UI.
        return f"{base_page}/{download_version}-x"

    @staticmethod
    def _cloak_dependencies(page: str) -> tuple[Any, Any]:
        """Load CloakBrowser lazily so flows that never touch Uptodown do not require a browser import."""
        try:
            return load_browser_dependencies()
        except ImportError as exc:
            msg = "Uptodown needs a browser to resolve its download token, but CloakBrowser is not installed."
            raise UptoDownAPKDownloadError(msg, url=page) from exc

    @staticmethod
    def _request_download_url(browser_page: Any, timeout_ms: int) -> Any:
        """Click the download button and return the token endpoint's response."""
        with browser_page.expect_response(
            lambda response: UPTODOWN_DOWNLOAD_URL_ENDPOINT_MARKER in response.url,
            timeout=timeout_ms,
        ) as token_response:
            logger.debug(f"Clicking {UPTODOWN_DOWNLOAD_BUTTON_SELECTOR}.")
            browser_page.click(UPTODOWN_DOWNLOAD_BUTTON_SELECTOR, timeout=UPTODOWN_CLICK_TIMEOUT_MS)
            # Each step logs because Turnstile can stall for minutes and an unattended build otherwise looks hung.
            logger.debug(f"Waiting up to {timeout_ms // 1000}s for uptodown to answer with a download token.")
        response = token_response.value
        logger.debug(f"Uptodown answered {response.status} from {response.url}.")
        return response

    def _resolve_download_token_with_cloak(self: Self, page: str, app: str) -> str:
        """Resolve the signed download token by letting Uptodown's own page clear Cloudflare Turnstile.

        Uptodown no longer ships the token in the markup: clicking the download button runs Turnstile and then
        posts to `/ajax/app/<app-id>/file/<file-id>/download-url`, whose JSON response carries the token. Driving
        the real page is the only way to obtain it, so the click is made in CloakBrowser and the response read back.
        """
        logger.debug(f"Resolving uptodown download token for {app} with CloakBrowser.")
        launch_browser, playwright_timeout_error = self._cloak_dependencies(page)
        # The first launch downloads CloakBrowser's Chromium (~200 MB) into its cache, which takes minutes.
        logger.debug("Launching CloakBrowser.")
        browser = launch_browser(args=CLOAK_BROWSER_ARGS)
        browser_page = None
        try:
            browser_page = browser.new_page()
            logger.debug(f"Loading {page} in CloakBrowser.")
            browser_page.goto(page, wait_until="domcontentloaded", timeout=CLOAK_REQUEST_TIMEOUT_MS)
            logger.debug(f"Loaded {page}.")
            try:
                payload = self._request_download_url(browser_page, UPTODOWN_TOKEN_TIMEOUT_MS).json()
            except playwright_timeout_error:
                # Silence here usually means Turnstile escalated to the interactive checkbox, so solve it and retry.
                logger.warning(f"Uptodown withheld the download token for {app}; trying the Cloudflare checkbox.")
                attempt_challenge_click(browser_page, page, playwright_timeout_error)
                payload = self._request_download_url(browser_page, UPTODOWN_TOKEN_TIMEOUT_MS).json()
        except Exception as exc:
            msg = f"Unable to resolve uptodown download token for {app}. {exc}"
            # A screenshot is the only way to tell a stuck Turnstile apart from changed markup after an unattended run.
            screenshot_path = save_debug_screenshot(browser_page, page) if browser_page else None
            if screenshot_path:
                msg += f" Screenshot saved to {screenshot_path}."
            raise UptoDownAPKDownloadError(msg, url=page) from exc
        finally:
            logger.debug("Closing CloakBrowser.")
            browser.close()

        token = payload.get("data", {}).get("downloadURL")
        if not token:
            msg = f"Uptodown returned no download token for {app}. {payload}"
            raise UptoDownAPKDownloadError(msg, url=page)
        return str(token)

    def extract_download_link(self: Self, page: str, app: str) -> tuple[str, str]:
        """Extract download link from uptodown url."""
        r = requests.get(page, headers=request_header, allow_redirects=True, timeout=request_timeout)
        handle_request_response(r, page)
        soup = BeautifulSoup(r.text, bs4_parser)
        detail_download_button = soup.find("button", id="detail-download-button")

        if not isinstance(detail_download_button, Tag):
            msg = f"Unable to download {app} from uptodown."
            raise UptoDownAPKDownloadError(msg, url=page)

        data_url = detail_download_button.get("data-url")
        if not isinstance(data_url, str) or not data_url:
            if self._is_xapk_store_bridge(detail_download_button, page):
                # Older Uptodown pages omitted the direct token, so keep the variant-page fallback for that shape.
                return self.extract_download_link(
                    self._resolve_xapk_variant_page(detail_download_button, page, app),
                    app,
                )

            # Current Uptodown pages hand out tokens only after Turnstile, so fall back to driving the page.
            data_url = self._resolve_download_token_with_cloak(page, app)

        download_url = f"https://dw.uptodown.com/dwn/{data_url}"
        # Generic pages may be labeled XAPK while redirecting to one APK; archive inspection handles splits later.
        file_name = f"{app}.xapk" if self._is_xapk_variant_page(page) else f"{app}.apk"
        # Uptodown signs direct download tokens against its API headers, so reuse the scrape auth for the binary GET.
        download_headers = {
            "User-Agent": request_header["User-Agent"],
            "Authorization": request_header["Authorization"],
        }
        self._download(download_url, file_name, extra_headers=download_headers)

        return file_name, download_url

    def specific_version(self: Self, app: APP, version: str) -> tuple[str, str]:
        """Function to download the specified version of app from uptodown.

        :param app: Name of the application
        :param version: Version of the application to download
        :return: Version of downloaded apk
        """
        logger.debug("downloading specified version of app from uptodown.")
        url = f"{app.download_source}/versions"
        html = requests.get(url, headers=request_header, timeout=request_timeout).text
        soup = BeautifulSoup(html, bs4_parser)
        detail_app_name = soup.find("h1", id="detail-app-name")

        if not isinstance(detail_app_name, Tag):
            msg = f"Unable to download {app} from uptodown."
            raise UptoDownAPKDownloadError(msg, url=url)

        app_code = detail_app_name.get("data-code")
        version_page = 1
        download_url = None
        version_found = False

        while not version_found:
            version_url = f"{app.download_source}/apps/{app_code}/versions/{version_page}"
            r = requests.get(version_url, headers=request_header, timeout=request_timeout)
            handle_request_response(r, version_url)
            json = r.json()

            if "data" not in json:
                break

            for item in json["data"]:
                if item["version"] == version:
                    version_url_data = item["versionURL"]
                    if isinstance(version_url_data, dict):
                        download_url = (
                            f"{version_url_data['url']}/{version_url_data['extraURL']}/"
                            f"{version_url_data['versionID']}"
                        )
                    else:
                        download_url = f"{version_url_data}-x"
                    version_found = True
                    break

            version_page += 1

        if download_url is None:
            msg = f"Unable to download {app.app_name} from uptodown."
            raise UptoDownAPKDownloadError(msg, url=url)

        return self.extract_download_link(download_url, app.app_name)

    def latest_version(self: Self, app: APP, **kwargs: Any) -> tuple[str, str]:
        """Function to download the latest version of app from uptodown."""
        logger.debug("downloading latest version of app from uptodown.")
        page = f"{app.download_source}/download"
        return self.extract_download_link(page, app.app_name)
