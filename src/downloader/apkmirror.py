"""Downloader Class."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4

from bs4 import BeautifulSoup, Tag
from loguru import logger

from src.app import APP
from src.downloader.download import Downloader
from src.downloader.sources import APK_MIRROR_BASE_URL
from src.exceptions import APKMirrorAPKDownloadError, ScrapingError
from src.utils import (
    apkmirror_scraper,
    bs4_parser,
    contains_any_word,
    handle_request_response,
    request_timeout,
    resource_folder,
    slugify,
)

if TYPE_CHECKING:
    from src.config import RevancedConfig

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
# APKMirror sometimes returns challenge HTML with HTTP 200, so the body needs explicit marker detection.
CLOAK_CHALLENGE_MARKERS = (
    "attention required",
    "captcha",
    "cf-chl",
    "cf-turnstile",
    "challenge-platform",
    "checking if the site connection is secure",
    "checking your browser",
    "just a moment",
    "turnstile",
)


class ApkMirror(Downloader):
    """Files downloader."""

    def __init__(self: Self, config: "RevancedConfig") -> None:
        super().__init__(config)
        # A single CloakBrowser page is reused across all steps so the Cloudflare clearance cookie persists and
        # later navigations skip the challenge entirely instead of re-solving it per page.
        self._cloak_browser: Any = None
        self._cloak_page: Any = None
        self._playwright_timeout_error: Any = None
        # Once Cloudflare challenges this run, cloudscraper only wastes a round-trip, so route straight to CloakBrowser.
        self._http_challenged = False

    @staticmethod
    def _is_cloudflare_challenge(source: str) -> bool:
        """Detect Cloudflare challenge HTML that can be returned with HTTP 200."""
        lowered_source = source.lower()
        return any(marker in lowered_source for marker in CLOAK_CHALLENGE_MARKERS)

    @staticmethod
    def _cloak_dependencies(url: str, cause: Exception | None = None) -> tuple[Any, Any]:
        """Load CloakBrowser lazily so non-APKMirror flows do not require a browser import."""
        try:
            from cloakbrowser import launch  # noqa: PLC0415
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: PLC0415
        except ImportError as exc:
            msg = "APKMirror returned a Cloudflare challenge, but CloakBrowser is not installed."
            raise APKMirrorAPKDownloadError(msg, url=url) from (cause or exc)

        return launch, PlaywrightTimeoutError

    @staticmethod
    def _locate_challenge_widget(page: Any) -> Any:
        """Return the on-page Turnstile element whose bounding box positions the "Verify you are human" checkbox.

        The managed challenge renders a hidden 0x0 orchestration iframe alongside the visible widget iframe, so
        candidates are filtered by a minimum clickable footprint instead of trusting the first DOM match.
        """
        for selector in CLOAK_CHALLENGE_WIDGET_SELECTORS:
            try:
                candidates = page.query_selector_all(selector)
            except Exception:  # noqa: BLE001
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

    @staticmethod
    def _locate_checkbox_via_cv(page: Any) -> tuple[float, float] | None:
        """Locate the checkbox by pixels via OpenCV template matching against a full-page screenshot."""
        try:
            # Import lazily so non-APKMirror flows never pay the OpenCV/numpy import cost.
            from src.downloader.turnstile_cv import locate_checkbox  # noqa: PLC0415

            point = locate_checkbox(page.screenshot(full_page=True))
        except Exception as exc:  # noqa: BLE001
            # CV is a best-effort locator; any failure falls back to DOM geometry rather than aborting the click.
            logger.debug(f"Turnstile CV locate failed: {exc}")
            return None
        if point is None:
            return None
        return float(point.x), float(point.y)

    @staticmethod
    def _locate_checkbox_via_dom(page: Any) -> tuple[float, float] | None:
        """Locate the checkbox from the widget iframe's on-page bounding box as a fallback to CV."""
        widget = ApkMirror._locate_challenge_widget(page)
        if widget is None:
            return None
        box = widget.bounding_box()
        if not box:
            return None
        # The checkbox sits at the left of the widget, vertically centered.
        return box["x"] + CLOAK_CHALLENGE_CHECKBOX_X_OFFSET, box["y"] + box["height"] / 2

    @staticmethod
    def _attempt_challenge_click(page: Any, url: str, playwright_timeout_error: Any) -> None:
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

        coordinates = ApkMirror._locate_checkbox_via_cv(page) or ApkMirror._locate_checkbox_via_dom(page)
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
            # A failed click must not mask the underlying challenge; fall through to screenshot+raise.
            logger.debug(f"Could not click Cloudflare challenge checkbox for {url}: {exc}")
            return

        try:
            # After the click Cloudflare validates and redirects, so wait for the real page to settle.
            page.wait_for_load_state("networkidle", timeout=CLOAK_CHALLENGE_SETTLE_TIMEOUT_MS)
        except playwright_timeout_error:
            logger.debug(f"Timed out waiting for APKMirror to settle after clicking the challenge for {url}.")

    @staticmethod
    def _save_debug_screenshot(page: Any, url: str) -> Path | None:
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

    def _cloak_session_page(self: Self, url: str, cause: Exception | None = None) -> Any:
        """Return the reusable CloakBrowser page, launching it once so Cloudflare clearance persists across steps."""
        if self._cloak_page is not None:
            return self._cloak_page
        launch_browser, playwright_timeout_error = self._cloak_dependencies(url, cause)
        self._playwright_timeout_error = playwright_timeout_error
        self._cloak_browser = launch_browser(args=CLOAK_BROWSER_ARGS)
        self._cloak_page = self._cloak_browser.new_page()
        return self._cloak_page

    def _close_cloak_session(self: Self) -> None:
        """Close the shared CloakBrowser session at the end of a download so no browser process leaks."""
        if self._cloak_browser is not None:
            try:
                self._cloak_browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Error closing CloakBrowser session: {exc}")
        self._cloak_browser = None
        self._cloak_page = None

    def _solve_challenge_if_present(self: Self, page: Any, url: str, source: str) -> str:
        """Click the Cloudflare checkbox when `source` is a challenge page and return the resulting HTML."""
        if not self._is_cloudflare_challenge(source):
            return source
        # Interactive "Verify you are human" challenges need a click before Cloudflare hands off the real page.
        logger.warning(f"APKMirror shows a Cloudflare challenge for {url}; attempting to click the checkbox.")
        self._attempt_challenge_click(page, url, self._playwright_timeout_error)
        return cast("str", page.content())

    def _fetch_source_with_cloak(self: Self, url: str, cause: Exception | None = None) -> str:
        """Fetch APKMirror HTML through the shared CloakBrowser page, solving a challenge only when one appears.

        Cleared pages return immediately after DOMContentLoaded; the clearance cookie set by the first solved
        challenge carries over on the reused page, so subsequent navigations never wait on the challenge path.
        """
        page = self._cloak_session_page(url, cause)
        page.goto(url, wait_until="domcontentloaded", timeout=CLOAK_REQUEST_TIMEOUT_MS)
        source = self._solve_challenge_if_present(page, url, cast("str", page.content()))

        if self._is_cloudflare_challenge(source):
            screenshot_path = self._save_debug_screenshot(page, url)
            msg = "APKMirror still returned a Cloudflare challenge after CloakBrowser loaded the page."
            if screenshot_path:
                msg += f" Screenshot saved to {screenshot_path}."
            raise APKMirrorAPKDownloadError(msg, url=url) from cause
        return source

    def _download_file_with_cloak(
        self: Self,
        url: str,
        file_name: str,
        referer: str,
        cause: Exception | None = None,
    ) -> None:
        """Download an APKMirror binary through CloakBrowser when the HTTP session is challenged."""
        if self.config.dry_run:
            logger.debug(f"Skipping CloakBrowser download of {file_name} from {url}. Dry run is enabled.")
            return

        page = self._cloak_session_page(referer, cause)
        target_path = self.config.temp_folder.joinpath(file_name)
        # Save into a unique partial path so failed browser downloads never poison the cache target.
        partial_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.part")
        try:
            # The download endpoint validates navigation context; keep CloakBrowser's own UA and add only the referer.
            page.set_extra_http_headers({"Referer": referer})
            page.goto(referer, wait_until="domcontentloaded", timeout=CLOAK_REQUEST_TIMEOUT_MS)
            # Reusing the cleared session usually skips the challenge; solve it only if the referer is still gated.
            self._solve_challenge_if_present(page, referer, cast("str", page.content()))

            with page.expect_download(timeout=CLOAK_REQUEST_TIMEOUT_MS) as download_info:
                # Triggering a same-page anchor preserves browser download behavior better than raw HTTP.
                page.evaluate(
                    """url => {
                        const link = document.createElement("a");
                        link.href = url;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                    }""",
                    url,
                )
            download_info.value.save_as(str(partial_path))
            partial_path.replace(target_path)
        except Exception as exc:
            partial_path.unlink(missing_ok=True)
            screenshot_path = self._save_debug_screenshot(page, referer)
            msg = f"Unable to download {file_name} from APKMirror with CloakBrowser."
            if screenshot_path:
                msg += f" Screenshot saved to {screenshot_path}."
            raise APKMirrorAPKDownloadError(msg, url=url) from exc

    @staticmethod
    def _select_download_extension(apk_type: str, *, preserve_bundle: bool) -> str:
        """Choose the local extension that preserves the patcher's expected input shape."""
        if apk_type == "BUNDLE" and preserve_bundle:
            # Morphe can patch APKM bundles directly, so preserving the bundle avoids APKEditor flattening split inputs.
            return "apkm"
        if apk_type == "BUNDLE":
            # ReVanced-style patchers still receive a merged APK, so bundles keep an archive suffix for APKEditor.
            return "zip"
        # Single APK variants are already patcher-ready and should keep the normal APK suffix.
        return "apk"

    def _extract_force_download_link(
        self: Self,
        link: str,
        app: str,
        *,
        preserve_bundle: bool = False,
    ) -> tuple[str, str]:
        """Extract force download link.

        The actual download.php file endpoint is also behind Cloudflare, so we
        must use apkmirror_scraper (instead of the plain requests session) and
        pass the download page URL as a Referer header — exactly what the
        twitter-apk reference implementation does — to satisfy Cloudflare checks.
        """
        link_page_source = self._extract_source(link)
        notes_divs = self._extracted_search_source_div(link_page_source, "tab-pane")
        apk_type = self._extracted_search_source_div(link_page_source, "apkm-badge").get_text()
        extension = self._select_download_extension(apk_type, preserve_bundle=preserve_bundle)
        possible_links = notes_divs.find_all("a")
        for possible_link in possible_links:
            if possible_link.get("href") and "download.php?id=" in possible_link.get("href"):
                file_name = f"{app}.{extension}"
                download_url = APK_MIRROR_BASE_URL + possible_link["href"]
                try:
                    # cloudscraper remains the fast path when APKMirror only serves a JavaScript challenge.
                    self._download(
                        download_url,
                        file_name,
                        http_session=apkmirror_scraper,
                        extra_headers={"Referer": link},
                    )
                except ScrapingError as exc:
                    # CAPTCHA/Turnstile challenges require a browser context rather than a raw HTTP retry.
                    self._download_file_with_cloak(download_url, file_name, link, exc)
                return file_name, download_url
        msg = f"Unable to extract force download for {app}"
        raise APKMirrorAPKDownloadError(msg, url=link)

    def _extract_download_link(self: Self, page: str, app: str, *, preserve_bundle: bool) -> tuple[str, str]:
        """Extract the APKMirror download link while honoring the selected input-shape policy.

        :param page: Url of the page
        :param app: Name of the app
        """
        logger.debug(f"Extracting download link from\n{page}")
        download_button = self._extracted_search_div(page, "center")
        download_links = download_button.find_all("a")
        if final_download_link := next(
            (
                download_link["href"]
                for download_link in download_links
                if download_link.get("href") and "download/?key=" in download_link.get("href")
            ),
            None,
        ):
            return self._extract_force_download_link(
                APK_MIRROR_BASE_URL + final_download_link,
                app,
                preserve_bundle=preserve_bundle,
            )
        msg = f"Unable to extract link from {app} version list"
        raise APKMirrorAPKDownloadError(msg, url=page)

    def extract_download_link(self: Self, page: str, app: str) -> tuple[str, str]:
        """Function to extract the download link from apkmirror html page.

        :param page: Url of the page
        :param app: Name of the app
        """
        # Public callers keep historical merged-bundle behavior unless they pass through the APP-aware path below.
        return self._extract_download_link(page, app, preserve_bundle=False)

    def extract_download_link_for_app(self: Self, page: str, app: APP) -> tuple[str, str]:
        """Extract the APKMirror download link using the app's patcher profile."""
        # Morphe's APKM support is profile-specific, so only Morphe apps preserve APKMirror bundles as `.apkm`.
        preserve_bundle = app.effective_cli_argsf == "morphe-cli"
        return self._extract_download_link(page, app.app_name, preserve_bundle=preserve_bundle)

    def get_download_page(self: Self, main_page: str) -> str:
        """Function to get the download page in apk_mirror.

        :param main_page: Main Download Page in APK mirror(Index)
        :return:
        """
        list_widget = self._extracted_search_div(main_page, "tab-pane noPadding")
        if list_widget is None:
            # APKMirror can return a normal 404 page for a guessed release URL, so fail before parsing variant rows.
            msg = "Unable to find APKMirror variants table on release page"
            raise APKMirrorAPKDownloadError(msg, url=main_page)
        table_rows = list_widget.find_all(class_="table-row headerFont")
        links: dict[str, str] = {}
        apk_archs = ["arm64-v8a", "universal", "noarch"]
        for row in table_rows:
            if row.find(class_="accent_color"):
                apk_type = row.find(class_="apkm-badge").get_text()
                sub_url = row.find(class_="accent_color")["href"]
                text = row.text.strip()
                if apk_type == "APK" and (not contains_any_word(text, apk_archs)):
                    continue
                links[apk_type] = f"{APK_MIRROR_BASE_URL}{sub_url}"
        if preferred_link := links.get("APK", links.get("BUNDLE")):
            return preferred_link
        msg = "Unable to extract download page"
        raise APKMirrorAPKDownloadError(msg, url=main_page)

    @staticmethod
    def _version_matches_title(version: str, title: str) -> bool:
        """Return whether an APKMirror app-row title refers to the requested version."""
        if version in title:
            return True
        # Piko advertises `release-ripped` versions while APKMirror stores the matching upstream `release` APK.
        apk_mirror_version = version.replace("-ripped", "")
        return apk_mirror_version in title

    @staticmethod
    def _guess_release_url(download_source: str, version: str) -> str:
        """Construct a direct APKMirror release URL from the app listing URL and version.

        APKMirror follows a predictable slug pattern: the last path segment of the listing URL
        (the app slug) is combined with the version (dots replaced by dashes) and a '-release'
        suffix. For example:
          source: https://www.apkmirror.com/apk/google-inc/youtube/
          version: 20.51.39
          result: https://www.apkmirror.com/apk/google-inc/youtube/youtube-20-51-39-release/
        """
        # Strip trailing slash and extract the app slug (last path segment of the listing URL).
        trimmed = download_source.rstrip("/")
        app_slug = trimmed.rsplit("/", maxsplit=1)[-1]
        # APKMirror normalizes version separators to dashes inside release slugs.
        version_slug = version.replace(".", "-")
        return f"{trimmed}/{app_slug}-{version_slug}-release/"

    def _find_specific_version_page(self: Self, app: APP, version: str) -> str:
        """Resolve a specific APKMirror release URL, trying a direct URL guess before listing scrape.

        The listing page only shows the most recent versions. Popular apps like YouTube push older
        versions off the first page quickly, so a direct URL construction is attempted first.
        """
        # Fast path: construct the release URL directly and verify that the release page exists.
        guessed_url = self._guess_release_url(app.download_source, version)
        try:
            page_source = self._extract_source(guessed_url)
            # A valid release page contains the variants table; a 404/soft-error page does not.
            if self._extracted_search_source_div(page_source, "tab-pane noPadding") is not None:
                logger.debug(f"Direct URL resolved for {app.app_name} {version}: {guessed_url}")
                return guessed_url
            logger.debug(f"Guessed URL {guessed_url} loaded but has no variants table; falling back to listing.")
        except (APKMirrorAPKDownloadError, ScrapingError):
            # The guessed URL returned a non-200 or challenge page; fall through to listing-based lookup.
            logger.debug(f"Guessed URL {guessed_url} failed; falling back to listing scrape.")

        # Slow path: scrape the first page of the version listing and match by title text.
        versions_div = self._extracted_search_div(app.download_source, "listWidget p-relative")
        if versions_div is None:
            # A missing listing container means the source page is not the expected APKMirror app listing.
            msg = f"Unable to find APKMirror version list for {app.app_name}"
            raise APKMirrorAPKDownloadError(msg, url=app.download_source)

        for app_row in versions_div.find_all(class_="appRow"):
            # APKMirror release slugs can differ from the app source slug, so links must come from the listing row.
            title = app_row.find(class_="appRowTitle")
            download_link = app_row.find(class_="downloadLink")
            if not title or not download_link or not download_link.get("href"):
                continue
            if self._version_matches_title(version, title.get_text(" ", strip=True)):
                return f"{APK_MIRROR_BASE_URL}{download_link['href']}"

        msg = f"Unable to find {app.app_name} version {version} on APKMirror"
        raise APKMirrorAPKDownloadError(msg, url=app.download_source)

    def _extract_source(self: Self, url: str) -> str:
        """Extracts the source from the url incase of reuse.

        Uses cloudscraper instead of plain requests because APKMirror is protected
        by Cloudflare. CloakBrowser is a heavier fallback for CAPTCHA/Turnstile
        pages that cloudscraper can no longer solve. Once a challenge is seen this run,
        cloudscraper is skipped entirely so the cleared CloakBrowser session is reused directly.
        """
        if self._http_challenged:
            return self._fetch_source_with_cloak(url)

        response = apkmirror_scraper.get(url, timeout=request_timeout)
        try:
            # Non-200 challenge responses need the same browser fallback as HTTP 200 challenge pages.
            handle_request_response(response, url)
        except ScrapingError as exc:
            logger.warning(f"APKMirror HTTP fetch failed for {url}; retrying with CloakBrowser.")
            self._http_challenged = True
            return self._fetch_source_with_cloak(url, exc)
        # cloudscraper's .text is typed as Any; cast to str to satisfy mypy
        source = cast("str", response.text)
        if self._is_cloudflare_challenge(source):
            logger.warning(f"APKMirror returned a Cloudflare challenge for {url}; retrying with CloakBrowser.")
            self._http_challenged = True
            return self._fetch_source_with_cloak(url)
        return source

    @staticmethod
    def _extracted_search_source_div(source: str, search_class: str) -> Tag:
        """Extract search div from source."""
        soup = BeautifulSoup(source, bs4_parser)
        return soup.find(class_=search_class)  # type: ignore[return-value]

    def _extracted_search_div(self: Self, url: str, search_class: str) -> Tag:
        """Extract search div from url."""
        return self._extracted_search_source_div(self._extract_source(url), search_class)

    def specific_version(self: Self, app: APP, version: str, main_page: str = "") -> tuple[str, str]:
        """Function to download the specified version of app from  apkmirror.

        :param app: Name of the application
        :param version: Version of the application to download
        :param main_page: Version of the application to download
        :return: Version of downloaded apk
        """
        try:
            if not main_page:
                # APKMirror may rename app slugs independently from source paths, so resolve release URLs from HTML.
                main_page = self._find_specific_version_page(app, version)
            download_page = self.get_download_page(main_page)
            if app.app_version == "latest":
                try:
                    logger.info(f"Trying to guess {app.app_name} version.")
                    appsec_val = self._extracted_search_div(download_page, "appspec-value")
                    appsec_version = str(appsec_val.find(text=lambda text: "Version" in text))
                    app.app_version = slugify(appsec_version.rsplit(":", maxsplit=1)[-1].strip())
                    logger.info(f"Guessed {app.app_version} for {app.app_name}")
                except ScrapingError:
                    pass
            return self.extract_download_link_for_app(download_page, app)
        finally:
            # Release the shared CloakBrowser session once this app's full download chain has finished.
            self._close_cloak_session()

    def latest_version(self: Self, app: APP, **kwargs: Any) -> tuple[str, str]:
        """Function to download whatever the latest version of app from apkmirror.

        :param app: Name of the application
        :return: Version of downloaded apk
        """
        try:
            app_main_page = app.download_source
            versions_div = self._extracted_search_div(app_main_page, "listWidget p-relative")
            if versions_div is None:
                # Without the listing widget there is no safe way to infer the latest APKMirror release.
                msg = f"Unable to find APKMirror version list for {app.app_name}"
                raise APKMirrorAPKDownloadError(msg, url=app_main_page)
            app_rows = versions_div.find_all(class_="appRow")
            version_urls = [
                app_row.find(class_="downloadLink")["href"]
                for app_row in app_rows
                if "beta" not in app_row.find(class_="appRowTitle").get_text().lower()
                and "alpha" not in app_row.find(class_="appRowTitle").get_text().lower()
            ]
            # specific_version reuses the same session and closes it in its own finally on the happy path.
            return self.specific_version(app, "latest", APK_MIRROR_BASE_URL + max(version_urls))
        finally:
            # Guard against leaking the session if the listing scrape raises before specific_version runs.
            self._close_cloak_session()
