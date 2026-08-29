"""Downloader Class."""

from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4

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
from src.downloader.sources import APK_MIRROR_BASE_URL
from src.exceptions import APKMirrorAPKDownloadError, ScrapingError
from src.utils import (
    apkmirror_scraper,
    bs4_parser,
    contains_any_word,
    handle_request_response,
    request_timeout,
    slugify,
)

if TYPE_CHECKING:
    from src.config import RevancedConfig

# Listing rows link to release pages whose URL always ends in `-release/`, which outlives APKMirror's class renames.
APK_MIRROR_RELEASE_HREF_MARKER = "-release/"
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
            return load_browser_dependencies()
        except ImportError as exc:
            msg = "APKMirror returned a Cloudflare challenge, but CloakBrowser is not installed."
            raise APKMirrorAPKDownloadError(msg, url=url) from (cause or exc)

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
        attempt_challenge_click(page, url, self._playwright_timeout_error)
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
            screenshot_path = save_debug_screenshot(page, url)
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
            screenshot_path = save_debug_screenshot(page, referer)
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
    def _release_link_from_row(app_row: Tag) -> str | None:
        """Return the release page href for one listing row.

        APKMirror keeps renaming the row's link class (`downloadLink` is gone; the link now rides on `fontBlack`
        and `infoLink`), so the release page is matched by URL shape instead. The comment anchor points at the same
        release page with a `#disqus_thread` fragment and is skipped so the returned URL stays a clean page URL.
        """
        for anchor in app_row.find_all("a"):
            href = anchor.get("href")
            if isinstance(href, str) and APK_MIRROR_RELEASE_HREF_MARKER in href and "#" not in href:
                return href
        return None

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
            release_link = self._release_link_from_row(app_row)
            if not title or not release_link:
                continue
            if self._version_matches_title(version, title.get_text(" ", strip=True)):
                return f"{APK_MIRROR_BASE_URL}{release_link}"

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
            version_urls = []
            for app_row in versions_div.find_all(class_="appRow"):
                title = app_row.find(class_="appRowTitle")
                release_link = self._release_link_from_row(app_row)
                if not title or not release_link:
                    continue
                title_text = title.get_text().lower()
                # Prerelease builds are not patchable targets, so they never qualify as the latest version.
                if "beta" in title_text or "alpha" in title_text:
                    continue
                version_urls.append(release_link)
            if not version_urls:
                # An empty list means the listing markup changed again; say so instead of failing on max().
                msg = f"Unable to find any APKMirror release links for {app.app_name}"
                raise APKMirrorAPKDownloadError(msg, url=app_main_page)
            # specific_version reuses the same session and closes it in its own finally on the happy path.
            return self.specific_version(app, "latest", APK_MIRROR_BASE_URL + max(version_urls))
        finally:
            # Guard against leaking the session if the listing scrape raises before specific_version runs.
            self._close_cloak_session()
