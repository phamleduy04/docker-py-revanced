"""Regression tests for Uptodown's app and XAPK download pages."""

# Uptodown changes markup often, so these tests pin the parser decisions that protect release artifacts.
# Private helper coverage is intentional because the public path would perform live Uptodown downloads.
# ruff: noqa: PT009, PT027, SLF001

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest import TestCase
from unittest.mock import patch

from src.downloader.uptodown import UptoDown
from src.exceptions import UptoDownAPKDownloadError
from src.utils import request_header, request_timeout

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.config import RevancedConfig


class _FakeTimeoutError(Exception):
    """Stand-in for Playwright's TimeoutError so the checkbox retry can be exercised without a browser."""


class _CloakPage:
    """Browser page double that answers the Turnstile-gated token request Uptodown makes on click."""

    def __init__(self: Self, payload: dict[str, Any], *, silent_attempts: int = 0) -> None:
        """Record the token payload plus how many clicks Turnstile swallows before answering."""
        self.payload = payload
        self.silent_attempts = silent_attempts
        self.visited: str | None = None
        self.clicks: list[str] = []

    def goto(self: Self, url: str, **_kwargs: object) -> None:
        """Record the navigated page because the token is only issued from the real download page."""
        self.visited = url

    @contextmanager
    def expect_response(self: Self, predicate: Any, **_kwargs: object) -> "Iterator[Any]":
        """Return the token endpoint response the button click triggers, or time out while Turnstile blocks it."""
        response = SimpleNamespace(
            url="https://wanderlog.en.uptodown.com/ajax/app/1286063/file/1183684099/download-url",
            status=200,
            json=lambda: self.payload,
        )
        if not predicate(response):
            msg = f"Token response predicate rejected {response.url}"
            raise AssertionError(msg)
        holder = SimpleNamespace(value=response)
        yield holder
        if self.silent_attempts > 0:
            self.silent_attempts -= 1
            raise _FakeTimeoutError

    def click(self: Self, selector: str, **_kwargs: object) -> None:
        """Record every clicked selector because only the download button starts the token exchange."""
        self.clicks.append(selector)


class _CloakBrowser:
    """Browser double that records closure while exposing one page instance."""

    def __init__(self: Self, payload: dict[str, Any], *, silent_attempts: int = 0) -> None:
        """Create state used to verify token resolution and cleanup."""
        self.closed = False
        self.page = _CloakPage(payload, silent_attempts=silent_attempts)

    def new_page(self: Self) -> _CloakPage:
        """Return the fake page used by the token fallback."""
        return self.page

    def close(self: Self) -> None:
        """Record cleanup because browser processes must not leak after token resolution."""
        self.closed = True


class _UptodownResponse(SimpleNamespace):
    """Small response double with the fields consumed by `handle_request_response` and BeautifulSoup."""

    status_code: int = 200
    text: str


def _config() -> "RevancedConfig":
    """Build the narrow config object needed while `_download` is mocked out."""
    return cast("RevancedConfig", SimpleNamespace())


def _cloak_dependencies(browser: _CloakBrowser) -> tuple[Any, Any]:
    """Mirror the real loader's (launch, TimeoutError) pair while handing back the browser double."""
    return lambda **_kwargs: browser, _FakeTimeoutError


def _download_headers() -> dict[str, str]:
    """Mirror the auth headers Uptodown expects when resolving signed direct download tokens."""
    return {
        "User-Agent": request_header["User-Agent"],
        "Authorization": request_header["Authorization"],
    }


class UptodownDownloaderTests(TestCase):
    """Verify that Uptodown pages resolve to app files, not Uptodown's installer app."""

    def test_generic_xapk_download_page_uses_current_direct_token(self: Self) -> None:
        """Current XAPK pages advertise the store while still exposing a direct app-file token."""
        generic_page = """
            <button id="detail-download-button" class="button download xapk"
                    data-download-version="1174126433"
                    data-url="youtube-music-token">
                Download with UPTODOWN app store
            </button>
        """
        downloader = UptoDown(_config())

        with (
            patch(
                "src.downloader.uptodown.requests.get",
                return_value=_UptodownResponse(text=generic_page),
            ) as request_get,
            patch.object(downloader, "_download") as download,
        ):
            file_name, download_url = downloader.extract_download_link(
                "https://youtube-music.en.uptodown.com/android/download/1164645913",
                "YOUTUBE_MUSIC_MORPHE",
            )

        self.assertEqual("YOUTUBE_MUSIC_MORPHE.apk", file_name)
        self.assertEqual("https://dw.uptodown.com/dwn/youtube-music-token", download_url)
        download.assert_called_once_with(
            "https://dw.uptodown.com/dwn/youtube-music-token",
            "YOUTUBE_MUSIC_MORPHE.apk",
            extra_headers=_download_headers(),
        )
        request_get.assert_called_once_with(
            "https://youtube-music.en.uptodown.com/android/download/1164645913",
            headers=request_header,
            allow_redirects=True,
            timeout=request_timeout,
        )

    def test_generic_xapk_download_page_without_token_resolves_legacy_variant_file(self: Self) -> None:
        """Legacy XAPK bridge pages without a direct token still need the variant file ID fallback."""
        generic_page = """
            <button id="detail-download-button" class="button download xapk"
                    data-download-version="1174126433">
                Download with UPTODOWN app store
            </button>
        """
        variant_page = """
            <button id="detail-download-button" class="button download"
                    data-url="reddit-xapk-token">
                Download 81.92 MB free
            </button>
        """
        downloader = UptoDown(_config())

        with (
            patch(
                "src.downloader.uptodown.requests.get",
                side_effect=[_UptodownResponse(text=generic_page), _UptodownResponse(text=variant_page)],
            ) as request_get,
            patch.object(downloader, "_download") as download,
        ):
            file_name, download_url = downloader.extract_download_link(
                "https://reddit-official-app.en.uptodown.com/android/download",
                "REDDIT_ANDEA",
            )

        self.assertEqual("REDDIT_ANDEA.xapk", file_name)
        self.assertEqual("https://dw.uptodown.com/dwn/reddit-xapk-token", download_url)
        download.assert_called_once_with(
            "https://dw.uptodown.com/dwn/reddit-xapk-token",
            "REDDIT_ANDEA.xapk",
            extra_headers=_download_headers(),
        )
        request_get.assert_any_call(
            "https://reddit-official-app.en.uptodown.com/android/download/1174126433-x",
            headers=request_header,
            allow_redirects=True,
            timeout=request_timeout,
        )

    def test_version_page_without_token_replaces_file_id_instead_of_nesting_it(self: Self) -> None:
        """Version pages already carry a file ID, so the variant URL must replace it rather than append to it."""
        version_page = """
            <button id="detail-download-button" class="button download xapk"
                    data-download-version="1183684099">
                Download with UPTODOWN app store
            </button>
        """
        variant_page = """
            <button id="detail-download-button" class="button download"
                    data-url="wanderlog-xapk-token">
                Download 30 MB free
            </button>
        """
        downloader = UptoDown(_config())

        with (
            patch(
                "src.downloader.uptodown.requests.get",
                side_effect=[_UptodownResponse(text=version_page), _UptodownResponse(text=variant_page)],
            ) as request_get,
            patch.object(downloader, "_download"),
        ):
            file_name, download_url = downloader.extract_download_link(
                "https://wanderlog.en.uptodown.com/android/download/1183684099",
                "WANDERLOG",
            )

        self.assertEqual("WANDERLOG.xapk", file_name)
        self.assertEqual("https://dw.uptodown.com/dwn/wanderlog-xapk-token", download_url)
        request_get.assert_any_call(
            "https://wanderlog.en.uptodown.com/android/download/1183684099-x",
            headers=request_header,
            allow_redirects=True,
            timeout=request_timeout,
        )

    def test_variant_page_without_token_resolves_it_through_the_browser(self: Self) -> None:
        """Uptodown now issues tokens only after Turnstile, so a tokenless variant page must drive the page."""
        variant_page = """
            <button id="detail-download-button" class="button download"
                    data-app-id="1286063" data-file-id="1183684099" data-only-xapk="1">
                Download 153.41 MB free
            </button>
        """
        browser = _CloakBrowser({"data": {"downloadURL": "wanderlog-signed-token"}})
        downloader = UptoDown(_config())

        with (
            patch("src.downloader.uptodown.requests.get", return_value=_UptodownResponse(text=variant_page)),
            patch.object(UptoDown, "_cloak_dependencies", return_value=_cloak_dependencies(browser)),
            patch.object(downloader, "_download") as download,
        ):
            file_name, download_url = downloader.extract_download_link(
                "https://wanderlog.en.uptodown.com/android/download/1183684099-x",
                "WANDERLOG",
            )

        self.assertEqual("WANDERLOG.xapk", file_name)
        self.assertEqual("https://dw.uptodown.com/dwn/wanderlog-signed-token", download_url)
        self.assertEqual("https://wanderlog.en.uptodown.com/android/download/1183684099-x", browser.page.visited)
        self.assertEqual(["#detail-download-button"], browser.page.clicks)
        self.assertTrue(browser.closed)
        download.assert_called_once_with(
            "https://dw.uptodown.com/dwn/wanderlog-signed-token",
            "WANDERLOG.xapk",
            extra_headers=_download_headers(),
        )

    def test_silent_token_request_retries_after_solving_the_cloudflare_checkbox(self: Self) -> None:
        """When Turnstile escalates to its checkbox the first click yields nothing, so the challenge must be solved."""
        browser = _CloakBrowser({"data": {"downloadURL": "wanderlog-signed-token"}}, silent_attempts=1)
        downloader = UptoDown(_config())

        with (
            patch.object(UptoDown, "_cloak_dependencies", return_value=_cloak_dependencies(browser)),
            patch("src.downloader.uptodown.attempt_challenge_click") as challenge_click,
        ):
            token = downloader._resolve_download_token_with_cloak(
                "https://wanderlog.en.uptodown.com/android/download/1183684099-x",
                "WANDERLOG",
            )

        self.assertEqual("wanderlog-signed-token", token)
        challenge_click.assert_called_once()
        # The button is clicked once per attempt, so a solved checkbox must be followed by a second click.
        self.assertEqual(["#detail-download-button", "#detail-download-button"], browser.page.clicks)
        self.assertTrue(browser.closed)

    def test_browser_token_failure_reports_the_page_that_withheld_it(self: Self) -> None:
        """A token response without a download URL must fail loudly instead of building a bogus dw.uptodown link."""
        browser = _CloakBrowser({"success": 0, "errorCode": -51})
        downloader = UptoDown(_config())

        with (
            patch.object(UptoDown, "_cloak_dependencies", return_value=_cloak_dependencies(browser)),
            self.assertRaises(UptoDownAPKDownloadError),
        ):
            downloader._resolve_download_token_with_cloak(
                "https://wanderlog.en.uptodown.com/android/download/1183684099-x",
                "WANDERLOG",
            )

        self.assertTrue(browser.closed)

    def test_plain_apk_download_page_keeps_apk_extension(self: Self) -> None:
        """Plain APK pages are already direct app downloads and should not be rewritten as XAPK variants."""
        apk_page = """
            <button id="detail-download-button" class="button download"
                    data-url="plain-apk-token">
                Download 20 MB free
            </button>
        """
        downloader = UptoDown(_config())

        with (
            patch("src.downloader.uptodown.requests.get", return_value=_UptodownResponse(text=apk_page)),
            patch.object(downloader, "_download") as download,
        ):
            file_name, download_url = downloader.extract_download_link(
                "https://example-app.en.uptodown.com/android/download",
                "EXAMPLE_APP",
            )

        self.assertEqual("EXAMPLE_APP.apk", file_name)
        self.assertEqual("https://dw.uptodown.com/dwn/plain-apk-token", download_url)
        download.assert_called_once_with(
            "https://dw.uptodown.com/dwn/plain-apk-token",
            "EXAMPLE_APP.apk",
            extra_headers=_download_headers(),
        )
