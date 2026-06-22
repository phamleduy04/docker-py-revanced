"""Downloader Factory."""

from src.config import RevancedConfig
from src.downloader.apkeep import Apkeep
from src.downloader.download import Downloader
from src.downloader.github import Github
from src.downloader.gitlab import Gitlab
from src.downloader.justapk import Justapk
from src.downloader.sources import (
    APK_MIRROR_BASE_URL,
    APKEEP,
    APKMIRROR_KEYWORD,
    GITHUB_BASE_URL,
    JUSTAPK,
    UPTODOWN_SUFFIX,
)
from src.downloader.uptodown import UptoDown
from src.exceptions import DownloadError


class DownloaderFactory(object):
    """Downloader Factory."""

    @staticmethod
    def create_downloader(config: RevancedConfig, apk_source: str) -> Downloader:
        """Returns appropriate downloader.

        Args:
        ----
            config : Config
            apk_source : Source URL for APK
        """
        if apk_source.startswith(GITHUB_BASE_URL):
            return Github(config)
        # GitLab app sources need release-asset discovery instead of generic HTML scraping.
        if Gitlab.is_gitlab_url(apk_source):
            return Gitlab(config)
        if apk_source.endswith(UPTODOWN_SUFFIX):
            return UptoDown(config)
        # APKMirror listings (full URL or the bare "apkmirror" keyword) are resolved by package name through
        # justapk, which handles Cloudflare and source fallback.
        if apk_source.startswith(APK_MIRROR_BASE_URL) or apk_source == APKMIRROR_KEYWORD:
            return Justapk(config)
        if apk_source.startswith(APKEEP):
            return Apkeep(config)
        if apk_source.startswith(JUSTAPK):
            return Justapk(config)
        msg = "No download factory found."
        raise DownloadError(msg, url=apk_source)
