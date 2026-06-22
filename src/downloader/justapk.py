"""justapk Downloader Class."""

import json
from subprocess import PIPE, Popen
from time import perf_counter
from typing import Any, Self

from loguru import logger

from src.app import APP
from src.downloader.download import Downloader
from src.exceptions import DownloadError


class Justapk(Downloader):
    """justapk-based multi-source Downloader.

    Resolves a clean APK by package name through the `justapk` CLI, which cycles
    through several public sources with automatic fallback and converts XAPK/split
    packages to a single APK before returning.
    """

    def _run_justapk(self: Self, package_name: str, version: str = "") -> tuple[str, str]:
        """Run the justapk CLI to fetch an APK.

        Returns
        -------
            tuple[str, str]: (file name inside the temp folder, version resolved by justapk).
        """
        # --no-convert keeps the raw XAPK/split so this project's APKEditor performs the merge,
        # matching every other downloader; justapk's built-in merge would bypass that pipeline.
        cmd = ["justapk", "download", package_name, "-o", self.config.temp_folder_name, "--no-convert"]
        # justapk defaults to the latest version when no -v flag is passed.
        if version and version != "latest":
            cmd.extend(["-v", version])
        logger.debug(f"Running command: {cmd}")

        start = perf_counter()
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            msg = (
                f"justapk failed with exit code {process.returncode} for app {package_name}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
            raise DownloadError(msg)

        # justapk prints a JSON object describing the resolved artifact to stdout.
        try:
            result = json.loads(stdout.decode(errors="replace"))
            downloaded_path = result["path"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            msg = f"Unable to parse justapk output for app {package_name}."
            raise DownloadError(msg) from error

        # The resolved version lets callers record the concrete version when "latest" was requested.
        resolved_version = str(result.get("version") or "")
        logger.info(f"Downloading completed for app {package_name} in {perf_counter() - start:.2f} seconds.")

        # justapk writes into temp_folder, so the patcher only needs the artifact's file name.
        file_name = self.config.temp_folder.joinpath(downloaded_path).name
        if not self.config.temp_folder.joinpath(file_name).exists():
            msg = f"APK file not found after justapk execution for app {package_name}."
            raise DownloadError(msg)
        return file_name, resolved_version

    @staticmethod
    def _record_resolved_version(app: APP, resolved_version: str) -> None:
        """Backfill the concrete version when the caller requested 'latest'.

        Output file names and the changelog read `app.app_version`, so an unresolved
        "latest" would otherwise leak into artifact names.
        """
        if resolved_version and (not app.app_version or app.app_version == "latest"):
            app.app_version = resolved_version

    def latest_version(self: Self, app: APP, **kwargs: Any) -> tuple[str, str]:
        """Download the latest version of an app by package name via justapk."""
        file_name, resolved_version = self._run_justapk(app.package_name)
        self._record_resolved_version(app, resolved_version)
        logger.info(f"Got file name as {file_name}")
        return file_name, f"justapk://{app.package_name}"

    def specific_version(self: Self, app: APP, version: str) -> tuple[str, str]:
        """Download a specific version of an app by package name via justapk."""
        file_name, resolved_version = self._run_justapk(app.package_name, version)
        self._record_resolved_version(app, resolved_version)
        logger.info(f"Got file name as {file_name}")
        return file_name, f"justapk://{app.package_name}@{resolved_version or version}"
