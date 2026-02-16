"""Apkeep Downloader Class."""

import zipfile
from subprocess import PIPE, Popen
from time import perf_counter
from typing import Any, Self

from loguru import logger

from src.app import APP
from src.downloader.download import Downloader
from src.exceptions import DownloadError


class Apkeep(Downloader):
    """Apkeep-based Downloader."""

    def _run_apkeep(self: Self, package_name: str) -> str:
        """Run apkeep CLI to fetch APK from Google Play.

        Google Play does not support downloading specific versions via apkeep.
        The @version syntax only works with APKPure and F-Droid sources.
        """
        email = self.config.env.str("APKEEP_EMAIL")
        token = self.config.env.str("APKEEP_TOKEN")

        if not email or not token:
            msg = "APKEEP_EMAIL and APKEEP_TOKEN must be set in environment."
            raise DownloadError(msg)

        file_name = f"{package_name}.apk"
        file_path = self.config.temp_folder / file_name
        folder_path = self.config.temp_folder / package_name
        zip_path = self.config.temp_folder / f"{package_name}.zip"

        if file_path.exists():
            return file_name
        if zip_path.exists():
            return zip_path.name

        cmd = [
            "apkeep",
            "-a",
            package_name,
            "-d",
            "google-play",
            "-e",
            email,
            "-t",
            token,
            "-o",
            "split_apk=true",
            self.config.temp_folder_name,
        ]

        start = perf_counter()
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            err_output = stderr.decode(errors="replace").strip() if stderr else "no details"
            msg = f"apkeep failed with exit code {process.returncode} for {package_name}: {err_output}"
            raise DownloadError(msg)
        logger.info(f"apkeep completed for {package_name} in {perf_counter() - start:.2f} seconds.")

        if file_path.exists():
            return file_name
        if folder_path.exists() and folder_path.is_dir():
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in folder_path.rglob("*"):
                    arcname = file.relative_to(self.config.temp_folder)
                    zipf.write(file, arcname)
            return zip_path.name
        msg = f"APK not found after apkeep. Expected {file_path} or {folder_path}"
        raise DownloadError(msg)

    def specific_version(self: Self, app: APP, version: str) -> tuple[str, str]:
        """Download from Google Play via Apkeep.

        Google Play does not support version pinning through apkeep.
        Downloads the latest available version instead.
        """
        logger.warning(
            f"apkeep with Google Play does not support specific versions. "
            f"Requested {app.package_name}@{version}, downloading latest instead."
        )
        file_name = self._run_apkeep(app.package_name)
        return file_name, f"apkeep://google-play/{app.package_name}"

    def latest_version(self: Self, app: APP, **kwargs: Any) -> tuple[str, str]:
        """Download latest version from Google Play via Apkeep."""
        file_name = self._run_apkeep(app.package_name)
        return file_name, f"apkeep://google-play/{app.package_name}"
