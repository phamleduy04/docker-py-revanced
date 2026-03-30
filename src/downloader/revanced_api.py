"""ReVanced API v5 — patches bundle metadata (works when GitHub releases are unavailable)."""

import re

import requests
from loguru import logger

from src.config import RevancedConfig
from src.exceptions import DownloadError
from src.utils import handle_request_response, request_timeout, update_changelog


def patch_resource(api_url: str, assets_filter: str, config: RevancedConfig) -> tuple[str, str]:
    """Resolve patches bundle URL from GET /v5/patches JSON (download_url, version)."""
    _ = config
    response = requests.get(api_url, timeout=request_timeout)
    handle_request_response(response, api_url)
    data = response.json()
    tag = (data.get("version") or "latest").strip()
    download_url = (data.get("download_url") or "").strip()
    if not download_url:
        msg = "ReVanced API v5 response missing download_url"
        raise DownloadError(msg)
    normalized = {
        "html_url": api_url,
        "tag_name": tag,
        "body": data.get("description") or "",
        "published_at": data.get("created_at") or "",
    }
    update_changelog("api.revanced.app/v5/patches", normalized)
    try:
        filter_pattern = re.compile(assets_filter)
    except re.error as e:
        msg = f"Invalid regex pattern: {assets_filter}"
        raise DownloadError(msg) from e
    if match := filter_pattern.search(download_url):
        logger.debug(f"Matched patch bundle URL: {match.group()}")
        return tag, download_url
    return "", ""
