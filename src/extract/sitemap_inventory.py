from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from config.logging_config import get_logger
from config.settings import SETTINGS


logger = get_logger(__name__)


def _normalize_url(value: str) -> str:
    """
    Normalize a URL for inventory and rotation tracking.
    """
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        parts = urlsplit(text)
    except ValueError:
        return ""

    if not parts.scheme or not parts.netloc:
        return ""

    path = parts.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return (
        f"{parts.scheme.lower()}://"
        f"{parts.netloc.lower()}"
        f"{path}"
    )


def fetch_sitemap_urls(
    sitemap_url: str | None = None,
    max_urls: int | None = None,
    request_timeout: int = 30,
    user_agent: str | None = None,
) -> list[str]:
    """
    Fetch page URLs from a sitemap and recursively follow
    nested sitemap files.

    Supports:
    - standard sitemap indexes
    - nested XML sitemap references
    - Demo Store's sitemap structure
    """
    sitemap_url = str(
        sitemap_url
        or SETTINGS.sitemap_url
        or ""
    ).strip()

    if not sitemap_url:
        logger.info(
            "Sitemap URL is not configured. "
            "Skipping sitemap inventory."
        )
        return []

    max_urls = int(
        max_urls
        or SETTINGS.sitemap_max_urls
    )

    user_agent = (
        user_agent
        or SETTINGS.crawl_user_agent
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "application/xml,"
                "text/xml,"
                "*/*;q=0.8"
            ),
        }
    )

    pending: list[str] = [
        sitemap_url,
    ]

    seen_sitemaps: set[str] = set()

    found: list[str] = []
    found_set: set[str] = set()

    while (
        pending
        and len(found) < max_urls
    ):
        current = pending.pop(0)

        if current in seen_sitemaps:
            continue

        seen_sitemaps.add(
            current
        )

        try:
            response = session.get(
                current,
                timeout=request_timeout,
            )

            response.raise_for_status()

            root = ET.fromstring(
                response.content
            )

        except (
            requests.RequestException,
            ET.ParseError,
        ) as exc:
            logger.warning(
                "Sitemap could not be read from %s: %s",
                current,
                exc,
            )
            continue

        root_name = (
            root.tag
            .rsplit("}", 1)[-1]
            .lower()
        )

        loc_nodes = root.findall(
            ".//{*}loc"
        )

        loc_values = [
            str(
                node.text
                or ""
            ).strip()
            for node in loc_nodes
            if str(
                node.text
                or ""
            ).strip()
        ]

        # ====================================================
        # STANDARD SITEMAP INDEX
        # ====================================================

        if root_name == "sitemapindex":
            for value in loc_values:
                if (
                    value
                    and value not in seen_sitemaps
                    and value not in pending
                ):
                    pending.append(
                        value
                    )

            continue

        # ====================================================
        # NESTED XML DETECTION
        # ====================================================

        nested_sitemaps = [
            value
            for value in loc_values
            if urlsplit(
                value
            ).path.lower().endswith(
                ".xml"
            )
        ]

        page_urls = [
            value
            for value in loc_values
            if not urlsplit(
                value
            ).path.lower().endswith(
                ".xml"
            )
        ]

        # If this sitemap only references XML files,
        # follow those files recursively.
        if nested_sitemaps and not page_urls:
            for value in nested_sitemaps:
                if (
                    value not in seen_sitemaps
                    and value not in pending
                ):
                    pending.append(
                        value
                    )

            continue

        # Mixed structure: recurse into XML references
        # and collect normal page/media URLs.
        for value in nested_sitemaps:
            if (
                value not in seen_sitemaps
                and value not in pending
            ):
                pending.append(
                    value
                )

        for value in page_urls:
            normalized = _normalize_url(
                value
            )

            if (
                not normalized
                or normalized in found_set
            ):
                continue

            found.append(
                normalized
            )

            found_set.add(
                normalized
            )

            if len(found) >= max_urls:
                break

    logger.info(
        "Sitemap inventory loaded: %d URLs "
        "from %d sitemap files.",
        len(found),
        len(seen_sitemaps),
    )

    return found


def build_url_inventory(
    sitemap_urls: Iterable[str],
    seo_dataframe: pd.DataFrame | None = None,
    allowed_host: str | None = None,
) -> pd.DataFrame:
    """
    Build a full-site page inventory from sitemap + GSC data.

    When allowed_host is provided:
    - only that website host is kept,
    - CDN/external hosts are excluded.

    Static/binary assets are always excluded.
    """
    sources: dict[str, set[str]] = {}

    normalized_allowed_host = str(
        allowed_host or ""
    ).strip().lower()

    excluded_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".pdf",
        ".zip",
        ".rar",
        ".xml",
        ".css",
        ".js",
        ".json",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp4",
        ".webm",
        ".mp3",
    )

    def add_url(
        value: str,
        source: str,
    ) -> None:
        normalized = _normalize_url(
            value
        )

        if not normalized:
            return

        try:
            parts = urlsplit(
                normalized
            )
        except ValueError:
            return

        host = (
            parts.netloc
            .lower()
        )

        path = (
            parts.path
            or "/"
        ).lower()

        if (
            normalized_allowed_host
            and host != normalized_allowed_host
        ):
            return

        if path.endswith(
            excluded_extensions
        ):
            return

        sources.setdefault(
            normalized,
            set(),
        ).add(
            source
        )

    for url in sitemap_urls:
        add_url(
            url,
            "sitemap",
        )

    if (
        seo_dataframe is not None
        and not seo_dataframe.empty
    ):
        page_column = next(
            (
                column
                for column in [
                    "page",
                    "Page",
                    "url",
                    "URL",
                ]
                if column in seo_dataframe.columns
            ),
            None,
        )

        if page_column:
            for value in (
                seo_dataframe[
                    page_column
                ]
                .dropna()
                .astype(str)
            ):
                add_url(
                    value,
                    "gsc",
                )

    rows = [
        {
            "url": url,
            "inventory_sources": ",".join(
                sorted(
                    source_set
                )
            ),
        }
        for url, source_set
        in sources.items()
    ]

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                "url",
                "inventory_sources",
            ]
        )

    return (
        result
        .sort_values(
            "url"
        )
        .reset_index(
            drop=True
        )
    )

def _load_rotation_state(
    state_file: str,
) -> dict[str, dict[str, object]]:
    """
    Load persisted crawl rotation state.
    """
    path = Path(
        state_file
    )

    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return (
            payload
            if isinstance(
                payload,
                dict,
            )
            else {}
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


def select_crawl_rotation_batch(
    inventory: pd.DataFrame,
    batch_size: int,
    state_file: str,
) -> pd.DataFrame:
    """
    Select crawl URLs using rotation logic.

    Priority:
    1. URLs never crawled before
    2. Oldest crawl timestamp
    3. Lowest crawl count
    4. URL alphabetical order
    """
    if inventory.empty:
        return inventory.copy()

    state = _load_rotation_state(
        state_file
    )

    working = inventory.copy()

    working[
        "last_crawled_at"
    ] = working["url"].map(
        lambda url: str(
            state.get(
                url,
                {},
            ).get(
                "last_crawled_at",
                "",
            )
        )
    )

    working[
        "crawl_count"
    ] = working["url"].map(
        lambda url: int(
            state.get(
                url,
                {},
            ).get(
                "crawl_count",
                0,
            )
            or 0
        )
    )

    working[
        "never_crawled"
    ] = working[
        "last_crawled_at"
    ].eq("")

    working = working.sort_values(
        [
            "never_crawled",
            "last_crawled_at",
            "crawl_count",
            "url",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )

    return (
        working
        .head(
            int(
                batch_size
            )
        )
        .reset_index(
            drop=True
        )
    )


def mark_crawl_batch_completed(
    crawled_urls: Iterable[str],
    state_file: str,
) -> None:
    """
    Persist completed crawl URLs.

    This allows the next pipeline run to continue
    from URLs that have not yet been crawled.
    """
    state = _load_rotation_state(
        state_file
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    for value in crawled_urls:
        url = _normalize_url(
            value
        )

        if not url:
            continue

        previous = state.get(
            url,
            {},
        )

        state[
            url
        ] = {
            "last_crawled_at": now,
            "crawl_count": int(
                previous.get(
                    "crawl_count",
                    0,
                )
                or 0
            )
            + 1,
        }

    path = Path(
        state_file
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )