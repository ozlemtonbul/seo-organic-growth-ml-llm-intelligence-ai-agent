from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from googleapiclient.discovery import build

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.extract.google_credentials import get_google_credentials
from src.utils.date_utils import resolve_date_range


logger = get_logger(__name__)


SEARCH_CONSOLE_SCOPE = (
    "https://www.googleapis.com/auth/webmasters.readonly"
)


def validate_search_console_settings() -> None:
    """
    Validate the Google Search Console extraction settings.
    """
    if not SETTINGS.gsc_site_url:
        raise ValueError(
            "GSC_SITE_URL is required for Search Console extraction."
        )

    if not SETTINGS.gsc_service_account_file:
        raise ValueError(
            "GSC_SERVICE_ACCOUNT_FILE is required for "
            "Search Console extraction."
        )

    if SETTINGS.gsc_row_limit < 1:
        raise ValueError(
            "GSC_ROW_LIMIT must be at least 1."
        )


def build_search_console_service():
    """
    Build an authenticated Google Search Console API service.
    """
    validate_search_console_settings()

    credentials = get_google_credentials(
        SETTINGS.gsc_service_account_file,
        [SEARCH_CONSOLE_SCOPE],
    )

    return build(
        "searchconsole",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


def parse_search_console_row(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a Search Console API response row into a normalized record.
    """
    keys = list(
        item.get(
            "keys",
            ["", "", "", "", ""],
        )
    )

    while len(keys) < 5:
        keys.append("")

    return {
        "date": keys[0],
        "page": keys[1],
        "query": keys[2],
        "device": keys[3],
        "country": keys[4],
        "clicks": item.get("clicks", 0),
        "impressions": item.get("impressions", 0),
        "ctr": item.get("ctr", 0),
        "position": item.get("position", 0),
    }


def fetch_search_console_data() -> pd.DataFrame:
    """
    Extract Search Console performance data.

    Dimensions
    ----------
    date
    page
    query
    device
    country

    Metrics
    -------
    clicks
    impressions
    ctr
    position

    Returns
    -------
    pandas.DataFrame
        Search Console performance records.
    """
    service = build_search_console_service()
    date_from, date_to = resolve_date_range()

    rows: List[Dict[str, Any]] = []
    start_row = 0
    page_size = SETTINGS.gsc_row_limit

    logger.info(
        "Starting Search Console extraction between %s and %s.",
        date_from,
        date_to,
    )

    while True:
        request_body = {
            "startDate": date_from,
            "endDate": date_to,
            "dimensions": [
                "date",
                "page",
                "query",
                "device",
                "country",
            ],
            "rowLimit": page_size,
            "startRow": start_row,
            "dataState": "final",
        }

        response = (
            service.searchanalytics()
            .query(
                siteUrl=SETTINGS.gsc_site_url,
                body=request_body,
            )
            .execute()
        )

        batch = response.get("rows", [])

        if not batch:
            break

        rows.extend(
            parse_search_console_row(item)
            for item in batch
        )

        logger.info(
            "Retrieved %d Search Console rows. "
            "Current total: %d.",
            len(batch),
            len(rows),
        )

        if len(batch) < page_size:
            break

        start_row += page_size

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "date",
            "page",
            "query",
            "device",
            "country",
            "clicks",
            "impressions",
            "ctr",
            "position",
        ],
    )

    logger.info(
        "Search Console extraction completed: %d rows.",
        len(dataframe),
    )

    return dataframe