from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.extract.google_credentials import get_google_credentials
from src.utils.date_utils import resolve_date_range


logger = get_logger(__name__)


GA4_SCOPE = "https://www.googleapis.com/auth/analytics.readonly"


GA4_DIMENSIONS = [
    "date",
    "landingPagePlusQueryString",
]


GA4_METRICS = [
    "sessions",
    "totalUsers",
    "engagedSessions",
    "engagementRate",
    "averageSessionDuration",
    "conversions",
    "purchaseRevenue",
    "ecommercePurchases",
    "addToCarts",
    "checkouts",
]


GA4_OUTPUT_COLUMNS = [
    "date",
    "landing_page",
    "sessions",
    "users",
    "engaged_sessions",
    "engagement_rate",
    "average_session_duration",
    "conversions",
    "revenue",
    "purchases",
    "add_to_carts",
    "checkouts",
]


GA4_NUMERIC_COLUMNS = [
    "sessions",
    "users",
    "engaged_sessions",
    "engagement_rate",
    "average_session_duration",
    "conversions",
    "revenue",
    "purchases",
    "add_to_carts",
    "checkouts",
]


def validate_ga4_settings() -> None:
    """
    Validate the Google Analytics 4 extraction settings.
    """
    if not SETTINGS.ga4_property_id:
        raise ValueError(
            "GA4_PROPERTY_ID is required for GA4 extraction."
        )

    if not SETTINGS.ga4_service_account_file:
        raise ValueError(
            "GA4_SERVICE_ACCOUNT_FILE is required for GA4 extraction."
        )

    if SETTINGS.ga4_row_limit < 1:
        raise ValueError(
            "GA4_ROW_LIMIT must be at least 1."
        )


def build_ga4_client() -> BetaAnalyticsDataClient:
    """
    Build an authenticated Google Analytics Data API client.
    """
    validate_ga4_settings()

    credentials = get_google_credentials(
        SETTINGS.ga4_service_account_file,
        [GA4_SCOPE],
    )

    return BetaAnalyticsDataClient(
        credentials=credentials,
    )


def build_ga4_report_request(
    date_from: str,
    date_to: str,
) -> RunReportRequest:
    """
    Build the GA4 landing-page report request.
    """
    return RunReportRequest(
        property=f"properties/{SETTINGS.ga4_property_id}",
        dimensions=[
            Dimension(name=dimension_name)
            for dimension_name in GA4_DIMENSIONS
        ],
        metrics=[
            Metric(name=metric_name)
            for metric_name in GA4_METRICS
        ],
        date_ranges=[
            DateRange(
                start_date=date_from,
                end_date=date_to,
            )
        ],
        limit=SETTINGS.ga4_row_limit,
    )


def parse_ga4_response_row(
    row: Any,
) -> Dict[str, Any]:
    """
    Convert a GA4 API response row into a normalized dictionary.
    """
    dimension_values = [
        value.value
        for value in row.dimension_values
    ]

    metric_values = [
        value.value
        for value in row.metric_values
    ]

    while len(dimension_values) < 2:
        dimension_values.append("")

    while len(metric_values) < 10:
        metric_values.append("0")

    return {
        "date": dimension_values[0],
        "landing_page": dimension_values[1],
        "sessions": metric_values[0],
        "users": metric_values[1],
        "engaged_sessions": metric_values[2],
        "engagement_rate": metric_values[3],
        "average_session_duration": metric_values[4],
        "conversions": metric_values[5],
        "revenue": metric_values[6],
        "purchases": metric_values[7],
        "add_to_carts": metric_values[8],
        "checkouts": metric_values[9],
    }


def standardize_ga4_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize GA4 date and numeric columns.
    """
    result = dataframe.copy()

    if result.empty:
        return pd.DataFrame(
            columns=GA4_OUTPUT_COLUMNS
        )

    result["date"] = pd.to_datetime(
        result["date"],
        format="%Y%m%d",
        errors="coerce",
    )

    for column in GA4_NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0)

    result = result.dropna(
        subset=["date"],
    ).reset_index(drop=True)

    return result


def fetch_ga4_landing_page_data() -> pd.DataFrame:
    """
    Extract GA4 landing-page performance data.

    Dimensions
    ----------
    date
    landingPagePlusQueryString

    Metrics
    -------
    sessions
    totalUsers
    engagedSessions
    engagementRate
    averageSessionDuration
    conversions
    purchaseRevenue
    ecommercePurchases
    addToCarts
    checkouts

    Returns
    -------
    pandas.DataFrame
        Standardized GA4 landing-page data.
    """
    client = build_ga4_client()
    date_from, date_to = resolve_date_range()

    request = build_ga4_report_request(
        date_from=date_from,
        date_to=date_to,
    )

    logger.info(
        "Starting GA4 extraction between %s and %s.",
        date_from,
        date_to,
    )

    response = client.run_report(
        request,
    )

    rows: List[Dict[str, Any]] = [
        parse_ga4_response_row(row)
        for row in response.rows
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=GA4_OUTPUT_COLUMNS,
    )

    dataframe = standardize_ga4_dataframe(
        dataframe,
    )

    logger.info(
        "GA4 extraction completed: %d rows.",
        len(dataframe),
    )

    return dataframe