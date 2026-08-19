from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd

from config.logging_config import get_logger
from src.extract.ga4_extractor import GA4_NUMERIC_COLUMNS
from src.utils.text_utils import (
    classify_keyword_intent,
    clean_text,
    infer_page_type,
)


logger = get_logger(__name__)


def normalize_page_key(
    value: str,
) -> str:
    """
    Normalize full URLs and GA4 landing-page paths
    into a common page key.
    """
    text = clean_text(value)

    if not text:
        return "/"

    parsed = urlparse(text)

    if parsed.scheme or parsed.netloc:
        path = parsed.path
    else:
        path = text.split("?", 1)[0]

    path = "/" + path.strip("/")

    if path != "/":
        path = path.rstrip("/")

    return path.lower()


def aggregate_gsc_page_data(
    gsc_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Search Console records to daily page level.

    Clicks and impressions are additive.
    CTR is recalculated from aggregated totals.
    Average position is impression-weighted.
    """

    if gsc_dataframe.empty:
        return gsc_dataframe.copy()

    result = gsc_dataframe.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["page_key"] = result[
        "page"
    ].map(
        normalize_page_key
    )

    result["query"] = (
        result.get(
            "query",
            pd.Series(
                [""] * len(result),
                index=result.index,
            ),
        )
        .fillna("")
        .astype(str)
    )

    for column in (
        "clicks",
        "impressions",
        "position",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result[
        "_position_weighted_sum"
    ] = (
        result["position"]
        * result["impressions"]
    )

    grouped = (
        result.groupby(
            [
                "date",
                "page",
                "page_key",
            ],
            as_index=False,
        )
        .agg(
            clicks=(
                "clicks",
                "sum",
            ),
            impressions=(
                "impressions",
                "sum",
            ),
            _position_weighted_sum=(
                "_position_weighted_sum",
                "sum",
            ),
            _simple_position=(
                "position",
                "mean",
            ),
            query=(
                "query",
                lambda values: " ".join(
                    sorted(
                        {
                            clean_text(value)
                            for value in values
                            if clean_text(value)
                        }
                    )
                ),
            ),
        )
    )

    grouped[
        "position"
    ] = (
        grouped[
            "_position_weighted_sum"
        ]
        / grouped[
            "impressions"
        ].replace(
            0,
            pd.NA,
        )
    )

    grouped[
        "position"
    ] = grouped[
        "position"
    ].fillna(
        grouped[
            "_simple_position"
        ]
    ).fillna(
        0.0
    )

    grouped["ctr"] = (
        grouped["clicks"]
        / grouped["impressions"].replace(
            0,
            pd.NA,
        )
    ).fillna(
        0.0
    )

    return (
        grouped.drop(
            columns=[
                "_position_weighted_sum",
                "_simple_position",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )


def aggregate_ga4_page_data(
    ga4_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate GA4 data to daily landing-page level.

    Additive metrics are summed. Rate and duration metrics are
    recomputed/weighted instead of being summed.
    """

    if ga4_dataframe.empty:
        return ga4_dataframe.copy()

    result = ga4_dataframe.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["page_key"] = result[
        "landing_page"
    ].map(
        normalize_page_key
    )

    available_numeric = [
        column
        for column in GA4_NUMERIC_COLUMNS
        if column in result.columns
    ]

    for column in available_numeric:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    non_additive = {
        "engagement_rate",
        "average_session_duration",
    }

    additive_columns = [
        column
        for column in available_numeric
        if column not in non_additive
    ]

    if (
        "average_session_duration"
        in result.columns
        and "sessions"
        in result.columns
    ):
        result[
            "_session_duration_weighted"
        ] = (
            result[
                "average_session_duration"
            ].fillna(0)
            * result[
                "sessions"
            ].fillna(0)
        )

    aggregation: dict[str, tuple[str, str]] = {
        column: (
            column,
            "sum",
        )
        for column in additive_columns
    }

    if (
        "_session_duration_weighted"
        in result.columns
    ):
        aggregation[
            "_session_duration_weighted"
        ] = (
            "_session_duration_weighted",
            "sum",
        )

    grouped = (
        result.groupby(
            [
                "date",
                "page_key",
            ],
            as_index=False,
        )
        .agg(
            **aggregation
        )
    )

    if (
        "engaged_sessions"
        in grouped.columns
        and "sessions"
        in grouped.columns
    ):
        grouped[
            "engagement_rate"
        ] = (
            grouped[
                "engaged_sessions"
            ]
            / grouped[
                "sessions"
            ].replace(
                0,
                pd.NA,
            )
        ).fillna(
            0.0
        )

    elif (
        "engagement_rate"
        in result.columns
    ):
        rate_mean = (
            result.groupby(
                [
                    "date",
                    "page_key",
                ],
                as_index=False,
            )
            .agg(
                engagement_rate=(
                    "engagement_rate",
                    "mean",
                )
            )
        )

        grouped = grouped.merge(
            rate_mean,
            on=[
                "date",
                "page_key",
            ],
            how="left",
        )

    if (
        "_session_duration_weighted"
        in grouped.columns
        and "sessions"
        in grouped.columns
    ):
        grouped[
            "average_session_duration"
        ] = (
            grouped[
                "_session_duration_weighted"
            ]
            / grouped[
                "sessions"
            ].replace(
                0,
                pd.NA,
            )
        ).fillna(
            0.0
        )

        grouped.drop(
            columns=[
                "_session_duration_weighted",
            ],
            inplace=True,
            errors="ignore",
        )

    elif (
        "average_session_duration"
        in result.columns
    ):
        duration_mean = (
            result.groupby(
                [
                    "date",
                    "page_key",
                ],
                as_index=False,
            )
            .agg(
                average_session_duration=(
                    "average_session_duration",
                    "mean",
                )
            )
        )

        grouped = grouped.merge(
            duration_mean,
            on=[
                "date",
                "page_key",
            ],
            how="left",
        )

    return grouped


def add_page_classifications(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add page-type and keyword-intent classifications.
    """
    result = dataframe.copy()

    if "page_type" not in result.columns:
        result["page_type"] = result.apply(
            infer_page_type,
            axis=1,
        )

    else:
        missing_page_type = (
            result["page_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )

        result.loc[
            missing_page_type,
            "page_type",
        ] = result.loc[
            missing_page_type
        ].apply(
            infer_page_type,
            axis=1,
        )

    if "keyword_intent" not in result.columns:
        result["keyword_intent"] = result[
            "query"
        ].map(
            classify_keyword_intent
        )

    else:
        missing_intent = (
            result["keyword_intent"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )

        result.loc[
            missing_intent,
            "keyword_intent",
        ] = result.loc[
            missing_intent,
            "query",
        ].map(
            classify_keyword_intent
        )

    return result


def merge_gsc_and_ga4(
    gsc_dataframe: pd.DataFrame,
    ga4_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Search Console and GA4 data using date and page key.
    """
    if gsc_dataframe.empty:
        raise ValueError(
            "Search Console or SEO input data cannot be empty."
        )

    gsc_daily = aggregate_gsc_page_data(
        gsc_dataframe
    )

    if ga4_dataframe.empty:
        result = gsc_daily.copy()

        for column in GA4_NUMERIC_COLUMNS:
            result[column] = 0.0

    else:
        ga4_daily = aggregate_ga4_page_data(
            ga4_dataframe
        )

        result = gsc_daily.merge(
            ga4_daily,
            on=[
                "date",
                "page_key",
            ],
            how="left",
        )

        for column in GA4_NUMERIC_COLUMNS:
            if column not in result.columns:
                result[column] = 0.0

            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="coerce",
                )
                .fillna(0)
            )

    result = add_page_classifications(
        result
    )

    logger.info(
        "GSC and GA4 integration completed: %d rows.",
        len(result),
    )

    return result.reset_index(
        drop=True
    )
