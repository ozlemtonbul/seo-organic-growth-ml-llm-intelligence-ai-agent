from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from src.features.holiday_features import (
    add_holiday_features,
)
from src.utils.text_utils import column_or_default


GA4_FEATURE_COLUMNS = [
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


LAG_METRICS = [
    "clicks",
    "impressions",
    "position",
    "CTR",
    "TrafficValue",
]


def compute_kpis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate SEO, traffic-value and organic-commerce KPIs.
    """
    result = dataframe.copy()

    required_columns = [
        "clicks",
        "impressions",
        "position",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    calculated_ctr = np.where(
        result["impressions"] > 0,
        result["clicks"] / result["impressions"],
        0,
    )

    supplied_ctr = pd.to_numeric(
        column_or_default(
            result,
            "ctr",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    result["CTR"] = np.where(
        supplied_ctr > 0,
        supplied_ctr,
        calculated_ctr,
    )

    result["TrafficValue"] = (
        result["clicks"]
        * SETTINGS.value_per_click
    )

    result["RankStrength"] = np.where(
        result["position"] > 0,
        1 / result["position"],
        0,
    )

    result["VisibilityScore"] = (
        result["impressions"]
        * result["CTR"]
    )

    result["Top3Flag"] = (
        (result["position"] > 0)
        & (result["position"] <= 3)
    ).astype(int)

    result["Top10Flag"] = (
        (result["position"] > 0)
        & (result["position"] <= 10)
    ).astype(int)

    result["Page2Flag"] = (
        (result["position"] > 10)
        & (result["position"] <= 20)
    ).astype(int)

    for column in GA4_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0

    result["OrganicConversionRate"] = np.where(
        result["sessions"] > 0,
        result["conversions"]
        / result["sessions"],
        0,
    )

    result["RevenuePerOrganicSession"] = np.where(
        result["sessions"] > 0,
        result["revenue"]
        / result["sessions"],
        0,
    )

    result["RevenuePerOrganicClick"] = np.where(
        result["clicks"] > 0,
        result["revenue"]
        / result["clicks"],
        0,
    )

    result["CartRate"] = np.where(
        result["sessions"] > 0,
        result["add_to_carts"]
        / result["sessions"],
        0,
    )

    result["CheckoutRate"] = np.where(
        result["sessions"] > 0,
        result["checkouts"]
        / result["sessions"],
        0,
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


def add_time_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar-based model features.
    """
    result = dataframe.copy()

    if "date" not in result.columns:
        raise ValueError(
            "The input DataFrame must include a date column."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        result["date"]
    ):
        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_month"] = (
        result["date"].dt.day
    )

    result["month_num"] = (
        result["date"].dt.month
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    result["is_weekend"] = (
        result["date"].dt.dayofweek >= 5
    ).astype(int)

    return result


def add_lag_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add previous-period, rolling-average and change features
    for each page.
    """
    result = dataframe.sort_values(
        ["page", "date"]
    ).copy()

    grouped = result.groupby(
        "page",
        group_keys=False,
    )

    for metric in LAG_METRICS:
        result[f"{metric}_lag_1"] = (
            grouped[metric].shift(1)
        )

        result[f"{metric}_lag_7_avg"] = (
            grouped[metric]
            .rolling(
                7,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    result["clicks_change"] = (
        grouped["clicks"].diff()
    )

    result["impressions_change"] = (
        grouped["impressions"].diff()
    )

    result["position_change"] = (
        grouped["position"].diff()
    )

    result["ctr_change"] = (
        grouped["CTR"].diff()
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


def prepare_training_data(
    seo_raw: pd.DataFrame,
    holiday_map: Optional[
        Dict[str, str]
    ] = None,
) -> pd.DataFrame:
    """
    Prepare page-level observations and next-period targets
    for machine-learning model training.
    """
    result = compute_kpis(
        seo_raw
    )

    result = add_time_features(
        result
    )

    if holiday_map:
        result = add_holiday_features(
            result,
            holiday_map,
        )

    result = add_lag_features(
        result
    )

    result = result.sort_values(
        ["page", "date"]
    ).copy()

    grouped = result.groupby(
        "page"
    )

    result["target_clicks_next"] = (
        grouped["clicks"].shift(-1)
    )

    result["target_impressions_next"] = (
        grouped["impressions"].shift(-1)
    )

    result = result.dropna(
        subset=[
            "target_clicks_next",
            "target_impressions_next",
        ]
    ).copy()

    return result


def get_feature_columns(
    with_holiday: bool = False,
) -> List[str]:
    """
    Return the machine-learning feature-column list.
    """
    columns = [
        "clicks",
        "impressions",
        "position",
        "CTR",
        "TrafficValue",
        "RankStrength",
        "VisibilityScore",
        "Top3Flag",
        "Top10Flag",
        "Page2Flag",
        "day_of_week",
        "day_of_month",
        "month_num",
        "quarter",
        "is_weekend",
        "clicks_lag_1",
        "clicks_lag_7_avg",
        "impressions_lag_1",
        "impressions_lag_7_avg",
        "position_lag_1",
        "position_lag_7_avg",
        "CTR_lag_1",
        "CTR_lag_7_avg",
        "TrafficValue_lag_1",
        "TrafficValue_lag_7_avg",
        "clicks_change",
        "impressions_change",
        "position_change",
        "ctr_change",
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
        "OrganicConversionRate",
        "RevenuePerOrganicSession",
        "RevenuePerOrganicClick",
        "CartRate",
        "CheckoutRate",
    ]

    if with_holiday:
        columns += [
            "is_holiday",
            "is_pre_holiday",
        ]

    return columns