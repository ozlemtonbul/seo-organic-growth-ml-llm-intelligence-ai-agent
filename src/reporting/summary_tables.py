from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.features.feature_engineering import compute_kpis
from src.features.holiday_features import add_holiday_features
from src.utils.text_utils import safe_divide


def build_keyword_intent_summary(
    seo_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an aggregated SEO performance summary by keyword intent.
    """
    if "keyword_intent" not in seo_raw.columns:
        raise ValueError(
            "The input DataFrame must include "
            "a keyword_intent column."
        )

    featured = compute_kpis(
        seo_raw
    )

    result = featured.groupby(
        "keyword_intent",
        as_index=False,
    ).agg(
        page_count=(
            "page",
            "nunique",
        ),
        total_clicks=(
            "clicks",
            "sum",
        ),
        total_impressions=(
            "impressions",
            "sum",
        ),
        avg_position=(
            "position",
            "mean",
        ),
    )

    result["CTR"] = safe_divide(
        result["total_clicks"],
        result["total_impressions"],
    ).fillna(0)

    return (
        result.sort_values(
            "total_clicks",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def build_page_type_summary(
    seo_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an aggregated SEO performance summary by page type.
    """
    if "page_type" not in seo_raw.columns:
        raise ValueError(
            "The input DataFrame must include "
            "a page_type column."
        )

    featured = compute_kpis(
        seo_raw
    )

    result = featured.groupby(
        "page_type",
        as_index=False,
    ).agg(
        page_count=(
            "page",
            "nunique",
        ),
        total_clicks=(
            "clicks",
            "sum",
        ),
        total_impressions=(
            "impressions",
            "sum",
        ),
        avg_position=(
            "position",
            "mean",
        ),
        total_traffic_value=(
            "TrafficValue",
            "sum",
        ),
    )

    result["CTR"] = safe_divide(
        result["total_clicks"],
        result["total_impressions"],
    ).fillna(0)

    return (
        result.sort_values(
            "total_clicks",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def build_seo_holiday_impact(
    seo_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compare SEO performance across normal, holiday and
    pre-holiday periods.
    """
    featured = add_holiday_features(
        seo_raw,
        holiday_map,
    )

    featured = compute_kpis(
        featured
    )

    result = featured.groupby(
        [
            "is_holiday",
            "is_pre_holiday",
        ],
        as_index=False,
    ).agg(
        clicks=(
            "clicks",
            "sum",
        ),
        impressions=(
            "impressions",
            "sum",
        ),
        avg_position=(
            "position",
            "mean",
        ),
        day_count=(
            "date",
            "nunique",
        ),
    )

    result["avg_daily_clicks"] = safe_divide(
        result["clicks"],
        result["day_count"],
    ).fillna(0)

    result["avg_daily_impressions"] = safe_divide(
        result["impressions"],
        result["day_count"],
    ).fillna(0)

    result["CTR"] = safe_divide(
        result["clicks"],
        result["impressions"],
    ).fillna(0)

    result["period_label"] = result.apply(
        lambda row: (
            "Holiday"
            if row["is_holiday"]
            else (
                "Pre-Holiday"
                if row["is_pre_holiday"]
                else "Normal Day"
            )
        ),
        axis=1,
    )

    return result


def build_daily_weekly_monthly_outputs(
    seo_raw: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build daily, weekly and monthly page-level SEO reports.
    """
    required_columns = {
        "date",
        "page",
        "page_type",
        "clicks",
        "impressions",
        "position",
    }

    missing_columns = sorted(
        required_columns
        - set(seo_raw.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required reporting columns: "
            f"{missing_columns}"
        )

    source = seo_raw.copy()

    if not pd.api.types.is_datetime64_any_dtype(
        source["date"]
    ):
        source["date"] = pd.to_datetime(
            source["date"],
            errors="coerce",
        )

    source = source.dropna(
        subset=["date"]
    )

    daily = source.groupby(
        [
            "date",
            "page",
            "page_type",
        ],
        as_index=False,
    ).agg(
        clicks=(
            "clicks",
            "sum",
        ),
        impressions=(
            "impressions",
            "sum",
        ),
        position=(
            "position",
            "mean",
        ),
    )

    daily = compute_kpis(
        daily
    )

    temporary = source.copy()

    temporary["week"] = (
        temporary["date"]
        .dt.to_period("W")
        .astype(str)
    )

    temporary["month"] = (
        temporary["date"]
        .dt.to_period("M")
        .astype(str)
    )

    weekly = temporary.groupby(
        [
            "week",
            "page",
            "page_type",
        ],
        as_index=False,
    ).agg(
        clicks=(
            "clicks",
            "sum",
        ),
        impressions=(
            "impressions",
            "sum",
        ),
        position=(
            "position",
            "mean",
        ),
    )

    weekly = compute_kpis(
        weekly
    )

    monthly = temporary.groupby(
        [
            "month",
            "page",
            "page_type",
        ],
        as_index=False,
    ).agg(
        clicks=(
            "clicks",
            "sum",
        ),
        impressions=(
            "impressions",
            "sum",
        ),
        position=(
            "position",
            "mean",
        ),
    )

    monthly = compute_kpis(
        monthly
    )

    return (
        daily,
        weekly,
        monthly,
    )


def build_recommendation_summary(
    recommendation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the final business-facing recommendation columns.
    """
    columns = [
        "page",
        "page_type",
        "keyword_intent",
        "ScenarioLabel",
        "ScenarioExplanation",
        "ScenarioNarrative",
        "CurrentClicks",
        "CurrentImpressions",
        "CurrentCTR",
        "CurrentPosition",
        "CurrentTrafficValue",
        "CurrentContentScore",
        "CurrentGeoReadinessScore",
        "Scenario",
        "RecommendedAction",
        "ConfidenceLevel",
        "PriorityTier",
        "EffortScore",
        "EstimatedImplementationCost",
        "PredictedNextClicks",
        "PredictedNextImpressions",
        "PredictedTrafficValue",
        "EstimatedClickChangePct",
        "EstimatedImpressionChangePct",
        "EstimatedPositionGain",
        "EstimatedGeoScoreGain",
        "EstimatedContentScoreGain",
        "ScenarioContentScore",
        "ScenarioGeoReadinessScore",
        "BaselinePredictedClicks",
        "ClicksUplift",
        "ClicksUpliftPct",
        "ExpectedIncrementalTrafficValue",
        "ExpectedNetValue",
        "EstimatedROI",
        "PaybackPeriod",
        "AdjustedNetValue",
        "RecommendationReason",
        "OpportunityScore",
    ]

    existing_columns = [
        column
        for column in columns
        if column in recommendation_df.columns
    ]

    return recommendation_df[
        existing_columns
    ].copy()