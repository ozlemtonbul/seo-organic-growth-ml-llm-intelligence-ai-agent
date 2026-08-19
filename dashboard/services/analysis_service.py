from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.filters import (
    DateRange,
    filter_seo_dataframe,
)
from dashboard.utils import (
    load_csv,
    safe_divide,
)


# ============================================================
# OUTPUT FILES
# ============================================================

INTEGRATED_FILE = "seo_integrated_data.csv"
LATEST_STATE_FILE = "seo_latest_page_state.csv"
SCENARIO_FILE = "seo_scenario_simulation.csv"
RECOMMENDATIONS_FILE = "seo_recommendations.csv"
INTENT_FILE = "seo_keyword_intent_summary.csv"
PAGE_TYPE_FILE = "seo_page_type_summary.csv"
HOLIDAY_FILE = "seo_holiday_impact.csv"
DAILY_FILE = "seo_daily_performance.csv"
WEEKLY_FILE = "seo_weekly_performance.csv"
MONTHLY_FILE = "seo_monthly_performance.csv"
MODEL_METRICS_FILE = "seo_model_metrics.csv"
MODEL_BENCHMARK_FILE = "seo_model_benchmark.csv"
FEATURE_IMPORTANCE_FILE = "seo_feature_importance.csv"
ML_FORECAST_DAILY_FILE = "seo_ml_forecast_daily.csv"
ML_FORECAST_HORIZONS_FILE = "seo_ml_forecast_horizons.csv"
ML_FORECAST_PORTFOLIO_FILE = "seo_ml_forecast_portfolio.csv"
ML_FORECAST_METRICS_FILE = "seo_ml_forecast_metrics.csv"
ML_FORECAST_BENCHMARK_FILE = "seo_ml_forecast_benchmark.csv"
ML_FORECAST_FEATURE_IMPORTANCE_FILE = "seo_ml_forecast_feature_importance.csv"
SHAP_SUMMARY_FILE = "seo_shap_summary.csv"
SHAP_DETAIL_FILE = "seo_shap_detail.csv"
TRAINING_FILE = "seo_training_data.csv"


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class AnalysisData:
    integrated: pd.DataFrame
    latest_page_state: pd.DataFrame
    scenarios: pd.DataFrame
    recommendations: pd.DataFrame
    keyword_intent: pd.DataFrame
    page_type: pd.DataFrame
    holiday_impact: pd.DataFrame
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    model_metrics: pd.DataFrame
    model_benchmark: pd.DataFrame
    feature_importance: pd.DataFrame
    ml_forecast_daily: pd.DataFrame
    ml_forecast_horizons: pd.DataFrame
    ml_forecast_portfolio: pd.DataFrame
    ml_forecast_metrics: pd.DataFrame
    ml_forecast_benchmark: pd.DataFrame
    ml_forecast_feature_importance: pd.DataFrame
    shap_summary: pd.DataFrame
    shap_detail: pd.DataFrame
    training: pd.DataFrame


# ============================================================
# LOADERS
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def load_analysis_data() -> AnalysisData:
    """
    Load all SEO pipeline outputs required by dashboard pages.
    """

    return AnalysisData(
        integrated=load_csv(
            INTEGRATED_FILE,
            parse_dates=["date", "Date"],
        ),
        latest_page_state=load_csv(
            LATEST_STATE_FILE,
            parse_dates=["date", "Date"],
        ),
        scenarios=load_csv(
            SCENARIO_FILE,
            parse_dates=["date", "Date"],
        ),
        recommendations=load_csv(
            RECOMMENDATIONS_FILE,
            parse_dates=["date", "Date"],
        ),
        keyword_intent=load_csv(
            INTENT_FILE,
        ),
        page_type=load_csv(
            PAGE_TYPE_FILE,
        ),
        holiday_impact=load_csv(
            HOLIDAY_FILE,
            parse_dates=["date", "Date"],
        ),
        daily=load_csv(
            DAILY_FILE,
            parse_dates=["date", "Date"],
        ),
        weekly=load_csv(
            WEEKLY_FILE,
            parse_dates=["date", "Date"],
        ),
        monthly=load_csv(
            MONTHLY_FILE,
            parse_dates=["date", "Date"],
        ),
        model_metrics=load_csv(
            MODEL_METRICS_FILE,
        ),
        model_benchmark=load_csv(
            MODEL_BENCHMARK_FILE,
            parse_dates=["FirstTestDate"],
        ),
        feature_importance=load_csv(
            FEATURE_IMPORTANCE_FILE,
        ),
        ml_forecast_daily=load_csv(
            ML_FORECAST_DAILY_FILE,
            parse_dates=["ForecastDate"],
        ),
        ml_forecast_horizons=load_csv(
            ML_FORECAST_HORIZONS_FILE,
            parse_dates=["ForecastStartDate", "ForecastEndDate"],
        ),
        ml_forecast_portfolio=load_csv(
            ML_FORECAST_PORTFOLIO_FILE,
            parse_dates=["ForecastStartDate", "ForecastEndDate"],
        ),
        ml_forecast_metrics=load_csv(
            ML_FORECAST_METRICS_FILE,
        ),
        ml_forecast_benchmark=load_csv(
            ML_FORECAST_BENCHMARK_FILE,
            parse_dates=["FirstTestDate"],
        ),
        ml_forecast_feature_importance=load_csv(
            ML_FORECAST_FEATURE_IMPORTANCE_FILE,
        ),
        shap_summary=load_csv(
            SHAP_SUMMARY_FILE,
        ),
        shap_detail=load_csv(
            SHAP_DETAIL_FILE,
            parse_dates=["ObservationDate"],
        ),
        training=load_csv(
            TRAINING_FILE,
            parse_dates=["date", "Date"],
        ),
    )


# ============================================================
# COLUMN HELPERS
# ============================================================

def resolve_date_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Resolve the available date column.
    """

    if dataframe.empty:
        return None

    for candidate in (
        "date",
        "Date",
        "ObservationDate",
    ):
        if candidate in dataframe.columns:
            return candidate

    return None


def resolve_page_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Resolve the available page / URL column.
    """

    if dataframe.empty:
        return None

    for candidate in (
        "page",
        "Page",
        "url",
        "URL",
    ):
        if candidate in dataframe.columns:
            return candidate

    return None


# ============================================================
# FILTERING
# ============================================================

def filter_analysis_data(
    dataframe: pd.DataFrame,
    date_range: DateRange | None = None,
    page_types: list[str] | None = None,
    keyword_intents: list[str] | None = None,
    pages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply standard SEO dashboard filters.

    Supports:
    - date range
    - page type
    - keyword intent
    - page / URL
    """

    if dataframe.empty:
        return dataframe.copy()

    date_column = resolve_date_column(dataframe)

    return filter_seo_dataframe(
        dataframe=dataframe,
        date_range=date_range,
        page_types=page_types,
        keyword_intents=keyword_intents,
        pages=pages,
        date_column=date_column,
    )


# ============================================================
# DATE BOUNDS
# ============================================================

def get_dataframe_date_bounds(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...] = (
        "date",
        "Date",
        "ObservationDate",
    ),
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Return min/max available dates without mutating
    the dataframe.
    """

    if dataframe.empty:
        return None, None

    for column in candidates:

        if column not in dataframe.columns:
            continue

        values = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            return (
                values.min(),
                values.max(),
            )

    return None, None


def get_available_date_bounds(
    dataframe: pd.DataFrame,
) -> tuple[object | None, object | None]:
    """
    Return min/max available dates as Python date objects.
    """

    minimum, maximum = get_dataframe_date_bounds(
        dataframe
    )

    if minimum is None or maximum is None:
        return None, None

    return (
        minimum.date(),
        maximum.date(),
    )


def fast_filter_period(
    dataframe: pd.DataFrame,
    start_date,
    end_date,
    candidates: tuple[str, ...] = (
        "date",
        "Date",
        "ObservationDate",
    ),
) -> pd.DataFrame:
    """
    Filter already-loaded dashboard data in memory.

    This function NEVER triggers the SEO pipeline.
    """

    if dataframe.empty:
        return dataframe.copy()

    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    for column in candidates:

        if column not in dataframe.columns:
            continue

        values = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

        mask = (
            values.dt.date.ge(normalized_start)
            & values.dt.date.le(normalized_end)
        )

        return (
            dataframe.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

    return dataframe.copy()


# ============================================================
# KPI AGGREGATION
# ============================================================

def aggregate_seo_kpis(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Aggregate SEO + GA4 KPIs from an integrated dataset.

    CTR is recalculated from total clicks / total impressions.
    Average position is impression-weighted whenever impressions
    are available, which matches the way aggregated GSC position
    should be interpreted across rows.
    """

    if dataframe.empty:
        return {
            "clicks": 0.0,
            "impressions": 0.0,
            "ctr": 0.0,
            "position": 0.0,
            "sessions": 0.0,
            "users": 0.0,
            "conversions": 0.0,
            "revenue": 0.0,
        }

    def numeric_series(
        candidates: list[str],
    ) -> pd.Series | None:

        for column in candidates:

            if column in dataframe.columns:
                return pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        return None

    def series_sum(
        candidates: list[str],
    ) -> float:

        values = numeric_series(
            candidates
        )

        if values is None:
            return 0.0

        return float(
            values.fillna(0).sum()
        )

    clicks = series_sum(
        [
            "clicks",
            "Clicks",
            "CurrentClicks",
        ]
    )

    impressions = series_sum(
        [
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ]
    )

    sessions = series_sum(
        [
            "sessions",
            "Sessions",
        ]
    )

    users = series_sum(
        [
            "users",
            "Users",
        ]
    )

    conversions = series_sum(
        [
            "conversions",
            "Conversions",
            "purchases",
            "Purchases",
        ]
    )

    revenue = series_sum(
        [
            "revenue",
            "Revenue",
        ]
    )

    position_values = numeric_series(
        [
            "position",
            "Position",
            "CurrentPosition",
        ]
    )

    impression_values = numeric_series(
        [
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ]
    )

    position = 0.0

    if position_values is not None:

        if impression_values is not None:
            valid = (
                position_values.notna()
                & impression_values.notna()
                & impression_values.gt(0)
            )

            if valid.any():
                weights = impression_values.loc[
                    valid
                ]

                position = float(
                    (
                        position_values.loc[
                            valid
                        ]
                        * weights
                    ).sum()
                    / weights.sum()
                )

            else:
                clean_position = (
                    position_values
                    .dropna()
                )

                if not clean_position.empty:
                    position = float(
                        clean_position.mean()
                    )

        else:
            clean_position = (
                position_values
                .dropna()
            )

            if not clean_position.empty:
                position = float(
                    clean_position.mean()
                )

    ctr = safe_divide(
        clicks,
        impressions,
    )

    return {
        "clicks": clicks,
        "impressions": impressions,
        "ctr": float(ctr),
        "position": position,
        "sessions": sessions,
        "users": users,
        "conversions": conversions,
        "revenue": revenue,
    }


# ============================================================
# TOP PAGES
# ============================================================

def get_top_pages(
    dataframe: pd.DataFrame,
    metric: str = "clicks",
    limit: int = 20,
) -> pd.DataFrame:
    """
    Aggregate and return top SEO pages.

    CTR is recalculated from aggregated clicks/impressions.
    Position is impression-weighted whenever possible.
    """

    if dataframe.empty:
        return pd.DataFrame()

    page_column = resolve_page_column(
        dataframe
    )

    if page_column is None:
        return pd.DataFrame()

    working = dataframe.copy()

    source_columns = {
        "clicks": (
            "clicks"
            if "clicks" in working.columns
            else "Clicks"
            if "Clicks" in working.columns
            else None
        ),
        "impressions": (
            "impressions"
            if "impressions" in working.columns
            else "Impressions"
            if "Impressions" in working.columns
            else None
        ),
        "sessions": (
            "sessions"
            if "sessions" in working.columns
            else "Sessions"
            if "Sessions" in working.columns
            else None
        ),
        "users": (
            "users"
            if "users" in working.columns
            else "Users"
            if "Users" in working.columns
            else None
        ),
        "conversions": (
            "conversions"
            if "conversions" in working.columns
            else "Conversions"
            if "Conversions" in working.columns
            else None
        ),
        "revenue": (
            "revenue"
            if "revenue" in working.columns
            else "Revenue"
            if "Revenue" in working.columns
            else None
        ),
        "position": (
            "position"
            if "position" in working.columns
            else "Position"
            if "Position" in working.columns
            else None
        ),
    }

    for source in {
        value
        for value in source_columns.values()
        if value is not None
    }:
        working[source] = pd.to_numeric(
            working[source],
            errors="coerce",
        )

    aggregation_map: dict[str, tuple[str, str]] = {}

    for output_name in (
        "clicks",
        "impressions",
        "sessions",
        "users",
        "conversions",
        "revenue",
    ):
        source = source_columns[
            output_name
        ]

        if source is not None:
            aggregation_map[
                output_name
            ] = (
                source,
                "sum",
            )

    if not aggregation_map:
        return pd.DataFrame()

    result = (
        working
        .groupby(
            page_column,
            dropna=False,
            as_index=False,
        )
        .agg(
            **aggregation_map
        )
    )

    if (
        "clicks" in result.columns
        and "impressions" in result.columns
    ):
        result["ctr"] = (
            result["clicks"]
            / result["impressions"]
            .replace(0, pd.NA)
        ).fillna(0.0)

    position_source = source_columns[
        "position"
    ]

    impression_source = source_columns[
        "impressions"
    ]

    if position_source is not None:

        position_work = working[
            [
                page_column,
                position_source,
            ]
            + (
                [impression_source]
                if impression_source is not None
                else []
            )
        ].copy()

        if impression_source is not None:
            valid = (
                position_work[
                    position_source
                ].notna()
                & position_work[
                    impression_source
                ].notna()
                & position_work[
                    impression_source
                ].gt(0)
            )

            position_work[
                "_weighted_position"
            ] = 0.0

            position_work.loc[
                valid,
                "_weighted_position",
            ] = (
                position_work.loc[
                    valid,
                    position_source,
                ]
                * position_work.loc[
                    valid,
                    impression_source,
                ]
            )

            weighted = (
                position_work
                .groupby(
                    page_column,
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    _weighted_position=(
                        "_weighted_position",
                        "sum",
                    ),
                    _position_weight=(
                        impression_source,
                        "sum",
                    ),
                )
            )

            weighted[
                "position"
            ] = (
                weighted[
                    "_weighted_position"
                ]
                / weighted[
                    "_position_weight"
                ].replace(
                    0,
                    pd.NA,
                )
            )

            fallback = (
                position_work
                .groupby(
                    page_column,
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    _fallback_position=(
                        position_source,
                        "mean",
                    )
                )
            )

            weighted = weighted.merge(
                fallback,
                on=page_column,
                how="left",
            )

            weighted[
                "position"
            ] = weighted[
                "position"
            ].fillna(
                weighted[
                    "_fallback_position"
                ]
            )

            result = result.merge(
                weighted[
                    [
                        page_column,
                        "position",
                    ]
                ],
                on=page_column,
                how="left",
            )

        else:
            simple_position = (
                position_work
                .groupby(
                    page_column,
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    position=(
                        position_source,
                        "mean",
                    )
                )
            )

            result = result.merge(
                simple_position,
                on=page_column,
                how="left",
            )

    if metric not in result.columns:

        fallback_metric = next(
            (
                candidate
                for candidate in (
                    "clicks",
                    "impressions",
                    "sessions",
                    "revenue",
                )
                if candidate in result.columns
            ),
            None,
        )

        if fallback_metric is None:
            return result.head(
                max(
                    1,
                    int(limit),
                )
            ).reset_index(
                drop=True
            )

        metric = fallback_metric

    return (
        result
        .sort_values(
            metric,
            ascending=False,
        )
        .head(
            max(
                1,
                int(limit),
            )
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# OPPORTUNITIES
# ============================================================

def get_priority_recommendations(
    recommendations: pd.DataFrame,
    limit: int = 50,
) -> pd.DataFrame:
    """
    Sort recommendation output by priority and expected value.
    """

    if recommendations.empty:
        return recommendations.copy()

    result = recommendations.copy()

    priority_order = {
        "High Priority": 0,
        "Medium Priority": 1,
        "Low Priority": 2,
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    if "PriorityTier" in result.columns:

        result["_priority_order"] = (
            result["PriorityTier"]
            .astype(str)
            .str.strip()
            .map(priority_order)
            .fillna(99)
        )

    sort_columns: list[str] = []
    ascending: list[bool] = []

    if "_priority_order" in result.columns:
        sort_columns.append(
            "_priority_order"
        )
        ascending.append(True)

    if "ExpectedNetValue" in result.columns:
        result["ExpectedNetValue"] = pd.to_numeric(
            result["ExpectedNetValue"],
            errors="coerce",
        ).fillna(0)

        sort_columns.append(
            "ExpectedNetValue"
        )
        ascending.append(False)

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            ascending=ascending,
        )

    return (
        result
        .drop(
            columns=["_priority_order"],
            errors="ignore",
        )
        .head(
            max(
                1,
                int(limit),
            )
        )
        .reset_index(
            drop=True
        )
    )
