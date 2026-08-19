from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dashboard.filters import DateRange, filter_dataframe_by_date
from dashboard.services.decision_engine import (
    DecisionEngineResult,
    build_decision_intelligence,
)

from dashboard.services.analysis_service import (
    AnalysisData,
    aggregate_seo_kpis,
    get_available_date_bounds,
    get_priority_recommendations,
    get_top_pages,
    load_analysis_data,
)


# ============================================================
# EXECUTIVE DATA MODEL
# ============================================================


@dataclass(frozen=True)
class ExecutiveData:
    daily: pd.DataFrame
    integrated: pd.DataFrame
    latest_page_state: pd.DataFrame
    recommendations: pd.DataFrame
    scenarios: pd.DataFrame
    keyword_intent: pd.DataFrame
    page_type: pd.DataFrame
    model_metrics: pd.DataFrame
    model_benchmark: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_summary: pd.DataFrame


# ============================================================
# LOAD
# ============================================================


def load_executive_data() -> ExecutiveData:
    """
    Load the dashboard's executive-level data.
    """
    data: AnalysisData = (
        load_analysis_data()
    )

    return ExecutiveData(
        daily=data.daily,
        integrated=data.integrated,
        latest_page_state=(
            data.latest_page_state
        ),
        recommendations=(
            data.recommendations
        ),
        scenarios=data.scenarios,
        keyword_intent=(
            data.keyword_intent
        ),
        page_type=data.page_type,
        model_metrics=(
            data.model_metrics
        ),
        model_benchmark=(
            data.model_benchmark
        ),
        feature_importance=(
            data.feature_importance
        ),
        shap_summary=(
            data.shap_summary
        ),
    )



# ============================================================
# EXECUTIVE DECISION INTELLIGENCE
# ============================================================

def build_executive_decision_intelligence(
    dataframe: pd.DataFrame,
    recommendations: pd.DataFrame,
    start_date,
    end_date,
    comparison_start_date=None,
    comparison_end_date=None,
    language: str = "tr",
    forecast_horizon_days: int = 7,
    limit: int = 30,
) -> DecisionEngineResult:
    """Build the Executive Overview from the shared Decision Engine.

    The dashboard date selector filters already collected historical data only.
    It does not trigger GSC/GA4 collection, crawling, model training, or SHAP.
    """
    current_period = filter_dataframe_by_date(
        dataframe=dataframe,
        date_range=DateRange(start_date=start_date, end_date=end_date),
    )

    if comparison_start_date is not None and comparison_end_date is not None:
        comparison_period = filter_dataframe_by_date(
            dataframe=dataframe,
            date_range=DateRange(
                start_date=comparison_start_date,
                end_date=comparison_end_date,
            ),
        )
    else:
        comparison_period = pd.DataFrame(columns=dataframe.columns)

    return build_decision_intelligence(
        current_period=current_period,
        comparison_period=comparison_period,
        recommendations=recommendations,
        language=language,
        forecast_horizon_days=forecast_horizon_days,
        limit=limit,
    )


# ============================================================
# EXECUTIVE KPI SUMMARY
# ============================================================


def build_executive_kpis(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Return main SEO + GA4 KPI summary.
    """
    return aggregate_seo_kpis(
        dataframe
    )


# ============================================================
# EXECUTIVE OPPORTUNITIES
# ============================================================


def build_executive_opportunities(
    recommendations: pd.DataFrame,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return highest-priority SEO recommendations.
    """
    return get_priority_recommendations(
        recommendations,
        limit=limit,
    )


# ============================================================
# EXECUTIVE TOP PAGES
# ============================================================


def build_executive_top_pages(
    dataframe: pd.DataFrame,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return top organic pages by clicks.
    """
    return get_top_pages(
        dataframe=dataframe,
        metric="clicks",
        limit=limit,
    )


# ============================================================
# EXECUTIVE MODEL SUMMARY
# ============================================================


def build_model_summary(
    model_metrics: pd.DataFrame,
) -> dict[str, float]:
    """
    Build model-level summary statistics.
    """
    if model_metrics.empty:
        return {
            "model_count": 0,
            "average_r2": 0.0,
            "best_r2": 0.0,
        }

    if "R2" not in model_metrics.columns:
        return {
            "model_count": int(
                len(
                    model_metrics
                )
            ),
            "average_r2": 0.0,
            "best_r2": 0.0,
        }

    r2_values = pd.to_numeric(
        model_metrics["R2"],
        errors="coerce",
    ).dropna()

    if r2_values.empty:
        return {
            "model_count": int(
                len(
                    model_metrics
                )
            ),
            "average_r2": 0.0,
            "best_r2": 0.0,
        }

    return {
        "model_count": int(
            len(
                model_metrics
            )
        ),
        "average_r2": float(
            r2_values.mean()
        ),
        "best_r2": float(
            r2_values.max()
        ),
    }


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================


def build_recommendation_summary(
    recommendations: pd.DataFrame,
) -> dict[str, float | int | str]:
    """
    Build executive recommendation KPIs.
    """
    if recommendations.empty:
        return {
            "recommendation_count": 0,
            "high_priority_count": 0,
            "expected_net_value": 0.0,
            "expected_incremental_value": 0.0,
            "average_roi": 0.0,
            "top_action": "N/A",
        }

    result = recommendations.copy()

    high_priority_count = 0

    if "PriorityTier" in result.columns:
        high_priority_count = int(
            result[
                "PriorityTier"
            ]
            .astype(str)
            .eq(
                "High Priority"
            )
            .sum()
        )

    def numeric_sum(
        column: str,
    ) -> float:
        if column not in result.columns:
            return 0.0

        return float(
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    def numeric_mean(
        column: str,
    ) -> float:
        if column not in result.columns:
            return 0.0

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return 0.0

        return float(
            values.mean()
        )

    top_action = "N/A"

    if "RecommendedAction" in result.columns:
        actions = (
            result[
                "RecommendedAction"
            ]
            .dropna()
            .astype(str)
        )

        if not actions.empty:
            mode = actions.mode()

            if not mode.empty:
                top_action = str(
                    mode.iloc[0]
                )

    return {
        "recommendation_count": int(
            len(
                result
            )
        ),
        "high_priority_count": (
            high_priority_count
        ),
        "expected_net_value": (
            numeric_sum(
                "ExpectedNetValue"
            )
        ),
        "expected_incremental_value": (
            numeric_sum(
                "ExpectedIncrementalTrafficValue"
            )
        ),
        "average_roi": (
            numeric_mean(
                "EstimatedROI"
            )
        ),
        "top_action": top_action,
    }


__all__ = [
    "ExecutiveData",
    "load_executive_data",
    "get_available_date_bounds",
    "build_executive_kpis",
    "build_executive_decision_intelligence",
    "build_executive_opportunities",
    "build_executive_top_pages",
    "build_model_summary",
    "build_recommendation_summary",
]
