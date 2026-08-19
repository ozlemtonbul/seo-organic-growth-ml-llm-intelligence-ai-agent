from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from dashboard.localization import (
    active_language,
    localize_dataframe,
    localized_column_config,
)


def select_existing_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    """
    Return only requested columns that exist.
    """
    return [
        column
        for column in columns
        if column in dataframe.columns
    ]


def render_dataframe(
    dataframe: pd.DataFrame,
    columns: Iterable[str] | None = None,
    height: int | None = None,
    hide_index: bool = True,
) -> None:
    """
    Render a DataFrame safely.

    Streamlit does not accept height=None, so height is only
    passed when a concrete integer value is provided.
    """
    if dataframe.empty:
        st.info(
            ("Veri bulunamadı." if st.session_state.get("dashboard_language", "tr") == "tr" else "No data available.")
        )
        return

    result = dataframe.copy()

    if columns is not None:
        existing_columns = (
            select_existing_columns(
                result,
                columns,
            )
        )

        if existing_columns:
            result = result[
                existing_columns
            ]

    language = active_language()
    result = localize_dataframe(
        result,
        language,
    )

    dataframe_kwargs = {
        "data": result,
        "width": "stretch",
        "hide_index": hide_index,
        "column_config": localized_column_config(
            result,
            language,
        ),
    }

    if height is not None:
        dataframe_kwargs[
            "height"
        ] = int(
            height
        )

    st.dataframe(
        **dataframe_kwargs
    )


def render_top_pages_table(
    dataframe: pd.DataFrame,
    limit: int = 20,
) -> None:
    """
    Render commonly used SEO page columns.
    """
    if dataframe.empty:
        st.info(
            ("Sayfa verisi bulunamadı." if st.session_state.get("dashboard_language", "tr") == "tr" else "No page data available.")
        )
        return

    preferred_columns = [
        "page",
        "Page",
        "page_type",
        "keyword_intent",
        "CurrentClicks",
        "clicks",
        "CurrentImpressions",
        "impressions",
        "CurrentCTR",
        "ctr",
        "CurrentPosition",
        "position",
        "sessions",
        "users",
        "conversions",
        "revenue",
        "PriorityTier",
        "ConfidenceLevel",
        "RecommendedAction",
        "RecommendationReason",
        "ExpectedIncrementalTrafficValue",
        "ExpectedNetValue",
        "EstimatedROI",
    ]

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result,
        columns=preferred_columns,
    )


def render_recommendations_table(
    dataframe: pd.DataFrame,
    limit: int = 50,
) -> None:
    """
    Render SEO recommendation columns.
    """
    if dataframe.empty:
        st.info(
            ("Öneri bulunamadı." if st.session_state.get("dashboard_language", "tr") == "tr" else "No recommendations available.")
        )
        return

    preferred_columns = [
        "page",
        "Page",
        "page_type",
        "keyword_intent",
        "PriorityTier",
        "ConfidenceLevel",
        "Scenario",
        "ScenarioLabel",
        "RecommendedAction",
        "RecommendationReason",
        "ExpectedIncrementalTrafficValue",
        "EstimatedImplementationCost",
        "ExpectedNetValue",
        "EstimatedROI",
        "PaybackPeriod",
        "ExecutiveCommentary",
        "CommentarySource",
    ]

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result,
        columns=preferred_columns,
    )


def render_model_metrics_table(
    dataframe: pd.DataFrame,
) -> None:
    """
    Render model benchmark / validation metrics.
    """
    preferred_columns = [
        "Target",
        "Model",
        "Selected",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
    ]

    render_dataframe(
        dataframe=dataframe,
        columns=preferred_columns,
    )


def render_feature_importance_table(
    dataframe: pd.DataFrame,
    limit: int = 30,
) -> None:
    """
    Render feature-importance data.
    """
    if dataframe.empty:
        st.info(
            ("Özellik önem verisi bulunamadı." if st.session_state.get("dashboard_language", "tr") == "tr" else "No feature importance data available.")
        )
        return

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result
    )


def render_shap_explanations_table(
    dataframe: pd.DataFrame,
    limit: int = 50,
) -> None:
    """
    Render page/query-level SHAP explanations when available.
    """
    if dataframe.empty:
        st.info(
            ("SHAP açıklama verisi bulunamadı." if st.session_state.get("dashboard_language", "tr") == "tr" else "No SHAP explanation data available.")
        )
        return

    preferred_columns = [
        "page",
        "Page",
        "query",
        "Query",
        "Target",
        "Model",
        "Prediction",
        "Driver1",
        "Driver1Value",
        "Driver2",
        "Driver2Value",
        "Driver3",
        "Driver3Value",
        "BaseValue",
    ]

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result,
        columns=preferred_columns,
    )


def render_technical_issues_table(
    dataframe: pd.DataFrame,
    limit: int = 100,
) -> None:
    """
    Render technical SEO audit findings.
    """
    preferred_columns = [
        "URL",
        "PageType",
        "IssueCategory",
        "Issue",
        "Severity",
        "Evidence",
        "CurrentValue",
        "RecommendedValue",
        "RecommendedAction",
        "Priority",
        "EstimatedImpact",
        "Confidence",
        "Source",
    ]

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result,
        columns=preferred_columns,
    )


def render_competitor_gap_table(
    dataframe: pd.DataFrame,
    limit: int = 100,
) -> None:
    """
    Render competitor / keyword / content-gap data.
    """
    preferred_columns = [
        "Keyword",
        "OurDomain",
        "OurPosition",
        "CompetitorDomain",
        "CompetitorPosition",
        "CompetitorURL",
        "GapType",
        "SearchVolume",
        "OpportunityScore",
        "Priority",
        "RecommendedAction",
        "Evidence",
    ]

    result = dataframe.head(
        max(
            1,
            int(limit),
        )
    )

    render_dataframe(
        dataframe=result,
        columns=preferred_columns,
    )
