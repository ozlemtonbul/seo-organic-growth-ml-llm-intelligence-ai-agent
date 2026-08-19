from __future__ import annotations

from typing import Dict

import pandas as pd


ACTION_MAP: Dict[str, str] = {
    "maintain": "Maintain",
    "title_meta_optimization": "Optimize Title and Meta",
    "content_refresh": "Refresh Content",
    "internal_linking_boost": "Improve Internal Linking",
    "category_expansion": "Expand Category SEO",
    "product_content_enrichment": "Enrich Product Content",
    "structured_data_upgrade": "Upgrade Structured Data",
    "geo_answer_optimization": "Optimize for Direct Answers",
    "entity_eet_upgrade": "Strengthen Entity and E-E-A-T Signals",
    "full_seo_geo_optimization": "Apply Full SEO and GEO Optimization",
}


REASON_MAP: Dict[str, str] = {
    "maintain": (
        "Current page performance supports maintaining "
        "the existing SEO setup."
    ),
    "title_meta_optimization": (
        "Visibility exists, but CTR can be improved through "
        "stronger title and meta messaging."
    ),
    "content_refresh": (
        "Ranking and relevance signals indicate a content "
        "refresh opportunity."
    ),
    "internal_linking_boost": (
        "Additional internal links may improve discoverability "
        "and authority distribution."
    ),
    "category_expansion": (
        "The category has potential for broader semantic coverage "
        "and additional organic demand."
    ),
    "product_content_enrichment": (
        "Richer product descriptions, benefits, attributes, image "
        "text and FAQs may improve relevance and commercial "
        "search performance."
    ),
    "structured_data_upgrade": (
        "Page-type-aligned structured data may improve machine "
        "readability and rich-result eligibility."
    ),
    "geo_answer_optimization": (
        "Direct answer blocks and structured entities may improve "
        "generative search visibility."
    ),
    "entity_eet_upgrade": (
        "Clear entity relationships, authorship, freshness and "
        "trust signals may strengthen E-E-A-T and generative-search "
        "readiness."
    ),
    "full_seo_geo_optimization": (
        "Combining metadata, content, internal linking, structured "
        "data, entity signals and GEO components offers the "
        "broadest growth opportunity."
    ),
}


CONFIDENCE_MULTIPLIERS: Dict[str, float] = {
    "High": 1.00,
    "Medium": 0.85,
    "Low": 0.60,
}


def build_recommendations(
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert selected SEO scenarios into actionable recommendations.
    """
    result = best_df.copy()

    if result.empty:
        return result

    if "Scenario" not in result.columns:
        raise ValueError(
            "The recommendation DataFrame must include "
            "a Scenario column."
        )

    result["RecommendedAction"] = (
        result["Scenario"]
        .map(ACTION_MAP)
        .fillna("Review")
    )

    result["RecommendationReason"] = (
        result["Scenario"]
        .map(REASON_MAP)
        .fillna("Manual review recommended.")
    )

    return result


def build_confidence_scores(
    recommendation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add page-level recommendation confidence labels.

    Confidence is based on:
    - historical observation count per page
    - average model R-squared performance
    """
    result = recommendation_df.copy()

    if result.empty:
        return result

    if "page" not in result.columns:
        raise ValueError(
            "The recommendation DataFrame must include "
            "a page column."
        )

    if "page" not in train_df.columns:
        raise ValueError(
            "The training DataFrame must include a page column."
        )

    history = (
        train_df.groupby(
            "page",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "HistoryRows",
            }
        )
    )

    result = result.merge(
        history,
        on="page",
        how="left",
    )

    result["HistoryRows"] = (
        result["HistoryRows"]
        .fillna(0)
        .astype(int)
    )

    average_r2 = (
        float(metrics_df["R2"].mean())
        if (
            not metrics_df.empty
            and "R2" in metrics_df.columns
        )
        else -1.0
    )

    def confidence_label(
        row: pd.Series,
    ) -> str:
        if (
            row["HistoryRows"] >= 20
            and average_r2 >= 0.60
        ):
            return "High"

        if (
            row["HistoryRows"] >= 10
            and average_r2 >= 0.30
        ):
            return "Medium"

        return "Low"

    result["ConfidenceLevel"] = result.apply(
        confidence_label,
        axis=1,
    )

    result["AverageModelR2"] = round(
        average_r2,
        4,
    )

    return result


def apply_confidence_guardrail(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace low-confidence automated actions with manual review.
    """
    result = dataframe.copy()

    if result.empty:
        return result

    if "ConfidenceLevel" not in result.columns:
        raise ValueError(
            "The DataFrame must include a ConfidenceLevel column."
        )

    low_confidence_mask = (
        result["ConfidenceLevel"] == "Low"
    )

    result.loc[
        low_confidence_mask,
        "RecommendedAction",
    ] = "Review"

    result.loc[
        low_confidence_mask,
        "RecommendationReason",
    ] = (
        "Low-confidence recommendation. "
        "Manual SEO validation is required."
    )

    return result


def add_priority_tier(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add confidence-adjusted net value and recommendation priority.
    """
    result = dataframe.copy()

    if result.empty:
        return result

    required_columns = [
        "ConfidenceLevel",
        "ExpectedNetValue",
        "EstimatedROI",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required priority columns: "
            f"{missing_columns}"
        )

    result["ConfidenceMultiplier"] = (
        result["ConfidenceLevel"]
        .map(CONFIDENCE_MULTIPLIERS)
        .fillna(0.75)
    )

    result["AdjustedNetValue"] = (
        result["ExpectedNetValue"]
        * result["ConfidenceMultiplier"]
    )

    def classify_priority(
        row: pd.Series,
    ) -> str:
        if (
            row["AdjustedNetValue"] > 50
            and row["EstimatedROI"] > 0.50
        ):
            return "High Priority"

        if row["AdjustedNetValue"] > 0:
            return "Medium Priority"

        return "Low Priority"

    result["PriorityTier"] = result.apply(
        classify_priority,
        axis=1,
    )

    return result