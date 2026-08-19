from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.text_utils import (
    classify_keyword_intent,
    humanize_slug,
    infer_page_type,
)


PAGE_NUMERIC_METRICS = [
    "clicks",
    "impressions",
    "sessions",
    "users",
    "conversions",
    "revenue",
    "purchases",
    "add_to_carts",
    "checkouts",
]


def _numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(
            0.0,
            index=dataframe.index,
            dtype="float64",
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(0.0)


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return (
        numerator
        / denominator.replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def _percentile_score(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if numeric.empty:
        return numeric

    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(
            50.0,
            index=numeric.index,
        )

    return (
        numeric.rank(
            method="average",
            pct=True,
        )
        * 100.0
    )


def _ranking_opportunity_score(
    positions: pd.Series,
) -> pd.Series:
    position = pd.to_numeric(
        positions,
        errors="coerce",
    ).fillna(0.0)

    score = pd.Series(
        0.0,
        index=position.index,
    )

    score = np.where(
        (position >= 4) & (position <= 10),
        100.0 - ((position - 4.0) * 5.0),
        score,
    )

    score = np.where(
        (position > 10) & (position <= 20),
        70.0 - ((position - 10.0) * 4.0),
        score,
    )

    score = np.where(
        (position > 20) & (position <= 40),
        25.0 - ((position - 20.0) * 0.75),
        score,
    )

    score = np.where(
        (position > 0) & (position < 4),
        35.0,
        score,
    )

    return pd.Series(
        np.clip(score, 0.0, 100.0),
        index=position.index,
    )


def _priority_label(
    score: float,
) -> str:
    if score >= 70:
        return "High"

    if score >= 45:
        return "Medium"

    return "Low"


def _page_opportunity_type(
    row: pd.Series,
) -> str:
    position = float(row.get("CurrentPosition", 0) or 0)
    ctr = float(row.get("CTR", 0) or 0)
    revenue = float(row.get("Revenue", 0) or 0)
    impressions = float(row.get("Impressions", 0) or 0)

    if 4 <= position <= 15 and impressions > 0:
        return "Ranking Quick Win"

    if 0 < position <= 10 and ctr < 0.02 and impressions > 0:
        return "CTR Opportunity"

    if 0 < position <= 3 and revenue > 0:
        return "Defend Revenue Page"

    if revenue > 0:
        return "Conversion Growth"

    return "Visibility Growth"


def _page_action(
    row: pd.Series,
) -> str:
    page_type = str(
        row.get("page_type", "other")
    ).lower()
    opportunity = str(
        row.get("OpportunityType", "")
    )

    if opportunity == "CTR Opportunity":
        return (
            "Improve title and meta description, align the snippet with the "
            "dominant commercial intent, and retest CTR before broader changes."
        )

    if opportunity == "Ranking Quick Win":
        if page_type == "product":
            return (
                "Strengthen product copy, product/entity schema, internal links, "
                "FAQ coverage and GEO-ready product attributes to move the page "
                "into stronger Top-10 positions."
            )

        if page_type == "blog":
            return (
                "Refresh the article around the target query cluster, improve "
                "answer-first sections, entities and internal links to relevant "
                "commercial pages."
            )

        return (
            "Expand category intent coverage, strengthen internal links, add "
            "structured answer sections and improve SEO/GEO relevance for the "
            "highest-opportunity query cluster."
        )

    if opportunity == "Defend Revenue Page":
        return (
            "Protect rankings and revenue: monitor technical health, refresh "
            "stale content carefully and strengthen internal linking without "
            "changing proven search intent."
        )

    if opportunity == "Conversion Growth":
        return (
            "Keep SEO visibility stable while improving commercial content, "
            "internal navigation and product/category paths that support "
            "add-to-cart and purchase behavior."
        )

    return (
        "Increase relevant search coverage with stronger content, internal "
        "linking and GEO-ready answers before applying higher-cost changes."
    )


def _dominant_value(
    values: Iterable[object],
    fallback: str = "Uncategorized",
) -> str:
    series = pd.Series(
        list(values),
        dtype="object",
    ).dropna()

    if series.empty:
        return fallback

    cleaned = (
        series.astype(str)
        .str.strip()
    )
    cleaned = cleaned[
        cleaned.ne("")
    ]

    if cleaned.empty:
        return fallback

    return str(
        cleaned.value_counts().index[0]
    )


def build_page_opportunity_intelligence(
    integrated_dataframe: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build one business-facing SEO/GEO opportunity row per page.

    The table combines organic visibility, ranking position, GA4 commerce,
    page type and the selected decision-engine recommendation so users can
    answer: which page should we work on next and why?
    """
    if integrated_dataframe.empty:
        return pd.DataFrame()

    source = integrated_dataframe.copy()

    if "page" not in source.columns:
        raise ValueError(
            "The integrated DataFrame must include a page column."
        )

    if "date" in source.columns:
        source["date"] = pd.to_datetime(
            source["date"],
            errors="coerce",
        )

    for column in PAGE_NUMERIC_METRICS:
        source[column] = _numeric_series(
            source,
            column,
        )

    source["position"] = _numeric_series(
        source,
        "position",
    )

    source["page_type"] = source.apply(
        infer_page_type,
        axis=1,
    )

    if "keyword_intent" not in source.columns:
        source["keyword_intent"] = source.get(
            "query",
            pd.Series("", index=source.index),
        ).map(classify_keyword_intent)

    grouped_rows = []

    for page, page_frame in source.groupby(
        "page",
        sort=False,
    ):
        if "date" in page_frame.columns:
            page_frame = page_frame.sort_values(
                "date"
            )

        latest = page_frame.iloc[-1]
        total_clicks = float(page_frame["clicks"].sum())
        total_impressions = float(page_frame["impressions"].sum())
        total_sessions = float(page_frame["sessions"].sum())
        total_revenue = float(page_frame["revenue"].sum())
        total_purchases = float(page_frame["purchases"].sum())
        total_add_to_carts = float(page_frame["add_to_carts"].sum())
        total_checkouts = float(page_frame["checkouts"].sum())
        total_conversions = float(page_frame["conversions"].sum())

        weighted_position = (
            float(
                np.average(
                    page_frame["position"],
                    weights=page_frame["impressions"],
                )
            )
            if total_impressions > 0
            else float(page_frame["position"].mean())
        )

        grouped_rows.append(
            {
                "page": page,
                "EntityName": humanize_slug(str(page)) or "Homepage",
                "page_type": _dominant_value(
                    page_frame["page_type"],
                    "other",
                ),
                "keyword_intent": _dominant_value(
                    page_frame["keyword_intent"],
                ),
                "Clicks": total_clicks,
                "Impressions": total_impressions,
                "CTR": (
                    total_clicks / total_impressions
                    if total_impressions > 0
                    else 0.0
                ),
                "AveragePosition": weighted_position,
                "CurrentPosition": float(
                    pd.to_numeric(
                        latest.get("position", 0),
                        errors="coerce",
                    )
                    or 0
                ),
                "Sessions": total_sessions,
                "Conversions": total_conversions,
                "AddToCarts": total_add_to_carts,
                "Checkouts": total_checkouts,
                "Purchases": total_purchases,
                "Revenue": total_revenue,
                "CartRate": (
                    total_add_to_carts / total_sessions
                    if total_sessions > 0
                    else 0.0
                ),
                "PurchaseRate": (
                    total_purchases / total_sessions
                    if total_sessions > 0
                    else 0.0
                ),
                "RevenuePerSession": (
                    total_revenue / total_sessions
                    if total_sessions > 0
                    else 0.0
                ),
            }
        )

    result = pd.DataFrame(grouped_rows)

    result["RankingOpportunityScore"] = (
        _ranking_opportunity_score(
            result["CurrentPosition"]
        )
    )
    result["DemandScore"] = _percentile_score(
        result["Impressions"]
    )
    result["CommerceScore"] = (
        _percentile_score(result["Revenue"]) * 0.55
        + _percentile_score(result["Purchases"]) * 0.25
        + _percentile_score(result["AddToCarts"]) * 0.20
    )

    result["PageOpportunityScore"] = (
        result["RankingOpportunityScore"] * 0.40
        + result["DemandScore"] * 0.25
        + result["CommerceScore"] * 0.35
    ).clip(0, 100)

    result["OpportunityType"] = result.apply(
        _page_opportunity_type,
        axis=1,
    )
    result["OpportunityPriority"] = result[
        "PageOpportunityScore"
    ].map(_priority_label)
    result["RecommendedFocus"] = result.apply(
        _page_action,
        axis=1,
    )

    if recommendations is not None and not recommendations.empty:
        decision_columns = [
            column
            for column in [
                "page",
                "Scenario",
                "ScenarioLabel",
                "RecommendedAction",
                "RecommendationReason",
                "ConfidenceLevel",
                "PriorityTier",
                "ExpectedNetValue",
                "EstimatedROI",
                "AdjustedNetValue",
                "BusinessDecisionScore",
                "CurrentGeoReadinessScore",
                "ScenarioGeoReadinessScore",
            ]
            if column in recommendations.columns
        ]

        if "page" in decision_columns:
            decisions = (
                recommendations[decision_columns]
                .drop_duplicates(
                    subset=["page"],
                    keep="first",
                )
            )
            result = result.merge(
                decisions,
                on="page",
                how="left",
            )

    return (
        result.sort_values(
            [
                "PageOpportunityScore",
                "Revenue",
                "Impressions",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def build_keyword_opportunity_intelligence(
    seo_dataframe: pd.DataFrame,
    integrated_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build query + landing-page opportunity intelligence.

    Raw GSC query rows are preserved here instead of relying on the integrated
    daily page table, where queries are combined during page-level aggregation.
    GA4 commerce metrics are attached at landing-page level so keyword priority
    reflects both ranking opportunity and commercial value.
    """
    if seo_dataframe.empty:
        return pd.DataFrame()

    required = {
        "page",
        "query",
        "clicks",
        "impressions",
        "position",
    }
    missing = sorted(
        required - set(seo_dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Keyword opportunity input is missing columns: "
            f"{missing}"
        )

    source = seo_dataframe.copy()
    source["query"] = (
        source["query"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    source = source[
        source["query"].ne("")
    ].copy()

    if source.empty:
        return pd.DataFrame()

    if "date" in source.columns:
        source["date"] = pd.to_datetime(
            source["date"],
            errors="coerce",
        )

    for column in ["clicks", "impressions", "position"]:
        source[column] = _numeric_series(
            source,
            column,
        )

    aggregate = source.groupby(
        ["page", "query"],
        as_index=False,
    ).agg(
        Clicks=("clicks", "sum"),
        Impressions=("impressions", "sum"),
        AveragePosition=("position", "mean"),
    )

    if "date" in source.columns:
        latest = (
            source.dropna(subset=["date"])
            .sort_values("date")
            .groupby(
                ["page", "query"],
                as_index=False,
            )
            .tail(1)[
                ["page", "query", "position"]
            ]
            .rename(
                columns={"position": "CurrentPosition"}
            )
        )
        aggregate = aggregate.merge(
            latest,
            on=["page", "query"],
            how="left",
        )
    else:
        aggregate["CurrentPosition"] = aggregate[
            "AveragePosition"
        ]

    aggregate["CTR"] = _safe_ratio(
        aggregate["Clicks"],
        aggregate["Impressions"],
    )
    aggregate["keyword_intent"] = aggregate[
        "query"
    ].map(classify_keyword_intent)

    page_context = build_page_opportunity_intelligence(
        integrated_dataframe=integrated_dataframe,
        recommendations=None,
    )

    context_columns = [
        column
        for column in [
            "page",
            "page_type",
            "Revenue",
            "Purchases",
            "AddToCarts",
            "Sessions",
            "RevenuePerSession",
            "CommerceScore",
        ]
        if column in page_context.columns
    ]

    aggregate = aggregate.merge(
        page_context[context_columns],
        on="page",
        how="left",
    )

    aggregate["RankingOpportunityScore"] = (
        _ranking_opportunity_score(
            aggregate["CurrentPosition"]
        )
    )
    aggregate["DemandScore"] = _percentile_score(
        aggregate["Impressions"]
    )
    aggregate["CommercialIntentScore"] = aggregate[
        "keyword_intent"
    ].map(
        {
            "Transactional": 100.0,
            "Commercial": 85.0,
            "Navigational": 55.0,
            "Informational": 45.0,
            "Uncategorized": 35.0,
        }
    ).fillna(35.0)
    aggregate["LandingPageCommerceScore"] = pd.to_numeric(
        aggregate.get("CommerceScore", 0),
        errors="coerce",
    ).fillna(0.0)

    aggregate["KeywordOpportunityScore"] = (
        aggregate["RankingOpportunityScore"] * 0.40
        + aggregate["DemandScore"] * 0.25
        + aggregate["CommercialIntentScore"] * 0.20
        + aggregate["LandingPageCommerceScore"] * 0.15
    ).clip(0, 100)

    aggregate["KeywordPriority"] = aggregate[
        "KeywordOpportunityScore"
    ].map(_priority_label)

    def keyword_action(row: pd.Series) -> str:
        position = float(row.get("CurrentPosition", 0) or 0)
        intent = str(row.get("keyword_intent", ""))

        if 4 <= position <= 15:
            return (
                "Prioritize this query on the current landing page: strengthen "
                "title/H1/content alignment, internal links and GEO answer coverage."
            )

        if 0 < position <= 3:
            return (
                "Defend the ranking and improve CTR/conversion without changing "
                "the proven query intent."
            )

        if intent in {"Transactional", "Commercial"}:
            return (
                "Build stronger commercial relevance and supporting internal "
                "links before targeting a Top-10 ranking."
            )

        return (
            "Use the query as supporting content or blog coverage and route "
            "qualified users toward the relevant commercial landing page."
        )

    aggregate["RecommendedKeywordAction"] = aggregate.apply(
        keyword_action,
        axis=1,
    )

    return (
        aggregate.sort_values(
            [
                "KeywordOpportunityScore",
                "Impressions",
                "Revenue",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def build_product_category_opportunities(
    page_intelligence: pd.DataFrame,
) -> pd.DataFrame:
    """Return the commercial product/category subset of page intelligence."""
    if page_intelligence.empty:
        return page_intelligence.copy()

    if "page_type" not in page_intelligence.columns:
        return pd.DataFrame()

    return (
        page_intelligence[
            page_intelligence["page_type"]
            .astype(str)
            .str.lower()
            .isin({"product", "category"})
        ]
        .copy()
        .reset_index(drop=True)
    )
