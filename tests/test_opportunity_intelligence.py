from __future__ import annotations

import pandas as pd

from src.features.opportunity_intelligence import (
    build_keyword_opportunity_intelligence,
    build_page_opportunity_intelligence,
    build_product_category_opportunities,
)


def build_integrated_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "page": "https://example.com/product/a",
                "query": "buy product a",
                "page_type": "product",
                "keyword_intent": "Transactional",
                "clicks": 20,
                "impressions": 300,
                "position": 12,
                "sessions": 50,
                "users": 40,
                "conversions": 5,
                "revenue": 500,
                "purchases": 4,
                "add_to_carts": 12,
                "checkouts": 7,
            },
            {
                "date": "2026-07-02",
                "page": "https://example.com/product/a",
                "query": "buy product a",
                "page_type": "product",
                "keyword_intent": "Transactional",
                "clicks": 25,
                "impressions": 350,
                "position": 9,
                "sessions": 60,
                "users": 45,
                "conversions": 6,
                "revenue": 650,
                "purchases": 5,
                "add_to_carts": 15,
                "checkouts": 8,
            },
            {
                "date": "2026-07-02",
                "page": "https://example.com/blog/guide",
                "query": "how to choose product",
                "page_type": "blog",
                "keyword_intent": "Informational",
                "clicks": 10,
                "impressions": 500,
                "position": 15,
                "sessions": 80,
                "users": 70,
                "conversions": 1,
                "revenue": 50,
                "purchases": 0,
                "add_to_carts": 3,
                "checkouts": 1,
            },
        ]
    )


def build_raw_gsc_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "page": "https://example.com/product/a",
                "query": "buy product a",
                "clicks": 10,
                "impressions": 150,
                "position": 12,
            },
            {
                "date": "2026-07-02",
                "page": "https://example.com/product/a",
                "query": "buy product a",
                "clicks": 15,
                "impressions": 200,
                "position": 9,
            },
            {
                "date": "2026-07-02",
                "page": "https://example.com/blog/guide",
                "query": "how to choose product",
                "clicks": 10,
                "impressions": 500,
                "position": 15,
            },
        ]
    )


def test_build_page_opportunity_intelligence() -> None:
    recommendations = pd.DataFrame(
        [
            {
                "page": "https://example.com/product/a",
                "ScenarioLabel": "Content Refresh",
                "RecommendedAction": "Refresh Content",
                "ConfidenceLevel": "High",
                "PriorityTier": "High Priority",
                "ExpectedNetValue": 300,
                "EstimatedROI": 2.5,
                "BusinessDecisionScore": 82,
            }
        ]
    )

    result = build_page_opportunity_intelligence(
        integrated_dataframe=build_integrated_data(),
        recommendations=recommendations,
    )

    assert len(result) == 2
    assert "PageOpportunityScore" in result.columns
    assert "OpportunityPriority" in result.columns
    assert "RecommendedFocus" in result.columns

    product = result[
        result["page"].eq(
            "https://example.com/product/a"
        )
    ].iloc[0]

    assert product["Revenue"] == 1150
    assert product["Purchases"] == 9
    assert product["CurrentPosition"] == 9
    assert product["ScenarioLabel"] == "Content Refresh"


def test_build_keyword_opportunity_intelligence() -> None:
    result = build_keyword_opportunity_intelligence(
        seo_dataframe=build_raw_gsc_data(),
        integrated_dataframe=build_integrated_data(),
    )

    assert len(result) == 2
    assert "KeywordOpportunityScore" in result.columns
    assert "KeywordPriority" in result.columns
    assert "RecommendedKeywordAction" in result.columns

    keyword = result[
        result["query"].eq("buy product a")
    ].iloc[0]

    assert keyword["Clicks"] == 25
    assert keyword["Impressions"] == 350
    assert keyword["CurrentPosition"] == 9
    assert keyword["keyword_intent"] == "Transactional"
    assert keyword["Revenue"] == 1150


def test_build_product_category_opportunities() -> None:
    page_intelligence = build_page_opportunity_intelligence(
        integrated_dataframe=build_integrated_data(),
    )

    result = build_product_category_opportunities(
        page_intelligence
    )

    assert len(result) == 1
    assert result.iloc[0]["page_type"] == "product"


def test_high_opportunity_position_gets_priority_score() -> None:
    result = build_keyword_opportunity_intelligence(
        seo_dataframe=build_raw_gsc_data(),
        integrated_dataframe=build_integrated_data(),
    )

    product_keyword = result[
        result["query"].eq("buy product a")
    ].iloc[0]

    assert product_keyword["RankingOpportunityScore"] > 0
    assert product_keyword["KeywordOpportunityScore"] > 0
