from __future__ import annotations

import pandas as pd

from src.features.blog_intelligence import (
    build_blog_keyword_content_gaps,
)


def test_content_gap_action_changes_for_corporate_landing_page() -> None:
    keywords = pd.DataFrame([
        {
            "page": "/contact/",
            "query": "cocuk ayakkabisi nasil secilir",
            "keyword_intent": "Informational",
            "KeywordOpportunityScore": 80,
            "Impressions": 5000,
            "Clicks": 20,
            "CurrentPosition": 12,
        }
    ])
    pages = pd.DataFrame([
        {
            "page": "/contact/",
            "page_type": "category",
        }
    ])

    result = build_blog_keyword_content_gaps(
        keywords,
        pages,
    )

    assert result.loc[0, "CurrentLandingPageType"] == "corporate"
    assert "corporate/utility page" in result.loc[0, "ContentGapReason"]
    assert "dedicated guide" in result.loc[0, "RecommendedContentAction"]


def test_content_gap_action_changes_for_product_landing_page() -> None:
    keywords = pd.DataFrame([
        {
            "page": "/urun/test-p-123",
            "query": "ilk adim ayakkabisi nasil secilir",
            "keyword_intent": "Informational",
            "KeywordOpportunityScore": 70,
            "Impressions": 2000,
            "Clicks": 40,
            "CurrentPosition": 8,
        }
    ])
    pages = pd.DataFrame([
        {
            "page": "/urun/test-p-123",
            "page_type": "product",
        }
    ])

    result = build_blog_keyword_content_gaps(
        keywords,
        pages,
    )

    assert result.loc[0, "CurrentLandingPageType"] == "product"
    assert "product page" in result.loc[0, "ContentGapReason"]
    assert "supporting guide/FAQ" in result.loc[0, "RecommendedContentAction"]
