from __future__ import annotations

import pandas as pd

from src.features.geo_intelligence import build_geo_ai_visibility_intelligence


def test_geo_intelligence_prioritizes_low_readiness_high_value_page() -> None:
    pages = pd.DataFrame([
        {
            "page": "/kategori/cocuk",
            "EntityName": "Cocuk",
            "page_type": "category",
            "PageOpportunityScore": 90,
            "DemandScore": 90,
            "CommerceScore": 85,
            "Revenue": 50000,
            "Purchases": 50,
            "AddToCarts": 150,
        },
        {
            "page": "/blog/rehber",
            "EntityName": "Rehber",
            "page_type": "blog",
            "PageOpportunityScore": 35,
            "DemandScore": 40,
            "CommerceScore": 5,
        },
    ])

    result = build_geo_ai_visibility_intelligence(pages)

    assert result.loc[0, "page"] == "/kategori/cocuk"
    assert (
        result.loc[0, "GEOOpportunityScore"]
        > result.loc[1, "GEOOpportunityScore"]
    )
    assert result.loc[0, "GEOPriority"] in {
        "High",
        "Medium",
    }


def test_geo_intelligence_detects_observable_readiness_signals() -> None:
    pages = pd.DataFrame([
        {
            "page": "/blog/rehber",
            "page_type": "blog",
            "PageOpportunityScore": 60,
            "DemandScore": 70,
            "CommerceScore": 20,
        }
    ])

    latest = pd.DataFrame([
        {
            "page": "/blog/rehber",
            "h1": "Cocuk ayakkabisi nasil secilir?",
            "meta_description": "Pratik cocuk ayakkabisi secim rehberi.",
            "schema_type": "Article",
            "brand": "Example",
            "author": "SEO Team",
            "date_modified": "2026-08-01",
            "content": (
                "Nasil secilir? Sık sorulan sorular. "
                + "detay " * 300
            ),
        }
    ])

    result = build_geo_ai_visibility_intelligence(
        pages,
        latest,
    )

    assert result.loc[0, "GEOReadinessScore"] >= 75
    assert (
        "No major readiness gap"
        in result.loc[0, "GEOMissingSignals"]
    )
    assert (
        "readiness proxy"
        in result.loc[0, "AIMeasurementStatus"]
    )


def test_geo_actions_are_page_type_specific() -> None:
    pages = pd.DataFrame([
        {
            "page": "/product/a",
            "page_type": "product",
            "PageOpportunityScore": 50,
            "DemandScore": 50,
            "CommerceScore": 50,
        }
    ])

    result = build_geo_ai_visibility_intelligence(
        pages
    )

    assert (
        "product attributes"
        in result.loc[
            0,
            "GEORecommendedActions",
        ]
    )
    assert (
        "Product structured data"
        in result.loc[
            0,
            "GEORecommendedActions",
        ]
    )


def test_geo_uses_detail_signal_when_base_column_is_empty() -> None:
    pages = pd.DataFrame([
        {
            "page": "/blog/rehber",
            "page_type": "blog",
            "h1": "",
            "PageOpportunityScore": 60,
            "DemandScore": 60,
            "CommerceScore": 10,
        }
    ])
    latest = pd.DataFrame([
        {
            "page": "/blog/rehber",
            "h1": "Gercek H1",
            "meta_description": "Aciklama",
            "schema_type": "Article",
            "brand": "Demo Store",
            "author": "SEO Team",
            "date_modified": "2026-08-01",
            "content": "Nasil secilir? " + "detay " * 300,
        }
    ])

    result = build_geo_ai_visibility_intelligence(
        pages,
        latest,
    )

    assert result.loc[0, "GEOReadinessScore"] >= 75


def test_geo_classifies_corporate_url_separately() -> None:
    pages = pd.DataFrame([
        {
            "page": "/contact/",
            "page_type": "category",
            "PageOpportunityScore": 40,
            "DemandScore": 20,
            "CommerceScore": 0,
        }
    ])

    result = build_geo_ai_visibility_intelligence(
        pages
    )

    assert result.loc[0, "page_type"] == "corporate"
    assert (
        "organisation/entity purpose"
        in result.loc[
            0,
            "GEORecommendedActions",
        ]
    )
