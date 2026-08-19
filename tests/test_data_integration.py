from __future__ import annotations

import pandas as pd

from src.features.data_integration import (
    add_page_classifications,
    aggregate_ga4_page_data,
    aggregate_gsc_page_data,
    merge_gsc_and_ga4,
    normalize_page_key,
)


def build_gsc_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "page": (
                    "https://example.com/product/a?ref=test"
                ),
                "query": "ürün fiyat",
                "clicks": 10,
                "impressions": 100,
                "position": 7,
            },
            {
                "date": "2026-07-01",
                "page": (
                    "https://example.com/product/a"
                ),
                "query": "ürün satın al",
                "clicks": 5,
                "impressions": 50,
                "position": 6,
            },
        ]
    )


def build_ga4_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "landing_page": (
                    "/product/a?source=organic"
                ),
                "sessions": 20,
                "users": 15,
                "engaged_sessions": 12,
                "engagement_rate": 0.60,
                "average_session_duration": 90,
                "conversions": 2,
                "revenue": 300,
                "purchases": 2,
                "add_to_carts": 5,
                "checkouts": 3,
            }
        ]
    )


def test_normalize_page_key() -> None:
    assert normalize_page_key(
        "https://example.com/Product/A?ref=test"
    ) == "/product/a"

    assert normalize_page_key(
        "/product/a?source=organic"
    ) == "/product/a"


def test_aggregate_gsc_page_data() -> None:
    result = aggregate_gsc_page_data(
        build_gsc_test_data()
    )

    assert len(result) == 2
    assert result["clicks"].sum() == 15
    assert result["impressions"].sum() == 150


def test_aggregate_ga4_page_data() -> None:
    result = aggregate_ga4_page_data(
        build_ga4_test_data()
    )

    assert len(result) == 1
    assert result.iloc[0]["page_key"] == "/product/a"
    assert result.iloc[0]["sessions"] == 20
    assert result.iloc[0]["revenue"] == 300


def test_add_page_classifications() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/product/sample"
                ),
                "query": "ürün fiyat satın al",
            }
        ]
    )

    result = add_page_classifications(
        dataframe
    )

    assert result.iloc[0]["page_type"] == "product"
    assert (
        result.iloc[0]["keyword_intent"]
        == "Transactional"
    )


def test_merge_gsc_and_ga4() -> None:
    result = merge_gsc_and_ga4(
        gsc_dataframe=build_gsc_test_data(),
        ga4_dataframe=build_ga4_test_data(),
    )

    assert len(result) == 2
    assert result["clicks"].sum() == 15
    assert result["impressions"].sum() == 150
    assert set(
        result["sessions"].tolist()
    ) == {20}
    assert set(
        result["revenue"].tolist()
    ) == {300}
    assert set(
        result["page_type"].tolist()
    ) == {"product"}
    assert set(
        result["keyword_intent"].tolist()
    ) == {"Transactional"}


def test_merge_without_ga4_adds_zero_metrics() -> None:
    result = merge_gsc_and_ga4(
        gsc_dataframe=build_gsc_test_data(),
        ga4_dataframe=pd.DataFrame(),
    )

    assert result["sessions"].sum() == 0
    assert result["revenue"].sum() == 0
