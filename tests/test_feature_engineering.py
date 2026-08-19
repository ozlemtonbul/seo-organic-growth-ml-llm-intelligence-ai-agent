from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_engineering import (
    GA4_FEATURE_COLUMNS,
    LAG_METRICS,
    add_lag_features,
    add_time_features,
    compute_kpis,
    get_feature_columns,
    prepare_training_data,
)
from src.features.holiday_features import (
    add_holiday_features,
    build_holiday_map,
    get_turkey_public_holidays,
)


def build_feature_test_data(
    number_of_days: int = 5,
) -> pd.DataFrame:
    """
    Build a small page-level SEO dataset for feature tests.
    """
    dates = pd.date_range(
        "2026-07-10",
        periods=number_of_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "page": [
                "https://example.com/product/a"
            ]
            * number_of_days,
            "date": dates,
            "clicks": [
                10 + index
                for index in range(number_of_days)
            ],
            "impressions": [
                100 + index * 10
                for index in range(number_of_days)
            ],
            "position": [
                8 - index * 0.5
                for index in range(number_of_days)
            ],
            "ctr": [
                0.10
            ]
            * number_of_days,
        }
    )


def test_feature_constant_lengths() -> None:
    assert len(
        GA4_FEATURE_COLUMNS
    ) == 10

    assert len(
        LAG_METRICS
    ) == 5


def test_compute_kpis() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "page": "https://example.com/product/a",
                "date": "2026-07-10",
                "clicks": 10,
                "impressions": 100,
                "position": 5,
                "ctr": 0.10,
            }
        ]
    )

    result = compute_kpis(
        dataframe
    )

    assert result.iloc[0]["CTR"] == pytest.approx(
        0.10
    )

    assert result.iloc[0]["TrafficValue"] == pytest.approx(
        5.0
    )

    assert result.iloc[0]["RankStrength"] == pytest.approx(
        0.20
    )

    assert result.iloc[0]["VisibilityScore"] == pytest.approx(
        10.0
    )

    assert result.iloc[0]["Top3Flag"] == 0
    assert result.iloc[0]["Top10Flag"] == 1
    assert result.iloc[0]["Page2Flag"] == 0


def test_compute_kpis_adds_missing_ga4_columns() -> None:
    result = compute_kpis(
        build_feature_test_data(
            number_of_days=1
        )
    )

    for column in GA4_FEATURE_COLUMNS:
        assert column in result.columns
        assert result.iloc[0][column] == 0


def test_compute_kpis_requires_core_columns() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "clicks": 10,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        compute_kpis(
            dataframe
        )


def test_add_time_features() -> None:
    dataframe = build_feature_test_data(
        number_of_days=1
    )

    result = add_time_features(
        dataframe
    )

    assert "day_of_week" in result.columns
    assert "day_of_month" in result.columns
    assert "month_num" in result.columns
    assert "quarter" in result.columns
    assert "is_weekend" in result.columns

    assert result.iloc[0]["month_num"] == 7
    assert result.iloc[0]["quarter"] == 3


def test_add_time_features_requires_date() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "clicks": 10,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="date column",
    ):
        add_time_features(
            dataframe
        )


def test_add_lag_features() -> None:
    dataframe = compute_kpis(
        build_feature_test_data(
            number_of_days=4
        )
    )

    result = add_lag_features(
        dataframe
    )

    assert "clicks_lag_1" in result.columns
    assert "clicks_lag_7_avg" in result.columns
    assert "position_change" in result.columns

    assert result.iloc[0]["clicks_lag_1"] == 0
    assert result.iloc[1]["clicks_lag_1"] == 10
    assert result.iloc[1]["clicks_change"] == 1


def test_get_turkey_public_holidays() -> None:
    holidays = get_turkey_public_holidays(
        2026
    )

    assert holidays[
        "2026-01-01"
    ] == "New Year"

    assert holidays[
        "2026-10-29"
    ] == "Republic Day"

    assert "2026-03-20" in holidays


def test_build_holiday_map() -> None:
    holiday_map = build_holiday_map(
        "2025-12-01",
        "2026-01-31",
    )

    assert "2025-12-01" not in holiday_map
    assert "2026-01-01" in holiday_map


def test_build_holiday_map_rejects_reverse_years() -> None:
    with pytest.raises(
        ValueError,
        match="start year",
    ):
        build_holiday_map(
            "2027-01-01",
            "2026-12-31",
        )


def test_add_holiday_features() -> None:
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-12",
                    "2026-07-15",
                ]
            )
        }
    )

    holiday_map = build_holiday_map(
        "2026-07-12",
        "2026-07-15",
    )

    result = add_holiday_features(
        dataframe,
        holiday_map,
    )

    assert result.iloc[0]["is_pre_holiday"] == 1
    assert result.iloc[1]["is_holiday"] == 1
    assert (
        result.iloc[1]["holiday_name"]
        == "Democracy and National Unity Day"
    )


def test_prepare_training_data() -> None:
    dataframe = build_feature_test_data(
        number_of_days=5
    )

    holiday_map = build_holiday_map(
        "2026-07-10",
        "2026-07-14",
    )

    result = prepare_training_data(
        dataframe,
        holiday_map,
    )

    assert len(result) == 4

    assert "target_clicks_next" in result.columns
    assert "target_impressions_next" in result.columns
    assert "is_holiday" in result.columns
    assert "is_pre_holiday" in result.columns

    assert result.iloc[0]["target_clicks_next"] == 11
    assert result.iloc[0]["target_impressions_next"] == 110


def test_feature_column_count() -> None:
    assert len(
        get_feature_columns(
            with_holiday=False
        )
    ) == 44

    assert len(
        get_feature_columns(
            with_holiday=True
        )
    ) == 46