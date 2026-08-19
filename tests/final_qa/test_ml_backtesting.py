from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ml_backtesting import (
    ALL_HORIZONS,
    _mae,
    _rmse,
    _safe_pct,
    _wape,
    source_coverage,
    prepare_model_source,
    minimum_coverage_days,
    standardize_gsc_page_daily,
)


def test_all_horizons_contract():
    assert ALL_HORIZONS == (
        7,
        14,
        30,
        90,
        180,
        365,
    )


def test_metric_helpers():
    actual = pd.Series(
        [
            100.0,
            200.0,
            300.0,
        ]
    )

    predicted = pd.Series(
        [
            110.0,
            190.0,
            330.0,
        ]
    )

    assert round(
        _mae(
            actual,
            predicted,
        ),
        4,
    ) == round(
        50.0
        / 3.0,
        4,
    )

    expected_rmse = np.sqrt(
        (
            10.0 ** 2
            + 10.0 ** 2
            + 30.0 ** 2
        )
        / 3.0
    )

    assert round(
        _rmse(
            actual,
            predicted,
        ),
        4,
    ) == round(
        expected_rmse,
        4,
    )

    assert round(
        _wape(
            actual,
            predicted,
        ),
        4,
    ) == round(
        50.0
        / 600.0
        * 100.0,
        4,
    )

    assert (
        _safe_pct(
            10.0,
            0.0,
        )
        == 0.0
    )


def test_raw_gsc_duplicate_rows_are_aggregated_page_daily():
    raw = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "page": [
                "/a",
                "/a",
                "/a",
            ],
            "query": [
                "shoe",
                "kids shoe",
                "shoe",
            ],
            "clicks": [
                10,
                5,
                8,
            ],
            "impressions": [
                100,
                50,
                80,
            ],
            "position": [
                2.0,
                4.0,
                3.0,
            ],
        }
    )

    daily = (
        standardize_gsc_page_daily(
            raw
        )
    )

    first = (
        daily.loc[
            daily[
                "date"
            ].eq(
                pd.Timestamp(
                    "2026-01-01"
                )
            )
        ]
        .iloc[
            0
        ]
    )

    assert len(
        daily
    ) == 2

    assert float(
        first[
            "clicks"
        ]
    ) == 15.0

    assert float(
        first[
            "impressions"
        ]
    ) == 150.0

    expected_position = (
        2.0
        * 100.0
        + 4.0
        * 50.0
    ) / 150.0

    assert round(
        float(
            first[
                "position"
            ]
        ),
        6,
    ) == round(
        expected_position,
        6,
    )


def test_source_coverage_calendar_days():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-12-31",
                ]
            ),
            "page": [
                "/a",
                "/a",
            ],
        }
    )

    coverage = (
        source_coverage(
            data
        )
    )

    assert (
        coverage[
            "calendar_days"
        ]
        == 366
    )



def test_integrated_model_source_preserves_extra_features():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "page": ["/a", "/a"],
            "clicks": [10, 12],
            "impressions": [100, 110],
            "position": [3.0, 2.8],
            "sessions": [50, 55],
            "revenue": [1000, 1200],
        }
    )

    prepared = prepare_model_source(
        frame
    )

    assert "sessions" in prepared.columns
    assert "revenue" in prepared.columns
    assert len(prepared) == 2


def test_strategic_coverage_requirements_are_independent():
    assert minimum_coverage_days(90) == 180
    assert minimum_coverage_days(180) == 360
    assert minimum_coverage_days(365) == 730
