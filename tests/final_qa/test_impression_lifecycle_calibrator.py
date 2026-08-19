from __future__ import annotations

import numpy as np

from src.models.impression_lifecycle_calibrator import (
    _bias_pct,
    _wape,
)


def test_wape_exact():
    actual = np.asarray(
        [
            100.0,
            200.0,
        ]
    )

    assert (
        _wape(
            actual,
            actual,
        )
        == 0.0
    )


def test_bias_exact():
    actual = np.asarray(
        [
            100.0,
            200.0,
        ]
    )

    assert (
        _bias_pct(
            actual,
            actual,
        )
        == 0.0
    )


def test_33_samples_are_allowed_for_development_walk_forward():
    import pandas as pd
    from src.models.impression_lifecycle_calibrator import _walk_forward_rank

    rows = []
    for i in range(33):
        rows.append(
            {
                "OriginDate": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i * 7),
                "imp_sum_7": 1000 + i,
                "imp_mean_7": 140 + i * 0.1,
                "imp_median_7": 140 + i * 0.1,
                "imp_std_7": 5.0,
                "active_mean_7": 100.0,
                "active_last_7": 100.0,
                "imp_sum_14": 2000 + i,
                "imp_mean_14": 140 + i * 0.1,
                "imp_median_14": 140 + i * 0.1,
                "imp_std_14": 5.0,
                "active_mean_14": 100.0,
                "active_last_14": 100.0,
                "imp_sum_28": 4000 + i,
                "imp_mean_28": 140 + i * 0.1,
                "imp_median_28": 140 + i * 0.1,
                "imp_std_28": 5.0,
                "active_mean_28": 100.0,
                "active_last_28": 100.0,
                "imp_sum_56": 8000 + i,
                "imp_mean_56": 140 + i * 0.1,
                "imp_median_56": 140 + i * 0.1,
                "imp_std_56": 5.0,
                "active_mean_56": 100.0,
                "active_last_56": 100.0,
                "imp_sum_90": 12600 + i,
                "imp_mean_90": 140 + i * 0.1,
                "imp_median_90": 140 + i * 0.1,
                "imp_std_90": 5.0,
                "active_mean_90": 100.0,
                "active_last_90": 100.0,
                "active_page_growth_28": 1.0,
                "recent28_active_pages": 100.0,
                "prev28_active_pages": 100.0,
                "impression_momentum_28": 1.0,
                "imp_per_active_page_28": 40.0,
                "top10_share_56": 0.2,
                "page_median_imp_56": 10.0,
                "page_p90_imp_56": 20.0,
                "known_pages": 1000.0,
                "new_pages_28": 20.0,
                "new_pages_56": 40.0,
                "month": float((i % 12) + 1),
                "dow": float(i % 7),
                "month_sin": 0.0,
                "month_cos": 1.0,
                "doy_sin": 0.0,
                "doy_cos": 1.0,
                "TargetTotal": 12500 + i * 5,
                "Recent28RunRate": 12600 + i,
                "Recent56RunRate": 12600 + i,
                "Recent90RunRate": 12600 + i,
                "TargetVsRecent28": 1.0,
            }
        )

    ranking = _walk_forward_rank(pd.DataFrame(rows))
    assert not ranking.empty
    assert ranking["FoldCount"].min() >= 2
