from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.strategic_forecast_stabilizer import (
    _build_damped_weekday_anchor,
    _reliability_weights,
    _wape,
)


def test_reliability_weight_decays_and_is_bounded():
    weights = _reliability_weights(
        horizon_days=180,
        base_r2=0.95,
    )

    assert len(weights) == 180
    assert weights[0] <= 0.90
    assert weights[-1] >= 0.15
    assert weights[-1] < weights[0]


def test_anchor_uses_positive_stable_values():
    index = pd.date_range(
        "2025-01-01",
        periods=200,
        freq="D",
    )

    series = pd.Series(
        100.0
        + np.sin(
            np.arange(
                200
            )
            / 7.0
        )
        * 5.0,
        index=index,
    )

    future = pd.Series(
        pd.date_range(
            "2025-07-20",
            periods=90,
            freq="D",
        )
    )

    anchor, diagnostics = (
        _build_damped_weekday_anchor(
            training_series=series,
            forecast_dates=future,
        )
    )

    assert len(anchor) == 90
    assert np.all(anchor >= 0)
    assert diagnostics["RecentLevel"] > 0


def test_wape_zero_for_exact_prediction():
    actual = pd.Series(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert _wape(
        actual,
        actual,
    ) == 0.0
