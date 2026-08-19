from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.strategic_direct_calibrator import (
    _bias_pct,
    _wape,
)


def test_wape_uses_positional_alignment():
    actual = pd.Series(
        [
            10.0,
            20.0,
            30.0,
        ],
        index=[
            0,
            1,
            2,
        ],
    )

    predicted = pd.Series(
        [
            11.0,
            18.0,
            33.0,
        ],
        index=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ),
    )

    expected = (
        1.0
        + 2.0
        + 3.0
    ) / 60.0 * 100.0

    assert round(
        _wape(
            actual,
            predicted,
        ),
        6,
    ) == round(
        expected,
        6,
    )


def test_bias_exact_prediction():
    actual = pd.Series(
        [
            10.0,
            20.0,
        ]
    )

    assert (
        _bias_pct(
            actual,
            actual,
        )
        == 0.0
    )
