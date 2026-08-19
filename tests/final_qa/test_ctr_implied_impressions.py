from __future__ import annotations

import numpy as np

from src.models.ctr_implied_impressions import (
    _bias_pct,
    _wape,
)


def test_wape_exact():
    actual = np.asarray([100.0, 200.0])
    assert _wape(actual, actual) == 0.0


def test_bias_exact():
    actual = np.asarray([100.0, 200.0])
    assert _bias_pct(actual, actual) == 0.0
