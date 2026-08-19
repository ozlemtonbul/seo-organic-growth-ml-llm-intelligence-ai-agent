
import numpy as np
import pandas as pd
from src.models.strategic_champion_selector import _wape, _bias_pct, _forecast_weekday_median

def test_wape_exact():
    s = pd.Series([10.0,20.0,30.0])
    assert _wape(s,s) == 0.0

def test_bias_exact():
    s = pd.Series([10.0,20.0,30.0])
    assert _bias_pct(s,s) == 0.0

def test_weekday_median_nonnegative():
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    s = pd.Series(100.0 + 10.0*np.sin(np.arange(120)*2*np.pi/7.0), index=idx)
    future = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=90, freq="D")
    pred = _forecast_weekday_median(s, future, weeks=8)
    assert len(pred) == 90
    assert np.all(pred >= 0)


def test_wape_ignores_index_labels_and_uses_position():
    actual = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
    predicted = pd.Series(
        [11.0, 18.0, 33.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    expected = (1.0 + 2.0 + 3.0) / 60.0 * 100.0
    assert round(_wape(actual, predicted), 6) == round(expected, 6)
