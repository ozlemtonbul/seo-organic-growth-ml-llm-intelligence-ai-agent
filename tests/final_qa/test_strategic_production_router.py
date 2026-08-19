
from __future__ import annotations

import pandas as pd

from src.models.strategic_production_router import (
    _resolve_column,
)


def test_column_resolver_case_insensitive():
    frame = pd.DataFrame(
        {
            "ForecastDate": [],
            "PredictedClicks": [],
        }
    )
    assert _resolve_column(
        frame,
        ["forecastdate"],
    ) == "ForecastDate"
