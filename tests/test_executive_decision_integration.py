from datetime import date

import pandas as pd

from dashboard.services.executive_service import build_executive_decision_intelligence


def test_executive_decision_filters_selected_and_comparison_periods():
    frame = pd.DataFrame(
        {
            "Date": ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
            "Page": ["/a", "/a", "/a", "/a"],
            "Clicks": [10, 20, 30, 40],
            "Impressions": [100, 200, 300, 400],
            "Position": [5, 5, 4, 4],
        }
    )
    result = build_executive_decision_intelligence(
        dataframe=frame,
        recommendations=pd.DataFrame(),
        start_date=date(2026, 8, 3),
        end_date=date(2026, 8, 4),
        comparison_start_date=date(2026, 8, 1),
        comparison_end_date=date(2026, 8, 2),
        language="tr",
        forecast_horizon_days=7,
    )
    assert result.comparison.current["clicks"] == 70.0
    assert result.comparison.previous["clicks"] == 30.0
    assert round(result.comparison.deltas["clicks_pct"], 2) == 133.33
