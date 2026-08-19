from __future__ import annotations

from datetime import date

import pandas as pd

from dashboard.demo_runtime import summarize_demo_frame


def test_demo_analysis_processes_bundled_shape_without_live_api():
    frame = pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-02", "2026-08-10"],
            "page": ["https://demo.example.com/a", "https://demo.example.com/a", "https://demo.example.com/b"],
            "clicks": [10, 15, 20],
            "impressions": [100, 150, 250],
        }
    )

    result = summarize_demo_frame(
        frame,
        date(2026, 8, 1),
        date(2026, 8, 2),
    )

    assert result["success"] is True
    assert result["rows"] == 2
    assert result["days"] == 2
    assert result["pages"] == 1
    assert result["clicks"] == 25.0
    assert result["impressions"] == 250.0
    assert result["ctr"] == 0.1
    assert result["source"] == "bundled_anonymized_gsc_ga4_demo"
    assert result["start_date"] == "2026-08-01"
    assert result["end_date"] == "2026-08-02"
