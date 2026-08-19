from __future__ import annotations

from datetime import date

from scripts.extract_gsc_historical_page_daily import (
    _date_windows,
)


def test_gsc_history_windows_cover_range_without_gaps():
    windows = list(
        _date_windows(
            date(2026, 1, 1),
            date(2026, 1, 20),
            window_days=7,
        )
    )

    assert windows == [
        (date(2026, 1, 1), date(2026, 1, 7)),
        (date(2026, 1, 8), date(2026, 1, 14)),
        (date(2026, 1, 15), date(2026, 1, 20)),
    ]
