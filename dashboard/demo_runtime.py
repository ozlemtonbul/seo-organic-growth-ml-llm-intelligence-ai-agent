from __future__ import annotations

from datetime import date

import pandas as pd

from dashboard.app_config import OUTPUT_DIR


DEMO_INTEGRATED_FILE = "seo_integrated_data.csv"


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def summarize_demo_frame(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    """Summarize a selected period from the bundled anonymized demo data."""
    if frame.empty:
        return {"success": False, "reason": "empty"}

    date_column = _first_existing(frame, ("date", "Date"))
    if date_column is None:
        return {"success": False, "reason": "date_column_missing"}

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = dates.dt.date.ge(start_date) & dates.dt.date.le(end_date)
    selected = frame.loc[mask].copy()
    selected_dates = dates.loc[mask].dropna()

    if selected.empty:
        return {"success": False, "reason": "period_empty"}

    click_column = _first_existing(
        selected,
        ("clicks", "Clicks", "CurrentClicks"),
    )
    impression_column = _first_existing(
        selected,
        ("impressions", "Impressions", "CurrentImpressions"),
    )
    page_column = _first_existing(
        selected,
        ("page", "Page", "url", "URL"),
    )

    clicks = (
        float(pd.to_numeric(selected[click_column], errors="coerce").fillna(0).sum())
        if click_column
        else 0.0
    )
    impressions = (
        float(pd.to_numeric(selected[impression_column], errors="coerce").fillna(0).sum())
        if impression_column
        else 0.0
    )
    pages = int(selected[page_column].nunique()) if page_column else 0
    days = int(selected_dates.dt.date.nunique())

    return {
        "success": True,
        "rows": int(len(selected)),
        "days": days,
        "pages": pages,
        "clicks": clicks,
        "impressions": impressions,
        "ctr": (clicks / impressions) if impressions > 0 else 0.0,
        "source": "bundled_anonymized_gsc_ga4_demo",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def run_demo_analysis(start_date: date, end_date: date) -> dict[str, object]:
    """Load and process the public sanitized dataset. No live API call is made."""
    path = OUTPUT_DIR / DEMO_INTEGRATED_FILE
    if not path.exists():
        return {"success": False, "reason": "demo_file_missing"}

    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return {"success": False, "reason": "demo_file_unreadable"}

    return summarize_demo_frame(frame, start_date, end_date)
