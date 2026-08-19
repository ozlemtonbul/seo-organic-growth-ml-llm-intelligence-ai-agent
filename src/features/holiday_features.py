from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import pandas as pd


def get_turkey_public_holidays(
    year: int,
) -> Dict[str, str]:
    """
    Return Turkish public holidays for a given year.

    Fixed-date public holidays are generated for every year.
    Religious holiday dates are included for supported years.
    """
    fixed_holidays = {
        f"{year}-01-01": "New Year",
        f"{year}-04-23": (
            "National Sovereignty and Children's Day"
        ),
        f"{year}-05-01": "Labour and Solidarity Day",
        f"{year}-05-19": (
            "Commemoration of Ataturk, Youth and Sports Day"
        ),
        f"{year}-07-15": (
            "Democracy and National Unity Day"
        ),
        f"{year}-08-30": "Victory Day",
        f"{year}-10-29": "Republic Day",
    }

    movable_holidays = {
        2024: {
            "Eid al-Fitr": [
                "2024-04-09",
                "2024-04-10",
                "2024-04-11",
                "2024-04-12",
            ],
            "Eid al-Adha": [
                "2024-06-15",
                "2024-06-16",
                "2024-06-17",
                "2024-06-18",
                "2024-06-19",
            ],
        },
        2025: {
            "Eid al-Fitr": [
                "2025-03-29",
                "2025-03-30",
                "2025-03-31",
                "2025-04-01",
            ],
            "Eid al-Adha": [
                "2025-06-05",
                "2025-06-06",
                "2025-06-07",
                "2025-06-08",
                "2025-06-09",
            ],
        },
        2026: {
            "Eid al-Fitr": [
                "2026-03-19",
                "2026-03-20",
                "2026-03-21",
                "2026-03-22",
            ],
            "Eid al-Adha": [
                "2026-05-26",
                "2026-05-27",
                "2026-05-28",
                "2026-05-29",
                "2026-05-30",
            ],
        },
    }

    result = dict(fixed_holidays)

    for holiday_name, holiday_dates in (
        movable_holidays.get(year, {}).items()
    ):
        for holiday_date in holiday_dates:
            result[holiday_date] = holiday_name

    return result


def build_holiday_map(
    date_from: str,
    date_to: str,
) -> Dict[str, str]:
    """
    Build a holiday lookup covering the supplied date range.
    """
    start_year = int(date_from[:4])
    end_year = int(date_to[:4])

    if start_year > end_year:
        raise ValueError(
            "The start year cannot be later than the end year."
        )

    result: Dict[str, str] = {}

    for year in range(start_year, end_year + 1):
        result.update(
            get_turkey_public_holidays(year)
        )

    return result


def add_holiday_features(
    dataframe: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Add holiday and pre-holiday indicators to SEO observations.
    """
    result = dataframe.copy()

    if "date" not in result.columns:
        raise ValueError(
            "The input DataFrame must include a date column."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        result["date"]
    ):
        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

    date_strings = result["date"].dt.strftime(
        "%Y-%m-%d"
    )

    result["is_holiday"] = (
        date_strings.isin(holiday_map).astype(int)
    )

    result["holiday_name"] = (
        date_strings.map(holiday_map).fillna("")
    )

    holiday_dates = {
        datetime.strptime(
            holiday_date,
            "%Y-%m-%d",
        ).date()
        for holiday_date in holiday_map
    }

    result["is_pre_holiday"] = (
        result["date"]
        .dt.date
        .map(
            lambda current_date: int(
                any(
                    current_date + timedelta(days=offset)
                    in holiday_dates
                    for offset in range(1, 4)
                )
            )
        )
    )

    return result