from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Tuple

from config.settings import SETTINGS


ROLLING_DATE_MODES: Dict[str, int] = {
    "last_30_days": 30,
    "last_60_days": 60,
    "last_90_days": 90,
    "last_180_days": 180,
    "last_365_days": 365,
}


def parse_iso_date(
    value: str,
    field_name: str,
) -> date:
    """
    Parse a date value using YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()

    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format. "
            f"Received: {value!r}"
        ) from exc


def resolve_custom_date_range() -> Tuple[str, str]:
    """
    Resolve a manually configured date range.
    """
    if not SETTINGS.date_from or not SETTINGS.date_to:
        raise ValueError(
            "DATE_FROM and DATE_TO are required "
            "when DATE_MODE is custom."
        )

    start_date = parse_iso_date(
        SETTINGS.date_from,
        "DATE_FROM",
    )

    end_date = parse_iso_date(
        SETTINGS.date_to,
        "DATE_TO",
    )

    if start_date > end_date:
        raise ValueError(
            "DATE_FROM cannot be later than DATE_TO."
        )

    return (
        start_date.isoformat(),
        end_date.isoformat(),
    )


def resolve_rolling_date_range(
    number_of_days: int,
) -> Tuple[str, str]:
    """
    Resolve a rolling date range ending after the configured
    API data-delay period.
    """
    if number_of_days < 1:
        raise ValueError(
            "The rolling date range must contain at least one day."
        )

    end_date = (
        datetime.now().date()
        - timedelta(days=SETTINGS.api_data_delay_days)
    )

    start_date = (
        end_date
        - timedelta(days=number_of_days - 1)
    )

    return (
        start_date.isoformat(),
        end_date.isoformat(),
    )


def resolve_date_range() -> Tuple[str, str]:
    """
    Resolve the reporting date range using DATE_MODE.

    Supported modes:
        custom
        last_30_days
        last_60_days
        last_90_days
        last_180_days
        last_365_days
    """
    date_mode = SETTINGS.date_mode.lower()

    if date_mode == "custom":
        return resolve_custom_date_range()

    if date_mode in ROLLING_DATE_MODES:
        return resolve_rolling_date_range(
            ROLLING_DATE_MODES[date_mode]
        )

    raise ValueError(
        "Unsupported DATE_MODE. "
        f"Received: {date_mode!r}"
    )