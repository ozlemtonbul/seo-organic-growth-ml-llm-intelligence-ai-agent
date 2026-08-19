from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from dashboard.app_config import OUTPUT_DIR


# ============================================================
# FILE HELPERS
# ============================================================


def get_output_path(
    filename: str,
) -> Path:
    """
    Return the absolute path of an output file.
    """
    return (
        OUTPUT_DIR
        / filename
    ).resolve()


def output_exists(
    filename: str,
) -> bool:
    """
    Return whether an output file exists.
    """
    return get_output_path(
        filename
    ).exists()


def get_latest_output_time() -> str:
    """
    Return the latest modification time among generated outputs.

    Returns '-' when no output file exists.
    """
    if not OUTPUT_DIR.exists():
        return "-"

    output_files = [
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
    ]

    if not output_files:
        return "-"

    latest_path = max(
        output_files,
        key=lambda path: path.stat().st_mtime,
    )

    latest_time = datetime.fromtimestamp(
        latest_path.stat().st_mtime
    )

    return latest_time.strftime(
        "%d.%m.%Y %H:%M"
    )


def count_output_files() -> int:
    """
    Count generated output files.
    """
    if not OUTPUT_DIR.exists():
        return 0

    return sum(
        1
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
    )


# ============================================================
# DATA LOADING
# ============================================================


def load_csv(
    filename: str,
    parse_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Load one pipeline CSV output safely.

    Returns an empty DataFrame when the file does not exist.
    """
    path = get_output_path(
        filename
    )

    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        path
    )

    if parse_dates:
        for column in parse_dates:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(
                    dataframe[column],
                    errors="coerce",
                )

    return dataframe


def load_text(
    filename: str,
) -> str:
    """
    Load one text output safely.

    Returns an empty string when the file does not exist.
    """
    path = get_output_path(
        filename
    )

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# NUMERIC HELPERS
# ============================================================


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Convert a value safely to float.
    """
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value,
    default: int = 0,
) -> int:
    """
    Convert a value safely to integer.
    """
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return int(
            float(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_divide(
    numerator,
    denominator,
    default: float = 0.0,
) -> float:
    """
    Divide values safely.

    Returns default when denominator is zero or invalid.
    """
    numerator_value = safe_float(
        numerator,
        default=0.0,
    )

    denominator_value = safe_float(
        denominator,
        default=0.0,
    )

    if denominator_value == 0:
        return default

    return (
        numerator_value
        / denominator_value
    )


# ============================================================
# FORMATTING
# ============================================================


def format_number(
    value,
    decimals: int = 0,
) -> str:
    """
    Format numeric values with thousands separators.
    """
    numeric_value = safe_float(
        value
    )

    return f"{numeric_value:,.{decimals}f}"


def format_integer(
    value,
) -> str:
    """
    Format a value as integer with thousands separators.
    """
    return f"{safe_int(value):,}"


def format_percent(
    value,
    decimals: int = 1,
    value_is_ratio: bool = True,
) -> str:
    """
    Format percentage values.

    Examples:
    0.05 -> 5.0% when value_is_ratio=True
    5.0  -> 5.0% when value_is_ratio=False
    """
    numeric_value = safe_float(
        value
    )

    if value_is_ratio:
        numeric_value *= 100

    return (
        f"{numeric_value:.{decimals}f}%"
    )


def format_currency(
    value,
    symbol: str = "₺",
    decimals: int = 2,
) -> str:
    """
    Format business value or revenue.
    """
    numeric_value = safe_float(
        value
    )

    return (
        f"{symbol}{numeric_value:,.{decimals}f}"
    )


def format_position(
    value,
    decimals: int = 2,
) -> str:
    """
    Format Google Search Console average position.
    """
    return (
        f"{safe_float(value):.{decimals}f}"
    )


def format_duration(
    seconds,
) -> str:
    """
    Convert seconds into a readable duration.
    """
    total_seconds = max(
        0,
        safe_int(seconds),
    )

    minutes, remaining_seconds = divmod(
        total_seconds,
        60,
    )

    hours, minutes = divmod(
        minutes,
        60,
    )

    if hours > 0:
        return (
            f"{hours}h "
            f"{minutes}m "
            f"{remaining_seconds}s"
        )

    if minutes > 0:
        return (
            f"{minutes}m "
            f"{remaining_seconds}s"
        )

    return (
        f"{remaining_seconds}s"
    )


# ============================================================
# DATAFRAME HELPERS
# ============================================================


def ensure_datetime_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Ensure one DataFrame column is datetime.
    """
    result = dataframe.copy()

    if column in result.columns:
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        )

    return result


def sort_dataframe(
    dataframe: pd.DataFrame,
    column: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Sort a DataFrame safely when the requested column exists.
    """
    if dataframe.empty:
        return dataframe.copy()

    if column not in dataframe.columns:
        return dataframe.copy()

    return (
        dataframe
        .sort_values(
            column,
            ascending=ascending,
        )
        .reset_index(
            drop=True
        )
    )


def top_n(
    dataframe: pd.DataFrame,
    column: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return top N rows using one numeric column.
    """
    sorted_dataframe = sort_dataframe(
        dataframe=dataframe,
        column=column,
        ascending=ascending,
    )

    return (
        sorted_dataframe
        .head(
            max(
                0,
                int(n),
            )
        )
        .reset_index(
            drop=True
        )
    )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Return the first matching column name.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


# ============================================================
# KPI HELPERS
# ============================================================


def calculate_ctr(
    clicks,
    impressions,
) -> float:
    """
    Calculate CTR as a ratio.
    """
    return safe_divide(
        clicks,
        impressions,
    )


def calculate_conversion_rate(
    conversions,
    sessions,
) -> float:
    """
    Calculate conversion rate as a ratio.
    """
    return safe_divide(
        conversions,
        sessions,
    )


def calculate_revenue_per_session(
    revenue,
    sessions,
) -> float:
    """
    Calculate revenue per session.
    """
    return safe_divide(
        revenue,
        sessions,
    )


def calculate_change(
    current,
    previous,
) -> float:
    """
    Calculate percentage change as a ratio.

    Example:
    current=120, previous=100 -> 0.20
    """
    current_value = safe_float(
        current
    )

    previous_value = safe_float(
        previous
    )

    if previous_value == 0:
        return 0.0

    return (
        current_value
        - previous_value
    ) / previous_value


def calculate_position_change(
    current_position,
    previous_position,
) -> float:
    """
    Calculate SEO position improvement.

    Positive value means improvement because a lower
    average Search Console position is better.

    Example:
    previous=8, current=5 -> +3 improvement
    """
    current_value = safe_float(
        current_position
    )

    previous_value = safe_float(
        previous_position
    )

    return (
        previous_value
        - current_value
    )