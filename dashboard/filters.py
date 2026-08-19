from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from dashboard.i18n import translate


# ============================================================
# DASHBOARD FILTER STATE
# ============================================================

@dataclass(frozen=True)
class DashboardFilters:
    """Shared dashboard filter state."""

    language: str
    preset: str
    start_date: date
    end_date: date
    comparison: str

    comparison_start_date: date | None = None
    comparison_end_date: date | None = None
    comparison_same_length: bool = True

    forecast_horizon_days: int = 7
    forecast_start_date: date | None = None
    forecast_end_date: date | None = None

    reference_date: date | None = None


# ============================================================
# FILTER OPTIONS
# ============================================================

DATE_PRESET_KEYS = [
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "last_7_days",
    "last_30_days",
    "last_60_days",
    "last_90_days",
    "this_month",
    "last_month",
    "this_quarter",
    "last_quarter",
    "this_year",
    "last_year",
    "custom_range",
]


COMPARISON_KEYS = [
    "no_comparison",
    "previous_period",
    "previous_month",
    "previous_quarter",
    "previous_year",
    "previous_year_ytd",
    "custom_comparison",
]


FORECAST_HORIZON_OPTIONS = [
    7,
    14,
    30,
    90,
    180,
    365,
]


# ============================================================
# DATE HELPERS
# ============================================================

def _month_start(value: date) -> date:
    return value.replace(day=1)


def _previous_month_start(value: date) -> date:
    first_day = _month_start(value)

    return (
        first_day - timedelta(days=1)
    ).replace(day=1)


def _quarter_start(value: date) -> date:
    quarter_month = (
        ((value.month - 1) // 3) * 3 + 1
    )

    return date(
        value.year,
        quarter_month,
        1,
    )


def _previous_quarter_start(value: date) -> date:
    current_start = _quarter_start(value)

    previous_quarter_end = (
        current_start - timedelta(days=1)
    )

    return _quarter_start(
        previous_quarter_end
    )


def _week_start(value: date) -> date:
    """Return Monday for the week containing value."""

    return (
        value
        - timedelta(days=value.weekday())
    )


def _safe_replace_year(
    value: date,
    year: int,
) -> date:
    """
    Move a date to another year without
    failing on 29 February.
    """

    day = min(
        value.day,
        monthrange(
            year,
            value.month,
        )[1],
    )

    return value.replace(
        year=year,
        day=day,
    )


def _period_length(
    start_date: date,
    end_date: date,
) -> int:
    """Return inclusive period length in days."""

    return (
        end_date - start_date
    ).days + 1


# ============================================================
# COMPARISON LENGTH
# ============================================================

def comparison_has_same_length(
    start_date: date,
    end_date: date,
    comparison_start_date: date | None,
    comparison_end_date: date | None,
) -> bool:
    """
    Return True when analysis and comparison
    periods have equal lengths.
    """

    if (
        comparison_start_date is None
        or comparison_end_date is None
    ):
        return True

    return (
        _period_length(
            start_date,
            end_date,
        )
        ==
        _period_length(
            comparison_start_date,
            comparison_end_date,
        )
    )


# ============================================================
# DATE PRESET RESOLUTION
# ============================================================

def resolve_date_range(
    preset: str,
    today: date | None = None,
) -> tuple[date, date]:
    """Resolve a preset into inclusive start and end dates."""

    current_date = today or date.today()

    if preset == "today":
        return (
            current_date,
            current_date,
        )

    if preset == "yesterday":
        yesterday = (
            current_date
            - timedelta(days=1)
        )

        return (
            yesterday,
            yesterday,
        )

    if preset == "this_week":
        return (
            _week_start(current_date),
            current_date,
        )

    if preset == "last_week":
        current_week_start = (
            _week_start(current_date)
        )

        previous_week_end = (
            current_week_start
            - timedelta(days=1)
        )

        previous_week_start = (
            previous_week_end
            - timedelta(days=6)
        )

        return (
            previous_week_start,
            previous_week_end,
        )

    if preset == "last_7_days":
        return (
            current_date
            - timedelta(days=6),
            current_date,
        )

    if preset == "last_30_days":
        return (
            current_date
            - timedelta(days=29),
            current_date,
        )

    if preset == "last_60_days":
        return (
            current_date
            - timedelta(days=59),
            current_date,
        )

    if preset == "last_90_days":
        return (
            current_date
            - timedelta(days=89),
            current_date,
        )

    if preset == "this_month":
        return (
            _month_start(current_date),
            current_date,
        )

    if preset == "last_month":
        start_date = _previous_month_start(
            current_date
        )

        end_date = (
            _month_start(current_date)
            - timedelta(days=1)
        )

        return (
            start_date,
            end_date,
        )

    if preset == "this_quarter":
        return (
            _quarter_start(current_date),
            current_date,
        )

    if preset == "last_quarter":
        start_date = _previous_quarter_start(
            current_date
        )

        end_date = (
            _quarter_start(current_date)
            - timedelta(days=1)
        )

        return (
            start_date,
            end_date,
        )

    if preset == "this_year":
        return (
            date(
                current_date.year,
                1,
                1,
            ),
            current_date,
        )

    if preset == "last_year":
        return (
            date(
                current_date.year - 1,
                1,
                1,
            ),
            date(
                current_date.year - 1,
                12,
                31,
            ),
        )

    # custom_range has no intrinsic range.
    # The toolbar handles its actual selected dates.
    return (
        current_date - timedelta(days=29),
        current_date,
    )


# ============================================================
# FORECAST RANGE
# ============================================================

def get_forecast_range(
    reference_date: date,
    horizon_days: int,
) -> tuple[date, date]:
    """
    Return the future window immediately after
    the latest real-data date.
    """

    horizon = int(
        horizon_days
    )

    if horizon not in FORECAST_HORIZON_OPTIONS:
        horizon = 7

    start_date = (
        reference_date
        + timedelta(days=1)
    )

    end_date = (
        start_date
        + timedelta(days=horizon - 1)
    )

    return (
        start_date,
        end_date,
    )


# ============================================================
# COMPARISON RANGE
# ============================================================

def get_comparison_range(
    start_date: date,
    end_date: date,
    comparison: str,
    custom_start_date: date | None = None,
    custom_end_date: date | None = None,
) -> tuple[date, date] | None:
    """
    Return an inclusive comparison period.
    """

    # IMPORTANT:
    # Normalize dates independently.
    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    start_date = normalized_start
    end_date = normalized_end

    # --------------------------------------------------------
    # No comparison
    # --------------------------------------------------------

    if comparison == "no_comparison":
        return None

    # --------------------------------------------------------
    # Previous equal-length period
    # --------------------------------------------------------

    if comparison == "previous_period":

        period_days = _period_length(
            start_date,
            end_date,
        )

        comparison_end = (
            start_date
            - timedelta(days=1)
        )

        comparison_start = (
            comparison_end
            - timedelta(days=period_days - 1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    # --------------------------------------------------------
    # Previous month
    # --------------------------------------------------------

    if comparison == "previous_month":

        comparison_start = (
            _previous_month_start(
                start_date
            )
        )

        comparison_end = (
            _month_start(start_date)
            - timedelta(days=1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    # --------------------------------------------------------
    # Previous quarter
    # --------------------------------------------------------

    if comparison == "previous_quarter":

        comparison_start = (
            _previous_quarter_start(
                start_date
            )
        )

        comparison_end = (
            _quarter_start(start_date)
            - timedelta(days=1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    # --------------------------------------------------------
    # Previous year
    # --------------------------------------------------------

    if comparison == "previous_year":

        return (
            _safe_replace_year(
                start_date,
                start_date.year - 1,
            ),
            _safe_replace_year(
                end_date,
                end_date.year - 1,
            ),
        )

    # --------------------------------------------------------
    # Previous year to same date
    # --------------------------------------------------------

    if comparison == "previous_year_ytd":

        return (
            _safe_replace_year(
                start_date,
                start_date.year - 1,
            ),
            _safe_replace_year(
                end_date,
                end_date.year - 1,
            ),
        )

    # --------------------------------------------------------
    # Custom comparison
    # --------------------------------------------------------

    if comparison == "custom_comparison":

        if (
            custom_start_date is None
            or custom_end_date is None
        ):
            return None

        return (
            min(
                custom_start_date,
                custom_end_date,
            ),
            max(
                custom_start_date,
                custom_end_date,
            ),
        )

    return None


# ============================================================
# LEGACY SIDEBAR FILTERS
# ============================================================

def render_sidebar_filters(
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
) -> DashboardFilters:
    """
    Render shared language and date filters.

    Legacy compatibility layer.
    """

    if "dashboard_language" not in st.session_state:
        st.session_state[
            "dashboard_language"
        ] = "tr"

    language_options = {
        "Türkçe": "tr",
        "English": "en",
    }

    current_language = st.session_state[
        "dashboard_language"
    ]

    selected_language_label = (
        st.sidebar.segmented_control(
            "Dil / Language",
            options=list(
                language_options.keys()
            ),
            default=(
                "Türkçe"
                if current_language == "tr"
                else "English"
            ),
            key="dashboard_language_selector",
        )
    )

    language = language_options.get(
        selected_language_label,
        current_language,
    )

    st.session_state[
        "dashboard_language"
    ] = language

    st.sidebar.divider()

    st.sidebar.subheader(
        translate(
            "filters",
            language,
        )
    )

    st.sidebar.caption(
        translate(
            "filter_help",
            language,
        )
    )

    # --------------------------------------------------------
    # Date presets
    # --------------------------------------------------------

    preset_labels = {
        translate(
            key,
            language,
        ): key
        for key in DATE_PRESET_KEYS
    }

    resolved_default_preset = (
        default_preset
        if default_preset in DATE_PRESET_KEYS
        else "last_30_days"
    )

    default_label = next(
        label
        for label, key
        in preset_labels.items()
        if key == resolved_default_preset
    )

    preset_options = list(
        preset_labels.keys()
    )

    selected_preset_label = (
        st.sidebar.selectbox(
            translate(
                "date_range",
                language,
            ),
            options=preset_options,
            index=preset_options.index(
                default_label
            ),
            key="dashboard_date_preset",
        )
    )

    preset = preset_labels[
        selected_preset_label
    ]

    default_start, default_end = (
        resolve_date_range(
            preset
        )
    )

    if preset == "custom_range":

        selected_dates = (
            st.sidebar.date_input(
                translate(
                    "date_range",
                    language,
                ),
                value=(
                    default_start,
                    default_end,
                ),
                key="dashboard_custom_date_range",
            )
        )

        if (
            isinstance(
                selected_dates,
                tuple,
            )
            and len(selected_dates) == 2
        ):
            start_date, end_date = (
                selected_dates
            )

        else:
            start_date, end_date = (
                default_start,
                default_end,
            )

    else:

        start_date, end_date = (
            default_start,
            default_end,
        )

        st.sidebar.caption(
            f"{start_date:%d.%m.%Y} — "
            f"{end_date:%d.%m.%Y}"
        )

    start_date, end_date = (
        min(start_date, end_date),
        max(start_date, end_date),
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_labels = {
        translate(
            key,
            language,
        ): key
        for key in COMPARISON_KEYS
    }

    resolved_default_comparison = (
        default_comparison
        if default_comparison in COMPARISON_KEYS
        else "previous_period"
    )

    default_comparison_label = next(
        label
        for label, key
        in comparison_labels.items()
        if key == resolved_default_comparison
    )

    comparison_options = list(
        comparison_labels.keys()
    )

    selected_comparison_label = (
        st.sidebar.selectbox(
            translate(
                "comparison",
                language,
            ),
            options=comparison_options,
            index=comparison_options.index(
                default_comparison_label
            ),
            key="dashboard_comparison",
        )
    )

    comparison = comparison_labels[
        selected_comparison_label
    ]

    custom_comparison_start = None
    custom_comparison_end = None

    if comparison == "custom_comparison":

        default_comparison_range = (
            get_comparison_range(
                start_date=start_date,
                end_date=end_date,
                comparison="previous_period",
            )
        )

        if default_comparison_range is None:
            default_comparison_range = (
                start_date,
                end_date,
            )

        selected_comparison_dates = (
            st.sidebar.date_input(
                translate(
                    "custom_comparison",
                    language,
                ),
                value=default_comparison_range,
                key=(
                    "dashboard_custom_"
                    "comparison_range"
                ),
            )
        )

        if (
            isinstance(
                selected_comparison_dates,
                tuple,
            )
            and len(
                selected_comparison_dates
            ) == 2
        ):
            (
                custom_comparison_start,
                custom_comparison_end,
            ) = selected_comparison_dates

        else:
            (
                custom_comparison_start,
                custom_comparison_end,
            ) = default_comparison_range

    comparison_range = (
        get_comparison_range(
            start_date=start_date,
            end_date=end_date,
            comparison=comparison,
            custom_start_date=(
                custom_comparison_start
            ),
            custom_end_date=(
                custom_comparison_end
            ),
        )
    )

    same_length = True

    if comparison_range is not None:

        comparison_start = (
            comparison_range[0]
        )

        comparison_end = (
            comparison_range[1]
        )

        st.sidebar.caption(
            f"{translate('comparison', language)}: "
            f"{comparison_start:%d.%m.%Y} — "
            f"{comparison_end:%d.%m.%Y}"
        )

        same_length = (
            comparison_has_same_length(
                start_date=start_date,
                end_date=end_date,
                comparison_start_date=(
                    comparison_start
                ),
                comparison_end_date=(
                    comparison_end
                ),
            )
        )

        if not same_length:

            st.sidebar.warning(
                translate(
                    "comparison_length_warning",
                    language,
                )
            )

    st.sidebar.caption(
        translate(
            "selection_applies_to_all",
            language,
        )
    )

    return DashboardFilters(
        language=language,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        comparison_start_date=(
            comparison_range[0]
            if comparison_range is not None
            else None
        ),
        comparison_end_date=(
            comparison_range[1]
            if comparison_range is not None
            else None
        ),
        comparison_same_length=same_length,
    )


# ============================================================
# SEO COMPATIBILITY LAYER
# ============================================================

from dataclasses import dataclass as _dataclass

import pandas as _pd


@_dataclass(frozen=True)
class DateRange:
    start_date: date
    end_date: date


def _resolve_dataframe_date_column(
    dataframe: _pd.DataFrame,
) -> str | None:
    """Resolve the most likely SEO date column."""

    candidates = [
        "Date",
        "date",
        "AnalysisDate",
        "analysis_date",
    ]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def filter_dataframe_by_date(
    dataframe: _pd.DataFrame,
    date_range: DateRange,
    date_column: str | None = None,
) -> _pd.DataFrame:
    """
    Filter a DataFrame to an inclusive SEO
    analysis date range.
    """

    if dataframe.empty:
        return dataframe.copy()

    resolved_date_column = (
        date_column
        or _resolve_dataframe_date_column(
            dataframe
        )
    )

    if resolved_date_column is None:
        return dataframe.copy()

    result = dataframe.copy()

    parsed_dates = _pd.to_datetime(
        result[resolved_date_column],
        errors="coerce",
    )

    mask = (
        parsed_dates.dt.date.ge(
            date_range.start_date
        )
        &
        parsed_dates.dt.date.le(
            date_range.end_date
        )
    )

    return (
        result.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


def filter_seo_dataframe(
    dataframe: _pd.DataFrame,
    date_range: DateRange | None = None,
    page_types: list[str] | None = None,
    keyword_intents: list[str] | None = None,
    pages: list[str] | None = None,
    date_column: str | None = None,
) -> _pd.DataFrame:
    """
    Apply shared SEO dashboard filters.

    Supports:
    - analysis date range
    - page type
    - keyword intent
    - page / URL
    """

    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    if date_range is not None:

        result = filter_dataframe_by_date(
            dataframe=result,
            date_range=date_range,
            date_column=date_column,
        )

    # --------------------------------------------------------
    # Page type filter
    # --------------------------------------------------------

    if page_types:

        page_type_column = None

        for candidate in [
            "page_type",
            "PageType",
            "pageType",
        ]:

            if candidate in result.columns:
                page_type_column = candidate
                break

        if page_type_column is not None:

            result = result[
                result[
                    page_type_column
                ]
                .astype(str)
                .isin(page_types)
            ]

    # --------------------------------------------------------
    # Keyword intent filter
    # --------------------------------------------------------

    if keyword_intents:

        intent_column = None

        for candidate in [
            "keyword_intent",
            "KeywordIntent",
            "intent",
            "Intent",
        ]:

            if candidate in result.columns:
                intent_column = candidate
                break

        if intent_column is not None:

            result = result[
                result[
                    intent_column
                ]
                .astype(str)
                .isin(keyword_intents)
            ]

    # --------------------------------------------------------
    # Page / URL filter
    # --------------------------------------------------------

    if pages:

        page_column = None

        for candidate in [
            "page",
            "Page",
            "url",
            "URL",
        ]:

            if candidate in result.columns:
                page_column = candidate
                break

        if page_column is not None:

            result = result[
                result[
                    page_column
                ]
                .astype(str)
                .isin(pages)
            ]

    return (
        result
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# LEGACY SIDEBAR COMPATIBILITY
# ============================================================

DATE_PRESETS = {
    "today": "Today",
    "yesterday": "Yesterday",
    "this_week": "This Week",
    "last_week": "Last Week",
    "last_7_days": "Last 7 Days",
    "last_30_days": "Last 30 Days",
    "last_60_days": "Last 60 Days",
    "last_90_days": "Last 90 Days",
    "this_month": "This Month",
    "last_month": "Last Month",
    "this_quarter": "This Quarter",
    "last_quarter": "Last Quarter",
    "this_year": "This Year",
    "last_year": "Last Year",
    "custom": "Custom",
}


COMPARISON_MODES = {
    "no_comparison": "No Comparison",
    "previous_period": "Previous Period",
    "previous_month": "Previous Month",
    "previous_quarter": "Previous Quarter",
    "previous_year": "Previous Year",
    "previous_year_ytd": "Previous Year YTD",
    "custom": "Custom Comparison",
}


def resolve_date_preset(
    preset: str,
    reference_date: date,
    available_start: date | None = None,
    available_end: date | None = None,
) -> DateRange:
    """
    Resolve an old sidebar preset to DateRange.
    """

    normalized_preset = (
        "custom_range"
        if preset == "custom"
        else preset
    )

    if (
        normalized_preset
        not in DATE_PRESET_KEYS
    ):
        normalized_preset = "last_30_days"

    start_date, end_date = (
        resolve_date_range(
            normalized_preset,
            today=reference_date,
        )
    )

    if available_start is not None:
        start_date = max(
            start_date,
            available_start,
        )

    if available_end is not None:
        end_date = min(
            end_date,
            available_end,
        )

    if start_date > end_date:

        fallback = (
            available_end
            or available_start
            or reference_date
        )

        start_date = fallback
        end_date = fallback

    return DateRange(
        start_date=start_date,
        end_date=end_date,
    )


def resolve_custom_range(
    start_date: date,
    end_date: date,
    available_start: date | None = None,
    available_end: date | None = None,
) -> DateRange:
    """
    Normalize and clamp an old custom sidebar range.
    """

    resolved_start = min(
        start_date,
        end_date,
    )

    resolved_end = max(
        start_date,
        end_date,
    )

    if available_start is not None:
        resolved_start = max(
            resolved_start,
            available_start,
        )

    if available_end is not None:
        resolved_end = min(
            resolved_end,
            available_end,
        )

    if resolved_start > resolved_end:

        fallback = (
            available_end
            or available_start
            or resolved_end
        )

        resolved_start = fallback
        resolved_end = fallback

    return DateRange(
        start_date=resolved_start,
        end_date=resolved_end,
    )


# ============================================================
# DIMENSION OPTIONS
# ============================================================

def _get_dimension_options(
    dataframe: _pd.DataFrame,
    candidates: list[str],
) -> list[str]:
    """
    Return unique, non-empty, sorted values
    from the first matching dimension column.
    """

    if dataframe is None or dataframe.empty:
        return []

    for candidate in candidates:

        if candidate not in dataframe.columns:
            continue

        values = (
            dataframe[candidate]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values.ne("")
        ]

        return sorted(
            values.unique().tolist()
        )

    return []


def get_page_type_options(
    dataframe: _pd.DataFrame,
) -> list[str]:
    """Return available SEO page-type values."""

    return _get_dimension_options(
        dataframe,
        [
            "page_type",
            "PageType",
            "pageType",
        ],
    )


def get_keyword_intent_options(
    dataframe: _pd.DataFrame,
) -> list[str]:
    """Return available keyword-intent values."""

    return _get_dimension_options(
        dataframe,
        [
            "keyword_intent",
            "KeywordIntent",
            "intent",
            "Intent",
        ],
    )


def get_page_options(
    dataframe: _pd.DataFrame,
) -> list[str]:
    """Return available page / URL values."""

    return _get_dimension_options(
        dataframe,
        [
            "page",
            "Page",
            "url",
            "URL",
        ],
    )
