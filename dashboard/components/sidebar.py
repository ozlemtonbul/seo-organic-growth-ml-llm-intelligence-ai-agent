from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.filters import (
    COMPARISON_MODES,
    DATE_PRESETS,
    DateRange,
    get_keyword_intent_options,
    get_page_options,
    get_page_type_options,
    resolve_custom_range,
    resolve_date_preset,
)
from dashboard.i18n import (
    normalize_language,
    t,
)


def render_date_preset_selector(
    language: str,
    default_preset: str = "last_30_days",
) -> str:
    """
    Render date preset selector.
    """
    resolved_language = (
        normalize_language(
            language
        )
    )

    options = list(
        DATE_PRESETS.keys()
    )

    default_index = (
        options.index(
            default_preset
        )
        if default_preset in options
        else 0
    )

    def format_preset(
        key: str,
    ) -> str:
        translated = t(
            key,
            resolved_language,
            default=DATE_PRESETS.get(
                key,
                key,
            ),
        )

        return translated

    return st.sidebar.selectbox(
        t(
            "date_range",
            resolved_language,
        ),
        options=options,
        index=default_index,
        format_func=format_preset,
        key="seo_date_preset",
    )


def render_comparison_selector(
    language: str,
    default_comparison: str = "no_comparison",
) -> str:
    """
    Render period comparison selector.
    """
    resolved_language = normalize_language(
        language
    )

    options = list(
        COMPARISON_MODES.keys()
    )

    default_index = (
        options.index(
            default_comparison
        )
        if default_comparison in options
        else 0
    )

    return st.sidebar.selectbox(
        t(
            "comparison",
            resolved_language,
        ),
        options=options,
        index=default_index,
        format_func=lambda key: t(
            key,
            resolved_language,
            default=COMPARISON_MODES.get(
                key,
                key,
            ),
        ),
        key="seo_comparison_mode",
    )


def render_date_range_filter(
    language: str,
    reference_date: date,
    available_start: date | None = None,
    available_end: date | None = None,
    default_preset: str = "last_30_days",
) -> tuple[
    DateRange,
    str,
]:
    """
    Render date preset/custom date controls.
    """
    preset = render_date_preset_selector(
        language=language,
        default_preset=default_preset,
    )

    if preset != "custom":
        return (
            resolve_date_preset(
                preset=preset,
                reference_date=reference_date,
                available_start=available_start,
                available_end=available_end,
            ),
            preset,
        )

    fallback_start = (
        available_start
        or reference_date
    )

    fallback_end = (
        available_end
        or reference_date
    )

    selected_start = st.sidebar.date_input(
        t(
            "start_date",
            language,
        ),
        value=fallback_start,
        key="seo_custom_start",
    )

    selected_end = st.sidebar.date_input(
        t(
            "end_date",
            language,
        ),
        value=fallback_end,
        key="seo_custom_end",
    )

    return (
        resolve_custom_range(
            start_date=selected_start,
            end_date=selected_end,
            available_start=available_start,
            available_end=available_end,
        ),
        preset,
    )


def render_seo_dimension_filters(
    dataframe: pd.DataFrame,
    language: str,
) -> dict[str, list[str]]:
    """
    Render page type, intent and page filters.
    """
    page_types = (
        get_page_type_options(
            dataframe
        )
    )

    intents = (
        get_keyword_intent_options(
            dataframe
        )
    )

    pages = (
        get_page_options(
            dataframe
        )
    )

    selected_page_types = (
        st.sidebar.multiselect(
            t(
                "page_type",
                language,
            ),
            options=page_types,
            key="seo_page_types",
        )
        if page_types
        else []
    )

    selected_intents = (
        st.sidebar.multiselect(
            t(
                "keyword_intent",
                language,
            ),
            options=intents,
            key="seo_keyword_intents",
        )
        if intents
        else []
    )

    selected_pages = (
        st.sidebar.multiselect(
            t(
                "page",
                language,
            ),
            options=pages,
            key="seo_pages",
        )
        if pages
        else []
    )

    return {
        "page_types": (
            selected_page_types
        ),
        "keyword_intents": (
            selected_intents
        ),
        "pages": selected_pages,
    }


def render_filter_summary(
    date_range: DateRange,
    comparison_mode: str,
    language: str,
) -> None:
    """
    Render compact active-filter summary.
    """
    st.sidebar.caption(
        (
            f"{date_range.start_date:%d.%m.%Y}"
            " – "
            f"{date_range.end_date:%d.%m.%Y}"
        )
    )

    st.sidebar.caption(
        t(
            comparison_mode,
            language,
            default=comparison_mode,
        )
    )