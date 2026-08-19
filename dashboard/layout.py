from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import streamlit as st

from dashboard.filters import (
    COMPARISON_KEYS,
    DATE_PRESET_KEYS,
    FORECAST_HORIZON_OPTIONS,
    DashboardFilters,
    comparison_has_same_length,
    get_comparison_range,
    get_forecast_range,
    resolve_date_range,
)

from dashboard.i18n import translate
from dashboard.url_state import (
    sync_language_to_url,
    sync_language_widget_to_url,
)


def _forecast_horizon_option_label(
    days: int,
    language: str,
) -> str:
    """Human-friendly label for operational and strategic ML horizons."""
    mapping = {
        7: ("7 Günlük ML Tahmini", "7-Day ML Forecast"),
        14: ("14 Günlük ML Tahmini", "14-Day ML Forecast"),
        30: ("30 Günlük ML Tahmini", "30-Day ML Forecast"),
        90: ("3 Aylık ML Tahmini (90 Gün)", "3-Month ML Forecast (90 Days)"),
        180: ("6 Aylık ML Tahmini (180 Gün)", "6-Month ML Forecast (180 Days)"),
        365: ("1 Yıllık ML Tahmini (365 Gün)", "1-Year ML Forecast (365 Days)"),
    }

    labels = mapping.get(
        int(days),
        (
            f"{days} Günlük ML Tahmini",
            f"{days}-Day ML Forecast",
        ),
    )

    return labels[0] if language == "tr" else labels[1]


# ============================================================
# DASHBOARD CONTEXT
# ============================================================

@dataclass(frozen=True)
class DashboardContext:
    """Shared context used by all dashboard pages."""

    filters: DashboardFilters
    language: str


# ============================================================
# GLOBAL STYLES
# ============================================================

def inject_global_styles() -> None:
    """Load common dashboard styles."""

    st.markdown(
        """
<style>

/* ============================================================
   MAIN LAYOUT
   ============================================================ */

.block-container {
    max-width: 1450px;
    padding-top: 4.5rem !important;
    padding-bottom: 5rem;
}

header[data-testid="stHeader"] {
    height: 3.5rem;
}

div[data-testid="stAppViewContainer"] {
    overflow: visible;
}




/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}


/* ============================================================
   PLATFORM BRAND
   ============================================================ */

.platform-brand {
    padding: 0.35rem 0 0.85rem 0;
}

.platform-brand-title {
    font-size: 1.15rem;
    font-weight: 850;
    line-height: 1.25;
}

.platform-brand-subtitle {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 0.35rem;
}


/* ============================================================
   PAGE HERO
   ============================================================ */

.page-hero {
    padding: 1.55rem 1.75rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 20px;

    background:
        radial-gradient(
            circle at top right,
            rgba(59, 130, 246, 0.16),
            transparent 35%
        ),
        rgba(15, 23, 42, 0.78);

    margin-bottom: 1.2rem;
}

.page-eyebrow {
    color: #7dd3fc;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}

.page-title {
    font-size: clamp(1.9rem, 3.2vw, 2.7rem);
    font-weight: 850;
    line-height: 1.12;
    margin: 0;
}

.page-subtitle {
    color: #aab4c3;
    line-height: 1.6;
    margin-top: 0.65rem;
    margin-bottom: 0;
    max-width: 920px;
}


/* ============================================================
   FILTER SUMMARY
   ============================================================ */

.filter-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 1rem;
}

.filter-chip {
    padding: 0.38rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.48);
    color: #cbd5e1;
    font-size: 0.8rem;
}


/* ============================================================
   BUTTONS
   ============================================================ */

div[data-testid="stButton"] > button {
    border-radius: 12px;
    font-weight: 700;
}


/* ============================================================
   METRICS
   ============================================================ */

div[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 15px;
    padding: 0.8rem;
    background: rgba(15, 23, 42, 0.34);
    min-height: 112px;
}

div[data-testid="stMetricValue"] {
    font-size: clamp(1.45rem, 2.25vw, 2.15rem) !important;
    line-height: 1.15 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    line-height: 1.25 !important;
    white-space: normal !important;
}


/* ============================================================
   SECTION DIVIDER
   ============================================================ */

.dashboard-divider {
    margin: 1.5rem 0;
    border-top: 1px solid rgba(148, 163, 184, 0.16);
}


/* ============================================================
   DETERMINISTIC NOTICE
   ============================================================ */

.deterministic-notice {
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: rgba(15, 23, 42, 0.42);
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0.75rem 0 1rem 0;
}


/* ============================================================
   FOOTER
   ============================================================ */

.read-only-footer {
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.5;
    margin-top: 1rem;
}

</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMPATIBILITY HELPERS
# ============================================================

def render_divider() -> None:
    """Render common dashboard divider."""

    st.markdown(
        '<div class="dashboard-divider"></div>',
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    """Render common section header."""

    st.subheader(title)

    if subtitle:
        st.caption(subtitle)


def render_deterministic_notice(
    language: str = "tr",
) -> None:
    """
    Explain that deterministic calculations are separate
    from LLM commentary.
    """

    if language == "tr":
        message = (
            "ℹ️ **Veri-temelli sonuç:** Bu bölümdeki skorlar, "
            "dashboard'a yüklenmiş gerçek SEO/Analytics verilerinden "
            "hesaplanan deterministik sonuçlardır. AI yorumu varsa "
            "bu sonuçların üzerine açıklama üretir; temel veriyi değiştirmez."
        )

    else:
        message = (
            "ℹ️ **Data-driven result:** The scores in this section are "
            "deterministic calculations based on the real SEO/Analytics "
            "data loaded into the dashboard. If AI commentary is used, "
            "it explains these results but does not replace the underlying data."
        )

    st.markdown(
        f'<div class="deterministic-notice">{message}</div>',
        unsafe_allow_html=True,
    )


def render_footer(
    language: str = "tr",
    demo_mode: bool = False,
) -> None:
    """
    Footer compatibility helper.

    demo_mode is retained for compatibility with existing pages.
    """

    render_read_only_footer(language)

    if demo_mode:
        st.caption(
            (
                "Demo Mode"
                if language == "en"
                else "Demo Modu"
            )
        )


# ============================================================
# SIDEBAR BRAND
# ============================================================

def render_sidebar_brand(
    language: str,
) -> None:
    """Show platform information in sidebar."""

    html = (
        '<div class="platform-brand">'

        '<div class="platform-brand-title">'
        f"{translate('app_name', language)}"
        "</div>"

        '<div class="platform-brand-subtitle">'
        f"{translate('platform_subtitle', language)}"
        "</div>"

        "</div>"
    )

    st.sidebar.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE HEADER
# ============================================================

def render_page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "SEO & GEO Decision Intelligence",
) -> None:
    """Render common page header."""

    html = (
        '<div class="page-hero">'

        '<div class="page-eyebrow">'
        f"{eyebrow}"
        "</div>"

        '<h1 class="page-title">'
        f"{title}"
        "</h1>"

        '<p class="page-subtitle">'
        f"{subtitle}"
        "</p>"

        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def render_header(
    title: str,
    subtitle: str = "",
    eyebrow: str = "SEO & GEO Decision Intelligence",
) -> None:
    """
    Backward-compatible alias for older dashboard pages.

    Existing pages may still import render_header.
    Internally it uses the current render_page_header implementation.
    """

    render_page_header(
        title=title,
        subtitle=subtitle,
        eyebrow=eyebrow,
    )


# ============================================================
# DATE / COMPARISON LABELS
# ============================================================

def _date_preset_label(
    value: str,
    language: str,
) -> str:

    labels = {
        "today": (
            "Bugün",
            "Today",
        ),
        "yesterday": (
            "Dün",
            "Yesterday",
        ),
        "this_week": (
            "Bu Hafta",
            "This Week",
        ),
        "last_week": (
            "Geçen Hafta",
            "Last Week",
        ),
        "last_7_days": (
            "Son 7 Gün",
            "Last 7 Days",
        ),
        "last_30_days": (
            "Son 30 Gün",
            "Last 30 Days",
        ),
        "last_60_days": (
            "Son 60 Gün",
            "Last 60 Days",
        ),
        "last_90_days": (
            "Son 90 Gün",
            "Last 90 Days",
        ),
        "this_month": (
            "Bu Ay",
            "This Month",
        ),
        "last_month": (
            "Geçen Ay",
            "Last Month",
        ),
        "this_quarter": (
            "Bu Çeyrek",
            "This Quarter",
        ),
        "last_quarter": (
            "Geçen Çeyrek",
            "Last Quarter",
        ),
        "this_year": (
            "Bu Yıl",
            "This Year",
        ),
        "last_year": (
            "Geçen Yıl",
            "Last Year",
        ),
        "custom_range": (
            "Özel Tarih Aralığı",
            "Custom Date Range",
        ),
    }

    pair = labels.get(
        value,
        (
            value.replace(
                "_",
                " ",
            ).title(),
            value.replace(
                "_",
                " ",
            ).title(),
        ),
    )

    return (
        pair[0]
        if language == "tr"
        else pair[1]
    )


def _comparison_label(
    value: str,
    language: str,
) -> str:

    labels = {
        "no_comparison": (
            "Karşılaştırma Yok",
            "No Comparison",
        ),
        "previous_period": (
            "Önceki Aynı Uzunluktaki Dönem",
            "Previous Equal-Length Period",
        ),
        "previous_month": (
            "Önceki Ay",
            "Previous Month",
        ),
        "previous_quarter": (
            "Önceki Çeyrek",
            "Previous Quarter",
        ),
        "previous_year": (
            "Geçen Yıl Aynı Dönem",
            "Same Period Last Year",
        ),
        "previous_year_ytd": (
            "Geçen Yıl Aynı Tarihe Kadar",
            "Previous Year to Same Date",
        ),
        "custom_comparison": (
            "Özel Karşılaştırma Tarihleri",
            "Custom Comparison Dates",
        ),
    }

    pair = labels.get(
        value,
        (
            value.replace(
                "_",
                " ",
            ).title(),
            value.replace(
                "_",
                " ",
            ).title(),
        ),
    )

    return (
        pair[0]
        if language == "tr"
        else pair[1]
    )


# ============================================================
# GLOBAL FILTER BAR
# ============================================================

def render_interactive_filter_bar(
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
    reference_date: date | None = None,
) -> DashboardFilters:
    """
    Manage shared dashboard date, comparison,
    projection and language filters.
    """

    if "dashboard_language" not in st.session_state:
        st.session_state["dashboard_language"] = "tr"

    date_reference = (
        reference_date
        or (
            date.today()
            - timedelta(days=1)
        )
    )

    # ========================================================
    # Applied values
    # ========================================================

    st.session_state.setdefault(
        "dashboard_selected_preset",
        default_preset,
    )

    st.session_state.setdefault(
        "dashboard_selected_comparison",
        default_comparison,
    )

    st.session_state.setdefault(
        "dashboard_selected_forecast_horizon",
        7,
    )

    if (
        st.session_state["dashboard_selected_preset"]
        not in DATE_PRESET_KEYS
    ):
        st.session_state["dashboard_selected_preset"] = default_preset

    if (
        st.session_state["dashboard_selected_comparison"]
        not in COMPARISON_KEYS
    ):
        st.session_state["dashboard_selected_comparison"] = default_comparison

    if (
        st.session_state["dashboard_selected_forecast_horizon"]
        not in FORECAST_HORIZON_OPTIONS
    ):
        st.session_state["dashboard_selected_forecast_horizon"] = 7

    # ========================================================
    # Draft toolbar values
    # ========================================================

    st.session_state.setdefault(
        "dashboard_toolbar_date_preset",
        st.session_state["dashboard_selected_preset"],
    )

    st.session_state.setdefault(
        "dashboard_toolbar_comparison",
        st.session_state["dashboard_selected_comparison"],
    )

    st.session_state.setdefault(
        "dashboard_toolbar_forecast_horizon",
        st.session_state["dashboard_selected_forecast_horizon"],
    )

    # ========================================================
    # APPLY
    # ========================================================

    def _apply_filters() -> None:

        st.session_state["dashboard_selected_preset"] = (
            st.session_state.get(
                "dashboard_toolbar_date_preset",
                default_preset,
            )
        )

        st.session_state["dashboard_selected_comparison"] = (
            st.session_state.get(
                "dashboard_toolbar_comparison",
                default_comparison,
            )
        )

        st.session_state["dashboard_selected_forecast_horizon"] = (
            st.session_state.get(
                "dashboard_toolbar_forecast_horizon",
                7,
            )
        )

        if (
            st.session_state["dashboard_selected_preset"]
            == "custom_range"
        ):

            selected = st.session_state.get(
                "dashboard_toolbar_custom_date_range"
            )

            if (
                isinstance(
                    selected,
                    (tuple, list),
                )
                and len(selected) == 2
            ):
                st.session_state[
                    "dashboard_applied_custom_date_range"
                ] = tuple(selected)

        if (
            st.session_state["dashboard_selected_comparison"]
            == "custom_comparison"
        ):

            selected = st.session_state.get(
                "dashboard_toolbar_custom_comparison_range"
            )

            if (
                isinstance(
                    selected,
                    (tuple, list),
                )
                and len(selected) == 2
            ):
                st.session_state[
                    "dashboard_applied_custom_comparison_range"
                ] = tuple(selected)

        st.session_state[
            "dashboard_filters_just_applied"
        ] = True

    # ========================================================
    # RESET
    # ========================================================

    def _reset_filters() -> None:

        st.session_state[
            "dashboard_selected_preset"
        ] = default_preset

        st.session_state[
            "dashboard_selected_comparison"
        ] = default_comparison

        st.session_state[
            "dashboard_selected_forecast_horizon"
        ] = 7

        st.session_state[
            "dashboard_toolbar_date_preset"
        ] = default_preset

        st.session_state[
            "dashboard_toolbar_comparison"
        ] = default_comparison

        st.session_state[
            "dashboard_toolbar_forecast_horizon"
        ] = 7

        st.session_state.pop(
            "dashboard_applied_custom_date_range",
            None,
        )

        st.session_state.pop(
            "dashboard_applied_custom_comparison_range",
            None,
        )

        st.session_state.pop(
            "dashboard_toolbar_custom_date_range",
            None,
        )

        st.session_state.pop(
            "dashboard_toolbar_custom_comparison_range",
            None,
        )

        st.session_state[
            "dashboard_filters_just_reset"
        ] = True

    # ========================================================
    # LANGUAGE + FILTERS
    # ========================================================

    language_column, date_column = st.columns([1, 3])

    with language_column:

        language = st.selectbox(
            (
                "Dil"
                if st.session_state.get("dashboard_language", "tr") == "tr"
                else "Language"
            ),
            options=[
                "tr",
                "en",
            ],
            format_func=lambda value: (
                "Dil: Türkçe"
                if value == "tr"
                else "Language: English"
            ),
            key="dashboard_language",
            label_visibility="collapsed",
            on_change=sync_language_widget_to_url,
        )

        language = sync_language_to_url(
            language
        )

    with date_column:

        with st.expander(
            (
                "Tarih ve Projeksiyon"
                if language == "tr"
                else "Date and Projection"
            ),
            expanded=False,
        ):

            c1, c2, c3 = st.columns(3)

            # ------------------------------------------------
            # Analysis period
            # ------------------------------------------------

            with c1:

                draft_preset = st.selectbox(
                    (
                        "Analiz Dönemi"
                        if language == "tr"
                        else "Analysis Period"
                    ),
                    options=DATE_PRESET_KEYS,
                    format_func=lambda value:
                        _date_preset_label(
                            value,
                            language,
                        ),
                    key="dashboard_toolbar_date_preset",
                )

            # ------------------------------------------------
            # Comparison
            # ------------------------------------------------

            with c2:

                draft_comparison = st.selectbox(
                    (
                        "Karşılaştırma"
                        if language == "tr"
                        else "Comparison"
                    ),
                    options=COMPARISON_KEYS,
                    format_func=lambda value:
                        _comparison_label(
                            value,
                            language,
                        ),
                    key="dashboard_toolbar_comparison",
                )

            # ------------------------------------------------
            # Forecast horizon
            # ------------------------------------------------

            with c3:

                draft_horizon = st.selectbox(
                    (
                        "ML Tahmin Ufku"
                        if language == "tr"
                        else "ML Forecast Horizon"
                    ),
                    options=FORECAST_HORIZON_OPTIONS,
                    format_func=lambda days:
                        _forecast_horizon_option_label(
                            days,
                            language,
                        ),
                    key="dashboard_toolbar_forecast_horizon",
                )

            # =================================================
            # ANALYSIS DATE RANGE
            # =================================================

            draft_start, draft_end = resolve_date_range(
                draft_preset,
                today=date_reference,
            )

            if draft_preset == "custom_range":

                default_custom = st.session_state.get(
                    "dashboard_applied_custom_date_range",
                    (
                        draft_start,
                        draft_end,
                    ),
                )

                selected_dates = st.date_input(
                    (
                        "Özel Analiz Tarihleri"
                        if language == "tr"
                        else "Custom Analysis Dates"
                    ),
                    value=default_custom,
                    max_value=date_reference,
                    key="dashboard_toolbar_custom_date_range",
                )

                if (
                    isinstance(
                        selected_dates,
                        (tuple, list),
                    )
                    and len(selected_dates) == 2
                ):

                    draft_start, draft_end = selected_dates

            draft_start, draft_end = (
                min(
                    draft_start,
                    draft_end,
                ),
                max(
                    draft_start,
                    draft_end,
                ),
            )

            draft_end = min(
                draft_end,
                date_reference,
            )

            draft_start = min(
                draft_start,
                draft_end,
            )

            # =================================================
            # CUSTOM COMPARISON
            # =================================================

            if draft_comparison == "custom_comparison":

                default_comparison_range = get_comparison_range(
                    draft_start,
                    draft_end,
                    "previous_period",
                )

                assert default_comparison_range is not None

                default_custom_comparison = st.session_state.get(
                    "dashboard_applied_custom_comparison_range",
                    default_comparison_range,
                )

                st.date_input(
                    (
                        "Özel Karşılaştırma Tarihleri"
                        if language == "tr"
                        else "Custom Comparison Dates"
                    ),
                    value=default_custom_comparison,
                    max_value=date_reference,
                    key="dashboard_toolbar_custom_comparison_range",
                )

            # =================================================
            # FORECAST PREVIEW
            # =================================================

            (
                preview_forecast_start,
                preview_forecast_end,
            ) = get_forecast_range(
                date_reference,
                int(draft_horizon),
            )

            st.caption(
                (
                    f"Son gerçek veri tarihi: "
                    f"**{date_reference:%d.%m.%Y}** · "
                    f"Seçili projeksiyon dönemi: "
                    f"**{preview_forecast_start:%d.%m.%Y} – "
                    f"{preview_forecast_end:%d.%m.%Y}**"
                    if language == "tr"
                    else
                    f"Latest real-data date: "
                    f"**{date_reference:%d.%m.%Y}** · "
                    f"Selected projection period: "
                    f"**{preview_forecast_start:%d.%m.%Y} – "
                    f"{preview_forecast_end:%d.%m.%Y}**"
                )
            )

            st.caption(
                (
                    "🤖 **ML Tahmin Ufku**, günlük takvim adımıyla eğitilmiş "
                    "üretim modelinin tahminlerini 7/14/30 gün ileriye yinelemeli "
                    "olarak üretir. Her tahmin edilen gün, bir sonraki günün model "
                    "girdisine dahil edilir; değerler tek-adımlı tahminin basitçe "
                    "gün sayısıyla çarpılmasıyla oluşturulmaz."
                    if language == "tr"
                    else
                    "🤖 **ML Forecast Horizon** uses a production model trained on "
                    "calendar-day steps and recursively forecasts 7/14/30 days ahead. "
                    "Each predicted day becomes part of the next day's model input; "
                    "the horizon values are not created by simply multiplying a "
                    "one-step prediction by the number of days."
                )
            )

            # =================================================
            # BUTTONS
            # =================================================

            b1, b2 = st.columns(2)

            with b1:

                st.button(
                    (
                        "Analizi Uygula"
                        if language == "tr"
                        else "Apply Analysis"
                    ),
                    type="primary",
                    use_container_width=True,
                    on_click=_apply_filters,
                    key="dashboard_apply_filters",
                )

            with b2:

                st.button(
                    (
                        "Filtreleri Sıfırla"
                        if language == "tr"
                        else "Reset Filters"
                    ),
                    use_container_width=True,
                    on_click=_reset_filters,
                    key="dashboard_reset_filters",
                )

            st.caption(
                (
                    "ℹ️ Seçimler yalnızca **Analizi Uygula** düğmesine "
                    "bastığınızda tarih-duyarlı performans bölümlerine uygulanır. "
                    "Model, SHAP, teknik denetim ve öneri snapshot'ları son başarılı "
                    "pipeline çalışmasını temsil eder. Dil ve uygulanmış filtreler "
                    "sayfalar arasında korunur."
                    if language == "tr"
                    else
                    "ℹ️ Selections affect date-aware performance sections only "
                    "after you press **Apply Analysis**. Model, SHAP, technical-audit "
                    "and recommendation snapshots represent the latest successful "
                    "pipeline run. Language and applied filters persist across pages."
                )
            )

    # ========================================================
    # APPLIED STATE
    # ========================================================

    preset = st.session_state[
        "dashboard_selected_preset"
    ]

    comparison = st.session_state[
        "dashboard_selected_comparison"
    ]

    forecast_horizon_days = int(
        st.session_state[
            "dashboard_selected_forecast_horizon"
        ]
    )

    start_date, end_date = resolve_date_range(
        preset,
        today=date_reference,
    )

    if preset == "custom_range":

        applied_custom = st.session_state.get(
            "dashboard_applied_custom_date_range"
        )

        if (
            isinstance(
                applied_custom,
                (tuple, list),
            )
            and len(applied_custom) == 2
        ):

            start_date, end_date = applied_custom

    start_date, end_date = (
        min(
            start_date,
            end_date,
        ),
        max(
            start_date,
            end_date,
        ),
    )

    end_date = min(
        end_date,
        date_reference,
    )

    start_date = min(
        start_date,
        end_date,
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    custom_comparison_start = None
    custom_comparison_end = None

    if comparison == "custom_comparison":

        applied_custom_comparison = st.session_state.get(
            "dashboard_applied_custom_comparison_range"
        )

        if (
            isinstance(
                applied_custom_comparison,
                (tuple, list),
            )
            and len(applied_custom_comparison) == 2
        ):

            (
                custom_comparison_start,
                custom_comparison_end,
            ) = applied_custom_comparison

    comparison_range = get_comparison_range(
        start_date,
        end_date,
        comparison,
        custom_start_date=custom_comparison_start,
        custom_end_date=custom_comparison_end,
    )

    # ========================================================
    # FORECAST
    # ========================================================

    (
        forecast_start_date,
        forecast_end_date,
    ) = get_forecast_range(
        date_reference,
        forecast_horizon_days,
    )

    # ========================================================
    # COMPARISON LENGTH CHECK
    # ========================================================

    comparison_same_length = comparison_has_same_length(
        start_date=start_date,
        end_date=end_date,
        comparison_start_date=(
            comparison_range[0]
            if comparison_range
            else None
        ),
        comparison_end_date=(
            comparison_range[1]
            if comparison_range
            else None
        ),
    )

    # ========================================================
    # FINAL FILTER OBJECT
    # ========================================================

    return DashboardFilters(
        language=language,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        comparison_start_date=(
            comparison_range[0]
            if comparison_range
            else None
        ),
        comparison_end_date=(
            comparison_range[1]
            if comparison_range
            else None
        ),
        comparison_same_length=comparison_same_length,
        forecast_horizon_days=forecast_horizon_days,
        forecast_start_date=forecast_start_date,
        forecast_end_date=forecast_end_date,
        reference_date=date_reference,
    )


# ============================================================
# FILTER SUMMARY
# ============================================================

def render_filter_summary(
    filters: DashboardFilters,
) -> None:
    """Show summary of applied filters."""

    comparison_label = _comparison_label(
        filters.comparison,
        filters.language,
    )

    def _fmt(value: date) -> str:

        if filters.language == "tr":
            return value.strftime("%d.%m.%Y")

        return value.strftime("%d %b %Y")

    comparison_dates = ""

    if (
        filters.comparison_start_date
        and filters.comparison_end_date
    ):

        comparison_dates = (
            " · "
            f"{_fmt(filters.comparison_start_date)}"
            " — "
            f"{_fmt(filters.comparison_end_date)}"
        )

    language_label = (
        "Türkçe"
        if filters.language == "tr"
        else "English"
    )

    forecast_label = (
        f"Projeksiyon: "
        f"{filters.forecast_horizon_days} gün"
        if filters.language == "tr"
        else
        f"Projection: "
        f"{filters.forecast_horizon_days} days"
    )

    html = (
        '<div class="filter-summary">'

        '<div class="filter-chip">'
        f"📅 {_fmt(filters.start_date)}"
        f" — {_fmt(filters.end_date)}"
        "&nbsp;·&nbsp;"
        f"- {comparison_label}"
        f"{comparison_dates}"
        "</div>"

        '<div class="filter-chip">'
        f"🔮 {forecast_label}"
        "</div>"

        '<div class="filter-chip">'
        f"🌐 {language_label}"
        "</div>"

        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD INITIALIZATION
# ============================================================

def initialize_dashboard(
    page_title: str,
    page_icon: str,
    title: str,
    subtitle: str,
    eyebrow: str = "SEO & GEO Decision Intelligence",
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
    reference_date: date | None = None,
) -> DashboardContext:
    """
    Prepare common dashboard settings.
    """

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
    )

    inject_global_styles()

    filters = render_interactive_filter_bar(
        default_preset=default_preset,
        default_comparison=default_comparison,
        reference_date=reference_date,
    )
    render_sidebar_brand(
        filters.language
    )

    render_page_header(
        title=title,
        subtitle=subtitle,
        eyebrow=eyebrow,
    )

    render_filter_summary(
        filters
    )

    return DashboardContext(
        filters=filters,
        language=filters.language,
    )


# ============================================================
# READ-ONLY FOOTER
# ============================================================

def render_read_only_footer(
    language: str,
) -> None:
    """Render common information footer."""

    html = (
        '<div class="read-only-footer">'
        f"{translate('read_only', language)}"
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# LOCALIZED TEXT
# ============================================================

def localized_text(
    language: str,
    turkish: str,
    english: str,
) -> str:
    """Return text according to active language."""

    return (
        turkish
        if language == "tr"
        else english
    )


# ============================================================
# STATUS FORMATTER
# ============================================================

def format_status_value(
    value: Any,
    language: str,
) -> str:
    """Format common system status values."""

    if isinstance(value, bool):

        return (
            translate(
                "online",
                language,
            )
            if value
            else
            translate(
                "offline",
                language,
            )
        )

    return str(value)
