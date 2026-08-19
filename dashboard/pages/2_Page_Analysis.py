from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROJECT_ROOT_STR = str(
    PROJECT_ROOT
)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(
        PROJECT_ROOT_STR
    )

sys.path.insert(
    0,
    PROJECT_ROOT_STR,
)


# ============================================================
# IMPORTS
# ============================================================

from dashboard.app_config import (
    APP_TITLE,
)
from dashboard.components import (
    render_export_buttons,
    render_kpi_row,
    render_line_chart,
    render_recommendations_table,
)
from dashboard.i18n import (
    t,
)
from dashboard.localization import render_localized_dataframe
from dashboard.layout import (
    initialize_dashboard,
    localized_text,
)
from dashboard.services import (
    build_executive_kpis,
    filter_analysis_data,
    get_available_date_bounds,
    load_analysis_data,
)
from dashboard.utils import (
    format_currency,
    format_integer,
    format_percent,
    format_position,
)


# ============================================================
# LOCAL UI / PAGE HELPERS
# ============================================================


def render_section_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    st.subheader(title)

    if subtitle:
        st.caption(subtitle)


def render_divider() -> None:
    st.divider()


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def _safe_numeric(
    value: object,
    default: float = 0.0,
) -> float:
    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return default

    return float(parsed)


def _normalize_page_type(
    value: object,
) -> str:
    raw = str(value or "").strip().lower()

    aliases = {
        "blog": "blog",
        "article": "blog",
        "content": "blog",
        "category": "category",
        "kategori": "category",
        "product": "product",
        "ürün": "product",
        "urun": "product",
    }

    return aliases.get(
        raw,
        raw or "unknown",
    )


def _filter_by_active_period(
    dataframe: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    date_column = _first_existing_column(
        dataframe,
        [
            "Date",
            "date",
            "AnalysisDate",
            "analysis_date",
        ],
    )

    if date_column is None:
        return dataframe.copy()

    result = dataframe.copy()

    parsed_dates = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    mask = (
        parsed_dates.dt.date.ge(start_date)
        & parsed_dates.dt.date.le(end_date)
    )

    return (
        result.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


def _resolve_effective_page_period(
    page_dataframe: pd.DataFrame,
    requested_start,
    requested_end,
):
    if page_dataframe.empty:
        return page_dataframe.copy(), requested_start, requested_end, False

    requested_rows = _filter_by_active_period(
        page_dataframe,
        requested_start,
        requested_end,
    )

    if not requested_rows.empty:
        return requested_rows, requested_start, requested_end, False

    date_column = _first_existing_column(
        page_dataframe,
        ["Date", "date", "AnalysisDate", "analysis_date"],
    )

    if date_column is None:
        return page_dataframe.copy(), requested_start, requested_end, False

    parsed_dates = pd.to_datetime(
        page_dataframe[date_column],
        errors="coerce",
    ).dropna()

    if parsed_dates.empty:
        return page_dataframe.copy(), requested_start, requested_end, False

    latest_date = parsed_dates.max().date()

    fallback_rows = _filter_by_active_period(
        page_dataframe,
        latest_date,
        latest_date,
    )

    return fallback_rows, latest_date, latest_date, True


def _filter_optional_period(
    dataframe: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Filter an optional dated output using the actual date column
    detected in that output.

    Snapshot outputs with no date column are returned unchanged.
    """

    if dataframe.empty:
        return dataframe.copy()

    date_column = _first_existing_column(
        dataframe,
        [
            "Date",
            "date",
            "AnalysisDate",
            "analysis_date",
            "AnalysisStartDate",
            "analysis_start_date",
            "StartDate",
            "start_date",
            "ObservationDate",
            "observation_date",
        ],
    )

    if date_column is None:
        return dataframe.copy()

    result = dataframe.copy()

    parsed_dates = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    mask = (
        parsed_dates.dt.date.ge(
            normalized_start
        )
        & parsed_dates.dt.date.le(
            normalized_end
        )
    )

    return (
        result.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


def _latest_row(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the latest row using the best available date column.

    If the dataset has no usable date column, preserve the existing
    row order and return the last row.
    """

    if dataframe.empty:
        return dataframe.copy()

    date_column = _first_existing_column(
        dataframe,
        [
            "Date",
            "date",
            "AnalysisDate",
            "analysis_date",
            "ObservationDate",
            "observation_date",
            "AnalysisStartDate",
            "analysis_start_date",
            "StartDate",
            "start_date",
        ],
    )

    if date_column is None:
        return dataframe.tail(
            1
        ).copy()

    working = dataframe.copy()

    working[
        "_resolved_sort_date"
    ] = pd.to_datetime(
        working[
            date_column
        ],
        errors="coerce",
    )

    if working[
        "_resolved_sort_date"
    ].notna().any():
        working = working.sort_values(
            "_resolved_sort_date"
        )

    return (
        working.tail(
            1
        )
        .drop(
            columns=[
                "_resolved_sort_date",
            ],
            errors="ignore",
        )
        .copy()
    )


def _latest_value(
    dataframe: pd.DataFrame,
    candidates: list[str],
    default: object = "-",
) -> object:
    if dataframe.empty:
        return default

    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return default

    values = dataframe[column].dropna()

    if values.empty:
        return default

    return values.iloc[-1]


def render_blog_analysis(
    page_data: pd.DataFrame,
    page_recommendations: pd.DataFrame,
    language: str,
) -> None:
    st.subheader(
        localized_text(
            language,
            "Blog Analizi",
            "Blog Analysis",
        )
    )

    st.caption(
        localized_text(
            language,
            (
                "Blog sayfasının organik görünürlük, kullanıcı davranışı, "
                "içerik ve GEO performansını değerlendirir."
            ),
            (
                "Evaluates the blog page's organic visibility, user behavior, "
                "content and GEO performance."
            ),
        )
    )

    latest = _latest_row(page_data)

    intent_value = _latest_value(
        latest,
        [
            "keyword_intent",
            "KeywordIntent",
            "intent",
            "Intent",
        ],
        "-",
    )

    content_score = _latest_value(
        latest,
        [
            "content_score",
            "ContentScore",
            "CurrentContentScore",
        ],
        "-",
    )

    geo_score = _latest_value(
        latest,
        [
            "geo_readiness",
            "GeoReadiness",
            "CurrentGeoReadiness",
        ],
        "-",
    )

    latest_sessions = _safe_numeric(
        _latest_value(
            latest,
            [
                "sessions",
                "Sessions",
            ],
            0,
        )
    )

    latest_engaged_sessions = _safe_numeric(
        _latest_value(
            latest,
            [
                "engaged_sessions",
                "EngagedSessions",
            ],
            0,
        )
    )

    if latest_sessions > 0:
        engagement_rate = (
            latest_engaged_sessions
            / latest_sessions
        )
    else:
        engagement_rate = _safe_numeric(
            _latest_value(
                latest,
                [
                    "engagement_rate",
                    "EngagementRate",
                ],
                0,
            )
        )

    blog_columns = st.columns(4)

    blog_columns[0].metric(
        localized_text(
            language,
            "Arama Niyeti",
            "Search Intent",
        ),
        str(intent_value),
    )

    blog_columns[1].metric(
        localized_text(
            language,
            "İçerik Skoru",
            "Content Score",
        ),
        str(content_score),
    )

    blog_columns[2].metric(
        localized_text(
            language,
            "GEO Hazırlığı",
            "GEO Readiness",
        ),
        str(geo_score),
    )

    blog_columns[3].metric(
        t(
            "engagement_rate",
            language,
        ),
        format_percent(
            _safe_numeric(
                engagement_rate
            )
        ),
    )

    st.info(
        localized_text(
            language,
            (
                "Blog Intelligence geliştirmesinde content decay, "
                "cannibalization, internal linking, content gap, yeni blog "
                "konusu önerileri ve rakip blog karşılaştırması da yer alacak."
            ),
            (
                "Blog Intelligence will also include content decay, "
                "cannibalization, internal linking, content gaps, new blog "
                "topic recommendations and competitor-blog comparison."
            ),
        )
    )

    if not page_recommendations.empty:
        render_export_buttons(
            dataframe=page_recommendations,
            basename="seo_blog_recommendations",
            csv_label=(
                "CSV İndir"
                if language == "tr"
                else "Download CSV"
            ),
            excel_label=(
                "Excel İndir"
                if language == "tr"
                else "Download Excel"
            ),
        )

        render_recommendations_table(
            page_recommendations,
            limit=20,
        )


# ============================================================
# LOAD DATA
# ============================================================

data = load_analysis_data()

integrated = data.integrated.copy()
recommendations = data.recommendations.copy()
scenarios = data.scenarios.copy()

available_start, available_end = (
    get_available_date_bounds(
        integrated
        if not integrated.empty
        else data.daily
    )
)


# ============================================================
# PAGE INITIALIZATION
# ============================================================

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{t('page_analysis', initial_language)}"
    ),
    page_icon="🔎",
    title=(
        "Sayfa Analizi"
        if initial_language == "tr"
        else "Page Analysis"
    ),
    subtitle=(
        (
            "Sayfa bazında GSC + GA4 performansını, içerik türünü, "
            "SEO önerilerini, fırsatları ve optimizasyon senaryolarını analiz eder."
        )
        if initial_language == "tr"
        else
        (
            "Analyzes page-level GSC + GA4 performance, content type, "
            "SEO recommendations, opportunities and optimization scenarios."
        )
    ),
    eyebrow=(
        "SEO & GEO Karar Zekâsı"
        if initial_language == "tr"
        else "SEO & GEO Decision Intelligence"
    ),
    default_preset="last_30_days",
    default_comparison="previous_period",
    reference_date=available_end,
)

language = context.language
filters = context.filters


# ============================================================
# PAGE COLUMN
# ============================================================

page_column = None

if "page" in integrated.columns:
    page_column = "page"

elif "Page" in integrated.columns:
    page_column = "Page"


# ============================================================
# PAGE SELECTOR
# ============================================================

if (
    integrated.empty
    or page_column is None
):
    st.info(
        t(
            "no_data",
            language,
        )
    )

    st.stop()


page_options = (
    integrated[
        page_column
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

page_options = page_options[
    page_options != ""
]

page_options = sorted(
    page_options
    .unique()
    .tolist()
)


if not page_options:
    st.info(
        t(
            "no_data",
            language,
        )
    )

    st.stop()


selected_page = st.selectbox(
    t(
        "select_page",
        language,
    ),
    options=page_options,
    index=0,
    key="page_analysis_selected_page",
)


selected_page_history = integrated[
    integrated[
        page_column
    ]
    .astype(str)
    .eq(
        str(
            selected_page
        )
    )
].copy()

selected_page_type = _normalize_page_type(
    _latest_value(
        _latest_row(
            selected_page_history
        ),
        [
            "page_type",
            "PageType",
            "pageType",
        ],
        "unknown",
    )
)

selector_status_columns = st.columns(2)

selector_status_columns[0].metric(
    t(
        "page_type",
        language,
    ),
    selected_page_type,
)

selector_status_columns[1].metric(
    localized_text(
        language,
        "Aktif Dönem",
        "Active Period",
    ),
    (
        f"{filters.start_date:%d.%m.%y} – "
        f"{filters.end_date:%d.%m.%y}"
    ),
)


# ============================================================
# FILTER SELECTED PAGE
# ============================================================

all_selected_page_data = filter_analysis_data(
    dataframe=integrated,
    pages=[
        selected_page
    ],
)

(
    page_data,
    effective_start_date,
    effective_end_date,
    used_latest_fallback,
) = _resolve_effective_page_period(
    all_selected_page_data,
    filters.start_date,
    filters.end_date,
)

if used_latest_fallback:
    st.warning(
        localized_text(
            language,
            (
                "Seçilen dönemde bu sayfa için GSC + GA4 verisi bulunmadı. "
                f"Bu nedenle sayfanın en güncel kullanılabilir verisi "
                f"({effective_start_date:%d.%m.%Y}) gösteriliyor."
            ),
            (
                "No GSC + GA4 data was available for this page in the selected "
                f"period, so the latest available page data "
                f"({effective_start_date:%d %b %Y}) is shown."
            ),
        )
    )

page_recommendations = (
    recommendations.copy()
)

recommendation_page_column = None

if "page" in page_recommendations.columns:
    recommendation_page_column = "page"

elif "Page" in page_recommendations.columns:
    recommendation_page_column = "Page"

if recommendation_page_column is not None:
    page_recommendations = (
        page_recommendations[
            page_recommendations[
                recommendation_page_column
            ]
            .astype(str)
            .eq(
                str(
                    selected_page
                )
            )
        ]
        .reset_index(
            drop=True
        )
    )

page_recommendations = _filter_optional_period(
    page_recommendations,
    effective_start_date,
    effective_end_date,
)


page_scenarios = scenarios.copy()

scenario_page_column = None

if "page" in page_scenarios.columns:
    scenario_page_column = "page"

elif "Page" in page_scenarios.columns:
    scenario_page_column = "Page"

if scenario_page_column is not None:
    page_scenarios = (
        page_scenarios[
            page_scenarios[
                scenario_page_column
            ]
            .astype(str)
            .eq(
                str(
                    selected_page
                )
            )
        ]
        .reset_index(
            drop=True
        )
    )

page_scenarios = _filter_optional_period(
    page_scenarios,
    effective_start_date,
    effective_end_date,
)


# ============================================================
# KPI SUMMARY
# ============================================================

render_section_header(
    t(
        "page_performance",
        language,
    ),
    selected_page,
)

kpis = build_executive_kpis(
    page_data
)

render_kpi_row(
    [
        {
            "label": t(
                "clicks",
                language,
            ),
            "value": format_integer(
                kpis["clicks"]
            ),
        },
        {
            "label": t(
                "impressions",
                language,
            ),
            "value": format_integer(
                kpis["impressions"]
            ),
        },
        {
            "label": t(
                "ctr",
                language,
            ),
            "value": format_percent(
                kpis["ctr"]
            ),
        },
        {
            "label": t(
                "average_position",
                language,
            ),
            "value": format_position(
                kpis["position"]
            ),
        },
    ]
)

render_kpi_row(
    [
        {
            "label": t(
                "sessions",
                language,
            ),
            "value": format_integer(
                kpis["sessions"]
            ),
        },
        {
            "label": t(
                "users",
                language,
            ),
            "value": format_integer(
                kpis["users"]
            ),
        },
        {
            "label": t(
                "conversions",
                language,
            ),
            "value": format_integer(
                kpis["conversions"]
            ),
        },
        {
            "label": t(
                "revenue",
                language,
            ),
            "value": format_currency(
                kpis["revenue"]
            ),
        },
    ]
)


# ============================================================
# PAGE ATTRIBUTES
# ============================================================

render_divider()

render_section_header(
    t(
        "page_details",
        language,
    )
)

latest_page_row = (
    page_data
    .tail(
        1
    )
)

detail_columns = st.columns(
    4
)

page_type_value = "N/A"
intent_value = "N/A"
engagement_rate_value = 0.0
avg_duration_value = 0.0

if not latest_page_row.empty:
    row = latest_page_row.iloc[0]

    if "page_type" in row.index:
        page_type_value = str(
            row.get(
                "page_type",
                "N/A",
            )
        )

    if "keyword_intent" in row.index:
        intent_value = str(
            row.get(
                "keyword_intent",
                "N/A",
            )
        )

    sessions_value = _safe_numeric(
        row.get(
            "sessions",
            0,
        )
    )

    engaged_sessions_value = _safe_numeric(
        row.get(
            "engaged_sessions",
            0,
        )
    )

    if sessions_value > 0:
        engagement_rate_value = (
            engaged_sessions_value
            / sessions_value
        )
    else:
        engagement_rate_value = _safe_numeric(
            row.get(
                "engagement_rate",
                0,
            )
        )

    if "average_session_duration" in row.index:
        avg_duration_value = float(
            pd.to_numeric(
                row.get(
                    "average_session_duration",
                    0,
                ),
                errors="coerce",
            )
            or 0
        )


detail_columns[0].metric(
    t(
        "page_type",
        language,
    ),
    page_type_value,
)

detail_columns[1].metric(
    t(
        "keyword_intent",
        language,
    ),
    intent_value,
)

detail_columns[2].metric(
    t(
        "engagement_rate",
        language,
    ),
    format_percent(
        engagement_rate_value
    ),
)

detail_columns[3].metric(
    t(
        "average_session_duration",
        language,
    ),
    f"{avg_duration_value:.1f}s",
)


if selected_page_type == "blog":
    render_divider()

    render_blog_analysis(
        page_data=page_data,
        page_recommendations=page_recommendations,
        language=language,
    )


# ============================================================
# PERFORMANCE TREND
# ============================================================

render_divider()

render_section_header(
    t(
        "performance_trend",
        language,
    )
)

trend_data = page_data.copy()

date_column = None

if "Date" in trend_data.columns:
    date_column = "Date"

elif "date" in trend_data.columns:
    date_column = "date"

if (
    not trend_data.empty
    and date_column is not None
):
    trend_data[
        date_column
    ] = pd.to_datetime(
        trend_data[
            date_column
        ],
        errors="coerce",
    )

    if "clicks" in trend_data.columns:
        render_line_chart(
            dataframe=trend_data,
            x=date_column,
            y="clicks",
            title=(
                "Tıklama Trendi"
                if language == "tr"
                else "Clicks Trend"
            ),
        )

    if "impressions" in trend_data.columns:
        render_line_chart(
            dataframe=trend_data,
            x=date_column,
            y="impressions",
            title=(
                "Gösterim Trendi"
                if language == "tr"
                else "Impressions Trend"
            ),
        )

    if "position" in trend_data.columns:
        render_line_chart(
            dataframe=trend_data,
            x=date_column,
            y="position",
            title=(
                "Ortalama Pozisyon Trendi"
                if language == "tr"
                else "Average Position Trend"
            ),
        )

else:
    st.info(
        t(
            "no_data",
            language,
        )
    )


# ============================================================
# PAGE RECOMMENDATIONS
# ============================================================

render_divider()

render_section_header(
    t(
        "page_recommendations",
        language,
    )
)

if page_recommendations.empty:
    st.info(
        (
            "Bu sayfa için öneri bulunamadı."
            if language == "tr"
            else
            "No recommendation was found for this page."
        )
    )

else:
    render_recommendations_table(
        page_recommendations,
        limit=50,
    )

    render_export_buttons(
        dataframe=page_recommendations,
        basename="seo_page_recommendations",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

render_divider()

render_section_header(
    t(
        "scenario_analysis",
        language,
    )
)

if page_scenarios.empty:
    st.info(
        (
            "Bu sayfa için senaryo verisi bulunamadı."
            if language == "tr"
            else
            "No scenario data was found for this page."
        )
    )

else:
    preferred_scenario_columns = [
        "Scenario",
        "ScenarioLabel",
        "ScenarioCTR",
        "ScenarioPosition",
        "EstimatedPositionGain",
        "ExpectedClicks",
        "ClickUplift",
        "ClickUpliftPct",
        "ExpectedIncrementalTrafficValue",
        "EstimatedImplementationCost",
        "ExpectedNetValue",
        "EstimatedROI",
        "PaybackPeriod",
        "CurrentContentScore",
        "ScenarioContentScore",
        "CurrentGeoReadiness",
        "ScenarioGeoReadiness",
    ]

    existing_scenario_columns = [
        column
        for column in preferred_scenario_columns
        if column in page_scenarios.columns
    ]

    if existing_scenario_columns:
        render_localized_dataframe(
            page_scenarios[
                existing_scenario_columns
            ],
            width="stretch",
            hide_index=True,
        )

    else:
        render_localized_dataframe(
            page_scenarios,
            width="stretch",
            hide_index=True,
        )

    render_export_buttons(
        dataframe=page_scenarios,
        basename="seo_page_scenarios",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# DETAILED DATA VIEW
# ============================================================

render_divider()

with st.expander(
    localized_text(
        language,
        "Detaylı Veri Görünümü",
        "Detailed Data View",
    ),
    expanded=False,
):
    detail_tabs = st.tabs(
        [
            localized_text(
                language,
                "Sayfa Verisi",
                "Page Data",
            ),
            localized_text(
                language,
                "Öneriler",
                "Recommendations",
            ),
            localized_text(
                language,
                "Senaryolar",
                "Scenarios",
            ),
        ]
    )

    with detail_tabs[0]:
        render_export_buttons(
            dataframe=page_data,
            basename="seo_page_data",
            csv_label="CSV",
            excel_label="Excel",
        )

        render_localized_dataframe(
            page_data,
            width="stretch",
            hide_index=True,
        )

    with detail_tabs[1]:
        if page_recommendations.empty:
            st.info(
                t(
                    "no_data",
                    language,
                )
            )
        else:
            render_localized_dataframe(
                page_recommendations,
                width="stretch",
                hide_index=True,
            )

    with detail_tabs[2]:
        if page_scenarios.empty:
            st.info(
                t(
                    "no_data",
                    language,
                )
            )
        else:
            render_localized_dataframe(
                page_scenarios,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# DATA PERIOD
# ============================================================

if (
    available_start is not None
    and available_end is not None
):
    st.caption(
        (
            f"Aktif sayfa veri dönemi: "
            f"{effective_start_date:%d.%m.%Y} – "
            f"{effective_end_date:%d.%m.%Y}"
        )
        if language == "tr"
        else
        (
            f"Active page data period: "
            f"{effective_start_date:%d.%m.%Y} – "
            f"{effective_end_date:%d.%m.%Y}"
        )
    )
