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
    render_bar_chart,
    render_export_buttons,
    render_kpi_row,
    render_recommendations_table,
    render_scatter_chart,
)
from dashboard.i18n import (
    t,
)
from dashboard.localization import (
    localize_value,
    render_localized_dataframe,
)
from dashboard.layout import (
    initialize_dashboard,
    render_divider,
    render_footer,
    render_header,
    render_section_header,
)
from dashboard.services import (
    get_available_date_bounds,
    get_priority_recommendations,
    load_analysis_data,
)
from dashboard.utils import (
    format_currency,
    format_integer,
    format_number,
    safe_float,
)


# ============================================================
# LOAD DATA
# ============================================================

data = load_analysis_data()

scenarios = data.scenarios.copy()
recommendations = data.recommendations.copy()
latest_page_state = (
    data.latest_page_state.copy()
)


# ============================================================
# PAGE INITIALIZATION
# ============================================================

available_start, available_end = get_available_date_bounds(
    data.integrated
)

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{t('seo_opportunity_optimizer', initial_language)}"
    ),
    page_icon="🎯",
    title=(
        "SEO Fırsat Optimizasyonu"
        if initial_language == "tr"
        else "SEO Opportunity Optimizer"
    ),
    subtitle=(
        "Hangi SEO/GEO fırsatına önce müdahale edilmesi gerektiğini; "
        "iş değeri, beklenen etki, maliyet ve model güveniyle önceliklendirin."
        if initial_language == "tr"
        else
        "Prioritize which SEO/GEO opportunity should be addressed first using "
        "business value, expected impact, cost, and model confidence."
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

st.caption(
    (
        "Bu ekran ileriye dönük optimizasyon için son başarılı model/senaryo "
        "snapshot'ını kullanır. Üstteki tarih filtresi diğer tarih-duyarlı "
        "performans ekranlarıyla ortak bağlamı korur; bu sayfada modeli "
        "yeniden eğitmez veya geçmiş tarih için sahte senaryo üretmez."
        if language == "tr"
        else
        "This optimizer uses the latest successful model/scenario snapshot. "
        "The global date filter preserves shared context with date-aware "
        "performance pages; it does not retrain the model or fabricate "
        "historical scenarios on this page."
    )
)


# ============================================================
# EMPTY STATE
# ============================================================

if scenarios.empty:
    st.info(
        (
            "SEO senaryo çıktısı bulunamadı."
            if language == "tr"
            else
            "No SEO scenario output was found."
        )
    )

    render_footer(
        language=language,
        demo_mode=False,
    )

    st.stop()


# ============================================================
# HELPERS
# ============================================================


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return one numeric series safely.
    """
    if column not in dataframe.columns:
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def numeric_sum(
    dataframe: pd.DataFrame,
    column: str,
) -> float:
    """
    Sum one numeric column safely.
    """
    values = numeric_series(
        dataframe,
        column,
    )

    if values.empty:
        return 0.0

    return float(
        values.fillna(0).sum()
    )


def numeric_mean(
    dataframe: pd.DataFrame,
    column: str,
) -> float:
    """
    Average one numeric column safely.
    """
    values = numeric_series(
        dataframe,
        column,
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.mean()
    )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return first existing column.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


# ============================================================
# FILTERS
# ============================================================

render_section_header(
    (
        "Optimizasyon Filtreleri"
        if language == "tr"
        else "Optimization Filters"
    )
)

filter_columns = st.columns(
    3
)

page_column = first_existing_column(
    scenarios,
    [
        "page",
        "Page",
    ],
)

scenario_column = first_existing_column(
    scenarios,
    [
        "Scenario",
        "scenario",
    ],
)

priority_column = first_existing_column(
    scenarios,
    [
        "PriorityTier",
        "priority_tier",
    ],
)


selected_pages: list[str] = []

if page_column is not None:
    page_options = sorted(
        scenarios[
            page_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with filter_columns[0]:
        all_token = "__ALL__"
        selected_page_value = st.selectbox(
            t("page", language),
            options=[all_token] + page_options,
            format_func=lambda value: (
                "Tümü"
                if value == all_token and language == "tr"
                else "All"
                if value == all_token
                else str(value)
            ),
            key="optimizer_page",
        )
        selected_pages = (
            []
            if selected_page_value == all_token
            else [selected_page_value]
        )


selected_scenarios: list[str] = []

if scenario_column is not None:
    scenario_options = sorted(
        scenarios[
            scenario_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with filter_columns[1]:
        all_token = "__ALL__"
        selected_scenario_value = st.selectbox(
            t("scenario", language),
            options=[all_token] + scenario_options,
            format_func=lambda value: (
                "Tümü"
                if value == all_token and language == "tr"
                else "All"
                if value == all_token
                else str(localize_value(value, language, "Scenario"))
            ),
            key="optimizer_scenario",
        )
        selected_scenarios = (
            []
            if selected_scenario_value == all_token
            else [selected_scenario_value]
        )


selected_priorities: list[str] = []

if priority_column is not None:
    priority_options = sorted(
        scenarios[
            priority_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with filter_columns[2]:
        all_token = "__ALL__"
        selected_priority_value = st.selectbox(
            t("priority", language),
            options=[all_token] + priority_options,
            format_func=lambda value: (
                "Tümü"
                if value == all_token and language == "tr"
                else "All"
                if value == all_token
                else str(localize_value(value, language, "PriorityTier"))
            ),
            key="optimizer_priority",
        )
        selected_priorities = (
            []
            if selected_priority_value == all_token
            else [selected_priority_value]
        )


filtered_scenarios = scenarios.copy()

if (
    selected_pages
    and page_column is not None
):
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios[
            page_column
        ].isin(
            selected_pages
        )
    ]

if (
    selected_scenarios
    and scenario_column is not None
):
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios[
            scenario_column
        ].isin(
            selected_scenarios
        )
    ]

if (
    selected_priorities
    and priority_column is not None
):
    filtered_scenarios = filtered_scenarios[
        filtered_scenarios[
            priority_column
        ].isin(
            selected_priorities
        )
    ]

filtered_scenarios = (
    filtered_scenarios
    .reset_index(
        drop=True
    )
)


# ============================================================
# KPI SUMMARY
# ============================================================

render_divider()

render_section_header(
    (
        "Senaryo Portföy Özeti"
        if language == "tr"
        else "Scenario Portfolio Summary"
    )
)

scenario_count = int(
    len(
        filtered_scenarios
    )
)

expected_incremental_value = (
    numeric_sum(
        filtered_scenarios,
        "ExpectedIncrementalTrafficValue",
    )
)

expected_net_value = (
    numeric_sum(
        filtered_scenarios,
        "ExpectedNetValue",
    )
)

implementation_cost = (
    numeric_sum(
        filtered_scenarios,
        "EstimatedImplementationCost",
    )
)

average_roi = (
    numeric_mean(
        filtered_scenarios,
        "EstimatedROI",
    )
)


render_kpi_row(
    [
        {
            "label": (
                "Senaryo Sayısı"
                if language == "tr"
                else "Scenario Count"
            ),
            "value": format_integer(
                scenario_count
            ),
        },
        {
            "label": t(
                "incremental_traffic_value",
                language,
            ),
            "value": format_currency(
                expected_incremental_value
            ),
        },
        {
            "label": t(
                "expected_net_value",
                language,
            ),
            "value": format_currency(
                expected_net_value
            ),
        },
        {
            "label": t(
                "implementation_cost",
                language,
            ),
            "value": format_currency(
                implementation_cost
            ),
        },
        {
            "label": t(
                "estimated_roi",
                language,
            ),
            "value": format_number(
                average_roi,
                decimals=2,
            ),
        },
    ]
)


# ============================================================
# OPPORTUNITY RANKING
# ============================================================

render_divider()

render_section_header(
    (
        "En Yüksek Değerli SEO Fırsatları"
        if language == "tr"
        else "Highest-Value SEO Opportunities"
    )
)

ranking = (
    filtered_scenarios.copy()
)

sort_columns: list[str] = []
ascending: list[bool] = []

if "ExpectedNetValue" in ranking.columns:
    sort_columns.append(
        "ExpectedNetValue"
    )
    ascending.append(
        False
    )

if "EstimatedROI" in ranking.columns:
    sort_columns.append(
        "EstimatedROI"
    )
    ascending.append(
        False
    )

if sort_columns:
    ranking = ranking.sort_values(
        sort_columns,
        ascending=ascending,
    )

ranking = ranking.reset_index(
    drop=True
)

preferred_columns = [
    "page",
    "Page",
    "Scenario",
    "ScenarioLabel",
    "PriorityTier",
    "ConfidenceLevel",
    "CurrentClicks",
    "ExpectedClicks",
    "ClickUplift",
    "ClickUpliftPct",
    "CurrentPosition",
    "ScenarioPosition",
    "EstimatedPositionGain",
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

visible_columns = [
    column
    for column in preferred_columns
    if column in ranking.columns
]

if ranking.empty:
    st.info(
        t(
            "no_data",
            language,
        )
    )

else:
    if visible_columns:
        render_localized_dataframe(
            ranking[
                visible_columns
            ],
            width="stretch",
            hide_index=True,
        )

    else:
        render_localized_dataframe(
            ranking,
            width="stretch",
            hide_index=True,
        )

    render_export_buttons(
        dataframe=ranking,
        basename="seo_opportunity_optimizer",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# EXPECTED NET VALUE BY SCENARIO
# ============================================================

render_divider()

render_section_header(
    (
        "Senaryo Bazında Beklenen Net Değer"
        if language == "tr"
        else "Expected Net Value by Scenario"
    )
)

if (
    scenario_column is not None
    and "ExpectedNetValue"
    in filtered_scenarios.columns
):
    scenario_value = (
        filtered_scenarios
        .groupby(
            scenario_column,
            as_index=False,
            dropna=False,
        )[
            "ExpectedNetValue"
        ]
        .sum()
        .sort_values(
            "ExpectedNetValue",
            ascending=False,
        )
    )

    render_bar_chart(
        dataframe=scenario_value,
        x=scenario_column,
        y="ExpectedNetValue",
        title=(
            "Senaryo İş Değeri"
            if language == "tr"
            else "Scenario Business Value"
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
# ROI VS NET VALUE
# ============================================================

render_divider()

render_section_header(
    (
        "ROI ve Net Değer Dağılımı"
        if language == "tr"
        else "ROI vs Net Value Distribution"
    )
)

if (
    "EstimatedROI"
    in filtered_scenarios.columns
    and "ExpectedNetValue"
    in filtered_scenarios.columns
):
    size_column = first_existing_column(
        filtered_scenarios,
        [
            "ExpectedIncrementalTrafficValue",
            "ClickUplift",
        ],
    )

    color_column = first_existing_column(
        filtered_scenarios,
        [
            "PriorityTier",
            "ConfidenceLevel",
            "Scenario",
        ],
    )

    render_scatter_chart(
        dataframe=filtered_scenarios,
        x="EstimatedROI",
        y="ExpectedNetValue",
        size=size_column,
        color=color_column,
        hover_name=page_column,
        title=(
            "SEO Fırsat Matrisi"
            if language == "tr"
            else "SEO Opportunity Matrix"
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
# SEO + GEO SCORE IMPROVEMENT
# ============================================================

render_divider()

render_section_header(
    (
        "SEO ve GEO Skor İyileşmesi"
        if language == "tr"
        else "SEO and GEO Score Improvement"
    )
)

score_columns = st.columns(
    2
)

with score_columns[0]:
    if (
        "CurrentContentScore"
        in filtered_scenarios.columns
        and "ScenarioContentScore"
        in filtered_scenarios.columns
    ):
        current_content = numeric_mean(
            filtered_scenarios,
            "CurrentContentScore",
        )

        scenario_content = numeric_mean(
            filtered_scenarios,
            "ScenarioContentScore",
        )

        st.metric(
            t(
                "content_score",
                language,
            ),
            f"{scenario_content:.2f}",
            delta=(
                f"{scenario_content - current_content:+.2f}"
            ),
        )

    else:
        st.info(
            t(
                "no_data",
                language,
            )
        )


with score_columns[1]:
    if (
        "CurrentGeoReadiness"
        in filtered_scenarios.columns
        and "ScenarioGeoReadiness"
        in filtered_scenarios.columns
    ):
        current_geo = numeric_mean(
            filtered_scenarios,
            "CurrentGeoReadiness",
        )

        scenario_geo = numeric_mean(
            filtered_scenarios,
            "ScenarioGeoReadiness",
        )

        st.metric(
            t(
                "geo_readiness",
                language,
            ),
            f"{scenario_geo:.2f}",
            delta=(
                f"{scenario_geo - current_geo:+.2f}"
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
# RECOMMENDED ACTIONS
# ============================================================

render_divider()

render_section_header(
    t(
        "recommended_action",
        language,
    )
)

priority_recommendations = (
    get_priority_recommendations(
        recommendations,
        limit=50,
    )
)

if priority_recommendations.empty:
    st.info(
        (
            "SEO önerisi bulunamadı."
            if language == "tr"
            else
            "No SEO recommendations were found."
        )
    )

else:
    render_recommendations_table(
        priority_recommendations,
        limit=50,
    )

    render_export_buttons(
        dataframe=priority_recommendations,
        basename="seo_priority_recommendations",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# CURRENT PAGE STATE
# ============================================================

render_divider()

with st.expander(
    (
        "Mevcut Sayfa Durumunu Gör"
        if language == "tr"
        else "View Current Page State"
    )
):
    if latest_page_state.empty:
        st.info(
            t(
                "no_data",
                language,
            )
        )

    else:
        render_localized_dataframe(
            latest_page_state,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# RAW SCENARIOS
# ============================================================

with st.expander(
    (
        "Tüm Senaryo Verisini Gör"
        if language == "tr"
        else "View All Scenario Data"
    )
):
    render_localized_dataframe(
        filtered_scenarios,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

render_footer(
    language=language,
    demo_mode=False,
)