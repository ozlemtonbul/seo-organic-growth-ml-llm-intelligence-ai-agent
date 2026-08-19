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

from dashboard.i18n import (
    t,
)

from dashboard.layout import (
    initialize_dashboard,
    localized_text,
)

from dashboard.filters import DateRange

from dashboard.services import (
    aggregate_seo_kpis,
    filter_analysis_data,
    get_available_date_bounds,
    load_analysis_data,
)

from dashboard.utils import (
    count_output_files,
    format_integer,
    format_percent,
    format_position,
    get_latest_output_time,
)

from src.llm.manager import (
    get_llm_runtime_info,
)


# ============================================================
# NAVIGATION CARD
# ============================================================


def render_navigation_card(
    title: str,
    description: str,
    page_path: str | None,
    button_label: str,
    key: str,
) -> None:
    """
    Render one dashboard navigation card.

    Navigation behavior follows the Ads dashboard:
    buttons use st.switch_page().
    """

    with st.container(
        border=True
    ):
        st.subheader(
            title
        )

        st.caption(
            description
        )

        if page_path is None:
            st.button(
                button_label,
                disabled=True,
                width="stretch",
                key=key,
            )
            return

        if st.button(
            button_label,
            width="stretch",
            key=key,
        ):
            st.switch_page(
                page_path
            )


# ============================================================
# LOAD DATA
# ============================================================

data = load_analysis_data()

available_source = (
    data.integrated
    if not data.integrated.empty
    else data.daily
)

available_start, available_end = (
    get_available_date_bounds(
        available_source
    )
)


# ============================================================
# INITIAL LANGUAGE
# ============================================================

initial_language = (
    st.session_state.get(
        "dashboard_language",
        "tr",
    )
)


# ============================================================
# INITIALIZE DASHBOARD
# ============================================================

context = initialize_dashboard(
    page_title=t("app_title", initial_language),
    page_icon="📈",
    title=t("app_title", initial_language),
    subtitle=(
        (
            "Google Search Console ve GA4 verilerini; "
            "makine öğrenmesi, SEO ve GEO senaryoları, "
            "tahmin modelleri ve AI destekli karar zekâsı "
            "ile birleştiren organik büyüme platformu."
        )
        if initial_language == "tr"
        else
        (
            "An organic growth intelligence platform combining "
            "Google Search Console and GA4 data with machine "
            "learning, SEO and GEO scenarios, forecasting models "
            "and AI-assisted decision intelligence."
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
# FILTERED KPI SOURCE
# ============================================================

kpi_source = (
    data.integrated
    if not data.integrated.empty
    else data.daily
)

period_kpi_source = filter_analysis_data(
    dataframe=kpi_source,
    date_range=DateRange(
        start_date=filters.start_date,
        end_date=filters.end_date,
    ),
)

kpis = aggregate_seo_kpis(
    period_kpi_source
)


# ============================================================
# LLM RUNTIME
# ============================================================

runtime_info = (
    get_llm_runtime_info()
)

llm_ready = bool(
    runtime_info.get(
        "ready",
        False,
    )
)


# ============================================================
# DETERMINISTIC / HYBRID NOTICE
# ============================================================

if not llm_ready:
    st.info(
        localized_text(
            language,
            (
                "API anahtarı yapılandırılmadığı için sistem "
                "deterministik SEO karar zekâsı modunda çalışıyor. "
                "GSC + GA4 analizi, KPI'lar, makine öğrenmesi, "
                "senaryo simülasyonları ve SEO önerileri "
                "kullanılmaya devam eder."
            ),
            (
                "The system is running in deterministic SEO "
                "decision-intelligence mode because no LLM API "
                "key is configured. GSC + GA4 analysis, KPIs, "
                "machine learning, scenario simulations and SEO "
                "recommendations remain available."
            ),
        ),
        icon="ℹ️",
    )


# ============================================================
# PLATFORM STATUS
# ============================================================

st.subheader(
    t(
        "platform_status",
        language,
    )
)


# ============================================================
# SELECTED PERIOD STATUS
# ============================================================

selected_row_count = int(
    len(
        period_kpi_source
    )
)

selected_day_count = 0
selected_date_column = None

for candidate in [
    "Date",
    "date",
    "AnalysisDate",
    "analysis_date",
]:
    if candidate in period_kpi_source.columns:
        selected_date_column = candidate
        break

if (
    selected_date_column is not None
    and not period_kpi_source.empty
):
    selected_day_count = int(
        pd.to_datetime(
            period_kpi_source[
                selected_date_column
            ],
            errors="coerce",
        )
        .dropna()
        .dt.date
        .nunique()
    )


# ============================================================
# PLATFORM STATUS ROW 1
# ============================================================

first_status_row = st.columns(
    3
)

first_status_row[0].metric(
    localized_text(
        language,
        "Seçili Dönem Veri Satırı",
        "Selected Period Rows",
    ),
    format_integer(
        selected_row_count
    ),
)

first_status_row[1].metric(
    localized_text(
        language,
        "Seçili Dönemde Veri Bulunan Gün",
        "Days With Data in Selected Period",
    ),
    format_integer(
        selected_day_count
    ),
)

first_status_row[2].metric(
    t(
        "generated_outputs",
        language,
    ),
    format_integer(
        count_output_files()
    ),
)


# ============================================================
# PLATFORM STATUS ROW 2
# ============================================================

second_status_row = st.columns(
    [1.35, 1, 1]
)

date_period = (
    f"{available_start:%d.%m.%y} – "
    f"{available_end:%d.%m.%y}"
    if (
        available_start is not None
        and available_end is not None
    )
    else t(
        "no_data",
        language,
    )
)

second_status_row[0].metric(
    localized_text(
        language,
        "Toplam Veri Kapsamı",
        "Total Data Coverage",
    ),
    date_period,
)

second_status_row[1].metric(
    t(
        "ai_runtime_mode",
        language,
    ),
    (
        t(
            "hybrid_llm",
            language,
        )
        if llm_ready
        else t(
            "deterministic",
            language,
        )
    ),
)

latest_output_value = get_latest_output_time()

if (
    isinstance(latest_output_value, str)
    and len(latest_output_value) >= 10
):
    latest_output_value = latest_output_value[:10]

second_status_row[2].metric(
    t(
        "latest_output",
        language,
    ),
    latest_output_value,
)


# ============================================================
# ACTIVE ANALYSIS PERIOD
# ============================================================

st.caption(
    localized_text(
        language,
        (
            f"Aktif analiz dönemi: "
            f"{filters.start_date:%d.%m.%Y} – "
            f"{filters.end_date:%d.%m.%Y}"
        ),
        (
            f"Active analysis period: "
            f"{filters.start_date:%d.%m.%Y} – "
            f"{filters.end_date:%d.%m.%Y}"
        ),
    )
)


# ============================================================
# QUICK KPI OVERVIEW
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "Hızlı SEO Özeti",
        "Quick SEO Overview",
    )
)

st.caption(
    localized_text(
        language,
        (
            "Aşağıdaki KPI'lar ve seçili dönem durum kartları aktif analiz "
            "dönemine göre hesaplanır; Toplam Veri Kapsamı ve sistem durumu "
            "kartları genel platform bilgisini gösterir."
        ),
        (
            "The KPIs and selected-period status cards are calculated for the "
            "active analysis period; Total Data Coverage and system-status cards "
            "describe the overall platform."
        ),
    )
)

st.caption(
    localized_text(
        language,
        (
            f"Seçili dönem: {filters.start_date:%d.%m.%Y} – "
            f"{filters.end_date:%d.%m.%Y} | "
            f"Filtrelenen kayıt: {format_integer(len(period_kpi_source))}"
        ),
        (
            f"Selected period: {filters.start_date:%d.%m.%Y} – "
            f"{filters.end_date:%d.%m.%Y} | "
            f"Filtered rows: {format_integer(len(period_kpi_source))}"
        ),
    )
)


quick_kpi_columns = st.columns(
    4
)


# ------------------------------------------------------------
# CLICKS
# ------------------------------------------------------------

quick_kpi_columns[0].metric(
    t(
        "clicks",
        language,
    ),
    format_integer(
        kpis["clicks"]
    ),
)


# ------------------------------------------------------------
# IMPRESSIONS
# ------------------------------------------------------------

quick_kpi_columns[1].metric(
    t(
        "impressions",
        language,
    ),
    format_integer(
        kpis["impressions"]
    ),
)


# ------------------------------------------------------------
# CTR
# ------------------------------------------------------------

quick_kpi_columns[2].metric(
    t(
        "ctr",
        language,
    ),
    format_percent(
        kpis["ctr"]
    ),
)


# ------------------------------------------------------------
# POSITION
# ------------------------------------------------------------

quick_kpi_columns[3].metric(
    t(
        "average_position",
        language,
    ),
    format_position(
        kpis["position"]
    ),
)


# ============================================================
# ANALYSIS AREAS
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "Analiz Alanları",
        "Analysis Areas",
    )
)

st.caption(
    localized_text(
        language,
        (
            "SEO karar zekâsının farklı analiz "
            "modüllerine erişin."
        ),
        (
            "Access the different analysis modules "
            "of the SEO decision-intelligence platform."
        ),
    )
)


# ============================================================
# FIRST NAVIGATION ROW
# ============================================================

first_navigation_row = st.columns(
    2
)


# ------------------------------------------------------------
# EXECUTIVE OVERVIEW
# ------------------------------------------------------------

with first_navigation_row[0]:
    render_navigation_card(
        title=t(
            "executive_overview",
            language,
        ),
        description=(
            (
                "Organik KPI'ları, veri kapsamını, "
                "fırsatları, model performansını "
                "ve SEO iş değerini inceleyin."
            )
            if language == "tr"
            else
            (
                "Review organic KPIs, data coverage, "
                "opportunities, model performance "
                "and SEO business value."
            )
        ),
        page_path=(
            "pages/"
            "1_Executive_Overview.py"
        ),
        button_label=(
            "Yönetici Özetini Aç"
            if language == "tr"
            else "Open Executive Overview"
        ),
        key=(
            "home_open_executive"
        ),
    )


# ------------------------------------------------------------
# ASK AI
# ------------------------------------------------------------

with first_navigation_row[1]:
    render_navigation_card(
        title=t(
            "ai_assistant",
            language,
        ),
        description=(
            (
                "SEO, GEO, Search Console, GA4, "
                "model sonuçları ve optimizasyon "
                "önerileri hakkında doğal dilde "
                "soru sorun."
            )
            if language == "tr"
            else
            (
                "Ask natural-language questions about "
                "SEO, GEO, Search Console, GA4, model "
                "results and optimization recommendations."
            )
        ),
        page_path=(
            "pages/"
            "5_Ask_AI.py"
        ),
        button_label=(
            "AI Asistanını Aç"
            if language == "tr"
            else "Open AI Assistant"
        ),
        key=(
            "home_open_ai"
        ),
    )


# ============================================================
# SECOND NAVIGATION ROW
# ============================================================

second_navigation_row = (
    st.columns(
        3
    )
)


navigation_pages = [
    # --------------------------------------------------------
    # PAGE ANALYSIS
    # --------------------------------------------------------
    {
        "title": t(
            "page_analysis",
            language,
        ),
        "description": (
            (
                "Sayfa bazında tıklama, gösterim, CTR, "
                "pozisyon, GA4 sonuçları ve SEO "
                "önerilerini inceleyin."
            )
            if language == "tr"
            else
            (
                "Review page-level clicks, impressions, "
                "CTR, position, GA4 outcomes and SEO "
                "recommendations."
            )
        ),
        "page_path": (
            "pages/"
            "2_Page_Analysis.py"
        ),
        "button_label": (
            "Sayfa Analizini Aç"
            if language == "tr"
            else "Open Page Analysis"
        ),
        "key": (
            "home_open_page_analysis"
        ),
    },

    # --------------------------------------------------------
    # SEO OPPORTUNITY OPTIMIZER
    # --------------------------------------------------------
    {
        "title": t(
            "seo_opportunity_optimizer",
            language,
        ),
        "description": (
            (
                "SEO ve GEO senaryolarını trafik değeri, "
                "uygulama maliyeti, beklenen net değer, "
                "ROI ve model güveni açısından karşılaştırın."
            )
            if language == "tr"
            else
            (
                "Compare SEO and GEO scenarios by traffic "
                "value, implementation cost, expected net "
                "value, ROI and model confidence."
            )
        ),
        "page_path": (
            "pages/"
            "3_SEO_Opportunity_Optimizer.py"
        ),
        "button_label": (
            "SEO Fırsatlarını Aç"
            if language == "tr"
            else "Open SEO Opportunities"
        ),
        "key": (
            "home_open_optimizer"
        ),
    },

    # --------------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------------
    {
        "title": t(
            "ai_insights",
            language,
        ),
        "description": (
            (
                "Model performansını, riskleri, "
                "fırsatları, feature importance ve "
                "AI karar gerekçelerini inceleyin."
            )
            if language == "tr"
            else
            (
                "Review model performance, risks, "
                "opportunities, feature importance "
                "and AI decision rationale."
            )
        ),
        "page_path": (
            "pages/"
            "4_AI_Insights.py"
        ),
        "button_label": (
            "AI Analizlerini Aç"
            if language == "tr"
            else "Open AI Insights"
        ),
        "key": (
            "home_open_ai_insights"
        ),
    },
]


for column, page in zip(
    second_navigation_row,
    navigation_pages,
):
    with column:
        render_navigation_card(
            title=page[
                "title"
            ],
            description=page[
                "description"
            ],
            page_path=page[
                "page_path"
            ],
            button_label=page[
                "button_label"
            ],
            key=page[
                "key"
            ],
        )


# ============================================================
# ADVANCED INTELLIGENCE MODULES
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "Gelişmiş SEO Zekâsı",
        "Advanced SEO Intelligence",
    )
)

st.caption(
    localized_text(
        language,
        (
            "Teknik SEO, içerik/GEO ve rakip zekâsı aynı karar mimarisini "
            "besler. Veri kaynağı bulunmayan modüller sonuç uydurmaz."
        ),
        (
            "Technical SEO, content/GEO and competitor intelligence feed the "
            "same decision architecture. Modules without a connected data "
            "source never fabricate results."
        ),
    )
)

advanced_columns = st.columns(3)

with advanced_columns[0]:
    render_navigation_card(
        title=localized_text(language, "Teknik SEO", "Technical SEO"),
        description=localized_text(
            language,
            (
                "Crawl ve PageSpeed sinyallerinden indexability, canonical, "
                "metadata, heading, schema, internal-link ve performans "
                "sorunlarını iş önceliğiyle analiz eder."
            ),
            (
                "Analyze indexability, canonical, metadata, headings, schema, "
                "internal links and performance issues from crawl and PageSpeed "
                "signals with business prioritization."
            ),
        ),
        page_path="pages/6_Technical_SEO.py",
        button_label=localized_text(
            language,
            "Teknik SEO'yu Aç",
            "Open Technical SEO",
        ),
        key="home_open_technical_seo",
    )

with advanced_columns[1]:
    render_navigation_card(
        title=localized_text(
            language,
            "İçerik + GEO Zekâsı",
            "Content + GEO Intelligence",
        ),
        description=localized_text(
            language,
            (
                "Keyword intent, content gap, content-to-commerce ve GEO "
                "readiness sinyallerini tek görünümde önceliklendirir."
            ),
            (
                "Prioritize keyword intent, content gaps, content-to-commerce "
                "and GEO-readiness signals in one view."
            ),
        ),
        page_path="pages/7_Content_GEO_Intelligence.py",
        button_label=localized_text(
            language,
            "İçerik + GEO'yu Aç",
            "Open Content + GEO",
        ),
        key="home_open_content_geo",
    )

with advanced_columns[2]:
    render_navigation_card(
        title=localized_text(
            language,
            "Rakip Zekâsı",
            "Competitor Intelligence",
        ),
        description=localized_text(
            language,
            (
                "DataForSEO bağlantısı geldiğinde SERP rakipleri, keyword gap, "
                "pozisyon ve görünürlük farklarını Decision Engine'e aktarır."
            ),
            (
                "When DataForSEO is connected, feed SERP competitors, keyword "
                "gaps, ranking and visibility gaps into the Decision Engine."
            ),
        ),
        page_path="pages/8_Competitor_Intelligence.py",
        button_label=localized_text(
            language,
            "Rakip Zekâsı Durumu",
            "Competitor Intelligence Status",
        ),
        key="home_open_competitor_intelligence",
    )