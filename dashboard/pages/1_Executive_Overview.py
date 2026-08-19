from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


# ============================================================
# IMPORTS
# ============================================================

from dashboard.app_config import APP_TITLE
from dashboard.components import (
    render_export_buttons,
    render_recommendations_table,
)
from dashboard.i18n import t
from dashboard.localization import render_localized_dataframe
from dashboard.layout import (
    initialize_dashboard,
    localized_text,
)
from dashboard.services import (
    build_executive_kpis,
    build_executive_decision_intelligence,
    build_executive_opportunities,
    build_executive_top_pages,
    build_model_summary,
    build_recommendation_summary,
    get_available_date_bounds,
    load_executive_data,
)
from dashboard.utils import (
    format_currency,
    format_integer,
    format_percent,
    format_position,
)
from src.llm.manager import get_llm_runtime_info


# ============================================================
# HELPERS
# ============================================================

def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _safe_value(
    mapping: dict,
    key: str,
    default=0,
):
    value = mapping.get(key, default)
    return default if value is None else value


def _localized_action(
    value: object,
    language: str,
) -> str:
    raw = str(value or "").strip()

    if not raw:
        return "-"

    mapping = {
        "Apply Full SEO + GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu"
            if language == "tr"
            else "Apply Full SEO + GEO Optimization"
        ),
        "full_seo_geo_optimization": (
            "Tam SEO + GEO Optimizasyonu"
            if language == "tr"
            else "Full SEO + GEO Optimization"
        ),
        "content_refresh": (
            "İçerik Güncellemesi"
            if language == "tr"
            else "Content Refresh"
        ),
        "technical_fix": (
            "Teknik SEO Düzeltmesi"
            if language == "tr"
            else "Technical SEO Fix"
        ),
        "internal_linking": (
            "İç Link Optimizasyonu"
            if language == "tr"
            else "Internal Linking Optimization"
        ),
        "Maintain": (
            "Mevcut Durumu Koru"
            if language == "tr"
            else "Maintain"
        ),
        "Maintain Current Setup": (
            "Mevcut Durumu Koru"
            if language == "tr"
            else "Maintain Current Setup"
        ),
    }

    return mapping.get(raw, raw)


def _build_exec_message(
    kpis: dict,
    recommendation_summary: dict,
    language: str,
) -> tuple[str, str, str]:
    clicks = float(_safe_value(kpis, "clicks", 0))
    ctr = float(_safe_value(kpis, "ctr", 0))
    position = float(_safe_value(kpis, "position", 0))
    rec_count = int(
        _safe_value(
            recommendation_summary,
            "recommendation_count",
            0,
        )
    )
    high_priority = int(
        _safe_value(
            recommendation_summary,
            "high_priority_count",
            0,
        )
    )

    general_tr = (
        f"Organik performans görünümü: {format_integer(clicks)} tıklama, "
        f"%{ctr * 100:.1f} CTR ve {position:.2f} ortalama pozisyon."
    )
    general_en = (
        f"Organic performance: {format_integer(clicks)} clicks, "
        f"{ctr * 100:.1f}% CTR and {position:.2f} average position."
    )

    opportunity_tr = (
        f"Toplam {rec_count} SEO önerisi üretildi. "
        f"En yüksek değerli fırsatlar aşağıda önceliklendirilmiştir."
    )
    opportunity_en = (
        f"{rec_count} SEO recommendations were generated. "
        f"The highest-value opportunities are prioritized below."
    )

    risk_tr = (
        f"Yüksek öncelikli öneri sayısı {high_priority}. "
        f"CTR, pozisyon ve ticari değer kaybı riski taşıyan sayfalar izlenmelidir."
    )
    risk_en = (
        f"There are {high_priority} high-priority recommendations. "
        f"Pages with CTR, ranking or commercial-value risk should be monitored."
    )

    return (
        general_tr if language == "tr" else general_en,
        opportunity_tr if language == "tr" else opportunity_en,
        risk_tr if language == "tr" else risk_en,
    )


def _top_three(
    dataframe: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    priority_col = _first_existing_column(
        result,
        [
            "PriorityOrder",
            "PriorityTier",
            "priority",
            "Priority",
        ],
    )

    value_col = _first_existing_column(
        result,
        [
            "ExpectedNetValue",
            "ExpectedIncrementalTrafficValue",
            "expected_net_value",
            "estimated_value",
            "TrafficValue",
        ],
    )

    confidence_col = _first_existing_column(
        result,
        [
            "ConfidenceScore",
            "confidence_score",
            "ConfidenceLevel",
        ],
    )

    if value_col is not None:
        numeric = pd.to_numeric(
            result[value_col],
            errors="coerce",
        )
        result = result.assign(_sort_value=numeric)

        result = result.sort_values(
            "_sort_value",
            ascending=(mode == "risk"),
            na_position="last",
        )

    elif priority_col is not None:
        result = result.sort_values(
            priority_col,
            ascending=(mode == "risk"),
            na_position="last",
        )

    if confidence_col is not None:
        result = result.sort_values(
            by=[confidence_col],
            ascending=False,
            na_position="last",
        )

    return result.head(3).drop(
        columns=["_sort_value"],
        errors="ignore",
    )


# ============================================================
# LOAD DATA
# ============================================================

data = load_executive_data()

available_start, available_end = get_available_date_bounds(
    data.daily
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
        f"{t('executive_overview', initial_language)}"
    ),
    page_icon="📈",
    title=(
        "Yönetici Özeti"
        if initial_language == "tr"
        else "Executive Overview"
    ),
    subtitle=(
        (
            "Google Search Console ve GA4 verilerini; organik görünürlük, "
            "SEO fırsatları, riskler ve iş değeriyle tek bir karar destek "
            "görünümünde birleştirir."
        )
        if initial_language == "tr"
        else
        (
            "Combines Google Search Console and GA4 data with organic "
            "visibility, SEO opportunities, risks and business value in "
            "a single decision-support view."
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
# RUNTIME
# ============================================================

runtime_info = get_llm_runtime_info()

llm_ready = bool(
    runtime_info.get(
        "ready",
        False,
    )
)

if not llm_ready:
    st.info(
        localized_text(
            language,
            (
                "LLM bağlantısı aktif değil. Yönetici Özeti deterministik "
                "SEO karar zekâsı modunda çalışıyor; GSC + GA4, KPI, model "
                "ve öneri analizleri kullanılmaya devam eder."
            ),
            (
                "No LLM connection is active. Executive Overview is running "
                "in deterministic SEO decision-intelligence mode; GSC + GA4, "
                "KPI, model and recommendation analyses remain available."
            ),
        ),
        icon="ℹ️",
    )


# ============================================================
# SELECTED-PERIOD DECISION INTELLIGENCE
# ============================================================

kpi_source = (
    data.integrated
    if not data.integrated.empty
    else data.daily
)

decision_result = build_executive_decision_intelligence(
    dataframe=kpi_source,
    recommendations=data.recommendations,
    start_date=filters.start_date,
    end_date=filters.end_date,
    comparison_start_date=filters.comparison_start_date,
    comparison_end_date=filters.comparison_end_date,
    language=language,
    forecast_horizon_days=getattr(filters, "forecast_horizon_days", 7),
    limit=30,
)

kpis = decision_result.comparison.current
comparison_kpis = decision_result.comparison.previous
kpi_deltas = decision_result.comparison.deltas
decisions = decision_result.decisions
page_changes = decision_result.page_changes

recommendation_summary = build_recommendation_summary(
    data.recommendations
)

model_summary = build_model_summary(
    data.model_metrics
)

# The Decision Engine is now the canonical opportunity source for this page.
opportunities = decisions.copy()

top_pages = (
    page_changes.sort_values("CurrentClicks", ascending=False).head(20)
    if not page_changes.empty and "CurrentClicks" in page_changes.columns
    else build_executive_top_pages(kpi_source, limit=20)
)

st.caption(
    localized_text(
        language,
        (
            f"Uygulanan analiz dönemi: "
            f"{filters.start_date:%d.%m.%Y} – {filters.end_date:%d.%m.%Y}. "
            "Tarih, karşılaştırma ve projeksiyon seçimleri üstteki "
            "'Analizi Uygula' düğmesiyle bu sayfaya doğrudan yansır. "
            "Bu işlem API çağrısı, crawl, model eğitimi veya SHAP üretimi başlatmaz."
        ),
        (
            f"Applied analysis period: "
            f"{filters.start_date:%d.%m.%Y} – {filters.end_date:%d.%m.%Y}. "
            "Date, comparison and projection selections are applied directly "
            "to this page with the 'Apply Analysis' button above. "
            "This does not trigger API collection, crawling, model training, "
            "or SHAP generation."
        ),
    )
)


# ============================================================
# EXECUTIVE DECISION SUMMARY
# ============================================================

st.subheader(
    localized_text(
        language,
        "Yönetici Karar Özeti",
        "Executive Decision Summary",
    )
)

def _delta_text(metric: str, label_tr: str, label_en: str) -> str:
    value = kpi_deltas.get(metric)
    label = label_tr if language == "tr" else label_en
    if value is None:
        return f"{label}: -"
    return f"{label}: {float(value):+.1f}%"

comparison_available = bool(
    getattr(decision_result.comparison, "comparison_available", True)
)

if comparison_available:
    change_message = " | ".join([
        _delta_text("clicks_pct", "Tıklama", "Clicks"),
        _delta_text("impressions_pct", "Gösterim", "Impressions"),
        _delta_text("ctr_pct", "CTR", "CTR"),
    ])
else:
    change_message = localized_text(
        language,
        (
            f"Mevcut dönem: {format_integer(kpis.get('clicks', 0))} tıklama | "
            f"{format_integer(kpis.get('impressions', 0))} gösterim | "
            f"%{float(kpis.get('ctr', 0)) * 100:.1f} CTR. "
            "Seçilen karşılaştırma dönemi için yeterli historical veri bulunmuyor; değişim hesaplanmadı."
        ),
        (
            f"Current period: {format_integer(kpis.get('clicks', 0))} clicks | "
            f"{format_integer(kpis.get('impressions', 0))} impressions | "
            f"{float(kpis.get('ctr', 0)) * 100:.1f}% CTR. "
            "There is not enough historical data for the selected comparison period, so no change was calculated."
        ),
    )

if not comparison_available:
    st.warning(
        localized_text(
            language,
            "Karşılaştırma dönemi için yeterli kayıtlı veri yok. Mevcut dönem analizi devam ediyor; karşılaştırmalı değişim üretilmiyor.",
            "There is not enough stored data for the comparison period. Current-period analysis continues, but comparative change is not generated.",
        )
    )

if decisions.empty:
    location_message = localized_text(
        language,
        "Seçilen dönem için sayfa bazlı karar üretilemedi.",
        "No page-level decision could be produced for the selected period.",
    )
    action_message = localized_text(
        language,
        "Karar için yeterli karşılaştırma/öneri verisi bulunmuyor.",
        "There is not enough comparison/recommendation data for a decision.",
    )
else:
    first = decisions.iloc[0]
    location_message = (
        f"{first.get('ProblemOpportunity', '-')}: {first.get('page', '-')}\n\n"
        f"{first.get('Evidence', '-')}"
    )
    decision_reason = first.get("Why", "-")
    if not comparison_available:
        decision_reason = localized_text(
            language,
            "Karşılaştırmalı değişim hesaplanmadı. Aşağıdaki aksiyon mevcut dönem model, iş kuralı ve öneri sinyallerine dayanır.",
            "Comparative change was not calculated. The action below is based on current-period model, business-rule, and recommendation signals.",
        )
    action_message = (
        f"{decision_reason}\n\n"
        f"{localized_text(language, 'Önerilen aksiyon', 'Recommended action')}: "
        f"{first.get('Action', '-')}\n\n"
        f"{localized_text(language, 'Beklenen etki', 'Expected impact')}: "
        f"{first.get('ExpectedImpact', '-')}"
    )

summary_columns = st.columns(3)
with summary_columns[0]:
    st.info(
        f"**{localized_text(language, 'Ne oldu?', 'What happened?')}**\n\n{change_message}"
    )
with summary_columns[1]:
    st.warning(
        f"**{localized_text(language, 'Nerede?', 'Where?')}**\n\n{location_message}"
    )
with summary_columns[2]:
    st.success(
        f"**{localized_text(language, 'Neden / Ne yapmalı?', 'Why / What next?')}**\n\n{action_message}"
    )

with st.expander(
    localized_text(language, "Karar Motoru Detayı", "Decision Engine Detail"),
    expanded=False,
):
    st.write(
        localized_text(
            language,
            "Bu özet seçilen dönem ile karşılaştırma dönemini aynı ortak Decision Engine içinde değerlendirir. Sayfa değişimleri, öneriler, öncelik ve beklenen etki aynı karar sözleşmesinden gelir.",
            "This summary evaluates the selected and comparison periods through the same shared Decision Engine. Page changes, recommendations, priority, and expected impact come from one decision contract.",
        )
    )


# ============================================================
# OPTIMIZATION RECOMMENDATIONS
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "Optimizasyon Önerileri",
        "Optimization Recommendations",
    )
)

st.info(
    localized_text(
        language,
        (
            "Bu öneriler seçili analiz dönemindeki SEO karar çıktısını temsil eder. "
            "Sayfa, sayfa türü, senaryo, güven ve beklenen iş değeri birlikte değerlendirilir."
        ),
        (
            "These recommendations represent the SEO decision output for the selected "
            "analysis period. Page, page type, scenario, confidence and expected business "
            "value are evaluated together."
        ),
    )
)

render_export_buttons(
    dataframe=opportunities,
    basename="seo_optimization_recommendations",
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
    opportunities,
    limit=20,
)


# ============================================================
# KPI SNAPSHOT
# ============================================================

st.divider()

kpi_columns = st.columns(5)

kpi_columns[0].metric(
    t("clicks", language),
    format_integer(_safe_value(kpis, "clicks", 0)),
    delta=(
        f"{kpi_deltas['clicks_pct']:+.1f}%"
        if kpi_deltas.get("clicks_pct") is not None
        else None
    ),
)

kpi_columns[1].metric(
    t("impressions", language),
    format_integer(_safe_value(kpis, "impressions", 0)),
    delta=(
        f"{kpi_deltas['impressions_pct']:+.1f}%"
        if kpi_deltas.get("impressions_pct") is not None
        else None
    ),
)

kpi_columns[2].metric(
    t("ctr", language),
    format_percent(_safe_value(kpis, "ctr", 0)),
    delta=(
        f"{kpi_deltas['ctr_pct']:+.1f}%"
        if kpi_deltas.get("ctr_pct") is not None
        else None
    ),
)

kpi_columns[3].metric(
    t("average_position", language),
    format_position(
        _safe_value(kpis, "position", 0)
    ),
)

kpi_columns[4].metric(
    t("revenue", language),
    format_currency(
        _safe_value(kpis, "revenue", 0)
    ),
)


# ============================================================
# TOP 3 OPPORTUNITIES / RISKS
# ============================================================

st.divider()

top_columns = st.columns(2)

with top_columns[0]:
    st.subheader(
        localized_text(
            language,
            "İlk 3 Fırsat",
            "Top 3 Opportunities",
        )
    )

    top_opportunities = _top_three(
        opportunities,
        "opportunity",
    )

    if top_opportunities.empty:
        st.info(
            t(
                "no_data",
                language,
            )
        )
    else:
        render_localized_dataframe(
            top_opportunities,
            width="stretch",
            hide_index=True,
        )

with top_columns[1]:
    st.subheader(
        localized_text(
            language,
            "İlk 3 Risk",
            "Top 3 Risks",
        )
    )

    top_risks = _top_three(
        opportunities,
        "risk",
    )

    if top_risks.empty:
        st.info(
            t(
                "no_data",
                language,
            )
        )
    else:
        render_localized_dataframe(
            top_risks,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# BUSINESS VALUE
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "SEO İş Değeri Özeti",
        "SEO Business Value Summary",
    )
)

value_columns = st.columns(4)

# Keep business value consistent with the same top Decision Engine decision
# shown in the Executive Decision Summary.
if not decisions.empty:
    canonical_decision = decisions.iloc[0]
    business_incremental_value = canonical_decision.get(
        "ExpectedIncrementalTrafficValue", 0
    )
    business_net_value = canonical_decision.get("ExpectedNetValue", 0)
    business_roi = canonical_decision.get("EstimatedROI", 0)
    business_action = canonical_decision.get("Action", "-")
else:
    business_incremental_value = 0
    business_net_value = 0
    business_roi = 0
    business_action = "-"

value_columns[0].metric(
    t(
        "incremental_traffic_value",
        language,
    ),
    format_currency(
        business_incremental_value
    ),
)

value_columns[1].metric(
    t(
        "expected_net_value",
        language,
    ),
    format_currency(
        business_net_value
    ),
)

value_columns[2].metric(
    t(
        "estimated_roi",
        language,
    ),
    f"{float(0 if pd.isna(business_roi) else business_roi):.2f}",
)

value_columns[3].metric(
    t(
        "recommended_action",
        language,
    ),
    _localized_action(
        business_action,
        language,
    ),
)


# ============================================================
# REPORT EXPORT
# ============================================================

st.divider()

st.subheader(
    localized_text(
        language,
        "Raporu Dışa Aktar",
        "Export Report",
    )
)

export_columns = st.columns(2)

with export_columns[0]:
    render_export_buttons(
        dataframe=opportunities,
        basename="seo_executive_recommendations",
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

with export_columns[1]:
    if st.button(
        localized_text(
            language,
            "AI Asistanına Sor",
            "Ask AI Assistant",
        ),
        width="stretch",
        key="executive_ask_ai",
    ):
        st.switch_page(
            "pages/5_Ask_AI.py"
        )


# ============================================================
# DETAIL DATA VIEW
# ============================================================

st.divider()

with st.expander(
    localized_text(
        language,
        "Detaylı Veri Görünümü",
        "Detailed Data View",
    ),
    expanded=False,
):
    tabs = st.tabs(
        [
            localized_text(
                language,
                "Öneri Özeti",
                "Recommendation Summary",
            ),
            localized_text(
                language,
                "Sayfa Verisi",
                "Page Data",
            ),
            localized_text(
                language,
                "Günlük Veri",
                "Daily Data",
            ),
            localized_text(
                language,
                "Model Sonuçları",
                "Model Results",
            ),
        ]
    )

    with tabs[0]:
        render_export_buttons(
            dataframe=opportunities,
            basename="seo_recommendation_summary",
            csv_label="CSV",
            excel_label="Excel",
        )

        render_recommendations_table(
            opportunities,
            limit=50,
        )

    with tabs[1]:
        render_export_buttons(
            dataframe=top_pages,
            basename="seo_page_data",
            csv_label="CSV",
            excel_label="Excel",
        )

        render_localized_dataframe(
            top_pages,
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        daily = data.daily.copy()

        if daily.empty:
            st.info(
                t(
                    "no_data",
                    language,
                )
            )
        else:
            render_export_buttons(
                dataframe=daily,
                basename="seo_daily_data",
                csv_label="CSV",
                excel_label="Excel",
            )

            render_localized_dataframe(
                daily,
                width="stretch",
                hide_index=True,
            )

    with tabs[3]:
        model_columns = st.columns(3)

        model_columns[0].metric(
            localized_text(
                language,
                "Model Sayısı",
                "Model Count",
            ),
            model_summary["model_count"],
        )

        model_columns[1].metric(
            localized_text(
                language,
                "Ortalama R²",
                "Average R²",
            ),
            f"{model_summary['average_r2']:.3f}",
        )

        model_columns[2].metric(
            localized_text(
                language,
                "En İyi R²",
                "Best R²",
            ),
            f"{model_summary['best_r2']:.3f}",
        )

        if not data.model_metrics.empty:
            render_export_buttons(
                dataframe=data.model_metrics,
                basename="seo_model_metrics",
                csv_label="CSV",
                excel_label="Excel",
            )

            render_localized_dataframe(
                data.model_metrics,
                width="stretch",
                hide_index=True,
            )
