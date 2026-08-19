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
    render_feature_importance_table,
    render_kpi_row,
    render_line_chart,
    render_model_metrics_table,
    render_recommendations_table,
)
from dashboard.i18n import (
    t,
)
from dashboard.localization import render_localized_dataframe
from dashboard.layout import (
    initialize_dashboard,
    render_read_only_footer,
)
from dashboard.services import (
    build_model_summary,
    build_recommendation_summary,
    get_available_date_bounds,
    get_priority_recommendations,
    load_analysis_data,
)
from dashboard.utils import (
    format_currency,
    format_integer,
    format_number,
)
from src.llm.manager import (
    get_llm_runtime_info,
)


# ============================================================
# LOCAL HELPERS
# ============================================================


def render_divider() -> None:
    """Render a standard dashboard divider."""
    st.divider()


def render_section_header(
    title: str,
) -> None:
    """Render a standard dashboard section title."""
    st.subheader(
        title
    )


def render_deterministic_notice(
    language: str,
) -> None:
    """Show deterministic-mode information."""
    st.info(
        (
            "LLM bağlantısı aktif değil. "
            "Analizler deterministik SEO karar zekâsı, "
            "GSC + GA4 verileri, tahmin modelleri ve model açıklamalarıyla "
            "çalışmaya devam eder."
            if language == "tr"
            else
            "No LLM connection is active. "
            "Analysis continues using deterministic SEO "
            "decision intelligence, GSC + GA4, model, "
            "benchmark and SHAP outputs."
        )
    )


def render_footer(
    language: str,
    demo_mode: bool = False,
) -> None:
    """Render the standard dashboard footer."""
    _ = demo_mode

    render_read_only_footer(
        language
    )


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first existing column name."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def _truthy_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize common boolean CSV values."""
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "selected"})
    )


def _localize_model_target(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "Next_Clicks": (
            "Tıklama Tahmini",
            "Click Forecast",
        ),
        "Next_Impressions": (
            "Gösterim Tahmini",
            "Impression Forecast",
        ),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw


def _localize_priority(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "High Priority": ("Yüksek Öncelik", "High Priority"),
        "Medium Priority": ("Orta Öncelik", "Medium Priority"),
        "Low Priority": ("Düşük Öncelik", "Low Priority"),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw


def _localize_confidence(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "High": ("Yüksek", "High"),
        "Medium": ("Orta", "Medium"),
        "Low": ("Düşük", "Low"),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw


def _forecast_horizon_label(
    days: int,
    language: str,
) -> str:
    mapping = {
        7: ("7 Gün", "7 Days"),
        14: ("14 Gün", "14 Days"),
        30: ("30 Gün", "30 Days"),
        90: ("3 Ay (90 Gün)", "3 Months (90 Days)"),
        180: ("6 Ay (180 Gün)", "6 Months (180 Days)"),
        365: ("1 Yıl (365 Gün)", "1 Year (365 Days)"),
    }
    labels = mapping.get(
        int(days),
        (f"{days} Gün", f"{days} Days"),
    )
    return labels[0] if language == "tr" else labels[1]


def _forecast_horizon_type(
    days: int,
    language: str,
) -> str:
    if int(days) <= 30:
        return (
            "Operasyonel ML Tahmini"
            if language == "tr"
            else "Operational ML Forecast"
        )

    return (
        "Stratejik ML Tahmini"
        if language == "tr"
        else "Strategic ML Forecast"
    )


def _localize_action(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "Maintain": ("Mevcut Durumu Koru", "Maintain Current State"),
        "Review": ("İncele", "Review"),
        "Optimize Title and Meta": (
            "Başlık ve Meta Optimizasyonu",
            "Optimize Title and Meta",
        ),
        "Apply Full SEO and GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Apply Full SEO + GEO Optimization",
        ),
        "Apply Full SEO + GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Apply Full SEO + GEO Optimization",
        ),
        "full_seo_geo_optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Full SEO + GEO Optimization",
        ),
        "content_refresh": (
            "İçerik Güncellemesi",
            "Content Refresh",
        ),
        "technical_fix": (
            "Teknik SEO Düzeltmesi",
            "Technical SEO Fix",
        ),
        "internal_linking": (
            "İç Link Optimizasyonu",
            "Internal Linking Optimization",
        ),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw.replace("_", " ").strip().title()


def _localize_recommendation_reason(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "Current page performance supports maintaining the existing SEO setup.": (
            "Mevcut performans sinyalleri SEO yapısının korunmasını destekliyor.",
            "Current performance signals support maintaining the existing SEO setup.",
        ),
        "Visibility exists, but CTR can be improved through stronger title and meta messaging.": (
            "Görünürlük var; ancak başlık ve meta mesajı güçlendirilerek CTR artırılabilir.",
            "Visibility exists, but CTR can be improved through stronger title and meta messaging.",
        ),
        "Ranking and relevance signals indicate a content refresh opportunity.": (
            "Sıralama ve alaka sinyalleri içerik güncelleme fırsatına işaret ediyor.",
            "Ranking and relevance signals indicate a content refresh opportunity.",
        ),
        "Additional internal links may improve discoverability and authority distribution.": (
            "Ek iç linkler keşfedilebilirliği ve sayfa otoritesi dağılımını iyileştirebilir.",
            "Additional internal links may improve discoverability and authority distribution.",
        ),
        "The category has potential for broader semantic coverage and additional organic demand.": (
            "Kategori daha geniş semantik kapsama ve ek organik talep kazanma potansiyeline sahip.",
            "The category has potential for broader semantic coverage and additional organic demand.",
        ),
        "Richer product descriptions, benefits, attributes, image text and FAQs may improve relevance and commercial search performance.": (
            "Daha zengin ürün açıklamaları, faydalar, özellikler, görsel metinleri ve SSS içeriği alaka ve ticari arama performansını artırabilir.",
            "Richer product descriptions, benefits, attributes, image text and FAQs may improve relevance and commercial search performance.",
        ),
        "Page-type-aligned structured data may improve machine readability and rich-result eligibility.": (
            "Sayfa türüne uygun yapılandırılmış veri makine okunabilirliğini ve zengin sonuç uygunluğunu artırabilir.",
            "Page-type-aligned structured data may improve machine readability and rich-result eligibility.",
        ),
        "Direct answer blocks and structured entities may improve generative search visibility.": (
            "Doğrudan yanıt blokları ve yapılandırılmış entity sinyalleri üretken arama görünürlüğünü artırabilir.",
            "Direct answer blocks and structured entities may improve generative search visibility.",
        ),
        "Clear entity relationships, authorship, freshness and trust signals may strengthen E-E-A-T and generative-search readiness.": (
            "Entity ilişkileri, yazarlık, güncellik ve güven sinyalleri E-E-A-T ve üretken arama hazırlığını güçlendirebilir.",
            "Clear entity relationships, authorship, freshness and trust signals may strengthen E-E-A-T and generative-search readiness.",
        ),
        "Combining metadata, content, internal linking, structured data, entity signals and GEO components offers the broadest growth opportunity.": (
            "Metadata, içerik, iç linkleme, yapılandırılmış veri, entity sinyalleri ve GEO bileşenlerinin birlikte iyileştirilmesi en geniş büyüme fırsatını sunuyor.",
            "Combining metadata, content, internal linking, structured data, entity signals and GEO components offers the broadest growth opportunity.",
        ),
        "Low-confidence recommendation. Manual SEO validation is required.": (
            "Model güveni düşük; uygulama öncesinde ek SEO doğrulaması gerekli.",
            "Model confidence is low; additional SEO validation is required before implementation.",
        ),
        "Manual review recommended.": (
            "Ek inceleme öneriliyor.",
            "Additional review is recommended.",
        ),
    }
    pair = mapping.get(raw)
    if pair is not None:
        return pair[0] if language == "tr" else pair[1]
    return raw


def _problem_statement(row: pd.Series, language: str) -> str:
    scenario = str(row.get("Scenario", "")).strip()
    action = str(row.get("RecommendedAction", "")).strip()
    key = scenario or action
    mapping = {
        "maintain": ("Acil optimizasyon sorunu görünmüyor.", "No urgent optimization issue is detected."),
        "title_meta_optimization": ("Görünürlük mevcut ancak tıklama kazanımı geliştirilebilir.", "Visibility exists, but click capture can be improved."),
        "content_refresh": ("İçerik güncelliği veya alaka sinyalleri büyümeyi sınırlıyor olabilir.", "Content freshness or relevance signals may be limiting growth."),
        "internal_linking_boost": ("İç link desteği ve otorite dağılımı yetersiz olabilir.", "Internal-link support and authority distribution may be insufficient."),
        "category_expansion": ("Kategori mevcut semantik talebin tamamını kapsamıyor olabilir.", "The category may not cover the full available semantic demand."),
        "product_content_enrichment": ("Ürün içeriği ticari arama niyetini yeterince karşılamıyor olabilir.", "Product content may not fully satisfy commercial search intent."),
        "structured_data_upgrade": ("Makine okunabilirliği ve zengin sonuç uygunluğu geliştirilebilir.", "Machine readability and rich-result eligibility can be improved."),
        "geo_answer_optimization": ("Sayfa doğrudan yanıt ve üretken arama görünürlüğü için yeterince yapılandırılmamış olabilir.", "The page may be under-optimized for direct answers and generative-search visibility."),
        "entity_eet_upgrade": ("Entity, güven ve E-E-A-T sinyalleri güçlendirilebilir.", "Entity, trust, and E-E-A-T signals can be strengthened."),
        "full_seo_geo_optimization": ("Birden fazla SEO/GEO sinyali birlikte iyileştirme gerektiriyor.", "Multiple SEO/GEO signals require coordinated improvement."),
    }
    pair = mapping.get(key)
    if pair is None:
        return "İnceleme gerektiren fırsat/sinyal bulundu." if language == "tr" else "An opportunity or signal requires review."
    return pair[0] if language == "tr" else pair[1]


def _expected_result_text(row: pd.Series, language: str) -> str:
    clicks = pd.to_numeric(pd.Series([row.get("ExpectedIncrementalClicks")]), errors="coerce").iloc[0]
    pct = pd.to_numeric(pd.Series([row.get("ExpectedClicksChangePct")]), errors="coerce").iloc[0]
    position = pd.to_numeric(pd.Series([row.get("ExpectedPositionImprovement")]), errors="coerce").iloc[0]
    value = pd.to_numeric(pd.Series([row.get("ExpectedIncrementalTrafficValue")]), errors="coerce").iloc[0]
    parts: list[str] = []
    if pd.notna(clicks):
        parts.append((f"+{clicks:.1f} ek tıklama" if language == "tr" else f"+{clicks:.1f} incremental clicks"))
    if pd.notna(pct):
        parts.append((f"%{pct:.1f} tıklama değişimi" if language == "tr" else f"{pct:.1f}% click change"))
    if pd.notna(position) and position != 0:
        parts.append((f"{position:.1f} sıra pozisyon iyileşmesi" if language == "tr" else f"{position:.1f}-position improvement"))
    if pd.notna(value):
        parts.append((f"{value:.2f} ek trafik değeri" if language == "tr" else f"{value:.2f} incremental traffic value"))
    if not parts:
        return "Etki, sonraki model koşularında doğrulanmalıdır." if language == "tr" else "Impact should be validated in subsequent model runs."
    return " · ".join(parts)


def _localize_scenario(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "maintain": ("Mevcut Durumu Koru", "Maintain Current State"),
        "product_content_enrichment": (
            "Ürün İçeriğini Zenginleştir",
            "Enrich Product Content",
        ),
        "title_meta_optimization": (
            "Başlık ve Meta Optimizasyonu",
            "Title and Meta Optimization",
        ),
        "internal_linking_boost": (
            "İç Link Güçlendirme",
            "Internal Linking Boost",
        ),
        "structured_data_upgrade": (
            "Yapısal Veri Geliştirme",
            "Structured Data Upgrade",
        ),
        "content_refresh": (
            "İçerik Güncellemesi",
            "Content Refresh",
        ),
        "geo_answer_optimization": (
            "GEO Yanıt Optimizasyonu",
            "GEO Answer Optimization",
        ),
        "entity_eet_upgrade": (
            "Entity ve E-E-A-T Geliştirme",
            "Entity and E-E-A-T Upgrade",
        ),
        "category_expansion": (
            "Kategori SEO Genişletme",
            "Category SEO Expansion",
        ),
        "full_seo_geo_optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Full SEO + GEO Optimization",
        ),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw.replace("_", " ").strip().title()


def _feature_label(value: object, language: str) -> str:
    raw = str(value or "").strip()
    mapping = {
        "clicks": ("Tıklamalar", "Clicks"),
        "impressions": ("Gösterimler", "Impressions"),
        "position": ("Ortalama Pozisyon", "Average Position"),
        "CTR": ("Tıklama Oranı (CTR)", "Click-Through Rate (CTR)"),
        "TrafficValue": ("Organik Trafik Değeri", "Organic Traffic Value"),
        "VisibilityScore": ("Görünürlük Skoru", "Visibility Score"),
        "clicks_lag_1": ("Bir Önceki Gözlem Tıklaması", "Previous Observation Clicks"),
        "clicks_lag_7_avg": ("7 Günlük Ortalama Tıklama", "7-Day Average Clicks"),
        "impressions_lag_1": ("Bir Önceki Gözlem Gösterimi", "Previous Observation Impressions"),
        "impressions_lag_7_avg": ("7 Günlük Ortalama Gösterim", "7-Day Average Impressions"),
        "position_lag_1": ("Bir Önceki Gözlem Pozisyonu", "Previous Observation Position"),
        "position_lag_7_avg": ("7 Günlük Ortalama Pozisyon", "7-Day Average Position"),
        "TrafficValue_lag_1": ("Bir Önceki Gözlem Trafik Değeri", "Previous Observation Traffic Value"),
        "TrafficValue_lag_7_avg": ("7 Günlük Ortalama Trafik Değeri", "7-Day Average Traffic Value"),
        "clicks_change": ("Tıklama Değişimi", "Click Change"),
        "impressions_change": ("Gösterim Değişimi", "Impression Change"),
        "position_change": ("Pozisyon Değişimi", "Position Change"),
        "ctr_change": ("CTR Değişimi", "CTR Change"),
        "ctr_lag_1": ("Bir Önceki Gözlem CTR", "Previous Observation CTR"),
        "ctr_lag_7_avg": ("7 Günlük Ortalama CTR", "7-Day Average CTR"),
        "sessions": ("Organik Oturumlar", "Organic Sessions"),
        "users": ("Kullanıcılar", "Users"),
        "engaged_sessions": ("Etkileşimli Oturumlar", "Engaged Sessions"),
        "engagement_rate": ("Etkileşim Oranı", "Engagement Rate"),
        "average_session_duration": ("Ortalama Oturum Süresi", "Average Session Duration"),
        "conversions": ("Dönüşümler", "Conversions"),
        "revenue": ("Gelir", "Revenue"),
        "purchases": ("Satın Almalar", "Purchases"),
        "add_to_carts": ("Sepete Ekleme", "Add to Carts"),
        "checkouts": ("Ödeme Adımı", "Checkouts"),
        "OrganicConversionRate": ("Organik Dönüşüm Oranı", "Organic Conversion Rate"),
        "RevenuePerOrganicSession": ("Organik Oturum Başına Gelir", "Revenue per Organic Session"),
        "RevenuePerOrganicClick": ("Organik Tıklama Başına Gelir", "Revenue per Organic Click"),
        "CartRate": ("Sepete Ekleme Oranı", "Cart Rate"),
        "CheckoutRate": ("Checkout Oranı", "Checkout Rate"),
        "day_of_week": ("Haftanın Günü", "Day of Week"),
        "day_of_month": ("Ayın Günü", "Day of Month"),
        "month_num": ("Ay", "Month"),
        "quarter": ("Çeyrek", "Quarter"),
        "is_weekend": ("Hafta Sonu Sinyali", "Weekend Signal"),
        "is_holiday": ("Tatil Günü Sinyali", "Holiday Signal"),
        "is_pre_holiday": ("Tatil Öncesi Sinyali", "Pre-Holiday Signal"),
    }
    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]
    return raw.replace("_", " ").strip().title()


def _display_dataframe(dataframe: pd.DataFrame, language: str) -> pd.DataFrame:
    result = dataframe.copy()

    if "Feature" in result.columns:
        result["Feature"] = result["Feature"].map(
            lambda value: _feature_label(value, language)
        )

    if "Model" in result.columns:
        result["Model"] = result["Model"].map(
            lambda value: _localize_model_target(value, language)
        )

    if "PriorityTier" in result.columns:
        result["PriorityTier"] = result["PriorityTier"].map(
            lambda value: _localize_priority(value, language)
        )

    if "ConfidenceLevel" in result.columns:
        result["ConfidenceLevel"] = result["ConfidenceLevel"].map(
            lambda value: _localize_confidence(value, language)
        )

    if "RecommendedAction" in result.columns:
        result["RecommendedAction"] = result["RecommendedAction"].map(
            lambda value: _localize_action(value, language)
        )

    if "RecommendationReason" in result.columns:
        result["RecommendationReason"] = result["RecommendationReason"].map(
            lambda value: _localize_recommendation_reason(value, language)
        )

    if "Scenario" in result.columns:
        result["Scenario"] = result["Scenario"].map(
            lambda value: _localize_scenario(value, language)
        )

    if "ScenarioLabel" in result.columns:
        result["ScenarioLabel"] = result["ScenarioLabel"].map(
            lambda value: _localize_scenario_label(
                value,
                language,
            )
        )

    for page_type_column in [
        "PageType",
        "page_type",
    ]:
        if page_type_column in result.columns:
            result[page_type_column] = result[
                page_type_column
            ].map(
                lambda value: _localize_page_type(
                    value,
                    language,
                )
            )

    for intent_column in [
        "KeywordIntent",
        "keyword_intent",
    ]:
        if intent_column in result.columns:
            result[intent_column] = result[
                intent_column
            ].map(
                lambda value: _localize_keyword_intent(
                    value,
                    language,
                )
            )

    if "Direction" in result.columns:
        result["Direction"] = result["Direction"].map(
            lambda value: _localize_direction(
                value,
                language,
            )
        )

    if "ValidationMethod" in result.columns:
        result["ValidationMethod"] = result[
            "ValidationMethod"
        ].map(
            lambda value: _localize_validation_method(
                value,
                language,
            )
        )

    if "ExecutiveCommentary" in result.columns:
        result["ExecutiveCommentary"] = result[
            "ExecutiveCommentary"
        ].map(
            lambda value: _localize_commentary(
                value,
                language,
            )
        )

    result = result.drop(
        columns=["CommentarySource", "RunID", "ModelRunTimestamp"],
        errors="ignore",
    )

    labels = {
        "Model": ("Tahmin Hedefi", "Forecast Target"),
        "Algorithm": ("Algoritma", "Algorithm"),
        "Selected": ("Seçilen Model", "Selected Model"),
        "MAE": ("MAE — Ortalama Mutlak Hata", "MAE — Mean Absolute Error"),
        "RMSE": ("RMSE — Tahmin Hatası", "RMSE — Prediction Error"),
        "R2": ("R² — Açıklama Gücü", "R² — Explained Variance"),
        "TrainRows": ("Eğitim Satırı", "Training Rows"),
        "TestRows": ("Test Satırı", "Test Rows"),
        "ValidationMethod": ("Doğrulama Yöntemi", "Validation Method"),
        "FirstTestDate": ("Test Başlangıcı", "Test Start"),
        "Status": ("Durum", "Status"),
        "Error": ("Hata", "Error"),
        "Feature": ("Etkileyen Değişken", "Feature"),
        "Importance": ("Önem Derecesi", "Importance"),
        "MeanAbsSHAP": ("Ortalama Etki Büyüklüğü", "Mean Absolute SHAP Impact"),
        "MeanSHAP": ("Ortalama Etki Yönü", "Mean SHAP Direction"),
        "PositiveImpactRows": ("Pozitif Etkilediği Gözlem", "Positive Impact Rows"),
        "NegativeImpactRows": ("Negatif Etkilediği Gözlem", "Negative Impact Rows"),
        "ZeroImpactRows": ("Sıfır Etkili Gözlem", "Zero Impact Rows"),
        "ImportanceRank": ("Etki Sırası", "Impact Rank"),
        "Page": ("Sayfa", "Page"),
        "page": ("Sayfa", "Page"),
        "ObservationDate": ("Analiz Tarihi", "Observation Date"),
        "PageType": ("Sayfa Türü", "Page Type"),
        "page_type": ("Sayfa Türü", "Page Type"),
        "KeywordIntent": ("Arama Niyeti", "Search Intent"),
        "keyword_intent": ("Arama Niyeti", "Search Intent"),
        "FeatureValue": ("Değişken Değeri", "Feature Value"),
        "SHAPValue": ("SHAP Etkisi", "SHAP Impact"),
        "AbsSHAPValue": ("Mutlak SHAP Etkisi", "Absolute SHAP Impact"),
        "Direction": ("Etki Yönü", "Impact Direction"),
        "BaseValue": ("Başlangıç Tahmini", "Baseline Prediction"),
        "Prediction": ("Model Tahmini", "Model Prediction"),
        "PriorityTier": ("Öncelik", "Priority"),
        "ConfidenceLevel": ("Güven Seviyesi", "Confidence"),
        "RecommendedAction": ("Önerilen Aksiyon", "Recommended Action"),
        "RecommendationReason": ("Neden / Kanıt", "Reason / Evidence"),
        "ExpectedNetValue": ("Beklenen Net Değer", "Expected Net Value"),
        "EstimatedROI": ("Ortalama Tahmini ROI", "Average Estimated ROI"),
        "Scenario": ("Senaryo", "Scenario"),
        "ScenarioLabel": ("Senaryo Açıklaması", "Scenario Label"),
        "ExecutiveCommentary": ("Yönetici Yorumu", "Executive Commentary"),
        "CommentarySource": ("Yorum Kaynağı", "Commentary Source"),
    }

    return result.rename(
        columns={
            column: pair[0] if language == "tr" else pair[1]
            for column, pair in labels.items()
            if column in result.columns
        }
    )


def _winner_reason(
    benchmark: pd.DataFrame,
    target: str,
    language: str,
) -> str:
    if benchmark.empty or "Selected" not in benchmark.columns:
        return ""

    data = benchmark.copy()
    data["_selected"] = _truthy_series(data["Selected"])
    rows = data[
        data["_selected"]
        & data["Model"].astype(str).eq(target)
    ]

    if rows.empty:
        return ""

    row = rows.iloc[0]
    algorithm = str(row.get("Algorithm", "-"))
    rmse = float(pd.to_numeric(pd.Series([row.get("RMSE")]), errors="coerce").fillna(0).iloc[0])
    mae = float(pd.to_numeric(pd.Series([row.get("MAE")]), errors="coerce").fillna(0).iloc[0])
    r2 = float(pd.to_numeric(pd.Series([row.get("R2")]), errors="coerce").fillna(0).iloc[0])
    target_label = _localize_model_target(target, language)

    if language == "tr":
        return (
            f"**Neden {algorithm}?** {target_label} için seçim önceliği en düşük "
            f"RMSE'ye verildi. Seçilen modelin RMSE değeri {rmse:.4f}, "
            f"MAE değeri {mae:.4f} ve R² değeri {r2:.4f}. Bu nedenle üretim "
            f"tahmininde **{algorithm}** kullanılıyor."
        )

    return (
        f"**Why {algorithm}?** For {target_label}, model selection prioritizes "
        f"the lowest RMSE. The selected model has RMSE {rmse:.4f}, "
        f"MAE {mae:.4f}, and R² {r2:.4f}; therefore **{algorithm}** is used "
        f"for production forecasting."
    )


def _shap_note(language: str) -> str:
    if language == "tr":
        return (
            "**Nasıl okunmalı?** Büyük mutlak SHAP değeri, ilgili değişkenin "
            "model tahminini daha güçlü etkilediğini gösterir. Pozitif SHAP "
            "tahmini yukarı, negatif SHAP ise aşağı yönlü etkiler."
        )
    return (
        "**How to read it?** A larger absolute SHAP value means the feature "
        "has a stronger influence on the prediction. Positive SHAP pushes the "
        "forecast upward; negative SHAP pushes it downward."
    )


def _localize_page_type(value: object, language: str) -> str:
    raw = str(value or "").strip().lower()

    mapping = {
        "category": ("Kategori", "Category"),
        "product": ("Ürün", "Product"),
        "blog": ("Blog", "Blog"),
        "homepage": ("Ana Sayfa", "Homepage"),
        "home": ("Ana Sayfa", "Homepage"),
        "landing": ("Açılış Sayfası", "Landing Page"),
        "other": ("Diğer", "Other"),
    }

    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]

    return str(value or "")


def _localize_keyword_intent(value: object, language: str) -> str:
    raw = str(value or "").strip()

    mapping = {
        "Transactional": ("İşlemsel", "Transactional"),
        "Commercial": ("Ticari", "Commercial"),
        "Commercial Investigation": ("Ticari Araştırma", "Commercial Investigation"),
        "Informational": ("Bilgilendirici", "Informational"),
        "Navigational": ("Navigasyonel", "Navigational"),
        "Uncategorized": ("Sınıflandırılmamış", "Uncategorized"),
    }

    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]

    return raw


def _localize_direction(value: object, language: str) -> str:
    raw = str(value or "").strip().lower()

    mapping = {
        "positive": ("Pozitif", "Positive"),
        "negative": ("Negatif", "Negative"),
        "increase": ("Artırıcı", "Increase"),
        "decrease": ("Azaltıcı", "Decrease"),
        "neutral": ("Nötr", "Neutral"),
        "zero": ("Sıfır", "Zero"),
    }

    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]

    return str(value or "")


def _localize_validation_method(value: object, language: str) -> str:
    raw = str(value or "").strip()

    mapping = {
        "time_aware_holdout": (
            "Zaman Duyarlı Holdout",
            "Time-Aware Holdout",
        ),
    }

    if raw in mapping:
        return mapping[raw][0] if language == "tr" else mapping[raw][1]

    return raw.replace("_", " ").title()


def _localize_commentary(value: object, language: str) -> str:
    text = str(value or "").strip()

    if not text:
        return text

    if language == "tr":
        replacements = {
            "Apply Full SEO and GEO Optimization": "Tam SEO + GEO Optimizasyonu",
            "Full SEO and GEO Optimization": "Tam SEO + GEO Optimizasyonu",
            "Optimize Title and Meta": "Başlık ve Meta Optimizasyonu",
            "Title and Meta Optimization": "Başlık ve Meta Optimizasyonu",
            "Maintain Current Setup": "Mevcut Durumu Koru",
            "Maintain current setup": "Mevcut Durumu Koru",
            "Maintain": "Mevcut durumu koru",
            "Review": "İncele",
            "High Priority": "Yüksek Öncelik",
            "Medium Priority": "Orta Öncelik",
            "Low Priority": "Düşük Öncelik",
            "model güveni High": "model güveni Yüksek",
            "model güveni Medium": "model güveni Orta",
            "model güveni Low": "model güveni Düşük",
            "Bu other sayfası": "Bu diğer sayfa",
            "Bu category sayfası": "Bu kategori sayfası",
            "Bu product sayfası": "Bu ürün sayfası",
            "category page": "kategori sayfası",
            "product page": "ürün sayfası",
            "blog page": "blog sayfası",
        }
    else:
        replacements = {
            "kategori sayfası": "category page",
            "ürün sayfası": "product page",
            "Ana Sayfa": "Homepage",
            "Mevcut durumu koru": "Maintain current state",
            "İncele": "Review",
            "Yüksek Öncelik": "High Priority",
            "Orta Öncelik": "Medium Priority",
            "Düşük Öncelik": "Low Priority",
        }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _section_explainer(
    language: str,
    tr_text: str,
    en_text: str,
) -> None:
    """Render a concise business-friendly explanation."""
    st.caption(
        tr_text
        if language == "tr"
        else en_text
    )


def _dataframe_date_bounds(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> tuple[object | None, object | None]:
    """Return min/max available dates for one output dataframe."""
    if dataframe.empty:
        return None, None

    for column in candidates:
        if column not in dataframe.columns:
            continue

        values = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if not values.empty:
            return (
                values.min().date(),
                values.max().date(),
            )

    return None, None


def _filter_by_period(
    dataframe: pd.DataFrame,
    start_date: object,
    end_date: object,
    candidates: list[str],
) -> pd.DataFrame:
    """Filter row-level output to the active dashboard period."""
    if dataframe.empty:
        return dataframe.copy()

    for column in candidates:
        if column not in dataframe.columns:
            continue

        parsed = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

        mask = (
            parsed.dt.date.ge(start_date)
            & parsed.dt.date.le(end_date)
        )

        return (
            dataframe.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

    return dataframe.copy()



def _safe_sum(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = _first_existing_column(
        dataframe,
        candidates,
    )
    if column is None or dataframe.empty:
        return 0.0
    return float(
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )


def _safe_mean(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = _first_existing_column(
        dataframe,
        candidates,
    )
    if column is None or dataframe.empty:
        return 0.0
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()
    return float(
        values.mean()
        if not values.empty
        else 0.0
    )


def _period_summary(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """Build actual-period KPI values from persisted daily output."""
    clicks = _safe_sum(
        dataframe,
        ["clicks", "Clicks"],
    )
    impressions = _safe_sum(
        dataframe,
        ["impressions", "Impressions"],
    )
    sessions = _safe_sum(
        dataframe,
        ["sessions", "Sessions"],
    )
    revenue = _safe_sum(
        dataframe,
        ["revenue", "Revenue"],
    )
    conversions = _safe_sum(
        dataframe,
        ["conversions", "Conversions", "purchases", "Purchases"],
    )
    position = _safe_mean(
        dataframe,
        ["position", "Position", "average_position", "AveragePosition"],
    )
    ctr = (
        clicks / impressions
        if impressions > 0
        else 0.0
    )
    return {
        "clicks": clicks,
        "impressions": impressions,
        "ctr": ctr,
        "position": position,
        "sessions": sessions,
        "revenue": revenue,
        "conversions": conversions,
    }


def _delta_percent(
    current: float,
    previous: float,
) -> float | None:
    if previous == 0:
        return None
    return (
        (current - previous)
        / abs(previous)
        * 100.0
    )


def _metric_delta_text(
    current: float,
    previous: float,
    *,
    reverse_good: bool = False,
) -> str | None:
    delta = _delta_percent(
        current,
        previous,
    )
    if delta is None:
        return None

    # Streamlit automatically uses the delta direction visually.
    # For position, a lower value is normally better, so invert the sign.
    display_delta = (
        -delta
        if reverse_good
        else delta
    )
    return f"{display_delta:+.1f}%"


def _page_change_table(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Build descriptive page-level change attribution."""
    if current.empty or previous.empty:
        return pd.DataFrame()

    page_column = _first_existing_column(current, ["page", "Page"])
    previous_page_column = _first_existing_column(previous, ["page", "Page"])
    click_column = _first_existing_column(current, ["clicks", "Clicks"])
    previous_click_column = _first_existing_column(previous, ["clicks", "Clicks"])
    impression_column = _first_existing_column(current, ["impressions", "Impressions"])
    previous_impression_column = _first_existing_column(previous, ["impressions", "Impressions"])

    if None in [page_column, previous_page_column, click_column, previous_click_column, impression_column, previous_impression_column]:
        return pd.DataFrame()

    cur = current[[page_column, click_column, impression_column]].copy()
    prev = previous[[previous_page_column, previous_click_column, previous_impression_column]].copy()

    cur[click_column] = pd.to_numeric(cur[click_column], errors="coerce").fillna(0)
    cur[impression_column] = pd.to_numeric(cur[impression_column], errors="coerce").fillna(0)
    prev[previous_click_column] = pd.to_numeric(prev[previous_click_column], errors="coerce").fillna(0)
    prev[previous_impression_column] = pd.to_numeric(prev[previous_impression_column], errors="coerce").fillna(0)

    cur = cur.groupby(page_column, as_index=False).agg(
        CurrentClicks=(click_column, "sum"),
        CurrentImpressions=(impression_column, "sum"),
    ).rename(columns={page_column: "Page"})

    prev = prev.groupby(previous_page_column, as_index=False).agg(
        PreviousClicks=(previous_click_column, "sum"),
        PreviousImpressions=(previous_impression_column, "sum"),
    ).rename(columns={previous_page_column: "Page"})

    merged = cur.merge(prev, on="Page", how="outer").fillna(0)
    merged["ClickChange"] = merged["CurrentClicks"] - merged["PreviousClicks"]
    merged["ImpressionChange"] = merged["CurrentImpressions"] - merged["PreviousImpressions"]
    merged["ClickChangePct"] = merged["ClickChange"] / merged["PreviousClicks"].replace(0, pd.NA) * 100
    merged["ImpressionChangePct"] = merged["ImpressionChange"] / merged["PreviousImpressions"].replace(0, pd.NA) * 100
    return merged.sort_values("ClickChange", ascending=False).reset_index(drop=True)


def _localize_scenario_label(
    value: object,
    language: str,
) -> str:
    raw = str(value or "").strip()
    mapping = {
        "Full SEO and GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Full SEO + GEO Optimization",
        ),
        "Title and Meta Optimization": (
            "Başlık ve Meta Optimizasyonu",
            "Title and Meta Optimization",
        ),
        "Maintain Current Setup": (
            "Mevcut Durumu Koru",
            "Maintain Current Setup",
        ),
        "Content Refresh": (
            "İçerik Güncellemesi",
            "Content Refresh",
        ),
        "Internal Linking Improvement": (
            "İç Link Güçlendirme",
            "Internal Linking Improvement",
        ),
        "Category SEO Expansion": (
            "Kategori SEO Genişletme",
            "Category SEO Expansion",
        ),
        "Structured Data Upgrade": (
            "Yapısal Veri Geliştirme",
            "Structured Data Upgrade",
        ),
        "GEO Answer Optimization": (
            "GEO Yanıt Optimizasyonu",
            "GEO Answer Optimization",
        ),
        "Entity and E-E-A-T Upgrade": (
            "Entity ve E-E-A-T Geliştirme",
            "Entity and E-E-A-T Upgrade",
        ),
    }
    if raw in mapping:
        return (
            mapping[raw][0]
            if language == "tr"
            else mapping[raw][1]
        )
    return raw




# ============================================================
# MODEL / SHAP HISTORY HELPERS
# ============================================================

SHAP_HISTORY_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "history"
)

SHAP_DETAIL_HISTORY_PATH = (
    SHAP_HISTORY_DIR
    / "seo_shap_detail_history.csv"
)

SHAP_SUMMARY_HISTORY_PATH = (
    SHAP_HISTORY_DIR
    / "seo_shap_summary_history.csv"
)


def _load_history_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Load an optional historical model artifact safely.

    Historical files are additive artifacts. The dashboard
    continues to work with the current/live output when a
    history file does not yet exist.
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except Exception as exc:
        st.warning(
            f"Historical model output could not be read: {exc}"
        )
        return pd.DataFrame()


def _build_shap_run_catalog(
    detail_history: pd.DataFrame,
    summary_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per historical SHAP/model-explanation run.
    """
    frames: list[pd.DataFrame] = []

    if (
        not detail_history.empty
        and "RunID" in detail_history.columns
    ):
        detail = detail_history.copy()

        if "ModelRunTimestamp" in detail.columns:
            detail["_RunTimestamp"] = pd.to_datetime(
                detail["ModelRunTimestamp"],
                errors="coerce",
            )
        else:
            detail["_RunTimestamp"] = pd.NaT

        observation_column = _first_existing_column(
            detail,
            [
                "ObservationDate",
                "observation_date",
                "Date",
                "date",
            ],
        )

        if observation_column is not None:
            detail["_ObservationDate"] = pd.to_datetime(
                detail[observation_column],
                errors="coerce",
            )
        else:
            detail["_ObservationDate"] = pd.NaT

        detail_catalog = (
            detail
            .groupby(
                "RunID",
                as_index=False,
                dropna=False,
            )
            .agg(
                ModelRunTimestamp=(
                    "_RunTimestamp",
                    "max",
                ),
                ObservationStart=(
                    "_ObservationDate",
                    "min",
                ),
                ObservationEnd=(
                    "_ObservationDate",
                    "max",
                ),
                DetailRows=(
                    "RunID",
                    "size",
                ),
            )
        )

        frames.append(
            detail_catalog
        )

    if (
        not summary_history.empty
        and "RunID" in summary_history.columns
    ):
        summary = summary_history.copy()

        if "ModelRunTimestamp" in summary.columns:
            summary["_RunTimestamp"] = pd.to_datetime(
                summary["ModelRunTimestamp"],
                errors="coerce",
            )
        else:
            summary["_RunTimestamp"] = pd.NaT

        summary_catalog = (
            summary
            .groupby(
                "RunID",
                as_index=False,
                dropna=False,
            )
            .agg(
                SummaryRunTimestamp=(
                    "_RunTimestamp",
                    "max",
                ),
                SummaryRows=(
                    "RunID",
                    "size",
                ),
            )
        )

        frames.append(
            summary_catalog
        )

    if not frames:
        return pd.DataFrame()

    catalog = frames[0].copy()

    for frame in frames[1:]:
        catalog = catalog.merge(
            frame,
            on="RunID",
            how="outer",
        )

    if "ModelRunTimestamp" not in catalog.columns:
        catalog["ModelRunTimestamp"] = pd.NaT

    if "SummaryRunTimestamp" in catalog.columns:
        catalog["ModelRunTimestamp"] = (
            pd.to_datetime(
                catalog["ModelRunTimestamp"],
                errors="coerce",
            )
            .fillna(
                pd.to_datetime(
                    catalog["SummaryRunTimestamp"],
                    errors="coerce",
                )
            )
        )

    for column in [
        "ObservationStart",
        "ObservationEnd",
        "ModelRunTimestamp",
    ]:
        if column in catalog.columns:
            catalog[column] = pd.to_datetime(
                catalog[column],
                errors="coerce",
            )

    return (
        catalog
        .sort_values(
            "ModelRunTimestamp",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def _candidate_shap_runs(
    catalog: pd.DataFrame,
    start_date: object,
    end_date: object,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer model-explanation runs whose observation window
    intersects the active reporting period.

    If no run intersects the reporting period, fall back to
    runs whose observation date is on/before the reporting
    period end. As a final fallback, expose all stored runs.
    """
    if catalog.empty:
        return (
            pd.DataFrame(),
            "none",
        )

    data = catalog.copy()

    if (
        "ObservationStart" in data.columns
        and "ObservationEnd" in data.columns
    ):
        observation_start = pd.to_datetime(
            data["ObservationStart"],
            errors="coerce",
        )
        observation_end = pd.to_datetime(
            data["ObservationEnd"],
            errors="coerce",
        )

        active_start = pd.Timestamp(
            start_date
        )
        active_end = pd.Timestamp(
            end_date
        )

        intersects = (
            observation_start.le(
                active_end
            )
            & observation_end.ge(
                active_start
            )
        )

        intersecting = data.loc[
            intersects.fillna(False)
        ].copy()

        if not intersecting.empty:
            return (
                intersecting,
                "intersects_period",
            )

        before_end = data.loc[
            observation_end.le(
                active_end
            ).fillna(False)
        ].copy()

        if not before_end.empty:
            return (
                before_end,
                "latest_before_period_end",
            )

    return (
        data,
        "all_history",
    )


def _shap_run_label(
    row: pd.Series,
    language: str,
) -> str:
    """
    Build a human-readable model-explanation run label.
    """
    run_id = str(
        row.get(
            "RunID",
            "-",
        )
    )

    run_timestamp = pd.to_datetime(
        row.get(
            "ModelRunTimestamp"
        ),
        errors="coerce",
    )

    observation_start = pd.to_datetime(
        row.get(
            "ObservationStart"
        ),
        errors="coerce",
    )

    observation_end = pd.to_datetime(
        row.get(
            "ObservationEnd"
        ),
        errors="coerce",
    )

    run_part = (
        run_timestamp.strftime(
            "%d.%m.%Y %H:%M"
        )
        if pd.notna(
            run_timestamp
        )
        else "-"
    )

    if (
        pd.notna(
            observation_start
        )
        and pd.notna(
            observation_end
        )
    ):
        if (
            observation_start.date()
            == observation_end.date()
        ):
            observation_part = (
                observation_start.strftime(
                    "%d.%m.%Y"
                )
            )
        else:
            observation_part = (
                observation_start.strftime(
                    "%d.%m.%Y"
                )
                + " – "
                + observation_end.strftime(
                    "%d.%m.%Y"
                )
            )
    else:
        observation_part = "-"

    if language == "tr":
        return (
            f"Model koşusu {run_part} "
            f"• Açıklanan veri {observation_part}"
        )

    return (
        f"Model run {run_part} "
        f"• Explained data {observation_part}"
    )


def _select_shap_run_data(
    dataframe: pd.DataFrame,
    run_id: str | None,
) -> pd.DataFrame:
    """
    Filter a historical SHAP dataframe to one RunID.
    """
    if (
        dataframe.empty
        or run_id is None
        or "RunID" not in dataframe.columns
    ):
        return pd.DataFrame()

    return (
        dataframe.loc[
            dataframe["RunID"]
            .astype(str)
            .eq(
                str(
                    run_id
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

# ============================================================
# LOAD DATA
# ============================================================

data = load_analysis_data()

recommendations = data.recommendations.copy()
model_metrics = data.model_metrics.copy()
model_benchmark = data.model_benchmark.copy()
feature_importance = data.feature_importance.copy()
shap_summary = data.shap_summary.copy()
shap_detail = data.shap_detail.copy()
scenarios = data.scenarios.copy()
ml_forecast_daily = data.ml_forecast_daily.copy()
ml_forecast_horizons = data.ml_forecast_horizons.copy()
ml_forecast_portfolio = data.ml_forecast_portfolio.copy()
ml_forecast_metrics = data.ml_forecast_metrics.copy()
ml_forecast_benchmark = data.ml_forecast_benchmark.copy()
ml_forecast_feature_importance = data.ml_forecast_feature_importance.copy()

shap_detail_history = _load_history_csv(
    SHAP_DETAIL_HISTORY_PATH
)

shap_summary_history = _load_history_csv(
    SHAP_SUMMARY_HISTORY_PATH
)


# ============================================================
# PAGE INITIALIZATION
# ============================================================

available_start, available_end = get_available_date_bounds(
    data.integrated.copy()
)

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{t('ai_insights', initial_language)}"
    ),
    page_icon="🧠",
    title=(
        "AI İçgörüleri"
        if initial_language == "tr"
        else "AI Insights"
    ),
    subtitle=(
        (
            "Seçilen dönemde ne olduğunu, hangi sayfaların değişimi etkilediğini, "
            "hangi aksiyonların öncelikli olduğunu ve beklenen etkileri inceleyin."
        )
        if initial_language == "tr"
        else
        (
            "See what happened in the selected period, which pages drove the change, "
            "what should be prioritized, and what impact is expected."
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
selected_start_date = filters.start_date
selected_end_date = filters.end_date
forecast_horizon_days = filters.forecast_horizon_days
forecast_start_date = filters.forecast_start_date
forecast_end_date = filters.forecast_end_date

# Fast date filtering: no API extraction or model retraining is triggered here.
# Period-aware tables use the selected dates immediately from persisted outputs.
period_integrated = _filter_by_period(
    data.integrated.copy(),
    selected_start_date,
    selected_end_date,
    ["date", "Date"],
)
period_training = _filter_by_period(
    data.training.copy(),
    selected_start_date,
    selected_end_date,
    ["date", "Date"],
)

period_daily = _filter_by_period(
    data.daily.copy(),
    selected_start_date,
    selected_end_date,
    ["date", "Date"],
)

comparison_daily = pd.DataFrame()
comparison_integrated = pd.DataFrame()

if (
    filters.comparison_start_date is not None
    and filters.comparison_end_date is not None
):
    comparison_daily = _filter_by_period(
        data.daily.copy(),
        filters.comparison_start_date,
        filters.comparison_end_date,
        ["date", "Date"],
    )

    comparison_integrated = _filter_by_period(
        data.integrated.copy(),
        filters.comparison_start_date,
        filters.comparison_end_date,
        ["date", "Date"],
    )

# Scenario outputs are a latest successful model snapshot.
# Historical dashboard date filters apply to realized performance,
# not to this model snapshot.
period_scenarios = scenarios.copy()

period_shap_detail = _filter_by_period(
    shap_detail,
    selected_start_date,
    selected_end_date,
    ["ObservationDate", "observation_date", "Date", "date"],
)

available_start, available_end = _dataframe_date_bounds(
    data.integrated.copy(),
    ["date", "Date"],
)

actual_start_date = selected_start_date
actual_end_date = selected_end_date

if available_start is not None:
    actual_start_date = max(
        selected_start_date,
        available_start,
    )

if available_end is not None:
    actual_end_date = min(
        selected_end_date,
        available_end,
    )

if available_start is not None and available_end is not None:
    if (
        selected_start_date < available_start
        or selected_end_date > available_end
    ):
        st.warning(
            (
                f"Seçilen dönem **{selected_start_date:%d.%m.%Y} – {selected_end_date:%d.%m.%Y}**, "
                f"ancak şu anda hızlı sorgulanabilen işlenmiş veri **{available_start:%d.%m.%Y} – {available_end:%d.%m.%Y}** aralığını kapsıyor. "
                f"Fiilen analiz edilen mevcut veri aralığı **{actual_start_date:%d.%m.%Y} – {actual_end_date:%d.%m.%Y}**. "
                "Seçilen dönemin veri kapsamı dışında kalan günleri KPI hesaplarına dahil edilmez."
                if language == "tr"
                else
                f"The selected period is **{selected_start_date:%d.%m.%Y} – {selected_end_date:%d.%m.%Y}**, "
                f"while the currently fast-queryable processed data covers **{available_start:%d.%m.%Y} – {available_end:%d.%m.%Y}**. "
                f"The actual analyzed data range is **{actual_start_date:%d.%m.%Y} – {actual_end_date:%d.%m.%Y}**. "
                "Days outside the available processed-data coverage are excluded from KPI calculations."
            )
        )
    else:
        st.caption(
            (
                f"⚡ Hızlı dönem filtresi aktif: {selected_start_date:%d.%m.%Y} – {selected_end_date:%d.%m.%Y}. "
                "Bu seçim modeli yeniden eğitmez."
                if language == "tr"
                else
                f"⚡ Fast period filter active: {selected_start_date:%d.%m.%Y} – {selected_end_date:%d.%m.%Y}. "
                "This selection does not retrain the model."
            )
        )

# Model metrics, benchmark, feature importance and global SHAP describe the
# latest successful production-model run and intentionally stay independent
# from the fast dashboard date filter.

# ============================================================
# LLM RUNTIME
# ============================================================

runtime_info = get_llm_runtime_info()

llm_ready = bool(
    runtime_info.get(
        "ready",
        False,
    )
)

if not llm_ready:
    render_deterministic_notice(
        language
    )


# ============================================================
# SUMMARIES
# ============================================================

model_summary = build_model_summary(
    model_metrics
)

recommendation_summary = (
    build_recommendation_summary(
        recommendations
    )
)


# ============================================================
# SELECTED PERIOD: WHAT HAPPENED?
# ============================================================

render_section_header(
    (
        "Seçilen Dönemde Ne Oldu?"
        if language == "tr"
        else "What Happened in the Selected Period?"
    )
)

current_period_summary = _period_summary(
    period_daily
)

comparison_period_summary = _period_summary(
    comparison_daily
)

comparison_requested = (
    filters.comparison_start_date is not None
    and filters.comparison_end_date is not None
)

has_comparison = (
    comparison_requested
    and not comparison_daily.empty
)

st.caption(
    (
        f"Bu bölüm fiilen analiz edilen **{actual_start_date:%d.%m.%Y} – {actual_end_date:%d.%m.%Y}** "
        "aralığındaki organik performansı gösterir. "
        + (
            f"Değişimler **{filters.comparison_start_date:%d.%m.%Y} – "
            f"{filters.comparison_end_date:%d.%m.%Y}** dönemiyle karşılaştırılır."
            if has_comparison
            else (
                f"Karşılaştırma dönemi **{filters.comparison_start_date:%d.%m.%Y} – "
                f"{filters.comparison_end_date:%d.%m.%Y}** seçildi; ancak bu dönem "
                "mevcut hızlı sorgulanabilir veri içinde bulunmadığı için değişim oranları gösterilemiyor."
                if comparison_requested
                else "Karşılaştırma dönemi seçilmemiştir."
            )
        )
        if language == "tr"
        else
        f"This section shows actual organic performance for "
        f"**{actual_start_date:%d.%m.%Y} – {actual_end_date:%d.%m.%Y}**. "
        + (
            f"Changes are compared with **{filters.comparison_start_date:%d.%m.%Y} – "
            f"{filters.comparison_end_date:%d.%m.%Y}**."
            if has_comparison
            else (
                f"The comparison period **{filters.comparison_start_date:%d.%m.%Y} – "
                f"{filters.comparison_end_date:%d.%m.%Y}** is selected, but it is not "
                "available in the current fast-queryable dataset, so deltas cannot be shown."
                if comparison_requested
                else "No comparison period is selected."
            )
        )
    )
)

period_has_sessions = (
    _first_existing_column(
        period_daily,
        ["sessions", "Sessions"],
    )
    is not None
)

period_has_conversions = (
    _first_existing_column(
        period_daily,
        ["conversions", "Conversions", "purchases", "Purchases"],
    )
    is not None
)

period_kpis = st.columns(6)

period_kpis[0].metric(
    "Tıklamalar" if language == "tr" else "Clicks",
    format_integer(current_period_summary["clicks"]),
    delta=(
        _metric_delta_text(
            current_period_summary["clicks"],
            comparison_period_summary["clicks"],
        )
        if has_comparison
        else None
    ),
)

period_kpis[1].metric(
    "Gösterimler" if language == "tr" else "Impressions",
    format_integer(current_period_summary["impressions"]),
    delta=(
        _metric_delta_text(
            current_period_summary["impressions"],
            comparison_period_summary["impressions"],
        )
        if has_comparison
        else None
    ),
)

period_kpis[2].metric(
    "CTR" if language == "tr" else "CTR",
    f"{current_period_summary['ctr'] * 100:.2f}%",
    delta=(
        _metric_delta_text(
            current_period_summary["ctr"],
            comparison_period_summary["ctr"],
        )
        if has_comparison
        else None
    ),
)

period_kpis[3].metric(
    "Ort. Pozisyon" if language == "tr" else "Avg. Position",
    format_number(
        current_period_summary["position"],
        decimals=2,
    ),
    delta=(
        _metric_delta_text(
            current_period_summary["position"],
            comparison_period_summary["position"],
            reverse_good=True,
        )
        if has_comparison
        else None
    ),
)

period_kpis[4].metric(
    "Organik Oturum" if language == "tr" else "Organic Sessions",
    (
        format_integer(
            current_period_summary["sessions"]
        )
        if period_has_sessions
        else ("Veri Yok" if language == "tr" else "No Data")
    ),
    delta=(
        _metric_delta_text(
            current_period_summary["sessions"],
            comparison_period_summary["sessions"],
        )
        if (
            has_comparison
            and period_has_sessions
        )
        else None
    ),
)

period_kpis[5].metric(
    "Dönüşüm" if language == "tr" else "Conversions",
    (
        format_integer(
            current_period_summary["conversions"]
        )
        if period_has_conversions
        else ("Veri Yok" if language == "tr" else "No Data")
    ),
    delta=(
        _metric_delta_text(
            current_period_summary["conversions"],
            comparison_period_summary["conversions"],
        )
        if (
            has_comparison
            and period_has_conversions
        )
        else None
    ),
)

if has_comparison:
    click_delta = _delta_percent(
        current_period_summary["clicks"],
        comparison_period_summary["clicks"],
    )
    impression_delta = _delta_percent(
        current_period_summary["impressions"],
        comparison_period_summary["impressions"],
    )
    ctr_delta = _delta_percent(
        current_period_summary["ctr"],
        comparison_period_summary["ctr"],
    )

    if language == "tr":
        st.info(
            "**Dönem özeti:** "
            f"Tıklamalar {click_delta:+.1f}% değişti, "
            f"gösterimler {impression_delta:+.1f}% değişti ve "
            f"CTR {ctr_delta:+.1f}% değişti."
            if (
                click_delta is not None
                and impression_delta is not None
                and ctr_delta is not None
            )
            else
            "Karşılaştırma döneminde yeterli veri olmadığı için bazı değişim oranları hesaplanamadı."
        )
    else:
        st.info(
            "**Period summary:** "
            f"Clicks changed {click_delta:+.1f}%, "
            f"impressions changed {impression_delta:+.1f}%, and "
            f"CTR changed {ctr_delta:+.1f}%."
            if (
                click_delta is not None
                and impression_delta is not None
                and ctr_delta is not None
            )
            else
            "Some changes could not be calculated because the comparison period has insufficient data."
        )

render_divider()


# ============================================================
# PERIOD CHANGE ATTRIBUTION
# ============================================================

if has_comparison:
    render_section_header(
        "Dönem Değişimi Hangi Sayfalardan Geldi?"
        if language == "tr"
        else "Which Pages Drove the Period Change?"
    )
    _section_explainer(
        language,
        "Bu bölüm seçilen dönem ile karşılaştırma dönemi arasındaki gerçek tıklama ve gösterim farkının hangi sayfalarda oluştuğunu gösterir. Bu bir nedensellik analizi değil, değişimin kaynak dağılımıdır.",
        "This section shows where the actual click and impression differences between the selected and comparison periods occurred. It is descriptive attribution, not causal analysis.",
    )
    page_changes = _page_change_table(period_integrated, comparison_integrated)
    if page_changes.empty:
        st.info("Sayfa bazında dönem karşılaştırması için yeterli veri bulunamadı." if language == "tr" else "There is not enough data for page-level period comparison.")
    else:
        positive_pages = page_changes[page_changes["ClickChange"] > 0].head(10).copy()
        negative_pages = page_changes[page_changes["ClickChange"] < 0].sort_values("ClickChange").head(10).copy()
        label_map = {
            "Page": ("Sayfa", "Page"),
            "CurrentClicks": ("Seçilen Dönem Tıklama", "Selected Period Clicks"),
            "PreviousClicks": ("Karşılaştırma Tıklama", "Comparison Clicks"),
            "ClickChange": ("Tıklama Farkı", "Click Change"),
            "ClickChangePct": ("Tıklama Değişimi (%)", "Click Change (%)"),
            "CurrentImpressions": ("Seçilen Dönem Gösterim", "Selected Period Impressions"),
            "PreviousImpressions": ("Karşılaştırma Gösterim", "Comparison Impressions"),
            "ImpressionChange": ("Gösterim Farkı", "Impression Change"),
        }
        def _change_display(frame: pd.DataFrame) -> pd.DataFrame:
            cols = ["Page", "CurrentClicks", "PreviousClicks", "ClickChange", "ClickChangePct", "CurrentImpressions", "PreviousImpressions", "ImpressionChange"]
            view = frame[cols].copy()
            return view.rename(columns={k: (v[0] if language == "tr" else v[1]) for k,v in label_map.items()})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**En Çok Tıklama Kazanan Sayfalar**" if language == "tr" else "**Top Click Gainers**")
            if positive_pages.empty:
                st.caption("Pozitif tıklama farkı bulunan sayfa yok." if language == "tr" else "No pages have a positive click change.")
            else:
                render_localized_dataframe(_change_display(positive_pages), width="stretch", hide_index=True)
        with c2:
            st.markdown("**En Çok Tıklama Kaybeden Sayfalar**" if language == "tr" else "**Top Click Decliners**")
            if negative_pages.empty:
                st.caption("Negatif tıklama farkı bulunan sayfa yok." if language == "tr" else "No pages have a negative click change.")
            else:
                render_localized_dataframe(_change_display(negative_pages), width="stretch", hide_index=True)
        parts=[]
        if not positive_pages.empty:
            r=positive_pages.iloc[0]
            parts.append((f"en büyük pozitif katkı **{r['Page']}** sayfasından geldi ({r['ClickChange']:+.0f} tıklama)" if language == "tr" else f"the largest positive contribution came from **{r['Page']}** ({r['ClickChange']:+.0f} clicks)"))
        if not negative_pages.empty:
            r=negative_pages.iloc[0]
            parts.append((f"en büyük negatif katkı **{r['Page']}** sayfasında oluştu ({r['ClickChange']:+.0f} tıklama)" if language == "tr" else f"the largest negative contribution came from **{r['Page']}** ({r['ClickChange']:+.0f} clicks)"))
        if parts:
            st.info(("**Değişim özeti:** " if language == "tr" else "**Change summary:** ") + "; ".join(parts) + ".")
    render_divider()


# ============================================================
# TRUE MULTI-HORIZON ML FORECAST
# ============================================================

render_section_header(
    (
        "Çok Ufuklu ML Tahmin Merkezi"
        if language == "tr"
        else "Multi-Horizon ML Forecast Center"
    )
)

_section_explainer(
    language,
    (
        "Bu bölüm senaryo değerlerini gün sayısıyla çarpmaz. Ayrı günlük takvim "
        "verisi üzerinde eğitilen üretim modeli, gelecekteki her günü sırayla tahmin eder. "
        "Bir günün tahmini bir sonraki günün lag ve trend girdilerine eklenir; böylece "
        "7, 14 ve 30 günlük ufuklar operasyonel; 90, 180 ve 365 günlük ufuklar stratejik ML tahminidir. Tüm toplamlar gerçek yinelemeli günlük ML tahmin yolundan hesaplanır."
    ),
    (
        "This section does not multiply scenario values by the number of days. A production "
        "model trained on calendar-day observations forecasts each future day sequentially. "
        "Each predicted day feeds the lag and trend features for the next step, so the 7, 14 "
        "7, 14 and 30-day horizons are operational forecasts; 90, 180 and 365-day horizons are strategic forecasts. All totals are derived from the genuine recursive daily ML forecast path."
    ),
)

selected_forecast = pd.DataFrame()

if (
    not ml_forecast_portfolio.empty
    and "HorizonDays" in ml_forecast_portfolio.columns
):
    selected_forecast = ml_forecast_portfolio.loc[
        pd.to_numeric(
            ml_forecast_portfolio["HorizonDays"],
            errors="coerce",
        ).eq(float(forecast_horizon_days))
    ].copy()

if selected_forecast.empty:
    st.info(
        (
            "Multi-horizon ML çıktıları henüz üretilmemiş. Yeni forecasting katmanını "
            "oluşturmak için pipeline'ı bir kez çalıştırın: `python main.py`."
            if language == "tr"
            else
            "Multi-horizon ML outputs have not been generated yet. Run the pipeline once "
            "to build the new forecasting layer: `python main.py`."
        )
    )
else:
    forecast_row = selected_forecast.iloc[0]

    predicted_clicks = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("PredictedClicks", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    predicted_impressions = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("PredictedImpressions", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    predicted_ctr = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("PredictedCTR", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    traffic_value = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("PredictedTrafficValue", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    click_change = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("ClickChangePct", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    reliability = float(
        pd.to_numeric(
            pd.Series([forecast_row.get("ForecastReliability", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )

    confidence_raw = str(
        forecast_row.get("ConfidenceLevel", "Low")
    )
    confidence_label = _localize_confidence(
        confidence_raw,
        language,
    )

    render_kpi_row(
        [
            {
                "label": "ML Tahmin Ufku" if language == "tr" else "ML Forecast Horizon",
                "value": _forecast_horizon_label(
                    forecast_horizon_days,
                    language,
                ),
                "help": _forecast_horizon_type(
                    forecast_horizon_days,
                    language,
                ),
            },
            {
                "label": "Tahmini Tıklamalar" if language == "tr" else "Forecast Clicks",
                "value": format_integer(predicted_clicks),
            },
            {
                "label": "Tahmini Gösterimler" if language == "tr" else "Forecast Impressions",
                "value": format_integer(predicted_impressions),
            },
            {
                "label": "Tahmini CTR" if language == "tr" else "Forecast CTR",
                "value": f"{predicted_ctr * 100:.2f}%",
            },
            {
                "label": "Tahmini Trafik Değeri" if language == "tr" else "Forecast Traffic Value",
                "value": format_currency(traffic_value),
            },
            {
                "label": "Tahmin Güveni" if language == "tr" else "Forecast Confidence",
                "value": confidence_label,
                "help": (
                    f"Reliability score: {reliability:.2f}"
                    if language == "en"
                    else f"Güvenilirlik skoru: {reliability:.2f}"
                ),
            },
        ]
    )

    st.caption(
        (
            (
                f"Seçilen {_forecast_horizon_label(forecast_horizon_days, language)} ML tahmininin, "
                f"hemen önceki eş uzunluktaki gerçek döneme göre tıklama değişimi "
                f"**{click_change:+.1f}%**. "
                + (
                    "Bu ufuk operasyonel tahmin sınıfındadır."
                    if forecast_horizon_days <= 30
                    else
                    "Bu ufuk stratejik tahmin sınıfındadır; uzun vadede belirsizlik biriktiği için "
                    "güvenilirlik skoru özellikle daha temkinli yorumlanmalıdır."
                )
                if language == "tr"
                else
                f"For the selected {_forecast_horizon_label(forecast_horizon_days, language)} ML horizon, "
                f"forecast clicks are **{click_change:+.1f}%** versus the immediately preceding real "
                f"period of equal length. "
                + (
                    "This is an operational forecast horizon."
                    if forecast_horizon_days <= 30
                    else
                    "This is a strategic forecast horizon; accumulated long-range uncertainty means "
                    "the reliability score should be interpreted more conservatively."
                )
            )
        )
    )

    if (
        not ml_forecast_daily.empty
        and "ForecastDate" in ml_forecast_daily.columns
        and "HorizonDay" in ml_forecast_daily.columns
    ):
        daily_view = ml_forecast_daily.copy()
        daily_view["HorizonDay"] = pd.to_numeric(
            daily_view["HorizonDay"], errors="coerce"
        )
        daily_view = daily_view.loc[
            daily_view["HorizonDay"].le(forecast_horizon_days)
        ].copy()

        if not daily_view.empty:
            portfolio_daily = (
                daily_view.groupby("ForecastDate", as_index=False)
                .agg(
                    PredictedClicks=("PredictedClicks", "sum"),
                    PredictedImpressions=("PredictedImpressions", "sum"),
                )
                .sort_values("ForecastDate")
            )

            chart_view = portfolio_daily.copy()
            chart_x = "ForecastDate"

            if forecast_horizon_days > 30:
                chart_view["ForecastDate"] = pd.to_datetime(
                    chart_view["ForecastDate"],
                    errors="coerce",
                )
                chart_view["ForecastMonth"] = (
                    chart_view["ForecastDate"]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )
                chart_view = (
                    chart_view.groupby(
                        "ForecastMonth",
                        as_index=False,
                    )
                    .agg(
                        PredictedClicks=("PredictedClicks", "sum"),
                        PredictedImpressions=("PredictedImpressions", "sum"),
                    )
                )
                chart_x = "ForecastMonth"

            chart_columns = st.columns(2)

            with chart_columns[0]:
                render_line_chart(
                    dataframe=chart_view,
                    x=chart_x,
                    y="PredictedClicks",
                    title=(
                        (
                            "Aylık Stratejik ML Tıklama Tahmini"
                            if forecast_horizon_days > 30
                            else "Günlük ML Tıklama Tahmini"
                        )
                        if language == "tr"
                        else (
                            "Monthly Strategic ML Click Forecast"
                            if forecast_horizon_days > 30
                            else "Daily ML Click Forecast"
                        )
                    ),
                    height=390,
                )

            with chart_columns[1]:
                render_line_chart(
                    dataframe=chart_view,
                    x=chart_x,
                    y="PredictedImpressions",
                    title=(
                        (
                            "Aylık Stratejik ML Gösterim Tahmini"
                            if forecast_horizon_days > 30
                            else "Günlük ML Gösterim Tahmini"
                        )
                        if language == "tr"
                        else (
                            "Monthly Strategic ML Impression Forecast"
                            if forecast_horizon_days > 30
                            else "Daily ML Impression Forecast"
                        )
                    ),
                    height=390,
                )

    if not ml_forecast_portfolio.empty:
        forecast_table = ml_forecast_portfolio[
            [
                column
                for column in [
                    "HorizonDays",
                    "ForecastStartDate",
                    "ForecastEndDate",
                    "PredictedClicks",
                    "PredictedImpressions",
                    "PredictedCTR",
                    "ClickChangePct",
                    "ImpressionChangePct",
                    "ForecastReliability",
                    "ConfidenceLevel",
                    "HorizonType",
                    "PageCount",
                ]
                if column in ml_forecast_portfolio.columns
            ]
        ].copy()

        if "ConfidenceLevel" in forecast_table.columns:
            forecast_table["ConfidenceLevel"] = forecast_table["ConfidenceLevel"].map(
                lambda value: _localize_confidence(value, language)
            )

        if "HorizonType" in forecast_table.columns:
            forecast_table["HorizonType"] = forecast_table["HorizonType"].map(
                lambda value: (
                    "Operasyonel"
                    if str(value) == "Operational" and language == "tr"
                    else "Stratejik"
                    if str(value) == "Strategic" and language == "tr"
                    else str(value)
                )
            )


        rename_map = {
            "HorizonDays": ("Tahmin Ufku (Gün)", "Forecast Horizon (Days)"),
            "ForecastStartDate": ("Tahmin Başlangıcı", "Forecast Start"),
            "ForecastEndDate": ("Tahmin Bitişi", "Forecast End"),
            "PredictedClicks": ("Tahmini Tıklamalar", "Forecast Clicks"),
            "PredictedImpressions": ("Tahmini Gösterimler", "Forecast Impressions"),
            "PredictedCTR": ("Tahmini CTR", "Forecast CTR"),
            "ClickChangePct": ("Tıklama Değişimi (%)", "Click Change (%)"),
            "ImpressionChangePct": ("Gösterim Değişimi (%)", "Impression Change (%)"),
            "ForecastReliability": ("Güvenilirlik Skoru", "Reliability Score"),
            "ConfidenceLevel": ("Güven Seviyesi", "Confidence"),
            "HorizonType": ("Tahmin Türü", "Forecast Type"),
            "PageCount": ("Tahmin Edilen Sayfa", "Forecast Pages"),
        }

        forecast_table = forecast_table.rename(
            columns={
                key: labels[0] if language == "tr" else labels[1]
                for key, labels in rename_map.items()
                if key in forecast_table.columns
            }
        )

        with st.expander(
            (
                "7 Gün – 1 Yıl ML Tahmin Karşılaştırması"
                if language == "tr"
                else "7-Day to 1-Year ML Forecast Comparison"
            ),
            expanded=False,
        ):
            render_localized_dataframe(
                forecast_table,
                width="stretch",
                hide_index=True,
            )

            if not ml_forecast_metrics.empty:
                st.markdown(
                    "**Günlük ML Model Doğrulaması**"
                    if language == "tr"
                    else "**Daily ML Model Validation**"
                )
                render_model_metrics_table(
                    ml_forecast_metrics,
                )

render_divider()


# ============================================================
# AI / MODEL STATUS
# ============================================================

render_section_header(
    (
        "Sistem ve Model Özeti"
        if language == "tr"
        else "System and Model Summary"
    )
)

_section_explainer(
    language,
    (
        "Bu alan sistemin şu anda LLM destekli mi yoksa deterministik modda mı "
        "çalıştığını ve seçilen üretim modellerinin genel doğruluk seviyesini gösterir."
    ),
    (
        "This section shows whether the system is running in LLM-assisted or "
        "deterministic mode and summarizes the accuracy of the selected production models."
    ),
)

render_kpi_row(
    [
        {
            "label": t(
                "ai_runtime_mode",
                language,
            ),
            "value": (
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
        },
        {
            "label": (
                "Üretim Modeli Sayısı"
                if language == "tr"
                else "Production Model Count"
            ),
            "value": format_integer(
                model_summary[
                    "model_count"
                ]
            ),
        },
        {
            "label": (
                "Ortalama R²"
                if language == "tr"
                else "Average R²"
            ),
            "value": format_number(
                model_summary[
                    "average_r2"
                ],
                decimals=3,
            ),
        },
        {
            "label": (
                "En İyi R²"
                if language == "tr"
                else "Best R²"
            ),
            "value": format_number(
                model_summary[
                    "best_r2"
                ],
                decimals=3,
            ),
        },
    ]
)


# ============================================================
# RECOMMENDATION INTELLIGENCE
# ============================================================

render_divider()

render_section_header(
    (
        "Şimdi Ne Yapmalıyız? — Karar Özeti"
        if language == "tr"
        else "What Should We Do Now? — Decision Summary"
    )
)

_section_explainer(
    language,
    (
        "Bu bölüm seçtiğiniz tarihte gerçekleşen performans değildir. "
        "Son başarılı model koşusunun ürettiği güncel SEO/GEO aksiyonlarını ve "
        "bunların tahmini ticari etkisini gösterir."
    ),
    (
        "This summary shows how many actions were generated, how many are high priority, "
        "and the estimated commercial impact of the recommendations."
    ),
)

render_kpi_row(
    [
        {
            "label": (
                "Toplam Öneri"
                if language == "tr"
                else "Total Recommendations"
            ),
            "value": format_integer(
                recommendation_summary[
                    "recommendation_count"
                ]
            ),
        },
        {
            "label": t(
                "high_priority",
                language,
            ),
            "value": format_integer(
                recommendation_summary[
                    "high_priority_count"
                ]
            ),
        },
        {
            "label": t(
                "incremental_traffic_value",
                language,
            ),
            "value": format_currency(
                recommendation_summary[
                    "expected_incremental_value"
                ]
            ),
        },
        {
            "label": t(
                "expected_net_value",
                language,
            ),
            "value": format_currency(
                recommendation_summary[
                    "expected_net_value"
                ]
            ),
        },
        {
            "label": t(
                "estimated_roi",
                language,
            ),
            "value": format_number(
                recommendation_summary[
                    "average_roi"
                ],
                decimals=2,
            ),
        },
    ]
)


# ============================================================

render_export_buttons(
    dataframe=recommendations,
    basename="seo_all_recommendations",
    csv_label="CSV",
    excel_label="Excel",
)

# ADVANCED MODEL DETAILS
# ============================================================

with st.expander(
    (
        "Gelişmiş Model Detayları"
        if language == "tr"
        else "Advanced Model Details"
    ),
    expanded=False,
):
    # MODEL METRICS
    # ============================================================

    render_divider()

    render_section_header(
        "Model Metrikleri" if language == "tr" else "Model Metrics"
    )

    st.info(
        (
            "Bu bölüm son başarılı model eğitiminin doğrulama sonuçlarıdır. "
            "Geçmiş model açıklaması seçimi yalnızca açıklama bağlamını değiştirir; "
            "MAE, RMSE ve R² kartları mevcut son üretim modelinin doğrulama sonuçlarıdır."
            if language == "tr"
            else
            "This section shows validation results from the latest successful production-model training. "
            "Selecting a historical SHAP run changes the explanation context only; "
            "MAE, RMSE, and R² remain the validation metrics of the current latest production model."
        )
    )

    st.caption(
        (
            "MAE ve RMSE tahmin hatasını ölçer; daha düşük değer daha iyidir. "
            "R² modelin verideki değişimi ne kadar iyi açıkladığını gösterir; "
            "1'e yaklaştıkça açıklama gücü artar."
            if language == "tr"
            else
            "MAE and RMSE measure prediction error; lower is better. "
            "R² shows how much variation in the data the model explains; "
            "values closer to 1 indicate stronger explanatory power."
        )
    )

    if model_metrics.empty:
        st.info(t("no_data", language))
    else:
        render_localized_dataframe(
            _display_dataframe(model_metrics, language),
            width="stretch",
            hide_index=True,
        )
        render_export_buttons(
            dataframe=model_metrics,
            basename="seo_model_metrics",
            csv_label="CSV",
            excel_label="Excel",
        )


    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    render_divider()

    render_section_header(
        "Özellik Önem Dereceleri"
        if language == "tr"
        else "Feature Importance"
    )

    st.caption(
        (
            "Bu bölüm, üretim modelinin tahmin oluştururken hangi değişkenlere "
            "daha fazla ağırlık verdiğini gösterir. Teknik adlar denetim ve "
            "geliştirici takibi için korunur."
            if language == "tr"
            else
            "This section shows which variables the production model weights most "
            "heavily when producing forecasts. Technical names are retained for "
            "auditability and developer traceability."
        )
    )

    if feature_importance.empty:
        st.info(t("no_data", language))
    else:
        feature_view = feature_importance.copy()

        if "Importance" in feature_view.columns:
            feature_view["_importance"] = pd.to_numeric(
                feature_view["Importance"],
                errors="coerce",
            )
            feature_view = feature_view.sort_values(
                "_importance",
                ascending=False,
                na_position="last",
            )

        feature_top = feature_view.head(30).drop(
            columns=["_importance"],
            errors="ignore",
        )

        render_localized_dataframe(
            _display_dataframe(feature_top, language),
            width="stretch",
            hide_index=True,
        )

        if "Feature" in feature_top.columns and "Importance" in feature_top.columns:
            chart_data = feature_top.head(15).copy()
            chart_data["FeatureLabel"] = chart_data["Feature"].map(
                lambda value: _feature_label(value, language)
            )
            chart_data["Importance"] = pd.to_numeric(
                chart_data["Importance"],
                errors="coerce",
            )

            importance_label = (
                "Önem Derecesi"
                if language == "tr"
                else "Importance"
            )
            feature_label = (
                "Etkileyen Değişken"
                if language == "tr"
                else "Feature Name"
            )

            chart_data = chart_data.rename(
                columns={
                    "FeatureLabel": feature_label,
                    "Importance": importance_label,
                }
            )

            render_bar_chart(
                dataframe=chart_data,
                x=feature_label,
                y=importance_label,
                title=(
                    "En Etkili Model Değişkenleri"
                    if language == "tr"
                    else "Most Influential Model Features"
                ),
            )

        render_export_buttons(
            dataframe=feature_importance,
            basename="seo_feature_importance",
            csv_label="CSV",
            excel_label="Excel",
        )


    # ============================================================
    # MODEL BENCHMARK / AUTOMATIC WINNER
    # ============================================================

    render_divider()

    render_section_header(
        (
            "Model Karşılaştırması ve Otomatik Seçim"
            if language == "tr"
            else "Model Benchmark and Automatic Selection"
        )
    )

    st.caption(
        (
            "Random Forest, XGBoost ve LightGBM aynı zaman duyarlı test setinde "
            "karşılaştırılır. Daha düşük RMSE ve MAE daha iyi; daha yüksek R² daha iyidir. "
            "Sistem bu ölçütlere göre üretim modelini otomatik seçer."
            if language == "tr"
            else
            "Random Forest, XGBoost and LightGBM are compared on the same time-aware "
            "holdout. Lower RMSE and MAE are better; higher R² is better. "
            "The system automatically selects the production model using these metrics."
        )
    )

    if model_benchmark.empty:
        st.info(t("no_data", language))
    else:
        benchmark_view = model_benchmark.copy()
        if "Selected" in benchmark_view.columns:
            benchmark_view["_SelectedBool"] = _truthy_series(benchmark_view["Selected"])
        else:
            benchmark_view["_SelectedBool"] = False

        selected_models = benchmark_view[benchmark_view["_SelectedBool"]].copy()
        benchmark_status_columns = st.columns(4)
        benchmark_status_columns[0].metric(
            "Benchmark Satırı" if language == "tr" else "Benchmark Rows",
            format_integer(len(benchmark_view)),
        )
        benchmark_status_columns[1].metric(
            "Seçilen Model" if language == "tr" else "Selected Models",
            format_integer(len(selected_models)),
        )

        def _winner(target: str) -> str:
            if selected_models.empty or "Model" not in selected_models.columns or "Algorithm" not in selected_models.columns:
                return "-"
            rows = selected_models[selected_models["Model"].astype(str).eq(target)]
            return str(rows["Algorithm"].iloc[0]) if not rows.empty else "-"

        benchmark_status_columns[2].metric(
            "Tıklama Modeli" if language == "tr" else "Click Forecast Model",
            _winner("Next_Clicks"),
        )
        benchmark_status_columns[3].metric(
            "Gösterim Modeli" if language == "tr" else "Impression Forecast Model",
            _winner("Next_Impressions"),
        )

        preferred_benchmark_columns = [
            "Model", "Algorithm", "Selected", "MAE", "RMSE", "R2",
            "TrainRows", "TestRows", "ValidationMethod", "FirstTestDate",
            "Status", "Error",
        ]
        visible_benchmark_columns = [c for c in preferred_benchmark_columns if c in benchmark_view.columns]
        benchmark_raw = benchmark_view[
            visible_benchmark_columns
        ].copy()

        render_localized_dataframe(
            _display_dataframe(
                benchmark_raw,
                language,
            ),
            width="stretch",
            hide_index=True,
        )

        clicks_reason = _winner_reason(
            model_benchmark,
            "Next_Clicks",
            language,
        )
        impressions_reason = _winner_reason(
            model_benchmark,
            "Next_Impressions",
            language,
        )

        if clicks_reason:
            st.info(clicks_reason)

        if impressions_reason:
            st.info(impressions_reason)
        render_export_buttons(
            dataframe=benchmark_view.drop(columns=["_SelectedBool"], errors="ignore"),
            basename="seo_model_benchmark",
            csv_label="CSV",
            excel_label="Excel",
        )



# ============================================================
# MODEL EXPLANATION HISTORY / RUN CONTEXT
# ============================================================

render_divider()

render_section_header(
    (
        "Model Açıklaması Geçmişi"
        if language == "tr"
        else "Model Explanation History"
    )
)

_section_explainer(
    language,
    (
        "Üstteki tarih filtresi gerçekleşen SEO performansını belirler. "
        "Bu alan ise model açıklamasının hangi saklanmış model koşusundan "
        "geldiğini gösterir. Model koşusu ile raporlama dönemi aynı kavram değildir."
    ),
    (
        "The date filter above defines actual SEO performance. "
        "This section identifies which stored model run supplies the SHAP explanation. "
        "A model run and the reporting period are different concepts."
    ),
)

shap_run_catalog = _build_shap_run_catalog(
    shap_detail_history,
    shap_summary_history,
)

candidate_run_catalog, run_selection_reason = (
    _candidate_shap_runs(
        shap_run_catalog,
        selected_start_date,
        selected_end_date,
    )
)

selected_shap_run_id = None

if candidate_run_catalog.empty:
    st.info(
        (
            "Henüz geçmiş model açıklaması kaydı bulunamadı. "
            "Aşağıdaki açıklamalar mevcut son snapshot üzerinden gösterilecektir."
            if language == "tr"
            else
            "No historical SHAP/model run is stored yet. "
            "The explanation below will use the current latest snapshot."
        )
    )

    active_shap_detail = shap_detail.copy()
    active_shap_summary = shap_summary.copy()

else:
    candidate_run_ids = (
        candidate_run_catalog[
            "RunID"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    run_rows = {
        str(row["RunID"]): row
        for _, row
        in candidate_run_catalog.iterrows()
        if pd.notna(
            row.get(
                "RunID"
            )
        )
    }

    selected_shap_run_id = st.selectbox(
        (
            "Model Koşusu"
            if language == "tr"
            else "Model Run"
        ),
        options=candidate_run_ids,
        format_func=lambda value: _shap_run_label(
            run_rows[
                str(
                    value
                )
            ],
            language,
        ),
        key="ai_insights_historical_shap_run",
        help=(
            "Seçilen raporlama dönemine en yakın veya dönemle kesişen model açıklama koşuları gösterilir."
            if language == "tr"
            else
            "Model-explanation runs that intersect or are closest to the reporting period are shown."
        ),
    )

    active_shap_detail = _select_shap_run_data(
        shap_detail_history,
        selected_shap_run_id,
    )

    active_shap_summary = _select_shap_run_data(
        shap_summary_history,
        selected_shap_run_id,
    )

    selected_run_row = run_rows.get(
        str(
            selected_shap_run_id
        )
    )

    if selected_run_row is not None:
        model_run_timestamp = pd.to_datetime(
            selected_run_row.get(
                "ModelRunTimestamp"
            ),
            errors="coerce",
        )

        observation_start = pd.to_datetime(
            selected_run_row.get(
                "ObservationStart"
            ),
            errors="coerce",
        )

        observation_end = pd.to_datetime(
            selected_run_row.get(
                "ObservationEnd"
            ),
            errors="coerce",
        )

        run_columns = st.columns(3)

        run_columns[0].metric(
            (
                "Model Koşu Zamanı"
                if language == "tr"
                else "Model Run Time"
            ),
            (
                model_run_timestamp.strftime(
                    "%d.%m.%Y %H:%M"
                )
                if pd.notna(
                    model_run_timestamp
                )
                else "-"
            ),
        )

        run_columns[1].metric(
            (
                "Açıklanan Veri Başlangıcı"
                if language == "tr"
                else "Explained Data Start"
            ),
            (
                observation_start.strftime(
                    "%d.%m.%Y"
                )
                if pd.notna(
                    observation_start
                )
                else "-"
            ),
        )

        run_columns[2].metric(
            (
                "Açıklanan Veri Sonu"
                if language == "tr"
                else "Explained Data End"
            ),
            (
                observation_end.strftime(
                    "%d.%m.%Y"
                )
                if pd.notna(
                    observation_end
                )
                else "-"
            ),
        )

    if run_selection_reason == "intersects_period":
        st.success(
            (
                "Seçilen model açıklaması, raporlama döneminizle kesişen bir SHAP snapshot'ından geliyor."
                if language == "tr"
                else
                "The selected model explanation comes from a SHAP snapshot that intersects the reporting period."
            )
        )

    elif run_selection_reason == "latest_before_period_end":
        st.info(
            (
                "Raporlama dönemiyle doğrudan kesişen bir SHAP snapshot'ı yok. "
                "Dönem sonuna en yakın geçmiş model açıklaması gösteriliyor."
                if language == "tr"
                else
                "No SHAP snapshot directly intersects the reporting period. "
                "The closest stored explanation before the period end is shown."
            )
        )

    elif run_selection_reason == "all_history":
        st.info(
            (
                "Raporlama dönemine göre uygun snapshot bulunamadığı için saklanan model koşuları gösteriliyor."
                if language == "tr"
                else
                "Stored model runs are shown because no period-matched snapshot was found."
            )
        )


# ============================================================
# SHAP GLOBAL EXPLAINABILITY
# ============================================================

render_divider()

render_section_header(
    "SHAP Genel Açıklanabilirlik"
    if language == "tr"
    else "SHAP Global Explainability"
)

st.caption(
    (
        "Model tahminini hangi değişkenlerin ne kadar etkilediğini gösterir. "
        "MeanAbsSHAP etki büyüklüğünü, MeanSHAP ortalama etki yönünü temsil eder."
        if language == "tr"
        else
        "Shows which variables influence the model forecast and by how much. "
        "MeanAbsSHAP represents impact magnitude; MeanSHAP represents average direction."
    )
)

st.info(_shap_note(language))

st.caption(
    (
        "Bu model şu anda iki üretim tahmini açıklar: **Tıklama Tahmini** ve "
        "**Gösterim Tahmini**. Bunlar tarih karşılaştırma seçenekleri değil, "
        "makine öğrenmesi modelinin iki hedef çıktısıdır."
        if language == "tr"
        else
        "The production model currently explains two forecast outputs: "
        "**Click Forecast** and **Impression Forecast**. These are model targets, "
        "not date-comparison options."
    )
)

if active_shap_summary.empty:
    st.info(t("no_data", language))
else:
    shap_models = (
        active_shap_summary["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "Model" in active_shap_summary.columns
        else []
    )

    selected_shap_model = (
        st.selectbox(
            "Model Açıklaması" if language == "tr" else "Model Explanation",
            options=shap_models,
            format_func=lambda value: _localize_model_target(value, language),
            key="ai_insights_shap_model",
        )
        if shap_models
        else None
    )

    shap_global_view = active_shap_summary.copy()

    if selected_shap_model is not None and "Model" in shap_global_view.columns:
        shap_global_view = shap_global_view[
            shap_global_view["Model"]
            .astype(str)
            .eq(str(selected_shap_model))
        ].copy()

    if "ImportanceRank" in shap_global_view.columns:
        shap_global_view["_rank"] = pd.to_numeric(
            shap_global_view["ImportanceRank"],
            errors="coerce",
        )
        shap_global_view = shap_global_view.sort_values(
            "_rank",
            ascending=True,
            na_position="last",
        )
    elif "MeanAbsSHAP" in shap_global_view.columns:
        shap_global_view["_importance"] = pd.to_numeric(
            shap_global_view["MeanAbsSHAP"],
            errors="coerce",
        )
        shap_global_view = shap_global_view.sort_values(
            "_importance",
            ascending=False,
            na_position="last",
        )

    top_shap_global = shap_global_view.head(20).drop(
        columns=["_rank", "_importance"],
        errors="ignore",
    )

    if (
        not top_shap_global.empty
        and "Feature" in top_shap_global.columns
        and "MeanAbsSHAP" in top_shap_global.columns
    ):
        chart_data = top_shap_global.copy()
        chart_data["FeatureLabel"] = chart_data["Feature"].map(
            lambda value: _feature_label(value, language)
        )
        chart_data["MeanAbsSHAP"] = pd.to_numeric(
            chart_data["MeanAbsSHAP"],
            errors="coerce",
        )

        shap_feature_label = (
            "Etkileyen Değişken"
            if language == "tr"
            else "Feature Name"
        )
        shap_value_label = (
            "SHAP Etki Büyüklüğü"
            if language == "tr"
            else "SHAP Impact Magnitude"
        )

        chart_data = chart_data.rename(
            columns={
                "FeatureLabel": shap_feature_label,
                "MeanAbsSHAP": shap_value_label,
            }
        )

        render_bar_chart(
            dataframe=chart_data,
            x=shap_feature_label,
            y=shap_value_label,
            title=(
                "Tahmini En Çok Etkileyen Değişkenler"
                if language == "tr"
                else "Variables With the Strongest Forecast Impact"
            ),
        )

        strongest = chart_data.sort_values(
            shap_value_label,
            ascending=False,
        ).head(3)

        if not strongest.empty:
            labels = ", ".join(
                strongest[shap_feature_label]
                .astype(str)
                .tolist()
            )
            st.success(
                (
                    f"Bu tahmin hedefinde en güçlü üç etken: **{labels}**. "
                    "Bu sıralama model etkisini gösterir; nedensellik iddiası değildir."
                    if language == "tr"
                    else
                    f"The three strongest drivers for this forecast target are "
                    f"**{labels}**. This ranking shows model impact, not causality."
                )
            )

    render_localized_dataframe(
        _display_dataframe(
            top_shap_global,
            language,
        ),
        width="stretch",
        hide_index=True,
    )

    render_export_buttons(
        dataframe=active_shap_summary,
        basename="seo_selected_run_shap_summary",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# SHAP PAGE-LEVEL EXPLAINABILITY
# ============================================================

render_divider()

render_section_header(
    "Sayfa Bazlı Model Açıklaması" if language == "tr" else "Page-Level Model Explanation"
)

st.caption(
    (
        "Bu bölüm gerçekleşen dönem performansını tekrar anlatmaz. Yukarıda seçilen model koşusunda belirli bir sayfanın tahminini hangi değişkenlerin yukarı veya aşağı taşıdığını açıklar."
        if language == "tr"
        else "This section does not repeat actual period performance. It explains which variables pushed a specific page forecast up or down in the model run selected above."
    )
)

if active_shap_detail.empty:
    st.info(
        "Seçilen model koşusunda kullanılabilir sayfa bazlı SHAP açıklaması bulunamadı."
        if language == "tr"
        else "No page-level SHAP explanation is available for the selected model run."
    )
else:
    detail_model_column = _first_existing_column(active_shap_detail, ["Model", "model"])
    detail_page_column = _first_existing_column(active_shap_detail, ["Page", "page"])
    detail_abs_column = _first_existing_column(active_shap_detail, ["AbsSHAPValue", "abs_shap_value"])

    detail_filter_columns = st.columns(2)
    selected_detail_model = None
    selected_detail_page = None

    if detail_model_column is not None:
        model_options = sorted(active_shap_detail[detail_model_column].dropna().astype(str).unique().tolist())
        with detail_filter_columns[0]:
            selected_detail_model = st.selectbox(
                "Model Açıklaması" if language == "tr" else "Model Explanation",
                options=model_options,
                format_func=lambda value: _localize_model_target(value, language),
                key="ai_insights_period_shap_detail_model",
                help=("Tıklama ve gösterim, mevcut üretim modelinin iki tahmin hedefidir." if language == "tr" else "Clicks and impressions are the two forecast targets of the current production model."),
            )

    detail_candidate = active_shap_detail.copy()
    if selected_detail_model is not None and detail_model_column is not None:
        detail_candidate = detail_candidate[detail_candidate[detail_model_column].astype(str).eq(str(selected_detail_model))].copy()

    if detail_page_column is not None:
        page_options = sorted(detail_candidate[detail_page_column].dropna().astype(str).unique().tolist())
        with detail_filter_columns[1]:
            selected_detail_page = st.selectbox(
                "Sayfa" if language == "tr" else "Page",
                options=page_options,
                key="ai_insights_period_shap_detail_page",
            )

    if selected_detail_page is not None and detail_page_column is not None:
        detail_candidate = detail_candidate[detail_candidate[detail_page_column].astype(str).eq(str(selected_detail_page))].copy()

    detail_date_column = _first_existing_column(detail_candidate, ["ObservationDate", "observation_date", "Date", "date"])
    selected_observation_date = None
    available_observation_dates = []
    if detail_date_column is not None:
        available_observation_dates = pd.to_datetime(detail_candidate[detail_date_column], errors="coerce").dropna().dt.date.drop_duplicates().sort_values(ascending=False).tolist()

    if len(available_observation_dates) == 1:
        selected_observation_date = available_observation_dates[0]
        st.info(
            (f"**Açıklanan Veri Tarihi: {selected_observation_date:%d.%m.%Y}** — Bu tarih, seçilen model koşusunda bu sayfa tahmininin hangi gözleme ait olduğunu gösterir." if language == "tr" else f"**Explained Data Date: {selected_observation_date:%d.%m.%Y}** — This is the observation explained for this page inside the selected model run.")
        )
    elif len(available_observation_dates) > 1:
        selected_observation_date = st.selectbox(
            "Açıklanan Veri Tarihi" if language == "tr" else "Explained Data Date",
            options=available_observation_dates,
            format_func=lambda value: value.strftime("%d.%m.%Y"),
            key="ai_insights_shap_observation_date",
        )

    if selected_observation_date is not None and detail_date_column is not None:
        parsed_dates = pd.to_datetime(detail_candidate[detail_date_column], errors="coerce").dt.date
        detail_candidate = detail_candidate[parsed_dates.eq(selected_observation_date)].copy()

    if detail_abs_column is not None:
        detail_candidate["_AbsSHAP"] = pd.to_numeric(detail_candidate[detail_abs_column], errors="coerce")
        detail_candidate = detail_candidate.sort_values("_AbsSHAP", ascending=False, na_position="last")

    top_detail = detail_candidate.head(20).drop(columns=["_AbsSHAP"], errors="ignore")
    prediction_column = _first_existing_column(top_detail, ["Prediction", "prediction"])
    base_value_column = _first_existing_column(top_detail, ["BaseValue", "base_value"])

    def _first_number(column: str | None) -> str:
        if column is None:
            return "-"
        values = pd.to_numeric(top_detail[column], errors="coerce").dropna()
        return format_number(values.iloc[0], decimals=2) if not values.empty else "-"

    s1, s2, s3 = st.columns(3)
    s1.metric("Açıklanan Değişken" if language == "tr" else "Explained Features", format_integer(len(top_detail)))
    s2.metric("Model Tahmini" if language == "tr" else "Model Prediction", _first_number(prediction_column))
    s3.metric("Model Başlangıç Değeri" if language == "tr" else "Model Baseline", _first_number(base_value_column))

    if not top_detail.empty and "Feature" in top_detail.columns and detail_abs_column is not None:
        detail_chart = top_detail.copy()
        detail_chart["FeatureLabel"] = detail_chart["Feature"].map(lambda value: _feature_label(value, language))
        detail_chart[detail_abs_column] = pd.to_numeric(detail_chart[detail_abs_column], errors="coerce")
        feature_label = "Etkileyen Değişken" if language == "tr" else "Feature Name"
        value_label = "Mutlak SHAP Etkisi" if language == "tr" else "Absolute SHAP Impact"
        detail_chart = detail_chart.rename(columns={"FeatureLabel": feature_label, detail_abs_column: value_label})
        render_bar_chart(
            dataframe=detail_chart, x=feature_label, y=value_label,
            title=("Bu Sayfa Tahminini En Çok Etkileyen Değişkenler" if language == "tr" else "Variables With the Largest Impact on This Page Forecast"),
        )
        strongest = detail_chart.sort_values(value_label, ascending=False).head(3)
        labels = ", ".join(strongest[feature_label].astype(str).tolist())
        if labels:
            st.success(
                (f"**Model açıklaması:** Bu sayfanın {_localize_model_target(selected_detail_model, language)} çıktısını en güçlü etkileyen değişkenler: **{labels}**. Bu sonuç model etkisini açıklar; tek başına SEO değişiminin nedeni olduğunu kanıtlamaz." if language == "tr" else f"**Model explanation:** The variables with the strongest impact on this page's {_localize_model_target(selected_detail_model, language)} are **{labels}**. This explains model influence; it does not by itself prove the cause of the SEO change.")
            )

    top_detail_display = top_detail.drop(
        columns=[
            "RunID",
            "ModelRunTimestamp",
        ],
        errors="ignore",
    )

    render_localized_dataframe(
        _display_dataframe(
            top_detail_display,
            language,
        ),
        width="stretch",
        hide_index=True,
    )
    render_export_buttons(
        dataframe=detail_candidate.drop(columns=["_AbsSHAP"], errors="ignore"),
        basename="seo_selected_run_shap_detail", csv_label="CSV", excel_label="Excel",
    )


# ============================================================
# TOP DECISIONS
# ============================================================

render_divider()

render_section_header(
    (
        "Öncelikli Aksiyonlar"
        if language == "tr"
        else "Priority Actions"
    )
)

_section_explainer(
    language,
    (
        "Bu tablo son başarılı model koşusundaki aksiyon önerilerini gösterir. "
        "Seçilen tarih aralığında ne yaşandığını üstteki dönem özeti gösterir; "
        "buradaki tablo ise şimdi ne yapılması gerektiğini anlatır."
    ),
    (
        "This table shows actions recommended jointly by the models and business rules. "
        "Priority, confidence, expected net value, and feasibility are evaluated together."
    ),
)

priority_recommendations = (
    get_priority_recommendations(
        recommendations,
        limit=30,
    )
)

if priority_recommendations.empty:
    st.info(
        (
            "Öneri verisi bulunamadı."
            if language == "tr"
            else
            "No recommendation data was found."
        )
    )

else:
    priority_agent = priority_recommendations.head(30).copy()
    priority_agent[
        "Problem" if language == "en" else "Problem / Fırsat"
    ] = priority_agent.apply(
        lambda row: _problem_statement(row, language),
        axis=1,
    )
    priority_agent[
        "Expected Result" if language == "en" else "Beklenen Sonuç"
    ] = priority_agent.apply(
        lambda row: _expected_result_text(row, language),
        axis=1,
    )

    page_column = _first_existing_column(
        priority_agent,
        ["page", "Page"],
    )
    preferred_columns = [
        page_column,
        "Problem" if language == "en" else "Problem / Fırsat",
        "RecommendationReason",
        "RecommendedAction",
        "PriorityTier",
        "ConfidenceLevel",
        "Expected Result" if language == "en" else "Beklenen Sonuç",
    ]
    preferred_columns = [
        column for column in preferred_columns
        if column is not None and column in priority_agent.columns
    ]
    priority_display = _display_dataframe(
        priority_agent[preferred_columns],
        language,
    )

    render_localized_dataframe(
        priority_display,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        (
            "Öncelik sırası; iş değeri, beklenen net değer, model güveni ve "
            "uygulanabilirlik sinyallerinin birlikte değerlendirilmesiyle oluşur."
            if language == "tr"
            else
            "Priority order combines business value, expected net value, "
            "model confidence, and implementation signals."
        )
    )

    render_export_buttons(
        dataframe=priority_recommendations,
        basename="seo_ai_priority_decisions",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# COMMENTARY ANALYSIS
# ============================================================

render_divider()

render_section_header(
    (
        "Yönetici Aksiyonları"
        if language == "tr"
        else "Executive Actions"
    )
)

_section_explainer(
    language,
    (
        "Bu bölüm teknik çıktıyı yönetici diline çevirir: hangi sayfaya neden müdahale "
        "edilmesi veya mevcut durumun neden korunması gerektiğini özetler."
    ),
    (
        "This section translates technical output into executive language: it summarizes "
        "why a page should be changed or why the current state should be maintained."
    ),
)

if recommendations.empty:
    st.info(
        (
            "Yönetici kararı bulunamadı."
            if language == "tr"
            else "No executive decision was found."
        )
    )
else:
    decision_view = recommendations.copy()

    page_column = _first_existing_column(
        decision_view,
        ["page", "Page"],
    )

    columns_to_keep = [
        column
        for column in [
            page_column,
            "PriorityTier",
            "RecommendedAction",
            "ConfidenceLevel",
            "ExpectedIncrementalClicks",
            "ExpectedClicksChangePct",
            "ExpectedPositionImprovement",
            "ExpectedIncrementalTrafficValue",
            "EstimatedROI",
            "ExecutiveCommentary",
        ]
        if column is not None
        and column in decision_view.columns
    ]

    decision_view = decision_view[
        columns_to_keep
    ].head(30).copy()

    if "ExecutiveCommentary" in decision_view.columns:
        decision_view["ExecutiveCommentary"] = decision_view[
            "ExecutiveCommentary"
        ].map(
            lambda value: _localize_commentary(
                value,
                language,
            )
        )

    decision_view = _display_dataframe(
        decision_view,
        language,
    )

    # User-facing names for business impact fields.
    impact_labels = {
        "ExpectedIncrementalClicks": (
            "Beklenen Ek Tıklama",
            "Expected Incremental Clicks",
        ),
        "ExpectedClicksChangePct": (
            "Beklenen Tıklama Değişimi (%)",
            "Expected Click Change (%)",
        ),
        "ExpectedPositionImprovement": (
            "Beklenen Pozisyon İyileşmesi",
            "Expected Position Improvement",
        ),
        "ExpectedIncrementalTrafficValue": (
            "Beklenen Ek Trafik Değeri",
            "Expected Incremental Traffic Value",
        ),
    }

    decision_view = decision_view.rename(
        columns={
            column: (
                pair[0]
                if language == "tr"
                else pair[1]
            )
            for column, pair in impact_labels.items()
            if column in decision_view.columns
        }
    )

    render_localized_dataframe(
        decision_view,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        (
            "Bu tablo 'seçilen tarihte bunlar oldu' demek için değil; "
            "gerçekleşen dönem performansını dikkate alarak son model koşusunun "
            "önerdiği sonraki aksiyonu okumak içindir."
            if language == "tr"
            else
            "This table is not a historical event log; it presents the next actions "
            "recommended by the latest model run, interpreted alongside the actual period performance above."
        )
    )

    render_export_buttons(
        dataframe=recommendations,
        basename="seo_executive_decisions",
        csv_label="CSV",
        excel_label="Excel",
    )


# ============================================================
# RISK / OPPORTUNITY SIGNALS
# ============================================================

render_divider()

render_section_header(
    (
        "Risk ve Fırsatlar"
        if language == "tr"
        else "Risks and Opportunities"
    )
)

_section_explainer(
    language,
    (
        "Pozitif net değer fırsatları ekonomik olarak uygulanabilir görünen aksiyonları; "
        "negatif net değer riskleri ise maliyeti beklenen getiriyi aşan aksiyonları gösterir. "
        "Düşük güvenli kararlar ek inceleme gerektirir."
    ),
    (
        "Positive net value opportunities are actions that appear economically viable; "
        "negative net value risks are actions whose estimated cost exceeds expected return. "
        "Low-confidence decisions require additional review."
    ),
)

signal_columns = st.columns(
    3
)

negative_net_value_count = 0
positive_net_value_count = 0
low_confidence_count = 0

if "ExpectedNetValue" in recommendations.columns:
    net_values = pd.to_numeric(
        recommendations[
            "ExpectedNetValue"
        ],
        errors="coerce",
    )

    negative_net_value_count = int(
        (
            net_values < 0
        ).sum()
    )

    positive_net_value_count = int(
        (
            net_values > 0
        ).sum()
    )


if "ConfidenceLevel" in recommendations.columns:
    low_confidence_count = int(
        recommendations[
            "ConfidenceLevel"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "low"
        )
        .sum()
    )


signal_columns[0].metric(
    (
        "Pozitif Net Değer Fırsatı"
        if language == "tr"
        else "Positive Net Value Opportunities"
    ),
    positive_net_value_count,
)

signal_columns[1].metric(
    (
        "Negatif Net Değer Riski"
        if language == "tr"
        else "Negative Net Value Risks"
    ),
    negative_net_value_count,
)

signal_columns[2].metric(
    (
        "Düşük Güvenli Karar"
        if language == "tr"
        else "Low-Confidence Decisions"
    ),
    low_confidence_count,
)

st.caption(
    (
        "Bu kartlar son model koşusunda seçilmiş önerilerin sonuçlarını özetler. "
        "Aşağıdaki Senaryo Zekâsı ise uygulanabilecek tüm alternatif aksiyonları simüle eder; "
        "bu nedenle iki bölümdeki pozitif/negatif sayılar doğrudan aynı ölçüm değildir."
        if language == "tr"
        else
        "These cards summarize selected recommendations from the latest model run. "
        "Scenario Intelligence below simulates all alternative actions, so the positive/negative "
        "counts in the two sections are not directly comparable."
    )
)


# ============================================================
# SCENARIO SIGNALS
# ============================================================

render_divider()

render_section_header(
    (
        "Projeksiyon Etkisi — Senaryo Zekâsı"
        if language == "tr"
        else "Projected Impact — Scenario Intelligence"
    )
)

_section_explainer(
    language,
    (
        "Bu bölüm gerçekleşmiş sonucu göstermez. Mevcut sayfa durumları üzerinde farklı "
        "SEO/GEO aksiyonları uygulanırsa oluşabilecek tahmini ticari etkiyi simüle eder. "
        "Beklenen Net Değer = tahmini fayda eksi tahmini uygulama maliyetidir."
    ),
    (
        "This section does not show realized results. It simulates the estimated commercial "
        "impact of alternative SEO/GEO actions on the current page states. Expected Net Value equals "
        "estimated benefit minus estimated implementation cost."
    ),
)

st.info(
    (
        "**Senaryo simülasyonu ile ML tahmini farklı katmanlardır.** Yukarıdaki bölüm gerçek "
        "7/14/30 günlük operasyonel ve 90/180/365 günlük stratejik yinelemeli ML baseline tahminini gösterir. Buradaki senaryo tablosu ise "
        "son sayfa durumu üzerinde alternatif SEO/GEO müdahalelerinin tek-adımlı etkisini karşılaştırır; "
        "senaryo değerleri gün sayısıyla çarpılmaz."
        if language == "tr"
        else
        "**Scenario simulation and ML forecasting are separate layers.** The section above shows the "
        "genuine recursive operational 7/14/30-day and strategic 90/180/365-day ML baseline forecast. The scenario table here compares the "
        "one-step impact of alternative SEO/GEO interventions on the latest page state; scenario "
        "values are not multiplied by the number of days."
    )
)

if not period_scenarios.empty:
    period_scenarios = period_scenarios.copy()

    incremental_value = pd.to_numeric(
        period_scenarios.get(
            "ExpectedIncrementalTrafficValue",
            pd.Series(
                0.0,
                index=period_scenarios.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0)

    implementation_cost = pd.to_numeric(
        period_scenarios.get(
            "EstimatedImplementationCost",
            pd.Series(
                0.0,
                index=period_scenarios.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0)

    native_net_value = pd.to_numeric(
        period_scenarios.get(
            "ExpectedNetValue",
            incremental_value
            - implementation_cost,
        ),
        errors="coerce",
    ).fillna(
        incremental_value
        - implementation_cost
    )

    native_roi = pd.to_numeric(
        period_scenarios.get(
            "EstimatedROI",
            pd.Series(
                0.0,
                index=period_scenarios.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0)

    period_scenarios[
        "SelectedHorizonDays"
    ] = forecast_horizon_days

    period_scenarios[
        "ProjectedIncrementalTrafficValue"
    ] = incremental_value

    period_scenarios[
        "ProjectedNetValue"
    ] = native_net_value

    period_scenarios[
        "ProjectedROI"
    ] = native_roi

    missing_roi = (
        period_scenarios[
            "ProjectedROI"
        ].eq(0)
        & implementation_cost.gt(0)
    )

    period_scenarios.loc[
        missing_roi,
        "ProjectedROI",
    ] = (
        period_scenarios.loc[
            missing_roi,
            "ProjectedNetValue",
        ]
        / implementation_cost.loc[
            missing_roi
        ]
    )

if period_scenarios.empty:
    st.info(
        t(
            "no_data",
            language,
        )
    )

else:
    scenario_column = None

    for candidate in [
        "Scenario",
        "scenario",
    ]:
        if candidate in period_scenarios.columns:
            scenario_column = candidate
            break

    if (
        scenario_column is not None
        and "ProjectedNetValue"
        in period_scenarios.columns
    ):
        scenario_summary = (
            period_scenarios
            .groupby(
                scenario_column,
                as_index=False,
                dropna=False,
            )
            .agg(
                ProjectedNetValue=(
                    "ProjectedNetValue",
                    "sum",
                ),
            )
            .sort_values(
                "ProjectedNetValue",
                ascending=False,
            )
        )

        scenario_label = (
            "Senaryo"
            if language == "tr"
            else "Scenario"
        )
        net_value_label = (
            "Beklenen Net Değer"
            if language == "tr"
            else "Expected Net Value"
        )

        scenario_summary[scenario_label] = scenario_summary[
            scenario_column
        ].map(
            lambda value: _localize_scenario(
                value,
                language,
            )
        )
        scenario_summary[net_value_label] = pd.to_numeric(
            scenario_summary["ProjectedNetValue"],
            errors="coerce",
        )

        render_bar_chart(
            dataframe=scenario_summary,
            x=scenario_label,
            y=net_value_label,
            title=(
                "Senaryo Bazında Beklenen Net Değer"
                if language == "tr"
                else "Expected Net Value by Scenario"
            ),
        )

        negative_scenario_count = int(
            (
                pd.to_numeric(
                    scenario_summary["ProjectedNetValue"],
                    errors="coerce",
                ) < 0
            ).sum()
        )

        if negative_scenario_count > 0:
            st.warning(
                (
                    f"{negative_scenario_count} senaryoda beklenen net değer negatif. "
                    "Bu, tahmini uygulama maliyetinin beklenen kısa vadeli ticari "
                    "getiriden yüksek olduğunu gösterir. Sistem bu senaryoları "
                    "otomatik olarak yüksek önceliğe taşımaz."
                    if language == "tr"
                    else
                    f"{negative_scenario_count} scenarios have negative expected net value. "
                    "This means estimated implementation cost exceeds expected short-term "
                    "commercial return, so the system does not automatically promote them "
                    "to high priority."
                )
            )

    with st.expander(
        (
            "Tüm Senaryo Verisini Gör"
            if language == "tr"
            else "View All Scenario Data"
        )
    ):
        scenario_view = period_scenarios.copy()
        scenario_columns = [
            _first_existing_column(scenario_view, ["page", "Page"]),
            "Scenario",
            "CurrentClicks",
            "CurrentImpressions",
            "CurrentCTR",
            "ScenarioCTR",
            "ScenarioPosition",
            "PredictedNextClicks",
            "PredictedNextImpressions",
            "EstimatedPositionGain",
            "EstimatedGeoScoreGain",
            "ProjectedIncrementalTrafficValue",
            "EstimatedImplementationCost",
            "ProjectedNetValue",
            "ProjectedROI",
            "ConfidenceLevel",
        ]
        scenario_columns = [
            column for column in scenario_columns
            if column is not None and column in scenario_view.columns
        ]
        scenario_view = scenario_view[scenario_columns].copy()

        extra_labels = {
            "CurrentClicks": ("Mevcut Tıklama", "Current Clicks"),
            "CurrentImpressions": ("Mevcut Gösterim", "Current Impressions"),
            "CurrentCTR": ("Mevcut CTR", "Current CTR"),
            "ScenarioCTR": ("Senaryo CTR", "Scenario CTR"),
            "ScenarioPosition": ("Senaryo Pozisyonu", "Scenario Position"),
            "PredictedNextClicks": ("Sonraki Gözlem Tıklama Tahmini", "Next-Observation Click Forecast"),
            "PredictedNextImpressions": ("Sonraki Gözlem Gösterim Tahmini", "Next-Observation Impression Forecast"),
            "EstimatedPositionGain": ("Tahmini Pozisyon Kazancı", "Estimated Position Gain"),
            "EstimatedGeoScoreGain": ("Tahmini GEO Skor Kazancı", "Estimated GEO Score Gain"),
            "ProjectedIncrementalTrafficValue": ("Tek-Adımlı Ek Trafik Değeri", "One-Step Incremental Traffic Value"),
            "EstimatedImplementationCost": ("Tahmini Uygulama Maliyeti", "Estimated Implementation Cost"),
            "ProjectedNetValue": ("Tek-Adımlı Beklenen Net Değer", "One-Step Expected Net Value"),
            "ProjectedROI": ("Tahmini ROI", "Estimated ROI"),
        }
        scenario_view = _display_dataframe(scenario_view, language)
        scenario_view = scenario_view.rename(
            columns={
                column: pair[0] if language == "tr" else pair[1]
                for column, pair in extra_labels.items()
                if column in scenario_view.columns
            }
        )
        render_localized_dataframe(
            scenario_view,
            width="stretch",
            hide_index=True,
        )

        render_export_buttons(
            dataframe=period_scenarios,
            basename="seo_period_scenario_intelligence",
            csv_label="CSV",
            excel_label="Excel",
        )
