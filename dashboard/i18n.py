from __future__ import annotations

from typing import Dict


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

DEFAULT_LANGUAGE = "tr"

SUPPORTED_LANGUAGES = {
    "tr": "Türkçe",
    "en": "English",
}


# ============================================================
# TRANSLATIONS
# ============================================================

TRANSLATIONS: Dict[str, Dict[str, str]] = {

    # ========================================================
    # COMMON
    # ========================================================

    "app_name": {
        "tr": "SEO Organik Büyüme Zekâsı AI Agentı",
        "en": "SEO Organic Growth Intelligence AI Agent",
    },

    "platform_subtitle": {
        "tr": "SEO + GEO Karar Zekâsı Platformu",
        "en": "SEO + GEO Decision Intelligence Platform",
    },

    "app_title": {
        "tr": "SEO Organik Büyüme Zekâsı AI Agentı",
        "en": "SEO Organic Growth Intelligence AI Agent",
    },

    "app_subtitle": {
        "tr": (
            "Yapay Zekâ Destekli SEO, GEO ve "
            "Organik Büyüme Karar Zekâsı Platformu"
        ),
        "en": (
            "AI-Powered SEO, GEO and Organic Growth "
            "Decision Intelligence Platform"
        ),
    },
    "language": {
        "tr": "Dil",
        "en": "Language",
    },

    "turkish": {
        "tr": "Türkçe",
        "en": "Turkish",
    },

    "english": {
        "tr": "İngilizce",
        "en": "English",
    },

    "loading": {
        "tr": "Yükleniyor...",
        "en": "Loading...",
    },

    "no_data": {
        "tr": "Veri bulunamadı.",
        "en": "No data found.",
    },

    "not_available": {
        "tr": "Kullanılamıyor",
        "en": "Not available",
    },

    "all": {
        "tr": "Tümü",
        "en": "All",
    },

    "yes": {
        "tr": "Evet",
        "en": "Yes",
    },

    "no": {
        "tr": "Hayır",
        "en": "No",
    },

    "download": {
        "tr": "İndir",
        "en": "Download",
    },

    "export": {
        "tr": "Dışa Aktar",
        "en": "Export",
    },

    "refresh": {
        "tr": "Yenile",
        "en": "Refresh",
    },

    "open": {
        "tr": "Aç",
        "en": "Open",
    },

    "close": {
        "tr": "Kapat",
        "en": "Close",
    },

    # ========================================================
    # NAVIGATION
    # ========================================================

    "navigation": {
        "tr": "Navigasyon",
        "en": "Navigation",
    },

    "executive_overview": {
        "tr": "Yönetici Özeti",
        "en": "Executive Overview",
    },

    "page_analysis": {
        "tr": "Sayfa Analizi",
        "en": "Page Analysis",
    },

    "seo_opportunity_optimizer": {
        "tr": "SEO Fırsat Optimizasyonu",
        "en": "SEO Opportunity Optimizer",
    },

    "ai_insights": {
        "tr": "AI Analizleri",
        "en": "AI Insights",
    },

    "ask_ai": {
        "tr": "AI Asistanı",
        "en": "Ask AI",
    },

    "system_status": {
        "tr": "Sistem Durumu",
        "en": "System Status",
    },

    "reports": {
        "tr": "Raporlar",
        "en": "Reports",
    },

    # ========================================================
    # PLATFORM STATUS
    # ========================================================

    "platform_status": {
        "tr": "Platform Durumu",
        "en": "Platform Status",
    },

    "daily_data_rows": {
        "tr": "Günlük Veri Satırı",
        "en": "Daily Data Rows",
    },

    "days_with_data": {
        "tr": "Veri Bulunan Gün",
        "en": "Days With Data",
    },

    "generated_outputs": {
        "tr": "Üretilen Çıktı",
        "en": "Generated Outputs",
    },

    "available_data_period": {
        "tr": "Mevcut Veri Dönemi",
        "en": "Available Data Period",
    },

    "latest_output": {
        "tr": "Son Çıktı",
        "en": "Latest Output",
    },

    "ai_runtime_mode": {
        "tr": "AI Çalışma Modu",
        "en": "AI Runtime Mode",
    },

    "hybrid_llm": {
        "tr": "Hibrit LLM",
        "en": "Hybrid LLM",
    },

    "deterministic": {
        "tr": "Deterministik",
        "en": "Deterministic",
    },

    "online": {
        "tr": "Çevrimiçi",
        "en": "Online",
    },

    "offline": {
        "tr": "Çevrimdışı",
        "en": "Offline",
    },

    # ========================================================
    # DATA SOURCES
    # ========================================================

    "data_sources": {
        "tr": "Veri Kaynakları",
        "en": "Data Sources",
    },

    "google_search_console": {
        "tr": "Google Search Console",
        "en": "Google Search Console",
    },

    "google_analytics_4": {
        "tr": "Google Analytics 4",
        "en": "Google Analytics 4",
    },

    "gsc": {
        "tr": "GSC",
        "en": "GSC",
    },

    "ga4": {
        "tr": "GA4",
        "en": "GA4",
    },

    "api_mode": {
        "tr": "API Modu",
        "en": "API Mode",
    },

    "csv_mode": {
        "tr": "CSV Modu",
        "en": "CSV Mode",
    },

    "hybrid_mode": {
        "tr": "Hibrit Mod",
        "en": "Hybrid Mode",
    },

    # ========================================================
    # FILTERS
    # ========================================================

    "filters": {
        "tr": "Filtreler",
        "en": "Filters",
    },

    "date_range": {
        "tr": "Tarih Aralığı",
        "en": "Date Range",
    },

    "comparison": {
        "tr": "Karşılaştırma",
        "en": "Comparison",
    },

    "no_comparison": {
        "tr": "Karşılaştırma Yok",
        "en": "No Comparison",
    },

    "previous_period": {
        "tr": "Önceki Dönem",
        "en": "Previous Period",
    },

    "previous_month": {
        "tr": "Önceki Ay",
        "en": "Previous Month",
    },

    "previous_year": {
        "tr": "Önceki Yıl",
        "en": "Previous Year",
    },

    "year_to_date": {
        "tr": "Yıl Başından Bugüne",
        "en": "Year to Date",
    },

    "custom": {
        "tr": "Özel",
        "en": "Custom",
    },

    "today": {
        "tr": "Bugün",
        "en": "Today",
    },

    "yesterday": {
        "tr": "Dün",
        "en": "Yesterday",
    },

    "last_7_days": {
        "tr": "Son 7 Gün",
        "en": "Last 7 Days",
    },

    "last_30_days": {
        "tr": "Son 30 Gün",
        "en": "Last 30 Days",
    },

    "last_60_days": {
        "tr": "Son 60 Gün",
        "en": "Last 60 Days",
    },

    "last_90_days": {
        "tr": "Son 90 Gün",
        "en": "Last 90 Days",
    },

    "this_month": {
        "tr": "Bu Ay",
        "en": "This Month",
    },

    "last_month": {
        "tr": "Geçen Ay",
        "en": "Last Month",
    },

    "this_quarter": {
        "tr": "Bu Çeyrek",
        "en": "This Quarter",
    },

    "this_year": {
        "tr": "Bu Yıl",
        "en": "This Year",
    },

    "start_date": {
        "tr": "Başlangıç Tarihi",
        "en": "Start Date",
    },

    "end_date": {
        "tr": "Bitiş Tarihi",
        "en": "End Date",
    },

    "page_type": {
        "tr": "Sayfa Türü",
        "en": "Page Type",
    },

    "keyword_intent": {
        "tr": "Anahtar Kelime Niyeti",
        "en": "Keyword Intent",
    },

    "device": {
        "tr": "Cihaz",
        "en": "Device",
    },

    "page": {
        "tr": "Sayfa",
        "en": "Page",
    },

    "query": {
        "tr": "Arama Sorgusu",
        "en": "Query",
    },

    # ========================================================
    # SEO KPIs
    # ========================================================

    "seo_performance": {
        "tr": "SEO Performansı",
        "en": "SEO Performance",
    },

    "clicks": {
        "tr": "Tıklamalar",
        "en": "Clicks",
    },

    "impressions": {
        "tr": "Gösterimler",
        "en": "Impressions",
    },

    "ctr": {
        "tr": "CTR",
        "en": "CTR",
    },

    "average_position": {
        "tr": "Ortalama Pozisyon",
        "en": "Average Position",
    },

    "position": {
        "tr": "Pozisyon",
        "en": "Position",
    },

    "sessions": {
        "tr": "Oturumlar",
        "en": "Sessions",
    },

    "users": {
        "tr": "Kullanıcılar",
        "en": "Users",
    },

    "engaged_sessions": {
        "tr": "Etkileşimli Oturumlar",
        "en": "Engaged Sessions",
    },

    "engagement_rate": {
        "tr": "Etkileşim Oranı",
        "en": "Engagement Rate",
    },

    "average_session_duration": {
        "tr": "Ort. Oturum Süresi",
        "en": "Avg. Session Duration",
    },

    "conversions": {
        "tr": "Dönüşümler",
        "en": "Conversions",
    },

    "revenue": {
        "tr": "Gelir",
        "en": "Revenue",
    },

    "purchases": {
        "tr": "Satın Almalar",
        "en": "Purchases",
    },

    "add_to_carts": {
        "tr": "Sepete Eklemeler",
        "en": "Add to Carts",
    },

    "checkouts": {
        "tr": "Ödeme Adımları",
        "en": "Checkouts",
    },

    "traffic_value": {
        "tr": "Trafik Değeri",
        "en": "Traffic Value",
    },

    # ========================================================
    # GEO
    # ========================================================

    "geo": {
        "tr": "GEO",
        "en": "GEO",
    },

    "geo_readiness": {
        "tr": "GEO Hazırlık Skoru",
        "en": "GEO Readiness Score",
    },

    "current_geo_readiness": {
        "tr": "Mevcut GEO Hazırlığı",
        "en": "Current GEO Readiness",
    },

    "scenario_geo_readiness": {
        "tr": "Senaryo GEO Hazırlığı",
        "en": "Scenario GEO Readiness",
    },

    "content_score": {
        "tr": "İçerik Skoru",
        "en": "Content Score",
    },

    "current_content_score": {
        "tr": "Mevcut İçerik Skoru",
        "en": "Current Content Score",
    },

    "scenario_content_score": {
        "tr": "Senaryo İçerik Skoru",
        "en": "Scenario Content Score",
    },

    # ========================================================
    # FORECASTING
    # ========================================================

    "forecast": {
        "tr": "Tahmin",
        "en": "Forecast",
    },

    "forecasting": {
        "tr": "Tahminleme",
        "en": "Forecasting",
    },

    "forecast_vs_actual": {
        "tr": "Tahmin vs Gerçekleşen",
        "en": "Forecast vs Actual",
    },

    "predicted_clicks": {
        "tr": "Tahmini Tıklamalar",
        "en": "Predicted Clicks",
    },

    "predicted_impressions": {
        "tr": "Tahmini Gösterimler",
        "en": "Predicted Impressions",
    },

    "predicted_next_clicks": {
        "tr": "Sonraki Dönem Tahmini Tıklama",
        "en": "Predicted Next Clicks",
    },

    "predicted_next_impressions": {
        "tr": "Sonraki Dönem Tahmini Gösterim",
        "en": "Predicted Next Impressions",
    },

    "actual": {
        "tr": "Gerçekleşen",
        "en": "Actual",
    },

    "predicted": {
        "tr": "Tahmin",
        "en": "Predicted",
    },

    # ========================================================
    # SCENARIOS & OPTIMIZATION
    # ========================================================

    "scenario": {
        "tr": "Senaryo",
        "en": "Scenario",
    },

    "scenario_analysis": {
        "tr": "Senaryo Analizi",
        "en": "Scenario Analysis",
    },

    "scenario_simulation": {
        "tr": "Senaryo Simülasyonu",
        "en": "Scenario Simulation",
    },

    "recommended_scenario": {
        "tr": "Önerilen Senaryo",
        "en": "Recommended Scenario",
    },

    "recommended_action": {
        "tr": "Önerilen Aksiyon",
        "en": "Recommended Action",
    },

    "recommendation_reason": {
        "tr": "Öneri Gerekçesi",
        "en": "Recommendation Reason",
    },

    "seo_opportunity": {
        "tr": "SEO Fırsatı",
        "en": "SEO Opportunity",
    },

    "opportunities": {
        "tr": "Fırsatlar",
        "en": "Opportunities",
    },

    "risks": {
        "tr": "Riskler",
        "en": "Risks",
    },

    "estimated_position_gain": {
        "tr": "Tahmini Pozisyon Kazancı",
        "en": "Estimated Position Gain",
    },

    "click_uplift": {
        "tr": "Tıklama Artışı",
        "en": "Click Uplift",
    },

    "click_uplift_pct": {
        "tr": "Tıklama Artış Oranı",
        "en": "Click Uplift %",
    },

    "incremental_traffic_value": {
        "tr": "Ek Trafik Değeri",
        "en": "Incremental Traffic Value",
    },

    "expected_net_value": {
        "tr": "Beklenen Net Değer",
        "en": "Expected Net Value",
    },

    "estimated_roi": {
        "tr": "Tahmini ROI",
        "en": "Estimated ROI",
    },

    "implementation_cost": {
        "tr": "Uygulama Maliyeti",
        "en": "Implementation Cost",
    },

    "payback_period": {
        "tr": "Geri Ödeme Süresi",
        "en": "Payback Period",
    },

    # ========================================================
    # PRIORITY & CONFIDENCE
    # ========================================================

    "priority": {
        "tr": "Öncelik",
        "en": "Priority",
    },

    "priority_tier": {
        "tr": "Öncelik Seviyesi",
        "en": "Priority Tier",
    },

    "high_priority": {
        "tr": "Yüksek Öncelik",
        "en": "High Priority",
    },

    "medium_priority": {
        "tr": "Orta Öncelik",
        "en": "Medium Priority",
    },

    "low_priority": {
        "tr": "Düşük Öncelik",
        "en": "Low Priority",
    },

    "confidence": {
        "tr": "Güven",
        "en": "Confidence",
    },

    "confidence_level": {
        "tr": "Model Güveni",
        "en": "Confidence Level",
    },

    "high": {
        "tr": "Yüksek",
        "en": "High",
    },

    "medium": {
        "tr": "Orta",
        "en": "Medium",
    },

    "low": {
        "tr": "Düşük",
        "en": "Low",
    },

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    "machine_learning": {
        "tr": "Makine Öğrenmesi",
        "en": "Machine Learning",
    },

    "model_performance": {
        "tr": "Model Performansı",
        "en": "Model Performance",
    },

    "model_metrics": {
        "tr": "Model Metrikleri",
        "en": "Model Metrics",
    },

    "feature_importance": {
        "tr": "Özellik Önem Dereceleri",
        "en": "Feature Importance",
    },

    "mae": {
        "tr": "MAE",
        "en": "MAE",
    },

    "mse": {
        "tr": "MSE",
        "en": "MSE",
    },

    "r2": {
        "tr": "R²",
        "en": "R²",
    },

    # ========================================================
    # AI
    # ========================================================

    "ai": {
        "tr": "Yapay Zekâ",
        "en": "AI",
    },

    "ai_assistant": {
        "tr": "AI Asistanı",
        "en": "AI Assistant",
    },

    "ai_commentary": {
        "tr": "AI Yönetici Yorumu",
        "en": "AI Executive Commentary",
    },

    "executive_commentary": {
        "tr": "Yönetici Yorumu",
        "en": "Executive Commentary",
    },

    "commentary_source": {
        "tr": "Yorum Kaynağı",
        "en": "Commentary Source",
    },

    "llm": {
        "tr": "LLM",
        "en": "LLM",
    },

    "ask_question": {
        "tr": "Sorunuzu yazın",
        "en": "Ask a question",
    },

    "send": {
        "tr": "Gönder",
        "en": "Send",
    },

    "clear_chat": {
        "tr": "Sohbeti Temizle",
        "en": "Clear Chat",
    },

    "deterministic_mode_message": {
        "tr": (
            "API anahtarı bulunmadığı için sistem deterministik "
            "analiz modunda çalışıyor. KPI, grafik, model ve "
            "optimizasyon hesapları kullanılabilir."
        ),
        "en": (
            "The system is running in deterministic analysis "
            "mode because no API key is configured. KPI, chart, "
            "model and optimization calculations remain available."
        ),
    },

    # ========================================================
    # EXECUTIVE OVERVIEW
    # ========================================================

    "organic_growth_overview": {
        "tr": "Organik Büyüme Özeti",
        "en": "Organic Growth Overview",
    },

    "top_opportunities": {
        "tr": "En Büyük Fırsatlar",
        "en": "Top Opportunities",
    },

    "top_pages": {
        "tr": "En İyi Sayfalar",
        "en": "Top Pages",
    },

    "top_queries": {
        "tr": "En İyi Arama Sorguları",
        "en": "Top Queries",
    },

    "seo_health": {
        "tr": "SEO Sağlığı",
        "en": "SEO Health",
    },

    "performance_trend": {
        "tr": "Performans Trendi",
        "en": "Performance Trend",
    },

    # ========================================================
    # PAGE ANALYSIS
    # ========================================================

    "page_performance": {
        "tr": "Sayfa Performansı",
        "en": "Page Performance",
    },

    "page_details": {
        "tr": "Sayfa Detayları",
        "en": "Page Details",
    },

    "page_recommendations": {
        "tr": "Sayfa Önerileri",
        "en": "Page Recommendations",
    },

    "page_opportunities": {
        "tr": "Sayfa Fırsatları",
        "en": "Page Opportunities",
    },

    "select_page": {
        "tr": "Sayfa Seç",
        "en": "Select Page",
    },

    # ========================================================
    # REPORTING
    # ========================================================

    "daily_performance": {
        "tr": "Günlük Performans",
        "en": "Daily Performance",
    },

    "weekly_performance": {
        "tr": "Haftalık Performans",
        "en": "Weekly Performance",
    },

    "monthly_performance": {
        "tr": "Aylık Performans",
        "en": "Monthly Performance",
    },

    "keyword_intent_summary": {
        "tr": "Anahtar Kelime Niyeti Özeti",
        "en": "Keyword Intent Summary",
    },

    "page_type_summary": {
        "tr": "Sayfa Türü Özeti",
        "en": "Page Type Summary",
    },

    "holiday_impact": {
        "tr": "Tatil / Özel Gün Etkisi",
        "en": "Holiday Impact",
    },

    # ========================================================
    # DEMO
    # ========================================================

    "public_demo": {
        "tr": "Public Demo",
        "en": "Public Demo",
    },

    "demo_mode": {
        "tr": "Demo Modu",
        "en": "Demo Mode",
    },

    "demo_notice": {
        "tr": (
            "Bu sürüm yalnızca anonimleştirilmiş demo verilerini "
            "kullanır. Canlı Google Search Console veya GA4 API "
            "çağrısı yapılmaz."
        ),
        "en": (
            "This version uses anonymized demo data only. "
            "No live Google Search Console or GA4 API request is made."
        ),
    },

    # ========================================================
    # FOOTER / SECURITY
    # ========================================================

    "read_only": {
        "tr": "Salt Okunur",
        "en": "Read Only",
    },

    "read_only_footer": {
        "tr": (
            "Bu dashboard karar destek ve analiz amacıyla "
            "salt okunur biçimde çalışır."
        ),
        "en": (
            "This dashboard operates in read-only mode "
            "for decision support and analytics."
        ),
    },

    "credentials_not_exposed": {
        "tr": "Kimlik bilgileri dashboard üzerinde gösterilmez.",
        "en": "Credentials are never exposed in the dashboard.",
    },
}


# ============================================================
# LANGUAGE HELPERS
# ============================================================


def normalize_language(
    language: str | None,
) -> str:
    """
    Normalize a requested dashboard language.

    Unsupported or empty values fall back to Turkish.
    """
    if language is None:
        return DEFAULT_LANGUAGE

    normalized = (
        str(language)
        .strip()
        .lower()
    )

    if normalized not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE

    return normalized


def translate(
    key: str,
    language: str | None = None,
    default: str | None = None,
) -> str:
    """
    Translate a dashboard key.

    Parameters
    ----------
    key:
        Translation dictionary key.

    language:
        Requested language code.

    default:
        Optional fallback when the key does not exist.
    """
    resolved_language = normalize_language(
        language
    )

    translation = TRANSLATIONS.get(
        key
    )

    if translation is None:
        return (
            default
            if default is not None
            else key
        )

    return (
        translation.get(
            resolved_language
        )
        or translation.get(
            DEFAULT_LANGUAGE
        )
        or default
        or key
    )


def t(
    key: str,
    language: str | None = None,
    default: str | None = None,
) -> str:
    """
    Short alias for translate().
    """
    return translate(
        key=key,
        language=language,
        default=default,
    )


def get_language_label(
    language: str,
) -> str:
    """
    Return the human-readable label for a language code.
    """
    normalized = normalize_language(
        language
    )

    return SUPPORTED_LANGUAGES[
        normalized
    ]


def get_language_options() -> Dict[str, str]:
    """
    Return a copy of supported dashboard languages.
    """
    return dict(
        SUPPORTED_LANGUAGES
    )
