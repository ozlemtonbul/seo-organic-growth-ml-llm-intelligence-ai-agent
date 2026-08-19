from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st


COLUMN_LABELS: dict[str, tuple[str, str]] = {
    # Core identifiers / dates
    "date": ("Tarih", "Date"),
    "Date": ("Tarih", "Date"),
    "ObservationDate": ("Gözlem Tarihi", "Observation Date"),
    "AnalysisDate": ("Analiz Tarihi", "Analysis Date"),
    "page": ("Sayfa", "Page"),
    "Page": ("Sayfa", "Page"),
    "URL": ("Sayfa", "Page"),
    "url": ("Sayfa", "Page"),
    "page_key": ("Sayfa Anahtarı", "Page Key"),
    "PageKey": ("Sayfa Anahtarı", "Page Key"),
    "page_type": ("Sayfa Türü", "Page Type"),
    "PageType": ("Sayfa Türü", "Page Type"),
    "CurrentLandingPageType": ("Mevcut Açılış Sayfası Türü", "Current Landing Page Type"),

    # Search / keyword
    "keyword_intent": ("Arama Niyeti", "Search Intent"),
    "KeywordIntent": ("Arama Niyeti", "Search Intent"),
    "SearchIntent": ("Arama Niyeti", "Search Intent"),
    "query": ("Arama Sorgusu", "Search Query"),
    "Query": ("Arama Sorgusu", "Search Query"),
    "PrimaryKeyword": ("Ana Anahtar Kelime", "Primary Keyword"),
    "keyword": ("Anahtar Kelime", "Keyword"),
    "Keyword": ("Anahtar Kelime", "Keyword"),
    "KeywordImpressions": ("Anahtar Kelime Gösterimleri", "Keyword Impressions"),
    "KeywordClicks": ("Anahtar Kelime Tıklamaları", "Keyword Clicks"),
    "KeywordCTR": ("Anahtar Kelime CTR", "Keyword CTR"),
    "KeywordOpportunityScore": ("Anahtar Kelime Fırsat Puanı", "Keyword Opportunity Score"),
    "KeywordPriority": ("Anahtar Kelime Önceliği", "Keyword Priority"),
    "RecommendedKeywordAction": ("Önerilen Anahtar Kelime Aksiyonu", "Recommended Keyword Action"),

    # GSC
    "clicks": ("Tıklamalar", "Clicks"),
    "Clicks": ("Tıklamalar", "Clicks"),
    "CurrentClicks": ("Mevcut Tıklamalar", "Current Clicks"),
    "ExpectedClicks": ("Beklenen Tıklamalar", "Expected Clicks"),
    "PredictedNextClicks": ("Tahmini Sonraki Tıklamalar", "Predicted Next Clicks"),
    "target_clicks_next": ("Sonraki Tıklama Hedefi", "Next Click Target"),
    "clicks_change": ("Tıklama Değişimi", "Click Change"),
    "impressions": ("Gösterimler", "Impressions"),
    "Impressions": ("Gösterimler", "Impressions"),
    "CurrentImpressions": ("Mevcut Gösterimler", "Current Impressions"),
    "PredictedNextImpressions": ("Tahmini Sonraki Gösterimler", "Predicted Next Impressions"),
    "target_impressions_next": ("Sonraki Gösterim Hedefi", "Next Impression Target"),
    "impressions_change": ("Gösterim Değişimi", "Impression Change"),
    "ctr": ("CTR", "CTR"),
    "CTR": ("CTR", "CTR"),
    "CurrentCTR": ("Mevcut CTR", "Current CTR"),
    "ScenarioCTR": ("Senaryo CTR", "Scenario CTR"),
    "ctr_change": ("CTR Değişimi", "CTR Change"),
    "position": ("Ortalama Pozisyon", "Average Position"),
    "Position": ("Ortalama Pozisyon", "Average Position"),
    "AveragePosition": ("Ortalama Pozisyon", "Average Position"),
    "AveragePosition_x": ("Ortalama Pozisyon (Kaynak 1)", "Average Position (Source 1)"),
    "AveragePosition_y": ("Ortalama Pozisyon (Kaynak 2)", "Average Position (Source 2)"),
    "CurrentPosition": ("Mevcut Pozisyon", "Current Position"),
    "CurrentPosition_x": ("Mevcut Pozisyon (Kaynak 1)", "Current Position (Source 1)"),
    "CurrentPosition_y": ("Mevcut Pozisyon (Kaynak 2)", "Current Position (Source 2)"),
    "ScenarioPosition": ("Senaryo Pozisyonu", "Scenario Position"),
    "EstimatedPositionGain": ("Tahmini Pozisyon Kazanımı", "Estimated Position Gain"),
    "position_change": ("Pozisyon Değişimi", "Position Change"),

    # GA4 / commerce
    "sessions": ("Oturumlar", "Sessions"),
    "Sessions": ("Oturumlar", "Sessions"),
    "users": ("Kullanıcılar", "Users"),
    "Users": ("Kullanıcılar", "Users"),
    "engaged_sessions": ("Etkileşimli Oturumlar", "Engaged Sessions"),
    "engagement_rate": ("Etkileşim Oranı", "Engagement Rate"),
    "average_session_duration": ("Ortalama Oturum Süresi", "Average Session Duration"),
    "conversions": ("Dönüşümler", "Conversions"),
    "Conversions": ("Dönüşümler", "Conversions"),
    "revenue": ("Gelir", "Revenue"),
    "Revenue": ("Gelir", "Revenue"),
    "purchases": ("Satın Almalar", "Purchases"),
    "Purchases": ("Satın Almalar", "Purchases"),
    "add_to_carts": ("Sepete Eklemeler", "Add to Carts"),
    "AddToCarts": ("Sepete Eklemeler", "Add to Carts"),
    "checkouts": ("Ödeme Adımları", "Checkouts"),
    "Checkouts": ("Ödeme Adımları", "Checkouts"),
    "CartRate": ("Sepete Ekleme Oranı", "Cart Rate"),
    "CheckoutRate": ("Ödeme Adımı Oranı", "Checkout Rate"),
    "PurchaseRate": ("Satın Alma Oranı", "Purchase Rate"),
    "RevenuePerSession": ("Oturum Başına Gelir", "Revenue per Session"),
    "RevenuePerOrganicSession": ("Organik Oturum Başına Gelir", "Revenue per Organic Session"),
    "RevenuePerOrganicClick": ("Organik Tıklama Başına Gelir", "Revenue per Organic Click"),
    "OrganicConversionRate": ("Organik Dönüşüm Oranı", "Organic Conversion Rate"),

    # Scenario / recommendations
    "Scenario": ("Senaryo", "Scenario"),
    "scenario": ("Senaryo", "Scenario"),
    "ScenarioLabel": ("Senaryo Açıklaması", "Scenario Label"),
    "PriorityTier": ("Öncelik", "Priority"),
    "priority_tier": ("Öncelik", "Priority"),
    "Priority": ("Öncelik", "Priority"),
    "ConfidenceLevel": ("Güven Seviyesi", "Confidence"),
    "RecommendedAction": ("Önerilen Aksiyon", "Recommended Action"),
    "RecommendationReason": ("Öneri Gerekçesi", "Recommendation Reason"),
    "ExpectedIncrementalClicks": ("Beklenen Ek Tıklama", "Expected Incremental Clicks"),
    "ExpectedClicksChangePct": ("Beklenen Tıklama Değişimi (%)", "Expected Click Change (%)"),
    "ClickUplift": ("Tıklama Artışı", "Click Uplift"),
    "ClickUpliftPct": ("Tıklama Artışı (%)", "Click Uplift (%)"),
    "ExpectedIncrementalTrafficValue": ("Beklenen Ek Trafik Değeri", "Expected Incremental Traffic Value"),
    "EstimatedImplementationCost": ("Tahmini Uygulama Maliyeti", "Estimated Implementation Cost"),
    "ExpectedNetValue": ("Beklenen Net Değer", "Expected Net Value"),
    "AdjustedNetValue": ("Düzeltilmiş Net Değer", "Adjusted Net Value"),
    "EstimatedROI": ("Tahmini ROI", "Estimated ROI"),
    "PaybackPeriod": ("Geri Ödeme Süresi", "Payback Period"),
    "BusinessDecisionScore": ("İş Karar Puanı", "Business Decision Score"),

    # Content / GEO / commerce intelligence
    "RecommendedCommercialPage": ("Önerilen Ticari Sayfa", "Recommended Commercial Page"),
    "RecommendedCommercialPageType": ("Önerilen Ticari Sayfa Türü", "Recommended Commercial Page Type"),
    "TargetCommerceScore": ("Hedef Ticaret Puanı", "Target Commerce Score"),
    "TargetRevenue": ("Hedef Gelir", "Target Revenue"),
    "TargetPurchases": ("Hedef Satın Alma", "Target Purchases"),
    "TargetAddToCarts": ("Hedef Sepete Ekleme", "Target Add to Carts"),
    "ContentToCommerceScore": ("İçerikten Ticarete Puan", "Content-to-Commerce Score"),
    "ContentPriority": ("İçerik Önceliği", "Content Priority"),
    "RecommendedBlogAction": ("Önerilen Blog Aksiyonu", "Recommended Blog Action"),
    "InternalLinkRecommendation": ("İç Link Önerisi", "Internal Link Recommendation"),
    "CommerceObjective": ("Ticari Hedef", "Commerce Objective"),
    "EntityName": ("Entity Adı", "Entity Name"),
    "CurrentContentScore": ("Mevcut İçerik Skoru", "Current Content Score"),
    "ScenarioContentScore": ("Senaryo İçerik Skoru", "Scenario Content Score"),
    "CurrentGeoReadiness": ("Mevcut GEO Hazırlığı", "Current GEO Readiness"),
    "ScenarioGeoReadiness": ("Senaryo GEO Hazırlığı", "Scenario GEO Readiness"),
    "CurrentGeoReadinessScore": ("Mevcut GEO Hazırlık Skoru", "Current GEO Readiness Score"),
    "ScenarioGeoReadinessScore": ("Senaryo GEO Hazırlık Skoru", "Scenario GEO Readiness Score"),
    "GEOReadinessScore": ("GEO Hazırlık Skoru", "GEO Readiness Score"),
    "GEOOpportunityScore": ("GEO Fırsat Puanı", "GEO Opportunity Score"),
    "ContentGapScore": ("İçerik Fırsat Puanı", "Content Opportunity Score"),
    "ContentGapPriority": ("İçerik Önceliği", "Content Priority"),
    "RankingOpportunityScore": ("Sıralama Fırsat Puanı", "Ranking Opportunity Score"),
    "DemandScore": ("Talep Puanı", "Demand Score"),
    "CommerceScore": ("Ticaret Puanı", "Commerce Score"),
    "PageOpportunityScore": ("Sayfa Fırsat Puanı", "Page Opportunity Score"),
    "OpportunityType": ("Fırsat Türü", "Opportunity Type"),
    "OpportunityPriority": ("Fırsat Önceliği", "Opportunity Priority"),
    "RecommendedFocus": ("Önerilen Odak", "Recommended Focus"),

    # SEO feature engineering
    "TrafficValue": ("Trafik Değeri", "Traffic Value"),
    "RankStrength": ("Sıralama Gücü", "Rank Strength"),
    "VisibilityScore": ("Görünürlük Puanı", "Visibility Score"),
    "Top3Flag": ("İlk 3 Sinyali", "Top 3 Flag"),
    "Top10Flag": ("İlk 10 Sinyali", "Top 10 Flag"),
    "Page2Flag": ("2. Sayfa Sinyali", "Page 2 Flag"),
    "day_of_week": ("Haftanın Günü", "Day of Week"),
    "day_of_month": ("Ayın Günü", "Day of Month"),
    "month_num": ("Ay", "Month"),
    "quarter": ("Çeyrek", "Quarter"),
    "is_weekend": ("Hafta Sonu mu?", "Is Weekend"),
    "is_holiday": ("Tatil mi?", "Is Holiday"),
    "holiday_name": ("Tatil Adı", "Holiday Name"),
    "is_pre_holiday": ("Tatil Öncesi mi?", "Is Pre-Holiday"),
    "clicks_lag_1": ("Tıklama Gecikmesi 1", "Clicks Lag 1"),
    "clicks_lag_7_avg": ("7 Günlük Tıklama Ortalaması", "7-Day Click Average"),
    "impressions_lag_1": ("Gösterim Gecikmesi 1", "Impressions Lag 1"),
    "impressions_lag_7_avg": ("7 Günlük Gösterim Ortalaması", "7-Day Impression Average"),
    "position_lag_1": ("Pozisyon Gecikmesi 1", "Position Lag 1"),
    "position_lag_7_avg": ("7 Günlük Pozisyon Ortalaması", "7-Day Position Average"),
    "CTR_lag_1": ("CTR Gecikmesi 1", "CTR Lag 1"),
    "CTR_lag_7_avg": ("7 Günlük CTR Ortalaması", "7-Day CTR Average"),
    "TrafficValue_lag_1": ("Trafik Değeri Gecikmesi 1", "Traffic Value Lag 1"),
    "TrafficValue_lag_7_avg": ("7 Günlük Trafik Değeri Ortalaması", "7-Day Traffic Value Average"),

    # Models / technical
    "Model": ("Model", "Model"),
    "Target": ("Tahmin Hedefi", "Forecast Target"),
    "Algorithm": ("Algoritma", "Algorithm"),
    "Selected": ("Seçilen", "Selected"),
    "Feature": ("Değişken", "Feature"),
    "Importance": ("Önem", "Importance"),
    "Prediction": ("Tahmin", "Prediction"),
    "Direction": ("Etki Yönü", "Impact Direction"),
    "Status": ("Durum", "Status"),
    "Severity": ("Önem", "Severity"),
    "Issue": ("Sorun", "Issue"),
    "value": ("Değer", "Value"),
    "variable": ("Metrik", "Metric"),
    "ForecastDate": ("Tahmin Tarihi", "Forecast Date"),
    "HorizonDay": ("Tahmin Günü", "Forecast Day"),
    "HorizonDays": ("Tahmin Ufku (Gün)", "Forecast Horizon (Days)"),
    "ForecastStartDate": ("Tahmin Başlangıcı", "Forecast Start"),
    "ForecastEndDate": ("Tahmin Bitişi", "Forecast End"),
    "PredictedClicks": ("Tahmini Tıklamalar", "Forecast Clicks"),
    "PredictedImpressions": ("Tahmini Gösterimler", "Forecast Impressions"),
    "PredictedCTR": ("Tahmini CTR", "Forecast CTR"),
    "PredictedTrafficValue": ("Tahmini Trafik Değeri", "Forecast Traffic Value"),
    "ReferenceClicks": ("Referans Tıklamalar", "Reference Clicks"),
    "ReferenceImpressions": ("Referans Gösterimler", "Reference Impressions"),
    "ClickChangePct": ("Tıklama Değişimi (%)", "Click Change (%)"),
    "ImpressionChangePct": ("Gösterim Değişimi (%)", "Impression Change (%)"),
    "ForecastReliability": ("Tahmin Güvenilirliği", "Forecast Reliability"),
    "ForecastMethod": ("Tahmin Yöntemi", "Forecast Method"),
    "PageCount": ("Sayfa Sayısı", "Page Count"),
    "ForecastFamily": ("Tahmin Ailesi", "Forecast Family"),
    "BaseStepDays": ("Temel Adım (Gün)", "Base Step (Days)"),
    "SupportedHorizons": ("Desteklenen Ufuklar", "Supported Horizons"),
}

SCENARIO_VALUES: dict[str, tuple[str, str]] = {
    "maintain": ("Mevcut Yapıyı Koru", "Maintain Current Setup"),
    "Maintain Current Setup": ("Mevcut Yapıyı Koru", "Maintain Current Setup"),
    "Maintain Current State": ("Mevcut Yapıyı Koru", "Maintain Current State"),
    "product_content_enrichment": ("Ürün İçeriğini Zenginleştir", "Product Content Enrichment"),
    "Enrich Product Content": ("Ürün İçeriğini Zenginleştir", "Enrich Product Content"),
    "title_meta_optimization": ("Başlık ve Meta Optimizasyonu", "Title and Meta Optimization"),
    "Title and Meta Optimization": ("Başlık ve Meta Optimizasyonu", "Title and Meta Optimization"),
    "content_refresh": ("İçerik Güncellemesi", "Content Refresh"),
    "Content Refresh": ("İçerik Güncellemesi", "Content Refresh"),
    "internal_linking_boost": ("İç Linkleme Güçlendirmesi", "Internal Linking Improvement"),
    "Internal Linking Improvement": ("İç Linkleme Güçlendirmesi", "Internal Linking Improvement"),
    "category_expansion": ("Kategori SEO Genişletme", "Category SEO Expansion"),
    "Category SEO Expansion": ("Kategori SEO Genişletme", "Category SEO Expansion"),
    "structured_data_upgrade": ("Yapılandırılmış Veri İyileştirmesi", "Structured Data Upgrade"),
    "Structured Data Upgrade": ("Yapılandırılmış Veri İyileştirmesi", "Structured Data Upgrade"),
    "geo_answer_optimization": ("GEO Cevap Optimizasyonu", "GEO Answer Optimization"),
    "GEO Answer Optimization": ("GEO Cevap Optimizasyonu", "GEO Answer Optimization"),
    "entity_eet_upgrade": ("Entity ve E-E-A-T İyileştirmesi", "Entity and E-E-A-T Upgrade"),
    "Entity and E-E-A-T Upgrade": ("Entity ve E-E-A-T İyileştirmesi", "Entity and E-E-A-T Upgrade"),
    "full_seo_geo_optimization": ("Tam SEO + GEO Optimizasyonu", "Full SEO + GEO Optimization"),
    "Full SEO and GEO Optimization": ("Tam SEO + GEO Optimizasyonu", "Full SEO + GEO Optimization"),
    "Full SEO + GEO Optimization": ("Tam SEO + GEO Optimizasyonu", "Full SEO + GEO Optimization"),
}

PAGE_TYPE_VALUES: dict[str, tuple[str, str]] = {
    "category": ("Kategori", "Category"),
    "product": ("Ürün", "Product"),
    "blog": ("Blog", "Blog"),
    "guide": ("Rehber", "Guide"),
    "faq": ("SSS", "FAQ"),
    "homepage": ("Ana Sayfa", "Homepage"),
    "home": ("Ana Sayfa", "Homepage"),
    "landing": ("Açılış Sayfası", "Landing Page"),
    "other": ("Diğer", "Other"),
    "unknown": ("Bilinmiyor", "Unknown"),
}

INTENT_VALUES: dict[str, tuple[str, str]] = {
    "Transactional": ("İşlemsel", "Transactional"),
    "Commercial": ("Ticari", "Commercial"),
    "Commercial Investigation": ("Ticari Araştırma", "Commercial Investigation"),
    "Informational": ("Bilgilendirici", "Informational"),
    "Navigational": ("Navigasyonel", "Navigational"),
    "Uncategorized": ("Sınıflandırılmamış", "Uncategorized"),
}

PRIORITY_VALUES: dict[str, tuple[str, str]] = {
    "High Priority": ("Yüksek Öncelik", "High Priority"),
    "Medium Priority": ("Orta Öncelik", "Medium Priority"),
    "Low Priority": ("Düşük Öncelik", "Low Priority"),
    "high": ("Yüksek", "High"),
    "medium": ("Orta", "Medium"),
    "low": ("Düşük", "Low"),
    "High": ("Yüksek", "High"),
    "Medium": ("Orta", "Medium"),
    "Low": ("Düşük", "Low"),
}

CONFIDENCE_VALUES = {
    "High": ("Yüksek", "High"),
    "Medium": ("Orta", "Medium"),
    "Low": ("Düşük", "Low"),
}

ACTION_VALUES = {
    "Maintain": ("Mevcut Durumu Koru", "Maintain"),
    "Review": ("İncele", "Review"),
    "Optimize Title and Meta": ("Başlık ve Meta Optimizasyonu", "Optimize Title and Meta"),
    "Apply Full SEO and GEO Optimization": ("Tam SEO + GEO Optimizasyonu", "Apply Full SEO + GEO Optimization"),
    "Apply Full SEO + GEO Optimization": ("Tam SEO + GEO Optimizasyonu", "Apply Full SEO + GEO Optimization"),
    **SCENARIO_VALUES,
}

REASON_VALUES: dict[str, tuple[str, str]] = {
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
        "Low-confidence recommendation. Manual SEO validation is required.",
    ),
    "Manual review recommended.": (
        "Ek inceleme öneriliyor.",
        "Manual review recommended.",
    ),
}

DIRECTION_VALUES = {
    "positive": ("Pozitif", "Positive"),
    "negative": ("Negatif", "Negative"),
    "neutral": ("Nötr", "Neutral"),
    "increase": ("Artırıcı", "Increase"),
    "decrease": ("Azaltıcı", "Decrease"),
    "zero": ("Sıfır", "Zero"),
}


def active_language(language: str | None = None) -> str:
    if language in {"tr", "en"}:
        return language
    return st.session_state.get("dashboard_language", "tr")


def _pair_value(
    mapping: dict[str, tuple[str, str]],
    value: Any,
    language: str,
) -> Any:
    if pd.isna(value):
        return value

    raw = str(value).strip()
    pair = mapping.get(raw)

    if pair is None:
        pair = mapping.get(raw.lower())

    if pair is None:
        return value

    return pair[0] if language == "tr" else pair[1]


def localize_column_name(
    column: str,
    language: str | None = None,
) -> str:
    lang = active_language(language)
    pair = COLUMN_LABELS.get(str(column))

    if pair is None:
        return str(column)

    return pair[0] if lang == "tr" else pair[1]


def localized_column_labels(
    columns: list[str] | pd.Index,
    language: str | None = None,
) -> dict[str, str]:
    """
    Build unique display labels without renaming the real DataFrame columns.

    Example:
      ctr + CTR -> CTR, CTR (2)
      Scenario + scenario -> Senaryo, Senaryo (2)

    Arrow/Plotly therefore continue to receive unique internal column names.
    """
    lang = active_language(language)
    raw_columns = [str(column) for column in columns]
    base_labels = [
        localize_column_name(column, lang)
        for column in raw_columns
    ]

    total = Counter(base_labels)
    seen: Counter[str] = Counter()
    labels: dict[str, str] = {}

    for raw, base in zip(raw_columns, base_labels):
        seen[base] += 1

        if total[base] == 1 or seen[base] == 1:
            label = base
        else:
            label = f"{base} ({seen[base]})"

        labels[raw] = label

    return labels


def localize_value(
    value: Any,
    language: str | None = None,
    column: str | None = None,
) -> Any:
    lang = active_language(language)
    name = str(column or "")

    if name in {"Scenario", "scenario", "ScenarioLabel"}:
        return _pair_value(SCENARIO_VALUES, value, lang)

    if name in {
        "page_type",
        "PageType",
        "CurrentLandingPageType",
        "RecommendedCommercialPageType",
    }:
        return _pair_value(PAGE_TYPE_VALUES, value, lang)

    if name in {
        "keyword_intent",
        "KeywordIntent",
        "SearchIntent",
    }:
        return _pair_value(INTENT_VALUES, value, lang)

    if name in {
        "PriorityTier",
        "priority_tier",
        "Priority",
        "ContentGapPriority",
        "ContentPriority",
        "GEOPriority",
        "OpportunityPriority",
        "KeywordPriority",
    }:
        return _pair_value(PRIORITY_VALUES, value, lang)

    if name == "ConfidenceLevel":
        return _pair_value(CONFIDENCE_VALUES, value, lang)

    if name in {
        "RecommendedAction",
        "RecommendedKeywordAction",
        "RecommendedBlogAction",
    }:
        return _pair_value(ACTION_VALUES, value, lang)

    if name == "RecommendationReason":
        return _pair_value(REASON_VALUES, value, lang)

    if name == "Direction":
        return _pair_value(DIRECTION_VALUES, value, lang)

    return value


VALUE_LOCALIZED_COLUMNS = {
    "Scenario",
    "scenario",
    "ScenarioLabel",
    "page_type",
    "PageType",
    "CurrentLandingPageType",
    "RecommendedCommercialPageType",
    "keyword_intent",
    "KeywordIntent",
    "SearchIntent",
    "PriorityTier",
    "priority_tier",
    "Priority",
    "ContentGapPriority",
    "ContentPriority",
    "GEOPriority",
    "OpportunityPriority",
    "KeywordPriority",
    "ConfidenceLevel",
    "RecommendedAction",
    "RecommendedKeywordAction",
    "RecommendedBlogAction",
    "RecommendationReason",
    "Direction",
}


def localize_dataframe(
    dataframe: pd.DataFrame,
    language: str | None = None,
) -> pd.DataFrame:
    """
    Localize categorical values while preserving internal column names.

    IMPORTANT:
    Column names are NOT renamed here. Display labels are supplied separately
    to Streamlit/Plotly. This prevents translation collisions.
    """
    lang = active_language(language)

    if dataframe is None:
        return pd.DataFrame()

    result = dataframe.copy()

    for column in list(result.columns):
        if str(column) in VALUE_LOCALIZED_COLUMNS:
            result[column] = result[column].map(
                lambda value, c=str(column): localize_value(
                    value,
                    lang,
                    c,
                )
            )

    return result


def localized_column_config(
    dataframe: pd.DataFrame,
    language: str | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build Streamlit column_config using collision-safe localized display labels.
    Existing page-specific configurations are preserved.
    """
    labels = localized_column_labels(
        dataframe.columns,
        language,
    )

    config: dict[str, Any] = dict(existing or {})

    for raw_column, display_label in labels.items():
        if raw_column not in config:
            config[raw_column] = display_label

    return config


def render_localized_dataframe(
    dataframe: pd.DataFrame,
    language: str | None = None,
    **kwargs: Any,
) -> None:
    lang = active_language(language)
    display = localize_dataframe(
        dataframe,
        lang,
    )

    existing_config = kwargs.pop(
        "column_config",
        None,
    )

    kwargs["column_config"] = localized_column_config(
        display,
        lang,
        existing=existing_config,
    )

    st.dataframe(
        display,
        **kwargs,
    )
