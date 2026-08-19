from __future__ import annotations

import re
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.app_config import APP_TITLE
from dashboard.i18n import t
from dashboard.localization import render_localized_dataframe
from dashboard.layout import initialize_dashboard, localized_text
from dashboard.services.analysis_service import (
    get_available_date_bounds,
    load_analysis_data,
)
from dashboard.services.intelligence_service import (
    content_summary,
    filter_period,
    geo_summary,
    load_advanced_intelligence,
)


PRIORITY_LABELS = {
    "High": ("Yüksek", "High"),
    "Medium": ("Orta", "Medium"),
    "Low": ("Düşük", "Low"),
    "Critical": ("Kritik", "Critical"),
}


def _priority(value: object, language: str) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    pair = PRIORITY_LABELS.get(raw)
    if pair:
        return pair[0] if language == "tr" else pair[1]
    return raw


def _content_action(value: object, language: str) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if language != "tr":
        return raw

    low = raw.lower()

    if low.startswith("create or expand a supporting") or "supporting content" in low:
        return "Destekleyici içerik oluşturun veya mevcut içeriği genişletin; ilgili arama niyetlerini ve alt konuları sayfada kapsayın."
    if low.startswith("create"):
        return "Yeni ve arama niyetine uygun içerik oluşturun."
    if "expand" in low:
        return "Mevcut içeriği ilgili alt konular, sorular ve dahili bağlantılarla genişletin."
    if "update" in low or "refresh" in low:
        return "Mevcut içeriği güncelleyin; güncellik, kapsam ve arama niyeti uyumunu güçlendirin."
    if "merge" in low or "consolidat" in low:
        return "Benzer/çakışan içerikleri tek güçlü sayfada birleştirmeyi değerlendirin."
    if "internal link" in low:
        return "İlgili sayfalardan bu URL'ye anlamlı dahili bağlantılar ekleyin."

    if raw:
        return "İçerik gap sinyalini kapatmak için sayfayı arama niyeti, konu kapsamı ve dahili bağlantılar açısından geliştirin."
    return "Sayfayı arama niyeti, konu kapsamı ve dahili bağlantılar açısından geliştirin."


def _content_reason(score: object, keyword: object, language: str) -> str:
    if language == "en":
        if keyword is not None and not pd.isna(keyword) and str(keyword).strip():
            return f"Pipeline signals indicate an uncovered or under-served search opportunity around “{str(keyword).strip()}”."
        return "Pipeline content-gap signals indicate uncovered or under-served search demand for this page."

    if keyword is not None and not pd.isna(keyword) and str(keyword).strip():
        return (
            f"Pipeline sinyalleri “{str(keyword).strip()}” çevresinde yeterince karşılanmayan "
            "bir arama talebi/fırsatı olduğunu gösteriyor."
        )

    return (
        "Pipeline'daki içerik gap sinyalleri bu URL'de yeterince karşılanmayan "
        "arama talebi veya konu kapsamı fırsatı olduğunu gösteriyor."
    )



def _num(row: pd.Series, candidates: tuple[str, ...]) -> float | None:
    for column in candidates:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _text(row: pd.Series, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in row.index:
            value = row[column]
            if value is not None and not pd.isna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _content_reason_from_row(row: pd.Series, language: str) -> str:
    keyword = _text(row, ("keyword", "Keyword"))
    page_type = _text(row, ("CurrentLandingPageType", "page_type", "PageType")).lower()
    impressions = _num(row, ("Impressions", "impressions"))
    clicks = _num(row, ("Clicks", "clicks"))
    score = _num(row, ("ContentGapScore", "GapScore"))

    if impressions is not None and impressions >= 1000 and (clicks is None or clicks / max(impressions, 1) < 0.02):
        return (
            f"Bu fırsat yüksek görünürlük sinyali taşıyor ({impressions:,.0f} gösterim) ancak tıklama kazanımı düşük. "
            "İçerik ve SERP mesajı güçlendirilirse mevcut görünürlük daha fazla organik tıklamaya dönüşebilir."
            if language == "tr"
            else
            f"This opportunity has strong visibility ({impressions:,.0f} impressions) but weak click capture. "
            "Improving content and SERP messaging may convert existing visibility into more organic clicks."
        )

    if keyword:
        return (
            f"“{keyword}” arama niyeti için pipeline içerik boşluğu tespit ediyor. "
            f"Fırsat puanı {score:.1f}/100." if language == "tr" and score is not None
            else f"Pipeline signals an under-served content opportunity for “{keyword}”. "
                 f"Opportunity score: {score:.1f}/100." if language == "en" and score is not None
            else f"“{keyword}” arama niyeti için yeterince karşılanmayan bir içerik fırsatı tespit edildi."
                 if language == "tr"
            else f"An under-served content opportunity was detected for the “{keyword}” search intent."
        )

    if "category" in page_type or "kategori" in page_type:
        return (
            "Kategori sayfasının konu kapsamı ve arama niyeti kapsaması geliştirilebilir."
            if language == "tr"
            else "The category page can improve topical coverage and search-intent alignment."
        )

    if "product" in page_type or "ürün" in page_type:
        return (
            "Ürün sayfasında kullanıcı sorularını ve ilgili arama niyetlerini kapsayan içerik alanı güçlendirilebilir."
            if language == "tr"
            else "The product page can better cover user questions and related search intents."
        )

    return (
        f"Pipeline bu URL için güçlü bir içerik fırsatı tespit ediyor{f' ({score:.1f}/100)' if score is not None else ''}."
        if language == "tr"
        else f"The pipeline identifies a strong content opportunity for this URL{f' ({score:.1f}/100)' if score is not None else ''}."
    )


def _content_action_from_row(row: pd.Series, language: str) -> str:
    page = _text(row, ("page", "Page", "URL", "url")).lower()
    page_type = _text(row, ("CurrentLandingPageType", "page_type", "PageType")).lower()
    keyword = _text(row, ("keyword", "Keyword"))

    if any(token in page for token in ("/contact", "/iletisim", "/hakkimizda", "/sales/guest")):
        return (
            "Bu sorgu için mevcut kurumsal/işlemsel sayfayı zorlamak yerine uygun bir kategori, landing page veya destekleyici içerik oluşturun."
            if language == "tr"
            else
            "Instead of forcing this corporate/transactional page to rank, create or strengthen a relevant category, landing page or supporting content asset."
        )

    if "category" in page_type or "kategori" in page_type:
        return (
            "Kategori açıklamasını arama niyetine göre genişletin; ilgili alt başlıklar, SSS ve güçlü dahili bağlantılar ekleyin."
            if language == "tr"
            else
            "Expand the category copy around search intent; add relevant subtopics, FAQs and stronger internal links."
        )

    if "product" in page_type or "ürün" in page_type:
        return (
            "Ürün açıklamasını özgünleştirin; fayda/özellik, kullanım soruları, ilgili kategori bağlantıları ve uygun schema alanlarını güçlendirin."
            if language == "tr"
            else
            "Strengthen unique product copy, benefits/features, user questions, related-category links and relevant schema fields."
        )

    if "blog" in page_type or "content" in page_type or keyword:
        return (
            f"İçeriği {('“' + keyword + '” ve ilişkili alt konuları') if keyword else 'ilgili alt konuları'} kapsayacak şekilde genişletin; "
            "soru-cevap blokları ve ilgili ticari sayfalara dahili bağlantılar ekleyin."
            if language == "tr"
            else
            f"Expand the content to cover {('“' + keyword + '” and related subtopics') if keyword else 'related subtopics'}; "
            "add Q&A blocks and internal links to relevant commercial pages."
        )

    return (
        "Sayfayı arama niyeti, konu kapsamı, heading yapısı ve dahili bağlantılar açısından geliştirin."
        if language == "tr"
        else
        "Improve the page for search intent, topical coverage, heading structure and internal linking."
    )


def _geo_reason_from_row(row: pd.Series, language: str) -> str:
    readiness = _num(row, ("GEOReadinessScore", "ReadinessScore"))
    opportunity = _num(row, ("GEOOpportunityScore", "OpportunityScore"))
    page_type = _text(row, ("page_type", "PageType", "CurrentLandingPageType"))
    missing = _text(row, ("GEOMissingSignals", "MissingSignals"))

    parts = []
    if readiness is not None:
        parts.append(
            f"GEO hazırlığı {readiness:.1f}/100"
            if language == "tr"
            else f"GEO readiness is {readiness:.1f}/100"
        )
    if opportunity is not None:
        parts.append(
            f"fırsat puanı {opportunity:.1f}/100"
            if language == "tr"
            else f"opportunity score is {opportunity:.1f}/100"
        )

    if missing:
        signal_text = missing[:160]
        return (
            f"{', '.join(parts)}. Eksik sinyaller: {signal_text}. Bu eksikler, içeriğin AI destekli arama/cevap sistemleri tarafından anlaşılmasını zorlaştırabilir."
            if language == "tr"
            else
            f"{', '.join(parts)}. Missing signals: {signal_text}. These gaps can make the content harder for AI-assisted search/answer systems to interpret."
        )

    if page_type:
        return (
            f"{', '.join(parts)}. {page_type} sayfa türünde AI tarafından anlaşılabilirlik ve doğrudan cevap yapısı geliştirilebilir."
            if language == "tr"
            else
            f"{', '.join(parts)}. This {page_type} page can improve machine-understandable structure and direct-answer readiness."
        )

    return (
        f"{', '.join(parts)}. Sayfada AI tarafından anlaşılabilirliği artıracak GEO sinyalleri geliştirilebilir."
        if language == "tr"
        else
        f"{', '.join(parts)}. The page has GEO signals that can be strengthened for better machine understanding."
    )


def _geo_action_from_row(row: pd.Series, language: str) -> str:
    missing = _text(row, ("GEOMissingSignals", "MissingSignals")).lower()
    page_type = _text(row, ("page_type", "PageType", "CurrentLandingPageType")).lower()
    actions = []

    if "faq" in missing:
        actions.append("net soru-cevap bölümleri ekleyin" if language == "tr" else "add clear Q&A/FAQ sections")
    if "schema" in missing or "structured" in missing:
        actions.append("uygun schema/structured data ekleyin" if language == "tr" else "add appropriate schema/structured data")
    if "entity" in missing:
        actions.append("entity bağlamını daha açık tanımlayın" if language == "tr" else "clarify entity context")
    if "citation" in missing or "source" in missing or "reference" in missing:
        actions.append("güvenilir kaynak/referans sinyallerini güçlendirin" if language == "tr" else "strengthen trustworthy source/reference signals")
    if "answer" in missing or "direct" in missing:
        actions.append("kısa ve doğrudan cevap blokları ekleyin" if language == "tr" else "add concise direct-answer blocks")
    if "author" in missing or "expert" in missing or "eeat" in missing:
        actions.append("uzmanlık/yazar güven sinyallerini güçlendirin" if language == "tr" else "strengthen author/expertise trust signals")

    if not actions:
        if "product" in page_type or "ürün" in page_type:
            actions = [
                "ürün varlığını ve temel özellikleri açık tanımlayın" if language == "tr" else "define the product entity and key attributes clearly",
                "Product schema alanlarını doğrulayın" if language == "tr" else "validate Product schema fields",
                "kullanıcı soruları için kısa cevap blokları ekleyin" if language == "tr" else "add concise answers to common user questions",
            ]
        elif "category" in page_type or "kategori" in page_type:
            actions = [
                "kategori kapsamını ve alt konu ilişkilerini açıklaştırın" if language == "tr" else "clarify category scope and subtopic relationships",
                "ilgili entity bağlantılarını güçlendirin" if language == "tr" else "strengthen related entity relationships",
                "SSS ve uygun schema sinyalleri ekleyin" if language == "tr" else "add FAQs and relevant schema signals",
            ]
        else:
            actions = [
                "içeriğin soru-cevap yapısını güçlendirin" if language == "tr" else "strengthen the Q&A structure",
                "entity bağlamını açıklaştırın" if language == "tr" else "clarify entity context",
                "uygun structured data sinyallerini tamamlayın" if language == "tr" else "complete relevant structured-data signals",
            ]

    return "; ".join(action[0].upper() + action[1:] for action in actions) + "."


def _to_excel_bytes(frame: pd.DataFrame, sheet_name: str) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def _render_export_buttons(frame: pd.DataFrame, language: str, prefix: str, sheet_name: str) -> None:
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            localized_text(language, "⬇ CSV İndir", "⬇ Download CSV"),
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{prefix}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{prefix}_csv",
        )
    with export_cols[1]:
        st.download_button(
            localized_text(language, "⬇ Excel İndir", "⬇ Download Excel"),
            _to_excel_bytes(frame, sheet_name),
            file_name=f"{prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{prefix}_xlsx",
        )



def _metric_or_no_data(has_data: bool, value: float, language: str, suffix: str = "") -> str:
    if not has_data:
        return localized_text(language, "Veri Yok", "No Data")
    return f"{value:.1f}{suffix}"


def _geo_actions_from_signals(missing: object, original: object, language: str) -> str:
    raw_missing = "" if pd.isna(missing) else str(missing)
    raw_action = "" if pd.isna(original) else str(original)

    if language != "tr":
        return raw_action or "Improve the page's answer structure, entity clarity and structured-data signals."

    text = f"{raw_missing} {raw_action}".lower()
    actions = []

    if "faq" in text:
        actions.append("net soru-cevap bölümleri ekleyin")
    if "schema" in text or "structured" in text:
        actions.append("sayfa türüne uygun yapılandırılmış veri/schema ekleyin")
    if "entity" in text:
        actions.append("marka, ürün, kategori ve konu varlıklarını daha açık tanımlayın")
    if "citation" in text or "source" in text or "reference" in text:
        actions.append("güvenilir kaynak ve referans sinyallerini güçlendirin")
    if "answer" in text or "direct" in text:
        actions.append("önemli sorular için kısa ve doğrudan cevap blokları ekleyin")
    if "author" in text or "expert" in text or "eeat" in text:
        actions.append("yazar/uzmanlık ve güven sinyallerini güçlendirin")
    if "heading" in text or "semantic" in text:
        actions.append("başlık ve içerik hiyerarşisini daha açık hale getirin")

    if not actions:
        actions = [
            "içeriğin soru-cevap yapısını güçlendirin",
            "entity bağlamını açıklaştırın",
            "uygun yapılandırılmış veri sinyallerini tamamlayın",
        ]

    return "; ".join(action.capitalize() for action in actions) + "."


def _geo_reason(score: object, why_now: object, language: str) -> str:
    if language != "tr":
        if why_now is not None and not pd.isna(why_now) and str(why_now).strip():
            return str(why_now)
        return "The page has GEO-readiness gaps that can make it harder for AI-assisted search systems to interpret and reuse its content."

    numeric = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
    if pd.notna(numeric) and numeric >= 70:
        return (
            "GEO fırsat puanı yüksek. Sayfanın AI destekli arama/cevap sistemleri tarafından "
            "daha kolay anlaşılması için eksik sinyallerin öncelikli olarak güçlendirilmesi gerekir."
        )
    return (
        "Sayfada AI tarafından anlaşılabilirliği ve doğrudan cevap üretimine uygunluğu "
        "güçlendirebilecek GEO sinyalleri eksik."
    )


base = load_analysis_data()
_available_start, available_end = get_available_date_bounds(base.daily)
initial_language = st.session_state.get("dashboard_language", "tr")

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{'İçerik + GEO Zekâsı' if initial_language == 'tr' else 'Content + GEO Intelligence'}"
    ),
    page_icon="🧭",
    title="İçerik + GEO Zekâsı" if initial_language == "tr" else "Content + GEO Intelligence",
    subtitle=(
        "Hangi sayfada hangi içerik fırsatının bulunduğunu ve AI destekli arama görünürlüğü için neyin geliştirilmesi gerektiğini aksiyona dönüştürür."
        if initial_language == "tr"
        else "Turns content opportunities and AI-search readiness gaps into clear page-level actions."
    ),
    eyebrow="SEO & GEO KARAR ZEKÂSI" if initial_language == "tr" else "SEO & GEO DECISION INTELLIGENCE",
    default_preset="last_30_days",
    default_comparison="no_comparison",
    reference_date=available_end,
)

language = context.language
filters = context.filters

intel = load_advanced_intelligence()
gaps = filter_period(intel.content_gaps, filters.start_date, filters.end_date)
commerce = filter_period(intel.content_commerce, filters.start_date, filters.end_date)
geo = filter_period(intel.geo, filters.start_date, filters.end_date)

cs = content_summary(gaps, commerce)
gs = geo_summary(geo)

content_tab, geo_tab = st.tabs([
    localized_text(language, "İçerik Zekâsı", "Content Intelligence"),
    localized_text(language, "GEO Zekâsı", "GEO Intelligence"),
])

with content_tab:
    with st.expander(
        localized_text(language, "İçerik Zekâsı ne söylüyor?", "What does Content Intelligence tell me?"),
        expanded=False,
    ):
        st.markdown(
            localized_text(
                language,
                """
Bu bölümün temel sorusu: **“Google'da büyümek için hangi sayfada hangi içerik işini yapmalıyım?”**

- **İçerik Fırsatı:** İçerik gap sinyali bulunan sayfa/fırsat sayısı.
- **Yüksek Öncelik:** Önce ele alınması önerilen fırsatlar.
- **Ort. Fırsat Puanı:** 0–100 arası içerik fırsatının ortalama gücü. Yüksek = daha güçlü fırsat.
- **Commerce Skoru:** İçerik ile ticari sonuç arasındaki sinyal. Veri üretilmediyse **Veri Yok** gösterilir.
- Tabloda **Neden Fırsat?** ve **Önerilen Aksiyon** alanları kullanıcıya doğrudan ne yapılacağını açıklar.
""",
                """
This section answers: **“Which content work should I do on which page to grow organic search?”**

- **Content Opportunities:** Pages/opportunities with content-gap signals.
- **High Priority:** Opportunities recommended for earlier action.
- **Avg Opportunity Score:** Average opportunity strength on a 0–100 scale. Higher = stronger opportunity.
- **Commerce Score:** Signal linking content to commercial outcomes. If it was not produced, **No Data** is shown.
- The table explains **Why it is an opportunity** and the **Recommended Action**.
""",
            )
        )

    cols = st.columns(4)
    cols[0].metric(
        localized_text(language, "İçerik Fırsatı", "Content Opportunities"),
        cs["content_gaps"],
    )
    cols[1].metric(
        localized_text(language, "Yüksek Öncelik", "High Priority"),
        cs["high_priority"],
    )
    cols[2].metric(
        localized_text(language, "Ort. Fırsat Puanı", "Avg Opportunity Score"),
        _metric_or_no_data(not gaps.empty, cs["avg_gap_score"], language),
    )
    cols[3].metric(
        localized_text(language, "Ort. Commerce Skoru", "Avg Commerce Score"),
        _metric_or_no_data(not commerce.empty, cs["avg_commerce_score"], language),
    )

    st.subheader(localized_text(language, "Öncelikli İçerik Fırsatları", "Priority Content Opportunities"))

    if gaps.empty:
        st.info(
            localized_text(
                language,
                "Seçilen görünüm için içerik fırsatı verisi yok.",
                "No content-opportunity data is available for this view.",
            )
        )
    else:
        page_col = next((c for c in ("page", "Page", "URL", "url") if c in gaps.columns), None)
        keyword_col = next((c for c in ("keyword", "Keyword") if c in gaps.columns), None)
        score_col = next((c for c in ("ContentGapScore", "GapScore") if c in gaps.columns), None)
        priority_col = next((c for c in ("ContentGapPriority", "Priority") if c in gaps.columns), None)
        action_col = next((c for c in ("RecommendedContentAction", "RecommendedAction", "Action") if c in gaps.columns), None)

        work = gaps.copy()
        if score_col:
            work = work.sort_values(score_col, ascending=False)

        result = pd.DataFrame(index=work.index)

        if page_col:
            result[localized_text(language, "Sayfa", "Page")] = work[page_col]

        if keyword_col:
            result[localized_text(language, "Anahtar Kelime", "Keyword")] = work[keyword_col]

        if score_col:
            result[localized_text(language, "İçerik Fırsat Puanı (0–100)", "Content Opportunity Score (0–100)")] = (
                pd.to_numeric(work[score_col], errors="coerce").round(1)
            )

        if priority_col:
            result[localized_text(language, "Öncelik", "Priority")] = work[priority_col].map(
                lambda value: _priority(value, language)
            )

        result[localized_text(language, "Neden Fırsat?", "Why Is This an Opportunity?")] = [
            _content_reason_from_row(work.loc[idx], language)
            for idx in work.index
        ]

        result[localized_text(language, "Önerilen İçerik Aksiyonu", "Recommended Content Action")] = [
            _content_action_from_row(work.loc[idx], language)
            for idx in work.index
        ]

        render_localized_dataframe(
            result.head(100),
            width="stretch",
            hide_index=True,
            column_config={
                localized_text(language, "Neden Fırsat?", "Why Is This an Opportunity?"): st.column_config.TextColumn(width="large"),
                localized_text(language, "Önerilen İçerik Aksiyonu", "Recommended Content Action"): st.column_config.TextColumn(width="large"),
            },
        )

        _render_export_buttons(
            result.head(100),
            language,
            "content_opportunities",
            "Content Opportunities",
        )

    if not commerce.empty:
        with st.expander(
            localized_text(language, "İçerikten Ticarete Detay", "Content-to-Commerce Detail")
        ):
            render_localized_dataframe(commerce.head(100), width="stretch", hide_index=True)

with geo_tab:
    with st.expander(
        localized_text(language, "GEO Zekâsı ne işe yarıyor?", "What does GEO Intelligence do?"),
        expanded=True,
    ):
        st.markdown(
            localized_text(
                language,
                """
**GEO (Generative Engine Optimization)** bu ekranda şu soruyu cevaplar:

> **“İçeriğim AI destekli arama ve cevap sistemlerinin anlayabileceği ve cevap üretirken kullanmaya elverişli bir yapıda mı?”**

Bu skorlar doğrudan ChatGPT/Gemini citation sayısı değildir. Bunlar **GEO/AI readiness proxy** sinyalleridir.

- **GEO Hazırlık Skoru:** Sayfanın AI tarafından anlaşılabilir içerik yapısına ne kadar hazır olduğunu gösterir.
- **GEO Fırsat Puanı:** Geliştirme potansiyelini gösterir. Yüksek = daha fazla iyileştirme fırsatı.
- **Eksikler:** Sayfada güçlendirilmesi gereken sinyaller.
- **Yapılacak İş:** Eksikleri kapatmak için uygulanacak net aksiyonlar.
""",
                """
**GEO (Generative Engine Optimization)** answers:

> **“Is my content structured so AI-assisted search and answer systems can understand and reuse it effectively?”**

These are not direct ChatGPT/Gemini citation counts. They are **GEO/AI-readiness proxy** signals.

- **GEO Readiness Score:** How prepared the page is for machine-understandable answer structure.
- **GEO Opportunity Score:** Improvement potential. Higher = more room to improve.
- **Missing Signals:** Signals that should be strengthened.
- **Action:** Concrete work to close the gaps.
""",
            )
        )

    cols = st.columns(4)
    cols[0].metric(
        localized_text(language, "Analiz Edilen Sayfa", "Pages Analyzed"),
        gs["pages"],
    )
    cols[1].metric(
        localized_text(language, "Yüksek GEO Önceliği", "High GEO Priority"),
        gs["high_priority"],
    )
    cols[2].metric(
        localized_text(language, "Ort. GEO Hazırlığı", "Avg GEO Readiness"),
        _metric_or_no_data(not geo.empty, gs["avg_readiness"], language, "/100"),
    )
    cols[3].metric(
        localized_text(language, "Ort. GEO Fırsatı", "Avg GEO Opportunity"),
        _metric_or_no_data(not geo.empty, gs["avg_opportunity"], language, "/100"),
    )

    if geo.empty:
        st.info(
            localized_text(
                language,
                "GEO intelligence verisi bulunamadı.",
                "No GEO intelligence data is available.",
            )
        )
    else:
        page_col = next((c for c in ("page", "Page", "URL", "url") if c in geo.columns), None)
        readiness_col = next((c for c in ("GEOReadinessScore", "ReadinessScore") if c in geo.columns), None)
        opportunity_col = next((c for c in ("GEOOpportunityScore", "OpportunityScore") if c in geo.columns), None)
        priority_col = next((c for c in ("GEOPriority", "Priority") if c in geo.columns), None)
        missing_col = next((c for c in ("GEOMissingSignals", "MissingSignals") if c in geo.columns), None)
        action_col = next((c for c in ("GEORecommendedActions", "RecommendedActions") if c in geo.columns), None)
        why_col = next((c for c in ("GEOWhyNow", "WhyNow") if c in geo.columns), None)

        work = geo.copy()
        if opportunity_col:
            work = work.sort_values(opportunity_col, ascending=False)

        result = pd.DataFrame(index=work.index)

        if page_col:
            result[localized_text(language, "Sayfa", "Page")] = work[page_col]

        if readiness_col:
            result[localized_text(language, "GEO Hazırlık Skoru", "GEO Readiness Score")] = (
                pd.to_numeric(work[readiness_col], errors="coerce").round(1)
            )

        if opportunity_col:
            result[localized_text(language, "GEO Fırsat Puanı", "GEO Opportunity Score")] = (
                pd.to_numeric(work[opportunity_col], errors="coerce").round(1)
            )

        if priority_col:
            result[localized_text(language, "Öncelik", "Priority")] = work[priority_col].map(
                lambda value: _priority(value, language)
            )

        result[localized_text(language, "Neden Önemli?", "Why It Matters")] = [
            _geo_reason_from_row(work.loc[idx], language)
            for idx in work.index
        ]

        result[localized_text(language, "Yapılacak İş", "Recommended Action")] = [
            _geo_action_from_row(work.loc[idx], language)
            for idx in work.index
        ]

        render_localized_dataframe(
            result.head(100),
            width="stretch",
            hide_index=True,
            column_config={
                localized_text(language, "Neden Önemli?", "Why It Matters"): st.column_config.TextColumn(width="large"),
                localized_text(language, "Yapılacak İş", "Recommended Action"): st.column_config.TextColumn(width="large"),
            },
        )

        _render_export_buttons(
            result.head(100),
            language,
            "geo_intelligence",
            "GEO Intelligence",
        )
