from __future__ import annotations

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
    filter_period,
    load_advanced_intelligence,
    technical_summary,
)


ISSUE_KNOWLEDGE = {
    "Orphan / No Internal Inlinks": {
        "tr_name": "İç Linki Olmayan Sayfa",
        "tr_why": (
            "Sayfaya site içinden bağlantı gelmemesi, arama motorlarının sayfayı keşfetmesini "
            "ve site içi otoritenin bu URL'ye aktarılmasını zorlaştırabilir."
        ),
        "tr_fix": (
            "İlgili kategori, ürün veya içerik sayfalarından bu URL'ye anlamlı dahili bağlantılar ekleyin."
        ),
        "en_why": (
            "A page with no internal inlinks can be harder for search engines to discover "
            "and may receive less internal authority."
        ),
        "en_fix": (
            "Add relevant internal links to this URL from related category, product or content pages."
        ),
    },
    "Structured Data Missing": {
        "tr_name": "Yapısal Veri Eksik",
        "tr_why": (
            "Uygun yapılandırılmış verinin eksik olması, arama motorlarının sayfanın içeriğini "
            "ve öğelerini daha net yorumlamasını zorlaştırabilir."
        ),
        "tr_fix": (
            "Sayfa türüne uygun schema.org işaretlemesini doğrulayın ve gerekli yapılandırılmış veriyi ekleyin."
        ),
        "en_why": (
            "Missing relevant structured data can make it harder for search engines to interpret "
            "the page and its entities clearly."
        ),
        "en_fix": (
            "Validate the page type and add the appropriate schema.org structured data."
        ),
    },
    "Canonical Points Elsewhere": {
        "tr_name": "Canonical Başka URL'ye İşaret Ediyor",
        "tr_why": (
            "Canonical etiketi başka bir URL'yi esas sayfa olarak gösterdiğinde, bu sayfanın "
            "indeksleme ve sıralama sinyalleri başka URL'ye aktarılabilir."
        ),
        "tr_fix": (
            "Canonical hedefinin gerçekten tercih edilen URL olup olmadığını kontrol edin; yanlışsa canonical etiketi düzeltin."
        ),
        "en_why": (
            "When the canonical points to another URL, indexing and ranking signals may be consolidated "
            "to that other page."
        ),
        "en_fix": (
            "Verify that the canonical target is the intended preferred URL; correct it if it is not."
        ),
    },
    "Missing Title": {
        "tr_name": "Title Eksik",
        "tr_why": "Title eksikliği sayfanın arama sonucundaki konu sinyalini ve tıklanabilirliğini zayıflatabilir.",
        "tr_fix": "Sayfanın ana arama niyetini açıklayan özgün ve kısa bir title ekleyin.",
        "en_why": "A missing title can weaken topical relevance and search-result clickability.",
        "en_fix": "Add a concise, unique title that reflects the page's primary search intent.",
    },
    "Missing Meta Description": {
        "tr_name": "Meta Açıklaması Eksik",
        "tr_why": "Meta açıklamasının olmaması arama sonucu mesajının kontrolünü ve CTR fırsatını azaltabilir.",
        "tr_fix": "Sayfanın değerini ve arama niyetini özetleyen özgün bir meta açıklaması ekleyin.",
        "en_why": "A missing meta description can reduce control over the search-result message and CTR opportunity.",
        "en_fix": "Add a unique meta description that summarizes the page value and search intent.",
    },
    "Duplicate Title": {
        "tr_name": "Tekrarlanan Title",
        "tr_why": "Aynı title'ların birden fazla sayfada kullanılması sayfaların birbirinden ayrışmasını zorlaştırabilir.",
        "tr_fix": "Her indekslenebilir sayfa için özgün ve amacına uygun title oluşturun.",
        "en_why": "Duplicate titles can make it harder to differentiate pages and their search intent.",
        "en_fix": "Create a unique, intent-specific title for each indexable page.",
    },
    "Duplicate Meta Description": {
        "tr_name": "Tekrarlanan Meta Açıklaması",
        "tr_why": "Tekrarlanan açıklamalar sayfa bazlı mesaj ve CTR optimizasyonunu zayıflatabilir.",
        "tr_fix": "Önemli indekslenebilir sayfalarda özgün meta açıklamaları kullanın.",
        "en_why": "Duplicate descriptions can weaken page-specific messaging and CTR optimization.",
        "en_fix": "Use unique meta descriptions on important indexable pages.",
    },
    "Broken Link": {
        "tr_name": "Kırık Link",
        "tr_why": "Kırık bağlantılar kullanıcı deneyimini ve crawler'ın site içinde ilerlemesini olumsuz etkileyebilir.",
        "tr_fix": "Kırık hedefi çalışan URL ile değiştirin veya artık gereksizse bağlantıyı kaldırın.",
        "en_why": "Broken links can hurt user experience and crawler navigation through the site.",
        "en_fix": "Replace the broken target with a working URL or remove the link if it is no longer needed.",
    },
    "Redirect Chain": {
        "tr_name": "Yönlendirme Zinciri",
        "tr_why": "Birden fazla ardışık yönlendirme tarama verimliliğini ve kullanıcı hızını olumsuz etkileyebilir.",
        "tr_fix": "Bağlantıları mümkün olduğunca nihai hedef URL'ye doğrudan yönlendirin.",
        "en_why": "Multiple consecutive redirects can reduce crawl efficiency and slow users down.",
        "en_fix": "Point links directly to the final destination URL wherever possible.",
    },
}


STATUS_LABELS = {
    "Audited": ("Doğrulandı", "Verified"),
    "Verified": ("Doğrulandı", "Verified"),
    "Ready": ("Hazır", "Ready"),
    "Pending": ("Bekliyor", "Pending"),
    "Not Audited": ("Denetlenmedi", "Not Audited"),
    "Open": ("Açık", "Open"),
    "Resolved": ("Çözüldü", "Resolved"),
}

SEVERITY_LABELS = {
    "Critical": ("Kritik", "Critical"),
    "High": ("Yüksek", "High"),
    "Medium": ("Orta", "Medium"),
    "Low": ("Düşük", "Low"),
}


def _pair(value: str, language: str, mapping: dict[str, tuple[str, str]]) -> str:
    pair = mapping.get(str(value).strip())
    if not pair:
        return str(value)
    return pair[0] if language == "tr" else pair[1]


def _issue_name(issue: object, language: str) -> str:
    raw = "" if pd.isna(issue) else str(issue).strip()
    knowledge = ISSUE_KNOWLEDGE.get(raw)
    if not knowledge:
        return raw
    return knowledge["tr_name"] if language == "tr" else raw


def _issue_why(issue: object, fallback: object, language: str) -> str:
    raw = "" if pd.isna(issue) else str(issue).strip()
    knowledge = ISSUE_KNOWLEDGE.get(raw)
    if knowledge:
        return knowledge["tr_why"] if language == "tr" else knowledge["en_why"]
    if fallback is not None and not pd.isna(fallback) and str(fallback).strip():
        return str(fallback)
    return localized_text(
        language,
        "Bu bulgu sayfanın taranabilirlik, indekslenebilirlik veya organik görünürlük performansını etkileyebilir.",
        "This finding may affect crawlability, indexability or organic visibility.",
    )


def _issue_fix(issue: object, fallback: object, language: str) -> str:
    raw = "" if pd.isna(issue) else str(issue).strip()
    knowledge = ISSUE_KNOWLEDGE.get(raw)
    if knowledge:
        return knowledge["tr_fix"] if language == "tr" else knowledge["en_fix"]
    if fallback is not None and not pd.isna(fallback) and str(fallback).strip():
        return str(fallback)
    return localized_text(
        language,
        "Sorunu ilgili sayfa üzerinde doğrulayın ve teknik SEO standardına göre düzeltin; ardından yeniden tarama ile kontrol edin.",
        "Validate the issue on the page, fix it according to technical SEO standards, then verify it with a new crawl.",
    )


def _priority_label(score: object, severity: object, language: str) -> str:
    numeric = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if numeric >= 80:
            key = "Critical"
        elif numeric >= 60:
            key = "High"
        elif numeric >= 40:
            key = "Medium"
        else:
            key = "Low"
        return _pair(key, language, SEVERITY_LABELS)

    severity_text = "" if pd.isna(severity) else str(severity).strip()
    return _pair(severity_text, language, SEVERITY_LABELS)



def _to_excel_bytes(frame: pd.DataFrame, sheet_name: str = "Technical SEO") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def _render_export_buttons(frame: pd.DataFrame, language: str) -> None:
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            label=localized_text(language, "⬇ CSV İndir", "⬇ Download CSV"),
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="technical_seo_findings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            label=localized_text(language, "⬇ Excel İndir", "⬇ Download Excel"),
            data=_to_excel_bytes(frame),
            file_name="technical_seo_findings.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


base = load_analysis_data()
_available_start, available_end = get_available_date_bounds(base.daily)
initial_language = st.session_state.get("dashboard_language", "tr")

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{'Teknik SEO Zekâsı' if initial_language == 'tr' else 'Technical SEO Intelligence'}"
    ),
    page_icon="🛠️",
    title="Teknik SEO Zekâsı" if initial_language == "tr" else "Technical SEO Intelligence",
    subtitle=(
        "Teknik sorunları yalnızca listelemez; her bulgunun neden önemli olduğunu, nasıl düzeltileceğini ve hangi sırada ele alınması gerektiğini gösterir."
        if initial_language == "tr"
        else "Does more than list technical issues: it explains why each finding matters, how to fix it and what should be handled first."
    ),
    eyebrow="SEO & GEO KARAR ZEKÂSI" if initial_language == "tr" else "SEO & GEO DECISION INTELLIGENCE",
    default_preset="last_30_days",
    default_comparison="no_comparison",
    reference_date=available_end,
)

language = context.language
filters = context.filters

intel = load_advanced_intelligence()
technical = filter_period(intel.technical, filters.start_date, filters.end_date)
summary = technical_summary(technical)

with st.expander(
    localized_text(language, "Bu ekranı nasıl okumalıyım?", "How should I read this page?"),
    expanded=False,
):
    st.markdown(
        localized_text(
            language,
            """
**Sorun:** URL'de tespit edilen teknik SEO problemi.  
**Neden önemli?** Problemin tarama, indeksleme, görünürlük veya kullanıcı deneyimine olası etkisi.  
**Nasıl düzeltilir?** Uygulanacak net teknik aksiyon.  
**Öncelik:** Hangi işin önce ele alınması gerektiğini sade biçimde gösterir.  
**Öncelik Puanı (0–100):** Puan yükseldikçe iş daha önce ele alınmalıdır.  
- **80–100:** Kritik  
- **60–79:** Yüksek  
- **40–59:** Orta  
- **0–39:** Düşük  

**Durum:** Bulgunun yaşam döngüsünü gösterir: Tespit Edildi → Doğrulandı → İşleme Alındı → Çözüldü → Yeniden Kontrol.
""",
            """
**Issue:** The technical SEO problem found on the URL.  
**Why it matters:** The potential impact on crawling, indexing, visibility or user experience.  
**How to fix it:** The concrete technical action to take.  
**Priority:** A simple indication of what should be handled first.  
**Priority Score (0–100):** Higher scores should be handled earlier. The score is used to prioritize findings using severity and available business/performance signals.  
**Status:** Shows whether the finding has been audited and verified.
""",
        )
    )

if summary["status"] == "no_data":
    st.warning(
        localized_text(
            language,
            "Teknik SEO çıktısı henüz bulunamadı. Pipeline çıktısı üretildiğinde bu ekran otomatik dolar.",
            "No Technical SEO output is available yet. This view populates automatically when the pipeline output is produced.",
        )
    )
elif summary["status"] == "not_audited":
    st.warning(
        localized_text(
            language,
            "Sayfalar için performans verisi var ancak crawl tabanlı teknik kontroller henüz doğrulanmamış. Sistem teknik sorun uydurmuyor.",
            "Performance data exists, but crawl-dependent technical checks have not yet been verified. The system does not fabricate technical findings.",
        )
    )

cols = st.columns(4)
cols[0].metric(localized_text(language, "Teknik Bulgu", "Technical Findings"), summary["issues"])
cols[1].metric(localized_text(language, "Kritik + Yüksek", "Critical + High"), summary["critical_high"])
cols[2].metric(localized_text(language, "Etkilenen URL", "Affected URLs"), summary["affected_urls"])
cols[3].metric(localized_text(language, "Denetlenen Kayıt", "Audited Records"), summary["audited"])

st.subheader(localized_text(language, "Öncelikli Teknik Sorunlar", "Priority Technical Issues"))

if technical.empty:
    st.info(localized_text(language, "Gösterilecek teknik veri yok.", "No technical data to display."))
else:
    issue_col = next((c for c in ("IssueType", "Issue", "issue_type") if c in technical.columns), None)
    page_col = next((c for c in ("Page", "page", "URL", "url") if c in technical.columns), None)
    severity_col = next((c for c in ("Severity", "severity") if c in technical.columns), None)
    score_col = next((c for c in ("BusinessPriorityScore", "PriorityScore") if c in technical.columns), None)
    status_col = next((c for c in ("AuditStatus", "Status", "status") if c in technical.columns), None)
    reason_col = next((c for c in ("Reason", "Why", "ImpactReason") if c in technical.columns), None)
    fix_col = next((c for c in ("Fix", "RecommendedFix", "Action") if c in technical.columns), None)

    work = technical.copy()

    if score_col:
        work = work.sort_values(score_col, ascending=False)

    result = pd.DataFrame(index=work.index)

    if page_col:
        result[localized_text(language, "Sayfa", "Page")] = work[page_col]

    if issue_col:
        result[localized_text(language, "Sorun", "Issue")] = work[issue_col].map(
            lambda value: _issue_name(value, language)
        )

    if severity_col:
        result[localized_text(language, "Önem", "Severity")] = work[severity_col].map(
            lambda value: _pair(value, language, SEVERITY_LABELS)
        )

    if issue_col:
        result[localized_text(language, "Neden Önemli?", "Why It Matters")] = [
            _issue_why(
                issue,
                work.at[idx, reason_col] if reason_col else None,
                language,
            )
            for idx, issue in work[issue_col].items()
        ]
        result[localized_text(language, "Nasıl Düzeltilir?", "How to Fix")] = [
            _issue_fix(
                issue,
                work.at[idx, fix_col] if fix_col else None,
                language,
            )
            for idx, issue in work[issue_col].items()
        ]

    result[localized_text(language, "Öncelik", "Priority")] = [
        _priority_label(
            work.at[idx, score_col] if score_col else None,
            work.at[idx, severity_col] if severity_col else None,
            language,
        )
        for idx in work.index
    ]

    if score_col:
        numeric_scores = pd.to_numeric(work[score_col], errors="coerce")
        result[localized_text(language, "Öncelik Puanı (0–100)", "Priority Score (0–100)")] = numeric_scores.round(1)

    if status_col:
        result[localized_text(language, "Durum", "Status")] = work[status_col].map(
            lambda value: _pair(value, language, STATUS_LABELS)
        )

    compact_columns = [
        col for col in [
            localized_text(language, "Sayfa", "Page"),
            localized_text(language, "Sorun", "Issue"),
            localized_text(language, "Önem", "Severity"),
            localized_text(language, "Öncelik", "Priority"),
            localized_text(language, "Öncelik Puanı (0–100)", "Priority Score (0–100)"),
            localized_text(language, "Durum", "Status"),
        ]
        if col in result.columns
    ]

    render_localized_dataframe(
        result[compact_columns].head(100),
        width="stretch",
        hide_index=True,
    )

    _render_export_buttons(result.head(100), language)

    st.markdown("---")
    st.subheader(localized_text(language, "Sorun Detayı", "Issue Detail"))

    detail_options = []
    for idx, row in result.head(100).iterrows():
        page_value = row.get(localized_text(language, "Sayfa", "Page"), "")
        issue_value = row.get(localized_text(language, "Sorun", "Issue"), "")
        detail_options.append((idx, f"{issue_value} — {page_value}"))

    if detail_options:
        selected_idx = st.selectbox(
            localized_text(language, "Detayını görmek istediğiniz sorunu seçin", "Select an issue to inspect"),
            options=[item[0] for item in detail_options],
            format_func=lambda selected: next(label for idx, label in detail_options if idx == selected),
        )
        selected = result.loc[selected_idx]

        with st.container(border=True):
            st.markdown(
                f"**{localized_text(language, 'Sorun', 'Issue')}:** "
                f"{selected.get(localized_text(language, 'Sorun', 'Issue'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Sayfa', 'Page')}:** "
                f"{selected.get(localized_text(language, 'Sayfa', 'Page'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Neden Önemli?', 'Why It Matters')}:** "
                f"{selected.get(localized_text(language, 'Neden Önemli?', 'Why It Matters'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Nasıl Düzeltilir?', 'How to Fix')}:** "
                f"{selected.get(localized_text(language, 'Nasıl Düzeltilir?', 'How to Fix'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Öncelik', 'Priority')}:** "
                f"{selected.get(localized_text(language, 'Öncelik', 'Priority'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Öncelik Puanı (0–100)', 'Priority Score (0–100)')}:** "
                f"{selected.get(localized_text(language, 'Öncelik Puanı (0–100)', 'Priority Score (0–100)'), '-')}"
            )
            st.markdown(
                f"**{localized_text(language, 'Durum', 'Status')}:** "
                f"{selected.get(localized_text(language, 'Durum', 'Status'), '-')}"
            )
