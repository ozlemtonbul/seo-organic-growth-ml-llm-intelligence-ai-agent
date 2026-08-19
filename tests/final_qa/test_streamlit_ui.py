from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit AppTest is not available in this Streamlit version.",
)
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGE_ROOT = PROJECT_ROOT / "dashboard" / "pages"


PAGES = [
    "0_Home.py",
    "1_Executive_Overview.py",
    "2_Page_Analysis.py",
    "3_SEO_Opportunity_Optimizer.py",
    "4_AI_Insights.py",
    "5_Ask_AI.py",
    "6_Technical_SEO.py",
    "7_Content_GEO_Intelligence.py",
    "8_Competitor_Intelligence.py",
]


def _exceptions(at: AppTest) -> list[str]:
    result = []
    try:
        for item in at.exception:
            value = getattr(item, "value", None)
            result.append(str(value if value is not None else item))
    except Exception:
        pass
    return result


def _collect_visible_text(at: AppTest) -> str:
    chunks = []

    element_groups = [
        "title", "header", "subheader", "caption", "markdown",
        "info", "warning", "error", "success", "metric",
        "button", "selectbox", "radio", "text_input", "text_area",
    ]

    for group_name in element_groups:
        group = getattr(at, group_name, None)
        if group is None:
            continue

        try:
            items = list(group)
        except Exception:
            continue

        for item in items:
            for attr in ("label", "value", "help"):
                try:
                    value = getattr(item, attr, None)
                except Exception:
                    value = None
                if isinstance(value, str) and value.strip():
                    chunks.append(value)

    return "\n".join(chunks)


@pytest.mark.parametrize("page_name", PAGES)
@pytest.mark.parametrize("language", ["tr", "en"])
def test_every_page_runs_in_both_languages(page_name, language):
    path = PAGE_ROOT / page_name
    at = AppTest.from_file(str(path))
    at.session_state["dashboard_language"] = language
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors, (
        f"{page_name} crashed in language={language}:\n"
        + "\n".join(errors)
    )


@pytest.mark.parametrize("page_name", PAGES)
def test_tr_mode_does_not_show_common_english_fallbacks(page_name):
    at = AppTest.from_file(str(PAGE_ROOT / page_name))
    at.session_state["dashboard_language"] = "tr"
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors

    text = _collect_visible_text(at).lower()

    forbidden = [
        "apply analysis",
        "analyze selected period",
        "no data available",
        "no page data available",
        "no recommendations available",
        "not connected",
    ]

    found = [phrase for phrase in forbidden if phrase in text]
    assert not found, (
        f"{page_name} shows English fallback text in TR mode: {found}"
    )


@pytest.mark.parametrize("page_name", PAGES)
def test_en_mode_does_not_show_common_turkish_fallbacks(page_name):
    at = AppTest.from_file(str(PAGE_ROOT / page_name))
    at.session_state["dashboard_language"] = "en"
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors

    text = _collect_visible_text(at).lower()

    forbidden = [
        "analizi uygula",
        "seçilen dönemi analiz et",
        "veri bulunamadı",
        "bağlı değil",
        "yönetici özeti",
    ]

    found = [phrase for phrase in forbidden if phrase in text]
    assert not found, (
        f"{page_name} shows Turkish fallback text in EN mode: {found}"
    )


def test_home_date_filter_controls_are_rendered():
    at = AppTest.from_file(str(PAGE_ROOT / "0_Home.py"))
    at.session_state["dashboard_language"] = "tr"
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors

    labels = []
    for group_name in ("selectbox", "button"):
        group = getattr(at, group_name, None)
        if group is None:
            continue
        for item in list(group):
            label = getattr(item, "label", None)
            if isinstance(label, str):
                labels.append(label)

    joined = "\n".join(labels)
    assert "Analizi Uygula" in joined


def test_executive_has_no_duplicate_analysis_action():
    at = AppTest.from_file(str(PAGE_ROOT / "1_Executive_Overview.py"))
    at.session_state["dashboard_language"] = "tr"
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors

    button_labels = [
        getattr(item, "label", "")
        for item in list(at.button)
    ]

    assert "Seçilen Dönemi Analiz Et" not in button_labels
    assert button_labels.count("Analizi Uygula") <= 1


EXPECTED_HERO = {
    "0_Home.py": {
        "tr": ("SEO Organik Büyüme Zekâsı AI Agentı", "SEO & GEO Karar Zekâsı"),
        "en": ("SEO Organic Growth Intelligence AI Agent", "SEO & GEO Decision Intelligence"),
    },
    "1_Executive_Overview.py": {
        "tr": ("Yönetici Özeti", "SEO & GEO Karar Zekâsı"),
        "en": ("Executive Overview", "SEO & GEO Decision Intelligence"),
    },
    "2_Page_Analysis.py": {
        "tr": ("Sayfa Analizi", "SEO & GEO Karar Zekâsı"),
        "en": ("Page Analysis", "SEO & GEO Decision Intelligence"),
    },
    "3_SEO_Opportunity_Optimizer.py": {
        "tr": ("SEO Fırsat Optimizasyonu", "SEO & GEO Karar Zekâsı"),
        "en": ("SEO Opportunity Optimizer", "SEO & GEO Decision Intelligence"),
    },
    "4_AI_Insights.py": {
        "tr": ("AI İçgörüleri", "SEO & GEO Karar Zekâsı"),
        "en": ("AI Insights", "SEO & GEO Decision Intelligence"),
    },
    "5_Ask_AI.py": {
        "tr": ("AI Asistan", "SEO & GEO Karar Zekâsı"),
        "en": ("AI Assistant", "SEO & GEO Decision Intelligence"),
    },
    "6_Technical_SEO.py": {
        "tr": ("Teknik SEO Zekâsı", "SEO & GEO KARAR ZEKÂSI"),
        "en": ("Technical SEO Intelligence", "SEO & GEO DECISION INTELLIGENCE"),
    },
    "7_Content_GEO_Intelligence.py": {
        "tr": ("İçerik + GEO Zekâsı", "SEO & GEO KARAR ZEKÂSI"),
        "en": ("Content + GEO Intelligence", "SEO & GEO DECISION INTELLIGENCE"),
    },
    "8_Competitor_Intelligence.py": {
        "tr": ("Rakip Zekâsı", "SEO & GEO KARAR ZEKÂSI"),
        "en": ("Competitor Intelligence", "SEO & GEO DECISION INTELLIGENCE"),
    },
}


@pytest.mark.parametrize("page_name", PAGES)
@pytest.mark.parametrize("language", ["tr", "en"])
def test_page_hero_title_and_eyebrow_match_active_language(page_name, language):
    at = AppTest.from_file(str(PAGE_ROOT / page_name))
    at.session_state["dashboard_language"] = language
    at.run(timeout=45)

    errors = _exceptions(at)
    assert not errors

    rendered = "\n".join(
        str(getattr(item, "value", ""))
        for item in list(at.markdown)
    )

    expected_title, expected_eyebrow = EXPECTED_HERO[page_name][language]
    assert expected_title in rendered, (
        f"{page_name} does not render the expected {language} hero title: "
        f"{expected_title!r}"
    )
    assert expected_eyebrow in rendered, (
        f"{page_name} does not render the expected {language} eyebrow: "
        f"{expected_eyebrow!r}"
    )

    opposite = "en" if language == "tr" else "tr"
    opposite_title, opposite_eyebrow = EXPECTED_HERO[page_name][opposite]

    if opposite_title != expected_title:
        assert opposite_title not in rendered, (
            f"{page_name} renders the opposite-language title in {language} mode: "
            f"{opposite_title!r}"
        )

    if opposite_eyebrow != expected_eyebrow:
        assert opposite_eyebrow not in rendered, (
            f"{page_name} renders the opposite-language eyebrow in {language} mode: "
            f"{opposite_eyebrow!r}"
        )
