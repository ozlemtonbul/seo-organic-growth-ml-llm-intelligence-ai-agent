from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

from dashboard.i18n import SUPPORTED_LANGUAGES, TRANSLATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
PAGE_ROOT = DASHBOARD_ROOT / "pages"


def test_translation_dictionary_has_tr_and_en_for_every_key():
    assert {"tr", "en"}.issubset(SUPPORTED_LANGUAGES)

    broken = []
    for key, mapping in TRANSLATIONS.items():
        for language in ("tr", "en"):
            value = mapping.get(language)
            if not isinstance(value, str) or not value.strip():
                broken.append(f"{key}:{language}")

    assert not broken, (
        "Missing/empty translation values: " + ", ".join(broken)
    )


def test_all_nine_dashboard_pages_exist():
    expected = [
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

    missing = [
        name for name in expected
        if not (PAGE_ROOT / name).exists()
    ]
    assert not missing, f"Missing dashboard pages: {missing}"


def test_single_router_navigation_contract():
    source = (DASHBOARD_ROOT / "app.py").read_text(encoding="utf-8-sig")

    assert "st.navigation(" in source
    assert "navigation.run()" in source
    assert "st.page_link(" not in source
    assert "st.switch_page(" not in source


def test_duplicate_executive_analysis_button_is_removed():
    source = (
        PAGE_ROOT / "1_Executive_Overview.py"
    ).read_text(encoding="utf-8-sig")

    assert "Seçilen Dönemi Analiz Et" not in source
    assert "Analyze Selected Period" not in source
    assert "executive_analyze_selected_period" not in source


def test_home_uses_selected_period_filter_for_kpis():
    source = (
        PAGE_ROOT / "0_Home.py"
    ).read_text(encoding="utf-8-sig")

    assert "filter_analysis_data(" in source
    assert "DateRange(" in source
    assert "period_kpi_source" in source
    assert "Seçili Dönemde Veri Bulunan Gün" in source
    assert "Toplam Veri Kapsamı" in source


def test_source_has_no_common_mojibake_sequences():
    suspicious = [
        "Ã¼", "Ã¶", "Ã§", "Ã‡", "Ä±", "ÅŸ", "ÄŸ",
        "â€“", "â€”", "â€™", "\ufffd",
    ]

    offenders = []
    for path in DASHBOARD_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        found = [token for token in suspicious if token in text]
        if found:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {found}")

    assert not offenders, (
        "Possible encoding/mojibake problems:\n" + "\n".join(offenders)
    )


UI_METHODS = {
    "title", "header", "subheader", "caption", "button",
    "info", "warning", "error", "success", "metric",
    "selectbox", "radio", "markdown", "write", "text",
    "text_input", "text_area",
}

NEUTRAL_CONSTANTS = {
    "---",
    "CSV",
    "Excel",
}

NEUTRAL_PREFIXES = (
    "<style",
    "<div ",
)


def _is_human_text(value: str) -> bool:
    cleaned = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return bool(cleaned)


def test_no_unlocalized_bare_streamlit_ui_strings():
    """
    Strict source-level localization check.

    A bare literal passed directly to a Streamlit UI element cannot switch
    between TR and EN. Conditional expressions, t(...), localized_text(...),
    variables, and generated values are accepted.
    """
    offenders = []

    for path in DASHBOARD_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in UI_METHODS:
                continue
            if not node.args:
                continue

            first = node.args[0]
            if not (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                continue

            value = first.value.strip()
            if not value:
                continue
            if value in NEUTRAL_CONSTANTS:
                continue
            if value.startswith(NEUTRAL_PREFIXES):
                continue
            if not _is_human_text(value):
                continue

            offenders.append(
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                f"{node.func.attr}({value!r})"
            )

    assert not offenders, (
        "Hard-coded UI strings cannot fully follow TR/EN mode. "
        "Localize these calls:\n" + "\n".join(offenders)
    )


def test_initialize_dashboard_headings_are_language_aware():
    """
    title/subtitle/eyebrow supplied as bare human-language literals are not
    allowed, because they cannot change after TR/EN selection.
    """
    offenders = []

    for path in PAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name != "initialize_dashboard":
                continue

            for kw in node.keywords:
                if kw.arg not in {"title", "subtitle", "eyebrow"}:
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, str
                ):
                    value = kw.value.value.strip()
                    if _is_human_text(value):
                        offenders.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                            f"{kw.arg}={value!r}"
                        )

    assert not offenders, (
        "Dashboard headings contain fixed-language literals:\n"
        + "\n".join(offenders)
    )


def test_router_contains_both_tr_and_en_labels():
    source = (DASHBOARD_ROOT / "app.py").read_text(encoding="utf-8-sig")

    required_tr = [
        "Ana Panel",
        "Yönetici Özeti",
        "Sayfa Analizi",
        "SEO Fırsat Optimizasyonu",
        "AI İçgörüleri",
        "AI'a Sor",
        "Teknik SEO",
        "İçerik + GEO Zekâsı",
        "Rakip Zekâsı",
    ]
    required_en = [
        "Home",
        "Executive Overview",
        "Page Analysis",
        "SEO Opportunity Optimizer",
        "AI Insights",
        "Ask AI",
        "Technical SEO",
        "Content + GEO Intelligence",
        "Competitor Intelligence",
    ]

    for label in required_tr + required_en:
        assert label in source, f"Router localization label missing: {label}"


def test_application_display_title_is_localized():
    from dashboard.i18n import t

    assert t("app_title", "tr") == "SEO Organik Büyüme Zekâsı AI Agentı"
    assert t("app_title", "en") == "SEO Organic Growth Intelligence AI Agent"
    assert t("app_title", "tr") != t("app_title", "en")


def test_page_title_does_not_force_english_language():
    offenders = []

    for path in PAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name != "initialize_dashboard":
                continue

            for kw in node.keywords:
                if kw.arg != "page_title":
                    continue

                segment = ast.get_source_segment(source, kw.value) or ""
                compact = segment.replace('"', "'")

                assert "t('executive_overview', 'en')" not in compact
                assert "t('page_analysis', 'en')" not in compact
                assert "t('seo_opportunity_optimizer', 'en')" not in compact
                assert "t('ai_insights', 'en')" not in compact
                assert "t('ask_ai', 'en')" not in compact

                if "initial_language" not in segment and "t(" not in segment:
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                        f"page_title={segment!r}"
                    )

    assert not offenders, (
        "Browser/site page titles are not language-aware:\n"
        + "\n".join(offenders)
    )


def test_pages_do_not_render_raw_dataframes_directly():
    offenders = []

    for path in PAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        if "st.dataframe(" in source:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, (
        "Pages must use render_localized_dataframe instead of raw st.dataframe: "
        + ", ".join(offenders)
    )


def test_ask_ai_does_not_render_raw_runtime_json():
    source = (
        PAGE_ROOT / "5_Ask_AI.py"
    ).read_text(encoding="utf-8-sig")

    assert "st.json(" not in source
    assert "AI Çalışma Durumu" in source
    assert "AI Runtime Status" in source


def test_optimizer_avoids_unlocalizable_multiselect_internal_ui():
    source = (
        PAGE_ROOT / "3_SEO_Opportunity_Optimizer.py"
    ).read_text(encoding="utf-8-sig")

    assert "st.multiselect(" not in source
    assert '"Tümü"' in source
    assert '"All"' in source


def test_shared_charts_use_display_localization_layer():
    source = (
        DASHBOARD_ROOT / "components" / "charts.py"
    ).read_text(encoding="utf-8-sig")

    assert "localize_dataframe" in source
    assert "localize_column_name" in source


def test_localization_preserves_internal_column_names_and_uniqueness():
    import pandas as pd

    from dashboard.localization import (
        localize_dataframe,
        localized_column_labels,
    )

    frame = pd.DataFrame(
        {
            "ctr": [0.10],
            "CTR": [0.20],
            "Scenario": ["maintain"],
            "scenario": ["title_meta_optimization"],
            "keyword_intent": ["Transactional"],
            "KeywordIntent": ["Informational"],
        }
    )

    localized = localize_dataframe(
        frame,
        "tr",
    )

    assert list(localized.columns) == list(frame.columns)
    assert localized.columns.is_unique

    labels = localized_column_labels(
        localized.columns,
        "tr",
    )

    assert len(labels) == len(frame.columns)
    assert len(set(labels.values())) == len(frame.columns)
    assert labels["ctr"] == "CTR"
    assert labels["CTR"] == "CTR (2)"
    assert labels["Scenario"] == "Senaryo"
    assert labels["scenario"] == "Senaryo (2)"


def test_localized_dataframe_values_follow_active_language():
    import pandas as pd

    from dashboard.localization import localize_dataframe

    frame = pd.DataFrame(
        {
            "Scenario": ["title_meta_optimization"],
            "page_type": ["category"],
            "keyword_intent": ["Transactional"],
            "PriorityTier": ["High Priority"],
            "ConfidenceLevel": ["High"],
        }
    )

    tr = localize_dataframe(frame, "tr")
    en = localize_dataframe(frame, "en")

    assert tr.loc[0, "Scenario"] == "Başlık ve Meta Optimizasyonu"
    assert tr.loc[0, "page_type"] == "Kategori"
    assert tr.loc[0, "keyword_intent"] == "İşlemsel"
    assert tr.loc[0, "PriorityTier"] == "Yüksek Öncelik"
    assert tr.loc[0, "ConfidenceLevel"] == "Yüksek"

    assert en.loc[0, "Scenario"] == "Title and Meta Optimization"
    assert en.loc[0, "page_type"] == "Category"
    assert en.loc[0, "keyword_intent"] == "Transactional"
    assert en.loc[0, "PriorityTier"] == "High Priority"
    assert en.loc[0, "ConfidenceLevel"] == "High"

def test_router_uses_clean_shareable_url_paths():
    source = (DASHBOARD_ROOT / "app.py").read_text(encoding="utf-8-sig")

    expected_paths = [
        'url_path="executive-overview"',
        'url_path="page-analysis"',
        'url_path="seo-opportunity-optimizer"',
        'url_path="ai-insights"',
        'url_path="ask-ai"',
        'url_path="technical-seo"',
        'url_path="content-geo-intelligence"',
        'url_path="competitor-intelligence"',
    ]

    for expected in expected_paths:
        assert expected in source, f"Clean Streamlit url_path missing: {expected}"

    # Streamlit url_path values must stay flat (no forward slashes).
    for expected in expected_paths:
        value = expected.split('"', 2)[1]
        assert "/" not in value


def test_language_is_synchronized_between_url_and_session_state():
    app_source = (DASHBOARD_ROOT / "app.py").read_text(encoding="utf-8-sig")
    layout_source = (DASHBOARD_ROOT / "layout.py").read_text(encoding="utf-8-sig")
    url_state_source = (DASHBOARD_ROOT / "url_state.py").read_text(encoding="utf-8-sig")

    assert "resolve_language_from_url()" in app_source
    assert "sync_language_to_url(" in layout_source
    assert "sync_language_widget_to_url" in layout_source
    assert "on_change=sync_language_widget_to_url" in layout_source

    assert 'LANG_QUERY_KEY = "lang"' in url_state_source
    assert "st.query_params" in url_state_source
    assert 'st.session_state["dashboard_language"]' in url_state_source
    assert "url_language or session_language" in url_state_source


def test_language_url_module_compiles_without_bom():
    source = (DASHBOARD_ROOT / "url_state.py").read_text(encoding="utf-8")
    assert not source.startswith("\\ufeff")
    compile(source, str(DASHBOARD_ROOT / "url_state.py"), "exec")

