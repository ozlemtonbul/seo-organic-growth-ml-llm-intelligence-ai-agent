from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


# ============================================================
# APP CONFIG
# ============================================================

from dashboard.app_config import APP_TITLE
from dashboard.i18n import t
from dashboard.url_state import resolve_language_from_url


# ============================================================
# ACTIVE LANGUAGE
# ============================================================

language = resolve_language_from_url()


# ============================================================
# GLOBAL PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=t("app_title", language),
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOCALIZED ROUTER LABELS
# ============================================================

labels = {
    "tr": {
        "overview_section": "Genel",
        "intelligence_section": "Karar Zekâsı",
        "advanced_section": "Gelişmiş SEO Zekâsı",
        "home": "Ana Panel",
        "executive": "Yönetici Özeti",
        "page_analysis": "Sayfa Analizi",
        "optimizer": "SEO Fırsat Optimizasyonu",
        "ai_insights": "AI İçgörüleri",
        "ask_ai": "AI'a Sor",
        "technical": "Teknik SEO",
        "content_geo": "İçerik + GEO Zekâsı",
        "competitor": "Rakip Zekâsı",
    },
    "en": {
        "overview_section": "Overview",
        "intelligence_section": "Decision Intelligence",
        "advanced_section": "Advanced SEO Intelligence",
        "home": "Home",
        "executive": "Executive Overview",
        "page_analysis": "Page Analysis",
        "optimizer": "SEO Opportunity Optimizer",
        "ai_insights": "AI Insights",
        "ask_ai": "Ask AI",
        "technical": "Technical SEO",
        "content_geo": "Content + GEO Intelligence",
        "competitor": "Competitor Intelligence",
    },
}

current = labels.get(
    language,
    labels["en"],
)


# ============================================================
# PAGE REGISTRY
# ============================================================

home_page = st.Page(
    "pages/0_Home.py",
    title=current["home"],
    icon="🏠",
    default=True,
)

executive_page = st.Page(
    "pages/1_Executive_Overview.py",
    url_path="executive-overview",
    title=current["executive"],
    icon="📊",
)

page_analysis_page = st.Page(
    "pages/2_Page_Analysis.py",
    url_path="page-analysis",
    title=current["page_analysis"],
    icon="📄",
)

optimizer_page = st.Page(
    "pages/3_SEO_Opportunity_Optimizer.py",
    url_path="seo-opportunity-optimizer",
    title=current["optimizer"],
    icon="🎯",
)

ai_insights_page = st.Page(
    "pages/4_AI_Insights.py",
    url_path="ai-insights",
    title=current["ai_insights"],
    icon="🧠",
)

ask_ai_page = st.Page(
    "pages/5_Ask_AI.py",
    url_path="ask-ai",
    title=current["ask_ai"],
    icon="💬",
)

technical_page = st.Page(
    "pages/6_Technical_SEO.py",
    url_path="technical-seo",
    title=current["technical"],
    icon="🛠️",
)

content_geo_page = st.Page(
    "pages/7_Content_GEO_Intelligence.py",
    url_path="content-geo-intelligence",
    title=current["content_geo"],
    icon="🧭",
)

competitor_page = st.Page(
    "pages/8_Competitor_Intelligence.py",
    url_path="competitor-intelligence",
    title=current["competitor"],
    icon="🔭",
)


# ============================================================
# SINGLE ROUTER
# ============================================================

navigation = st.navigation(
    {
        current["overview_section"]: [
            home_page,
            executive_page,
        ],
        current["intelligence_section"]: [
            page_analysis_page,
            optimizer_page,
            ai_insights_page,
            ask_ai_page,
        ],
        current["advanced_section"]: [
            technical_page,
            content_geo_page,
            competitor_page,
        ],
    },
    position="sidebar",
)

navigation.run()
