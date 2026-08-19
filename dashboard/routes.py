from __future__ import annotations

from typing import Final


SUPPORTED_LANGUAGES: Final[tuple[str, str]] = ("tr", "en")

# Stable semantic keys let the router keep the same logical page while
# switching the visible pathname between Turkish and English.
PAGE_SPECS: Final[dict[str, dict[str, object]]] = {
    "home": {
        "source": "pages/0_Home.py",
        "icon": "🏠",
        "titles": {"tr": "Ana Panel", "en": "Home"},
        "slugs": {"tr": "ana-panel", "en": "home"},
    },
    "executive": {
        "source": "pages/1_Executive_Overview.py",
        "icon": "📊",
        "titles": {"tr": "Yönetici Özeti", "en": "Executive Overview"},
        "slugs": {"tr": "yonetici-ozeti", "en": "executive-overview"},
    },
    "page_analysis": {
        "source": "pages/2_Page_Analysis.py",
        "icon": "📄",
        "titles": {"tr": "Sayfa Analizi", "en": "Page Analysis"},
        "slugs": {"tr": "sayfa-analizi", "en": "page-analysis"},
    },
    "optimizer": {
        "source": "pages/3_SEO_Opportunity_Optimizer.py",
        "icon": "🎯",
        "titles": {"tr": "SEO Fırsat Optimizasyonu", "en": "SEO Opportunity Optimizer"},
        "slugs": {"tr": "seo-firsat-optimizasyonu", "en": "seo-opportunity-optimizer"},
    },
    "ai_insights": {
        "source": "pages/4_AI_Insights.py",
        "icon": "🧠",
        "titles": {"tr": "AI İçgörüleri", "en": "AI Insights"},
        "slugs": {"tr": "ai-icgoruleri", "en": "ai-insights"},
    },
    "ask_ai": {
        "source": "pages/5_Ask_AI.py",
        "icon": "💬",
        "titles": {"tr": "AI'a Sor", "en": "Ask AI"},
        "slugs": {"tr": "ai-a-sor", "en": "ask-ai"},
    },
    "technical": {
        "source": "pages/6_Technical_SEO.py",
        "icon": "🛠️",
        "titles": {"tr": "Teknik SEO", "en": "Technical SEO"},
        "slugs": {"tr": "teknik-seo", "en": "technical-seo"},
    },
    "content_geo": {
        "source": "pages/7_Content_GEO_Intelligence.py",
        "icon": "🧭",
        "titles": {"tr": "İçerik + GEO Zekâsı", "en": "Content + GEO Intelligence"},
        "slugs": {"tr": "icerik-geo-zekasi", "en": "content-geo-intelligence"},
    },
    "competitor": {
        "source": "pages/8_Competitor_Intelligence.py",
        "icon": "🔭",
        "titles": {"tr": "Rakip Zekâsı", "en": "Competitor Intelligence"},
        "slugs": {"tr": "rakip-zekasi", "en": "competitor-intelligence"},
    },
}

OVERVIEW_PAGE_KEYS: Final[tuple[str, ...]] = ("home", "executive")
INTELLIGENCE_PAGE_KEYS: Final[tuple[str, ...]] = (
    "page_analysis",
    "optimizer",
    "ai_insights",
    "ask_ai",
)
ADVANCED_PAGE_KEYS: Final[tuple[str, ...]] = (
    "technical",
    "content_geo",
    "competitor",
)


def page_slug(page_key: str, language: str) -> str:
    return str(PAGE_SPECS[page_key]["slugs"][language])


def page_title(page_key: str, language: str) -> str:
    return str(PAGE_SPECS[page_key]["titles"][language])


def page_source(page_key: str) -> str:
    return str(PAGE_SPECS[page_key]["source"])


def page_icon(page_key: str) -> str:
    return str(PAGE_SPECS[page_key]["icon"])


SLUG_TO_PAGE_KEY: Final[dict[str, str]] = {
    page_slug(page_key, language): page_key
    for page_key in PAGE_SPECS
    for language in SUPPORTED_LANGUAGES
}


def page_key_from_slug(slug: str) -> str | None:
    return SLUG_TO_PAGE_KEY.get(str(slug).strip().strip("/"))

SLUG_TO_LANGUAGE: Final[dict[str, str]] = {
    page_slug(page_key, language): language
    for page_key in PAGE_SPECS
    for language in SUPPORTED_LANGUAGES
}


def language_from_slug(slug: str) -> str | None:
    """Return the language encoded by a localized page slug."""
    return SLUG_TO_LANGUAGE.get(str(slug).strip().strip("/"))
