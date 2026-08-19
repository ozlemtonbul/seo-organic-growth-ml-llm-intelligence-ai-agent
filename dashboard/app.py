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

from dashboard.i18n import t
from dashboard.routes import (
    ADVANCED_PAGE_KEYS,
    INTELLIGENCE_PAGE_KEYS,
    OVERVIEW_PAGE_KEYS,
    PAGE_SPECS,
    SUPPORTED_LANGUAGES,
    page_icon,
    page_key_from_slug,
    page_slug,
    page_source,
    page_title,
)
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
# LOCALIZED NAVIGATION SECTIONS
# ============================================================

section_labels = {
    "tr": {
        "overview": "Genel",
        "intelligence": "Karar Zekâsı",
        "advanced": "Gelişmiş SEO Zekâsı",
    },
    "en": {
        "overview": "Overview",
        "intelligence": "Decision Intelligence",
        "advanced": "Advanced SEO Intelligence",
    },
}

current_sections = section_labels.get(language, section_labels["en"])


# ============================================================
# BILINGUAL PAGE REGISTRY
# ============================================================

# Streamlit requires every directly addressable page to be registered during
# the session's initial st.navigation call. Therefore both TR and EN pathname
# aliases are registered at all times. Only the active language is visible in
# the native sidebar; the counterpart aliases stay routable but hidden.
pages_by_language: dict[str, dict[str, object]] = {
    lang: {} for lang in SUPPORTED_LANGUAGES
}

for lang in SUPPORTED_LANGUAGES:
    visibility = "visible" if lang == language else "hidden"
    for page_key in PAGE_SPECS:
        pages_by_language[lang][page_key] = st.Page(
            page_source(page_key),
            url_path=page_slug(page_key, lang),
            title=page_title(page_key, lang),
            icon=page_icon(page_key),
            visibility=visibility,
        )


def _root_redirect() -> None:
    """Canonicalize the app root to the localized Home pathname."""
    st.switch_page(
        pages_by_language[language]["home"],
        query_params={"lang": language},
    )


root_redirect_page = st.Page(
    _root_redirect,
    title="Localized Home",
    default=True,
    visibility="hidden",
)


def _localized_pair(page_key: str) -> list[object]:
    """Return active route first and the alternate-language alias second."""
    alternate_language = "en" if language == "tr" else "tr"
    return [
        pages_by_language[language][page_key],
        pages_by_language[alternate_language][page_key],
    ]


def _section_pages(page_keys: tuple[str, ...], include_root: bool = False) -> list[object]:
    pages: list[object] = [root_redirect_page] if include_root else []
    for page_key in page_keys:
        pages.extend(_localized_pair(page_key))
    return pages


# ============================================================
# SINGLE ROUTER
# ============================================================

navigation = st.navigation(
    {
        current_sections["overview"]: _section_pages(
            OVERVIEW_PAGE_KEYS,
            include_root=True,
        ),
        current_sections["intelligence"]: _section_pages(
            INTELLIGENCE_PAGE_KEYS,
        ),
        current_sections["advanced"]: _section_pages(
            ADVANCED_PAGE_KEYS,
        ),
    },
    position="sidebar",
)


# ============================================================
# CANONICAL LOCALIZED URL
# ============================================================

# Root is intentionally a hidden redirect so Home itself can have a localized
# pathname: /ana-panel?lang=tr or /home?lang=en.
if navigation.url_path == "":
    st.switch_page(
        pages_by_language[language]["home"],
        query_params={"lang": language},
    )

# If the language selector changed while the user stayed on the same logical
# page, st.navigation may initially resolve the old-language alias. Redirect
# to the counterpart alias before executing the page so pathname + ?lang stay
# in sync (e.g. /teknik-seo?lang=tr <-> /technical-seo?lang=en).
page_key = page_key_from_slug(navigation.url_path)
if page_key is not None:
    canonical_page = pages_by_language[language][page_key]
    if navigation.url_path != canonical_page.url_path:
        st.switch_page(
            canonical_page,
            query_params={"lang": language},
        )

navigation.run()
