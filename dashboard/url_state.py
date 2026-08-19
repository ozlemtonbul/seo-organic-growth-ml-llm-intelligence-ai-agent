from __future__ import annotations

import streamlit as st


VALID_LANGUAGES = {"tr", "en"}
DEFAULT_LANGUAGE = "tr"
LANG_QUERY_KEY = "lang"


def normalize_language(value: object, default: str | None = None) -> str | None:
    """Normalize a language value to a supported URL/session code."""
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in VALID_LANGUAGES:
        return normalized

    return default


def _query_language() -> str | None:
    """Read a supported language from the current URL query string."""
    try:
        return normalize_language(
            st.query_params.get(LANG_QUERY_KEY),
            None,
        )
    except Exception:
        # Keeps direct page tests/bare execution resilient.
        return None


def _write_query_language(language: str) -> None:
    """Keep the shareable URL synchronized with the active language."""
    try:
        current = normalize_language(
            st.query_params.get(LANG_QUERY_KEY),
            None,
        )
        if current != language:
            st.query_params[LANG_QUERY_KEY] = language
    except Exception:
        # Query parameters are a URL enhancement; session state remains
        # the continuity source if a test/runtime cannot expose them.
        pass


def resolve_language_from_url(
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Resolve active language before language widgets are instantiated.

    Precedence:
        URL query parameter -> session state -> default.

    This function is intentionally used from dashboard/app.py before
    navigation.run(), so assigning dashboard_language is safe here.
    """
    default_language = (
        normalize_language(default, DEFAULT_LANGUAGE)
        or DEFAULT_LANGUAGE
    )

    url_language = _query_language()
    session_language = normalize_language(
        st.session_state.get("dashboard_language"),
        None,
    )

    language = url_language or session_language or default_language

    # Safe because app.py resolves URL state before the keyed language
    # widget is instantiated during the current Streamlit run.
    if st.session_state.get("dashboard_language") != language:
        st.session_state["dashboard_language"] = language

    _write_query_language(language)
    return language


def sync_language_to_url(language: object) -> str:
    """
    Synchronize the active language to the URL only.

    IMPORTANT: do not assign st.session_state['dashboard_language'] here.
    This helper is called after the selectbox with key='dashboard_language'
    has been instantiated, and Streamlit forbids mutating that widget key
    later in the same run.
    """
    normalized = (
        normalize_language(language, DEFAULT_LANGUAGE)
        or DEFAULT_LANGUAGE
    )

    _write_query_language(normalized)
    return normalized


def sync_language_widget_to_url() -> None:
    """
    Streamlit widget callback.

    The widget itself has already written the selected value to
    st.session_state['dashboard_language']. The callback only mirrors that
    value to ?lang=tr/en; it does not rewrite the widget-owned state key.
    """
    selected = normalize_language(
        st.session_state.get("dashboard_language"),
        DEFAULT_LANGUAGE,
    ) or DEFAULT_LANGUAGE

    _write_query_language(selected)
