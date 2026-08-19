from __future__ import annotations

from urllib.parse import urlparse

import streamlit as st

from dashboard.routes import language_from_slug


VALID_LANGUAGES = {"tr", "en"}
DEFAULT_LANGUAGE = "tr"
LANG_QUERY_KEY = "lang"
PENDING_LANGUAGE_KEY = "_dashboard_language_requested"


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
        return None


def _pathname_language() -> str | None:
    """
    Infer language directly from the localized pathname.

    This keeps /ana-panel, /teknik-seo, /home and /technical-seo stable
    even when Streamlit clears query parameters during multipage navigation.
    """
    try:
        url = str(st.context.url)
        path = urlparse(url).path.rstrip("/")
        if not path:
            return None

        slug = path.rsplit("/", 1)[-1]
        return normalize_language(
            language_from_slug(slug),
            None,
        )
    except Exception:
        # Bare page tests may not expose a browser URL.
        return None


def _write_query_language(language: str) -> None:
    """Mirror the active language to ?lang= without owning widget state."""
    try:
        current = normalize_language(
            st.query_params.get(LANG_QUERY_KEY),
            None,
        )
        if current != language:
            st.query_params[LANG_QUERY_KEY] = language
    except Exception:
        # Query params are an enhancement; pathname + session remain valid.
        pass


def choose_language(
    *,
    requested_language: object = None,
    query_language: object = None,
    pathname_language: object = None,
    session_language: object = None,
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Pure language precedence rule.

    A just-selected widget value must win over the old URL for one rerun;
    otherwise the selector can visually snap back to the previous language.
    """
    normalized_default = (
        normalize_language(default, DEFAULT_LANGUAGE)
        or DEFAULT_LANGUAGE
    )

    for candidate in (
        requested_language,
        query_language,
        pathname_language,
        session_language,
    ):
        normalized = normalize_language(candidate, None)
        if normalized is not None:
            return normalized

    return normalized_default


def resolve_language_from_url(
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Resolve active language before the language widget is instantiated.

    Precedence:
        fresh widget intent -> ?lang -> localized pathname -> session -> default.

    The pending intent marker is consumed once. It prevents an old pathname
    such as /ana-panel from overriding a user's new English selection during
    the rerun that redirects to /home.
    """
    try:
        requested_language = st.session_state.pop(
            PENDING_LANGUAGE_KEY,
            None,
        )
    except Exception:
        requested_language = None

    language = choose_language(
        requested_language=requested_language,
        query_language=_query_language(),
        pathname_language=_pathname_language(),
        session_language=st.session_state.get(
            "dashboard_language"
        ),
        default=default,
    )

    # Safe here: app.py calls this before the keyed selectbox exists.
    if st.session_state.get("dashboard_language") != language:
        st.session_state["dashboard_language"] = language

    _write_query_language(language)
    return language


def sync_language_to_url(language: object) -> str:
    """
    Synchronize active language to the URL only.

    Do not write dashboard_language here because the selectbox already owns
    that Session State key during page execution.
    """
    normalized = (
        normalize_language(language, DEFAULT_LANGUAGE)
        or DEFAULT_LANGUAGE
    )
    _write_query_language(normalized)
    return normalized


def sync_language_widget_to_url() -> None:
    """
    Callback for the TR/EN selectbox.

    Streamlit has already stored the newly selected value in
    session_state["dashboard_language"]. We record that value as one-run
    navigation intent *before* touching the URL. On the next entrypoint rerun,
    this intent wins over the previous pathname/query and the router switches
    to the matching localized slug.
    """
    selected = (
        normalize_language(
            st.session_state.get("dashboard_language"),
            DEFAULT_LANGUAGE,
        )
        or DEFAULT_LANGUAGE
    )

    st.session_state[PENDING_LANGUAGE_KEY] = selected
    _write_query_language(selected)
