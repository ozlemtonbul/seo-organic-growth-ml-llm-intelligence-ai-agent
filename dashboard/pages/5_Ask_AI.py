from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROJECT_ROOT_STR = str(
    PROJECT_ROOT
)

if PROJECT_ROOT_STR in sys.path:

    sys.path.remove(
        PROJECT_ROOT_STR
    )

sys.path.insert(
    0,
    PROJECT_ROOT_STR,
)


# ============================================================
# IMPORTS
# ============================================================

from dashboard.agent_engine import (
    ask_agent,
)

from dashboard.app_config import (
    APP_TITLE,
)

from dashboard.i18n import (
    t,
)

from dashboard.layout import (
    initialize_dashboard,
    localized_text,
    render_deterministic_notice,
    render_divider,
    render_footer,
    render_section_header,
)

from dashboard.services import (
    fast_filter_period,
    get_available_date_bounds,
    load_analysis_data,
)

from src.llm.manager import (
    get_llm_runtime_info,
)


# ============================================================
# LOAD DATA
# ============================================================

data = load_analysis_data()


# ============================================================
# PAGE INITIALIZATION
# ============================================================

available_start, available_end = (
    get_available_date_bounds(
        data.integrated
    )
)

initial_language = (
    st.session_state.get(
        "dashboard_language",
        "tr",
    )
)


context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{t('ask_ai', initial_language)}"
    ),

    page_icon="💬",

    title=(
        "AI Asistan"
        if initial_language == "tr"
        else "AI Assistant"
    ),

    subtitle=(
        "SEO, GEO, Search Console, GA4, model çıktıları "
        "ve optimizasyon önerileri hakkında doğal dilde "
        "soru sorun."
        if initial_language == "tr"
        else
        "Ask natural-language questions about SEO, GEO, "
        "Search Console, GA4, model outputs, and "
        "optimization recommendations."
    ),

    eyebrow=(
        "SEO & GEO Karar Zekâsı"
        if initial_language == "tr"
        else "SEO & GEO Decision Intelligence"
    ),

    default_preset=(
        "last_30_days"
    ),

    default_comparison=(
        "previous_period"
    ),

    reference_date=(
        available_end
    ),
)


# ============================================================
# ACTIVE CONTEXT
# ============================================================

language = context.language

filters = context.filters


# ============================================================
# FILTER CURRENT DATA
# ============================================================

period_integrated = (
    fast_filter_period(
        data.integrated,
        filters.start_date,
        filters.end_date,
        candidates=(
            "date",
            "Date",
            "AnalysisDate",
            "ObservationDate",
        ),
    )
)


# Recommendations are a latest successful model snapshot.
# The historical date filter applies to realized performance data.
period_recommendations = (
    data.recommendations.copy()
)

st.caption(
    (
        "AI Asistan seçilen tarih aralığındaki gerçekleşmiş GSC + GA4 "
        "performansını ve son başarılı model çalışmasının güncel önerilerini "
        "birlikte kullanır."
        if language == "tr"
        else
        "The AI Assistant combines realized GSC + GA4 performance from the "
        "selected period with the latest successful model-run recommendations."
    )
)


# ============================================================
# RUNTIME STATUS
# ============================================================

runtime_info = (
    get_llm_runtime_info()
)


llm_ready = bool(
    runtime_info.get(
        "ready",
        False,
    )
)


if not llm_ready:

    render_deterministic_notice(
        language
    )


# ============================================================
# CHAT STATE
# ============================================================

if (
    "seo_chat_history"
    not in st.session_state
):

    st.session_state[
        "seo_chat_history"
    ] = []


# ============================================================
# SAMPLE QUESTIONS
# ============================================================

render_section_header(
    (
        "Örnek Sorular"
        if language == "tr"
        else "Example Questions"
    )
)


example_columns = st.columns(
    3
)


examples = (

    [
        "SEO performansı nasıl?",
        "En önemli SEO fırsatları neler?",
        "Dönüşüm ve gelir tarafında durum nasıl?",
    ]

    if language == "tr"

    else

    [
        "How is SEO performance?",
        "What are the most important SEO opportunities?",
        "How are conversions and revenue performing?",
    ]
)


for (
    index,
    example,
) in enumerate(examples):

    with example_columns[index]:

        if st.button(
            example,
            width="stretch",
            key=(
                f"seo_example_question_"
                f"{index}"
            ),
        ):

            st.session_state[
                "seo_prefilled_question"
            ] = example


# ============================================================
# CHAT HISTORY
# ============================================================

render_divider()


render_section_header(
    t(
        "ai_assistant",
        language,
    )
)


for message in (
    st.session_state[
        "seo_chat_history"
    ]
):

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )


# ============================================================
# QUESTION INPUT
# ============================================================

prefilled_question = (
    st.session_state.pop(
        "seo_prefilled_question",
        None,
    )
)


question = st.chat_input(
    t(
        "ask_question",
        language,
    )
)


if (
    prefilled_question
    and not question
):

    question = (
        prefilled_question
    )


# ============================================================
# AGENT EXECUTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state[
        "seo_chat_history"
    ].append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Execute agent
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            (
                "SEO verileri analiz ediliyor..."
                if language == "tr"
                else
                "Analyzing SEO data..."
            )
        ):

            response = ask_agent(

                question=question,

                dataframe=(
                    period_integrated
                ),

                recommendations=(
                    period_recommendations
                ),

                model_metrics=(
                    data.model_metrics
                ),

                language=language,

                max_tokens=500,
            )


        st.markdown(
            response.answer
        )


        # ----------------------------------------------------
        # Source label
        # ----------------------------------------------------

        source_label = (

            "LLM"

            if response.source == "llm"

            else

            (
                "Deterministik"
                if language == "tr"
                else "Deterministic"
            )
        )


        # ----------------------------------------------------
        # Runtime caption
        # ----------------------------------------------------

        runtime_caption = (

            f"Kaynak: {source_label}"

            if language == "tr"

            else

            f"Source: {source_label}"
        )


        if response.provider:

            runtime_caption += (
                f" · "
                f"{response.provider}"
            )


        if response.model:

            runtime_caption += (
                f" · "
                f"{response.model}"
            )


        st.caption(
            runtime_caption
        )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state[
        "seo_chat_history"
    ].append(
        {
            "role": "assistant",
            "content": response.answer,
            "source": response.source,
        }
    )


    st.rerun()


# ============================================================
# CLEAR CHAT
# ============================================================

render_divider()


clear_column, info_column = (
    st.columns(
        [
            1,
            3,
        ]
    )
)


with clear_column:

    if st.button(
        t(
            "clear_chat",
            language,
        ),
        width="stretch",
    ):

        st.session_state[
            "seo_chat_history"
        ] = []

        st.rerun()


with info_column:

    st.caption(
        (
            "AI yanıtları yalnızca mevcut SEO/GSC/GA4 "
            "ve model çıktılarına dayanır. Eksik veri "
            "uydurulmaz."

            if language == "tr"

            else

            "AI responses use only the available "
            "SEO/GSC/GA4 and model outputs. "
            "Missing data is not invented."
        )
    )


# ============================================================
# DATA CONTEXT
# ============================================================

with st.expander(
    (
        "AI Veri Kapsamını Gör"
        if language == "tr"
        else "View AI Data Context"
    )
):

    context_columns = (
        st.columns(4)
    )


    # --------------------------------------------------------
    # Integrated rows
    # --------------------------------------------------------

    context_columns[0].metric(
        (
            "Entegre Veri Satırı"
            if language == "tr"
            else "Integrated Rows"
        ),
        len(
            data.integrated
        ),
    )


    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    context_columns[1].metric(
        (
            "Öneri Sayısı"
            if language == "tr"
            else "Recommendations"
        ),
        len(
            data.recommendations
        ),
    )


    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    context_columns[2].metric(
        (
            "Model Metrik Satırı"
            if language == "tr"
            else "Model Metric Rows"
        ),
        len(
            data.model_metrics
        ),
    )


    # --------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------

    context_columns[3].metric(
        (
            "Senaryo Sayısı"
            if language == "tr"
            else "Scenarios"
        ),
        len(
            data.scenarios
        ),
    )


    # --------------------------------------------------------
    # AI runtime status — user-friendly view
    # --------------------------------------------------------

    st.markdown(
        (
            "**AI Çalışma Durumu**"
            if language == "tr"
            else "**AI Runtime Status**"
        )
    )

    runtime_columns = st.columns(5)

    enabled = bool(runtime_info.get("enabled", False))
    ready = bool(runtime_info.get("ready", False))
    provider = str(runtime_info.get("provider", "") or "").strip()
    model = str(runtime_info.get("model", "") or "").strip()
    daily_requests = int(runtime_info.get("daily_requests", 0) or 0)
    daily_limit = int(runtime_info.get("daily_limit", 0) or 0)

    runtime_columns[0].metric(
        "AI Durumu" if language == "tr" else "AI Status",
        (
            "Aktif" if enabled and language == "tr"
            else "Active" if enabled
            else "Pasif" if language == "tr"
            else "Inactive"
        ),
    )

    runtime_columns[1].metric(
        "Hazır" if language == "tr" else "Ready",
        (
            "Evet" if ready and language == "tr"
            else "Yes" if ready
            else "Hayır" if language == "tr"
            else "No"
        ),
    )

    runtime_columns[2].metric(
        "Sağlayıcı" if language == "tr" else "Provider",
        provider or (
            "Yapılandırılmadı"
            if language == "tr"
            else "Not configured"
        ),
    )

    runtime_columns[3].metric(
        localized_text(
            language,
            "Model",
            "Model",
        ),
        model or (
            "Yapılandırılmadı"
            if language == "tr"
            else "Not configured"
        ),
    )

    runtime_columns[4].metric(
        "Günlük Kullanım" if language == "tr" else "Daily Usage",
        f"{daily_requests} / {daily_limit}" if daily_limit else str(daily_requests),
    )

    st.caption(
        (
            "Ham sistem/debug JSON'u son kullanıcı arayüzünde gösterilmez."
            if language == "tr"
            else
            "Raw system/debug JSON is not displayed in the user interface."
        )
    )


# ============================================================
# FOOTER
# ============================================================

render_footer(
    language=language,
    demo_mode=False,
)