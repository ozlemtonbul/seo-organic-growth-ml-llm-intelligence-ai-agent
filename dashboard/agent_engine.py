from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from dashboard.i18n import (
    normalize_language,
)
from dashboard.utils import (
    safe_float,
    safe_int,
)
from src.llm.manager import (
    generate_text,
    get_llm_runtime_info,
)


# ============================================================
# AGENT RESPONSE
# ============================================================


@dataclass(frozen=True)
class AgentResponse:
    """
    Standard response returned by the dashboard AI agent.
    """

    answer: str
    source: str
    provider: str
    model: str
    ready: bool


# ============================================================
# QUESTION NORMALIZATION
# ============================================================


def normalize_question(
    question: str | None,
) -> str:
    """
    Normalize a user question.
    """
    if question is None:
        return ""

    return (
        str(question)
        .strip()
    )


# ============================================================
# DATAFRAME HELPERS
# ============================================================


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Return the first matching DataFrame column.
    """
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def _sum_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> float:
    """
    Sum the first matching numeric column.
    """
    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return 0.0

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(0)

    return float(
        values.sum()
    )


def _mean_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> float:
    """
    Average the first matching numeric column.
    """
    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return 0.0

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.mean()
    )


def _mode_value(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str:
    """
    Return the most common value from the first matching column.
    """
    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return "N/A"

    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    if values.empty:
        return "N/A"

    mode = values.mode()

    if mode.empty:
        return "N/A"

    return str(
        mode.iloc[0]
    )


# ============================================================
# SEO CONTEXT
# ============================================================


def build_agent_context(
    dataframe: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
    model_metrics: pd.DataFrame | None = None,
) -> dict[str, object]:
    """
    Build a compact SEO intelligence context for the agent.
    """
    data = (
        dataframe.copy()
        if dataframe is not None
        else pd.DataFrame()
    )

    recs = (
        recommendations.copy()
        if recommendations is not None
        else pd.DataFrame()
    )

    metrics = (
        model_metrics.copy()
        if model_metrics is not None
        else pd.DataFrame()
    )

    total_clicks = _sum_column(
        data,
        [
            "clicks",
            "Clicks",
            "CurrentClicks",
        ],
    )

    total_impressions = _sum_column(
        data,
        [
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ],
    )

    average_position = _mean_column(
        data,
        [
            "position",
            "Position",
            "CurrentPosition",
        ],
    )

    sessions = _sum_column(
        data,
        [
            "sessions",
            "Sessions",
        ],
    )

    conversions = _sum_column(
        data,
        [
            "conversions",
            "Conversions",
            "purchases",
            "Purchases",
        ],
    )

    revenue = _sum_column(
        data,
        [
            "revenue",
            "Revenue",
        ],
    )

    ctr = (
        total_clicks
        / total_impressions
        if total_impressions > 0
        else 0.0
    )

    conversion_rate = (
        conversions
        / sessions
        if sessions > 0
        else 0.0
    )

    total_pages = 0

    page_column = _first_existing_column(
        data,
        [
            "page",
            "Page",
        ],
    )

    if page_column is not None:
        total_pages = int(
            data[
                page_column
            ]
            .dropna()
            .nunique()
        )

    recommendation_count = int(
        len(
            recs
        )
    )

    high_priority_count = 0

    priority_column = _first_existing_column(
        recs,
        [
            "PriorityTier",
            "priority_tier",
        ],
    )

    if priority_column is not None:
        high_priority_count = int(
            (
                recs[
                    priority_column
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "high priority"
                )
            ).sum()
        )

    expected_net_value = _sum_column(
        recs,
        [
            "ExpectedNetValue",
            "expected_net_value",
        ],
    )

    expected_incremental_value = _sum_column(
        recs,
        [
            "ExpectedIncrementalTrafficValue",
            "expected_incremental_traffic_value",
        ],
    )

    average_roi = _mean_column(
        recs,
        [
            "EstimatedROI",
            "estimated_roi",
        ],
    )

    top_action = _mode_value(
        recs,
        [
            "RecommendedAction",
            "recommended_action",
        ],
    )

    top_intent = _mode_value(
        data,
        [
            "keyword_intent",
            "KeywordIntent",
        ],
    )

    average_model_r2 = _mean_column(
        metrics,
        [
            "R2",
            "r2",
        ],
    )

    return {
        "total_rows": int(
            len(
                data
            )
        ),
        "total_pages": total_pages,
        "total_clicks": total_clicks,
        "total_impressions": total_impressions,
        "ctr": ctr,
        "average_position": average_position,
        "sessions": sessions,
        "conversions": conversions,
        "conversion_rate": conversion_rate,
        "revenue": revenue,
        "recommendation_count": recommendation_count,
        "high_priority_count": high_priority_count,
        "expected_net_value": expected_net_value,
        "expected_incremental_value": (
            expected_incremental_value
        ),
        "average_roi": average_roi,
        "top_action": top_action,
        "top_intent": top_intent,
        "average_model_r2": average_model_r2,
    }


# ============================================================
# PROMPT
# ============================================================


def build_agent_prompt(
    question: str,
    context: dict[str, object],
    language: str = "tr",
) -> str:
    """
    Build a concise prompt using only provided SEO context.
    """
    resolved_language = normalize_language(
        language
    )

    language_instruction = (
        "Yanıtı Türkçe ver."
        if resolved_language == "tr"
        else "Respond in English."
    )

    return f"""
You are an SEO, GEO and organic growth decision intelligence assistant.

Use only the supplied dashboard context.
Do not invent unavailable metrics.
Clearly distinguish observed performance from forecasts or recommendations.
Keep the response concise, actionable and suitable for a business user.

{language_instruction}

Dashboard context:

Total data rows: {safe_int(context.get("total_rows"))}
Total pages: {safe_int(context.get("total_pages"))}
Organic clicks: {safe_float(context.get("total_clicks")):.0f}
Organic impressions: {safe_float(context.get("total_impressions")):.0f}
CTR: {safe_float(context.get("ctr")):.4f}
Average organic position: {safe_float(context.get("average_position")):.2f}
Sessions: {safe_float(context.get("sessions")):.0f}
Conversions: {safe_float(context.get("conversions")):.0f}
Conversion rate: {safe_float(context.get("conversion_rate")):.4f}
Revenue: {safe_float(context.get("revenue")):.2f}

Recommendation count: {safe_int(context.get("recommendation_count"))}
High-priority opportunities: {safe_int(context.get("high_priority_count"))}
Expected incremental traffic value: {safe_float(context.get("expected_incremental_value")):.2f}
Expected net value: {safe_float(context.get("expected_net_value")):.2f}
Average estimated ROI: {safe_float(context.get("average_roi")):.2f}
Most common recommended action: {context.get("top_action", "N/A")}
Dominant keyword intent: {context.get("top_intent", "N/A")}
Average model R2: {safe_float(context.get("average_model_r2")):.3f}

User question:
{question}
""".strip()


# ============================================================
# DETERMINISTIC ANSWER
# ============================================================


def generate_deterministic_answer(
    question: str,
    context: dict[str, object],
    language: str = "tr",
) -> str:
    """
    Generate a deterministic dashboard answer without an LLM.
    """
    resolved_language = normalize_language(
        language
    )

    normalized_question = (
        question
        .lower()
        .strip()
    )

    clicks = safe_float(
        context.get(
            "total_clicks"
        )
    )

    impressions = safe_float(
        context.get(
            "total_impressions"
        )
    )

    ctr = safe_float(
        context.get(
            "ctr"
        )
    )

    position = safe_float(
        context.get(
            "average_position"
        )
    )

    sessions = safe_float(
        context.get(
            "sessions"
        )
    )

    conversions = safe_float(
        context.get(
            "conversions"
        )
    )

    revenue = safe_float(
        context.get(
            "revenue"
        )
    )

    recommendation_count = safe_int(
        context.get(
            "recommendation_count"
        )
    )

    high_priority_count = safe_int(
        context.get(
            "high_priority_count"
        )
    )

    expected_value = safe_float(
        context.get(
            "expected_incremental_value"
        )
    )

    expected_net_value = safe_float(
        context.get(
            "expected_net_value"
        )
    )

    average_roi = safe_float(
        context.get(
            "average_roi"
        )
    )

    top_action = str(
        context.get(
            "top_action",
            "N/A",
        )
    )

    top_intent = str(
        context.get(
            "top_intent",
            "N/A",
        )
    )

    if resolved_language == "tr":
        if any(
            keyword in normalized_question
            for keyword in [
                "fırsat",
                "öner",
                "aksiyon",
                "ne yap",
                "optimiz",
            ]
        ):
            return (
                f"Sistemde {recommendation_count} SEO önerisi bulunuyor; "
                f"{high_priority_count} tanesi yüksek öncelikli. "
                f"En sık önerilen aksiyon: {top_action}. "
                f"Senaryoların tahmini ek trafik değeri "
                f"{expected_value:,.2f}, beklenen net değeri "
                f"{expected_net_value:,.2f} ve ortalama tahmini ROI "
                f"{average_roi:.2f}. Öncelikle yüksek güvenli ve yüksek "
                f"öncelikli sayfalardan başlanması önerilir."
            )

        if any(
            keyword in normalized_question
            for keyword in [
                "ctr",
                "tıklama",
                "gösterim",
                "pozisyon",
                "organik",
                "seo perform",
            ]
        ):
            return (
                f"Mevcut organik görünümde {clicks:,.0f} tıklama ve "
                f"{impressions:,.0f} gösterim bulunuyor. CTR %{ctr * 100:.2f}, "
                f"ortalama pozisyon {position:.2f}. Baskın arama niyeti "
                f"{top_intent}. CTR ve pozisyon birlikte değerlendirilerek "
                f"yüksek gösterim alan fakat düşük tıklama üreten sayfalar "
                f"önceliklendirilebilir."
            )

        if any(
            keyword in normalized_question
            for keyword in [
                "gelir",
                "dönüşüm",
                "conversion",
                "revenue",
                "satış",
            ]
        ):
            conversion_rate = (
                conversions
                / sessions
                if sessions > 0
                else 0.0
            )

            return (
                f"GA4 tarafında {sessions:,.0f} oturum, "
                f"{conversions:,.0f} dönüşüm ve {revenue:,.2f} gelir "
                f"görülüyor. Dönüşüm oranı %{conversion_rate * 100:.2f}. "
                f"SEO kararlarında yalnızca trafik hacmine değil, gelir ve "
                f"dönüşüm katkısı yüksek sayfalara da öncelik verilmelidir."
            )

        return (
            f"SEO görünümünde {clicks:,.0f} organik tıklama, "
            f"{impressions:,.0f} gösterim ve %{ctr * 100:.2f} CTR bulunuyor. "
            f"Ortalama pozisyon {position:.2f}. Sistemde "
            f"{recommendation_count} öneri ve {high_priority_count} yüksek "
            f"öncelikli fırsat mevcut. En sık önerilen aksiyon "
            f"{top_action}. Daha spesifik olarak performans, fırsatlar, "
            f"dönüşüm veya belirli bir SEO metriği hakkında sorabilirsin."
        )

    if any(
        keyword in normalized_question
        for keyword in [
            "opportunity",
            "recommend",
            "action",
            "optimiz",
            "what should",
        ]
    ):
        return (
            f"The system contains {recommendation_count} SEO recommendations, "
            f"including {high_priority_count} high-priority opportunities. "
            f"The most common recommended action is {top_action}. "
            f"Scenario outputs indicate {expected_value:,.2f} in expected "
            f"incremental traffic value, {expected_net_value:,.2f} in expected "
            f"net value and an average estimated ROI of {average_roi:.2f}. "
            f"Start with high-confidence, high-priority pages."
        )

    if any(
        keyword in normalized_question
        for keyword in [
            "ctr",
            "click",
            "impression",
            "position",
            "organic",
            "seo performance",
        ]
    ):
        return (
            f"Current organic performance includes {clicks:,.0f} clicks and "
            f"{impressions:,.0f} impressions. CTR is {ctr * 100:.2f}% and "
            f"average position is {position:.2f}. The dominant search intent "
            f"is {top_intent}. Pages with high impressions but relatively low "
            f"CTR can be prioritized for title, meta and content improvements."
        )

    if any(
        keyword in normalized_question
        for keyword in [
            "revenue",
            "conversion",
            "sales",
        ]
    ):
        conversion_rate = (
            conversions
            / sessions
            if sessions > 0
            else 0.0
        )

        return (
            f"GA4 reports {sessions:,.0f} sessions, "
            f"{conversions:,.0f} conversions and {revenue:,.2f} in revenue. "
            f"The conversion rate is {conversion_rate * 100:.2f}%. "
            f"SEO prioritization should consider both organic visibility "
            f"and commercial contribution."
        )

    return (
        f"The SEO view contains {clicks:,.0f} organic clicks, "
        f"{impressions:,.0f} impressions and a CTR of {ctr * 100:.2f}%. "
        f"Average position is {position:.2f}. There are "
        f"{recommendation_count} recommendations, including "
        f"{high_priority_count} high-priority opportunities. "
        f"The most common recommended action is {top_action}. "
        f"You can ask about performance, opportunities, conversions "
        f"or a specific SEO metric."
    )


# ============================================================
# AGENT EXECUTION
# ============================================================


def ask_agent(
    question: str,
    dataframe: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
    model_metrics: pd.DataFrame | None = None,
    language: str = "tr",
    max_tokens: int = 500,
) -> AgentResponse:
    """
    Answer an SEO dashboard question.

    Uses the shared multi-provider LLM manager when available.
    Falls back to deterministic logic when LLM generation is
    disabled, unavailable, limited or unsuccessful.
    """
    resolved_question = normalize_question(
        question
    )

    resolved_language = normalize_language(
        language
    )

    runtime_info = (
        get_llm_runtime_info()
    )

    ready = bool(
        runtime_info.get(
            "ready",
            False,
        )
    )

    provider = str(
        runtime_info.get(
            "provider",
            "",
        )
        or ""
    )

    model = str(
        runtime_info.get(
            "model",
            "",
        )
        or ""
    )

    if not resolved_question:
        empty_answer = (
            "Lütfen bir soru yazın."
            if resolved_language == "tr"
            else "Please enter a question."
        )

        return AgentResponse(
            answer=empty_answer,
            source="validation",
            provider=provider,
            model=model,
            ready=ready,
        )

    context = build_agent_context(
        dataframe=dataframe,
        recommendations=recommendations,
        model_metrics=model_metrics,
    )

    prompt = build_agent_prompt(
        question=resolved_question,
        context=context,
        language=resolved_language,
    )

    generated_answer = generate_text(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    if generated_answer:
        return AgentResponse(
            answer=generated_answer,
            source="llm",
            provider=provider,
            model=model,
            ready=ready,
        )

    deterministic_answer = (
        generate_deterministic_answer(
            question=resolved_question,
            context=context,
            language=resolved_language,
        )
    )

    return AgentResponse(
        answer=deterministic_answer,
        source="deterministic",
        provider=provider,
        model=model,
        ready=ready,
    )