from __future__ import annotations

from typing import List

import pandas as pd

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.llm.manager import (
    generate_text,
    get_llm_runtime_info,
)


logger = get_logger(__name__)


# ============================================================
# HELPERS
# ============================================================


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Convert a value safely to float.
    """
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def normalize_language(
    language: str | None,
) -> str:
    """
    Resolve commentary language.

    Supported:
    - tr
    - en
    """
    resolved = (
        language
        or SETTINGS.llm_language
        or "tr"
    )

    resolved = (
        str(resolved)
        .strip()
        .lower()
    )

    if resolved not in {
        "tr",
        "en",
    }:
        return "tr"

    return resolved


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================


def build_deterministic_page_commentary(
    row: pd.Series,
    language: str = "tr",
) -> str:
    """
    Build page-level commentary without an external LLM.

    This fallback keeps the pipeline useful when:
    - LLM is disabled,
    - API key is missing,
    - daily usage limit is reached,
    - provider request fails.
    """
    language = normalize_language(
        language
    )

    page_type = str(
        row.get(
            "page_type",
            "page",
        )
        or "page"
    )

    scenario_label = str(
        row.get(
            "ScenarioLabel",
            "SEO optimization",
        )
        or "SEO optimization"
    )

    recommended_action = str(
        row.get(
            "RecommendedAction",
            scenario_label,
        )
        or scenario_label
    )

    priority = str(
        row.get(
            "PriorityTier",
            "",
        )
        or ""
    )

    confidence = str(
        row.get(
            "ConfidenceLevel",
            "",
        )
        or ""
    )

    current_clicks = safe_float(
        row.get(
            "CurrentClicks",
            0,
        )
    )

    click_uplift = safe_float(
        row.get(
            "ClicksUplift",
            0,
        )
    )

    click_uplift_pct = safe_float(
        row.get(
            "ClicksUpliftPct",
            0,
        )
    )

    position_gain = safe_float(
        row.get(
            "EstimatedPositionGain",
            0,
        )
    )

    incremental_value = safe_float(
        row.get(
            "ExpectedIncrementalTrafficValue",
            0,
        )
    )

    roi = safe_float(
        row.get(
            "EstimatedROI",
            0,
        )
    )

    current_geo = safe_float(
        row.get(
            "CurrentGeoReadinessScore",
            0,
        )
    )

    scenario_geo = safe_float(
        row.get(
            "ScenarioGeoReadinessScore",
            current_geo,
        )
    )

    geo_gain = max(
        0.0,
        scenario_geo
        - current_geo,
    )

    if language == "tr":
        parts: List[str] = []

        parts.append(
            f"Bu {page_type} sayfası mevcut durumda "
            f"{current_clicks:.0f} organik tıklama üretmektedir."
        )

        parts.append(
            f"Önerilen aksiyon {recommended_action}; "
            f"model senaryosu {scenario_label} olarak belirlenmiştir."
        )

        impact_parts: List[str] = []

        if click_uplift != 0:
            impact_parts.append(
                f"{click_uplift:.1f} ek tıklama"
            )

        if click_uplift_pct != 0:
            impact_parts.append(
                f"%{click_uplift_pct:.1f} tıklama değişimi"
            )

        if position_gain > 0:
            impact_parts.append(
                f"{position_gain:.1f} sıra pozisyonu iyileşmesi"
            )

        if incremental_value != 0:
            impact_parts.append(
                f"{incremental_value:.2f} tahmini ek trafik değeri"
            )

        if impact_parts:
            parts.append(
                "Beklenen etki: "
                + ", ".join(
                    impact_parts
                )
                + "."
            )

        focus_parts: List[str] = []

        if geo_gain > 0:
            focus_parts.append(
                f"GEO hazırlığında +{geo_gain:.0f} puan"
            )

        if priority:
            focus_parts.append(
                f"öncelik seviyesi {priority}"
            )

        if confidence:
            focus_parts.append(
                f"model güveni {confidence}"
            )

        if roi != 0:
            focus_parts.append(
                f"tahmini ROI {roi:.2f}"
            )

        if focus_parts:
            parts.append(
                "Uygulama odağı: "
                + ", ".join(
                    focus_parts
                )
                + "."
            )

        return " ".join(
            parts[:4]
        )

    parts = [
        (
            f"This {page_type} page currently generates "
            f"{current_clicks:.0f} organic clicks."
        ),
        (
            f"The recommended action is {recommended_action}, "
            f"based on the {scenario_label} scenario."
        ),
    ]

    impact_parts = []

    if click_uplift != 0:
        impact_parts.append(
            f"{click_uplift:.1f} incremental clicks"
        )

    if click_uplift_pct != 0:
        impact_parts.append(
            f"{click_uplift_pct:.1f}% click change"
        )

    if position_gain > 0:
        impact_parts.append(
            f"{position_gain:.1f} positions of ranking improvement"
        )

    if incremental_value != 0:
        impact_parts.append(
            f"{incremental_value:.2f} estimated incremental traffic value"
        )

    if impact_parts:
        parts.append(
            "Expected impact: "
            + ", ".join(
                impact_parts
            )
            + "."
        )

    focus_parts = []

    if geo_gain > 0:
        focus_parts.append(
            f"+{geo_gain:.0f} GEO-readiness points"
        )

    if priority:
        focus_parts.append(
            f"priority {priority}"
        )

    if confidence:
        focus_parts.append(
            f"model confidence {confidence}"
        )

    if roi != 0:
        focus_parts.append(
            f"estimated ROI {roi:.2f}"
        )

    if focus_parts:
        parts.append(
            "Implementation focus: "
            + ", ".join(
                focus_parts
            )
            + "."
        )

    return " ".join(
        parts[:4]
    )


def build_deterministic_portfolio_commentary(
    summary_df: pd.DataFrame,
    intent_df: pd.DataFrame,
    language: str = "tr",
) -> str:
    """
    Build portfolio-level SEO commentary without an external LLM.
    """
    language = normalize_language(
        language
    )

    if summary_df.empty:
        if language == "tr":
            return (
                "SEO portföy özeti oluşturulamadı çünkü "
                "öneri verisi bulunmuyor."
            )

        return (
            "SEO portfolio commentary could not be generated "
            "because recommendation data is empty."
        )

    total_pages = len(
        summary_df
    )

    priority_series = summary_df.get(
        "PriorityTier",
        pd.Series(
            dtype=str
        ),
    )

    high_priority_pages = int(
        (
            priority_series
            == "High Priority"
        ).sum()
    )

    total_clicks = safe_float(
        summary_df.get(
            "CurrentClicks",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    total_incremental_value = safe_float(
        summary_df.get(
            "ExpectedIncrementalTrafficValue",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    total_net_value = safe_float(
        summary_df.get(
            "ExpectedNetValue",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    roi_series = pd.to_numeric(
        summary_df.get(
            "EstimatedROI",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    ).dropna()

    average_roi = (
        float(
            roi_series.mean()
        )
        if not roi_series.empty
        else 0.0
    )

    recommended_actions = (
        summary_df.get(
            "RecommendedAction",
            pd.Series(
                dtype=str
            ),
        )
        .fillna("")
        .astype(str)
    )

    recommended_actions = (
        recommended_actions[
            recommended_actions.str.strip()
            != ""
        ]
    )

    top_action = (
        recommended_actions
        .value_counts()
        .idxmax()
        if not recommended_actions.empty
        else "N/A"
    )

    top_intent = "N/A"

    if not intent_df.empty:
        top_intent = str(
            intent_df.iloc[0].get(
                "keyword_intent",
                "N/A",
            )
            or "N/A"
        )

    if language == "tr":
        return (
            f"SEO portföyünde {total_pages} sayfa analiz edildi ve "
            f"{high_priority_pages} sayfa yüksek öncelikli olarak "
            f"sınıflandırıldı. Mevcut toplam organik tıklama "
            f"{total_clicks:.0f} seviyesinde; önerilen aksiyonların "
            f"tahmini ek trafik değeri {total_incremental_value:.2f} "
            f"ve net değeri {total_net_value:.2f}. "
            f"Portföyde en sık önerilen strateji {top_action}, "
            f"baskın arama niyeti ise {top_intent}. "
            f"Ortalama tahmini ROI {average_roi:.2f}; uygulama "
            f"önceliği yüksek değer ve yüksek öncelik grubundaki "
            f"sayfalara verilmelidir."
        )

    return (
        f"The SEO portfolio contains {total_pages} analyzed pages, "
        f"with {high_priority_pages} classified as high priority. "
        f"Current organic clicks total {total_clicks:.0f}, while the "
        f"recommended actions represent an estimated incremental traffic "
        f"value of {total_incremental_value:.2f} and net value of "
        f"{total_net_value:.2f}. The most common strategic action is "
        f"{top_action}, while the dominant search intent is {top_intent}. "
        f"Average estimated ROI is {average_roi:.2f}; implementation "
        f"should prioritize the highest-value and highest-priority pages."
    )


# ============================================================
# PAGE COMMENTARY PROMPT
# ============================================================


def build_page_commentary_prompt(
    row: pd.Series,
    language: str = "en",
) -> str:
    """
    Build an executive SEO commentary prompt for one page.
    """
    language = normalize_language(
        language
    )

    language_instruction = (
        "Respond in Turkish."
        if language == "tr"
        else "Respond in English."
    )

    return f"""
You are an experienced SEO, GEO and organic-growth director.

Write a concise executive commentary for the page below.

Requirements:
- Maximum 4 sentences.
- Explain the current situation.
- Explain the recommended action.
- Explain the expected business impact.
- Explain the main implementation focus.
- Use only information supplied below.
- Do not invent facts.
- Do not mechanically repeat raw field names.
- Clearly distinguish predicted values from observed values.
- Avoid claiming that a forecast is guaranteed.
{language_instruction}

Page:
{row.get("page", "")}

Page type:
{row.get("page_type", "")}

Keyword intent:
{row.get("keyword_intent", "")}

Current clicks:
{safe_float(row.get("CurrentClicks", 0)):.2f}

Current impressions:
{safe_float(row.get("CurrentImpressions", 0)):.2f}

Current CTR:
{safe_float(row.get("CurrentCTR", 0)):.4f}

Current average position:
{safe_float(row.get("CurrentPosition", 0)):.2f}

Current traffic value:
{safe_float(row.get("CurrentTrafficValue", 0)):.2f}

Recommended scenario:
{row.get("ScenarioLabel", "")}

Recommended action:
{row.get("RecommendedAction", "")}

Recommendation reason:
{row.get("RecommendationReason", "")}

Priority tier:
{row.get("PriorityTier", "")}

Confidence level:
{row.get("ConfidenceLevel", "")}

Predicted next clicks:
{safe_float(row.get("PredictedNextClicks", 0)):.2f}

Predicted next impressions:
{safe_float(row.get("PredictedNextImpressions", 0)):.2f}

Expected click uplift:
{safe_float(row.get("ClicksUplift", 0)):.2f}

Expected click uplift percentage:
{safe_float(row.get("ClicksUpliftPct", 0)):.2f}%

Expected incremental traffic value:
{safe_float(row.get("ExpectedIncrementalTrafficValue", 0)):.2f}

Expected net value:
{safe_float(row.get("ExpectedNetValue", 0)):.2f}

Estimated ROI:
{safe_float(row.get("EstimatedROI", 0)):.2f}

Current content score:
{safe_float(row.get("CurrentContentScore", 0)):.2f}

Scenario content score:
{safe_float(row.get("ScenarioContentScore", 0)):.2f}

Current GEO readiness score:
{safe_float(row.get("CurrentGeoReadinessScore", 0)):.2f}

Scenario GEO readiness score:
{safe_float(row.get("ScenarioGeoReadinessScore", 0)):.2f}

Executive SEO commentary:
""".strip()


# ============================================================
# PAGE COMMENTARY GENERATION
# ============================================================


def generate_page_commentaries(
    summary_df: pd.DataFrame,
    language: str | None = None,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """
    Generate executive commentary for selected SEO pages.

    LLM architecture:
    - Uses src.llm.manager.generate_text().
    - Provider selection is handled centrally.
    - Supports Claude, OpenAI and Gemini.
    - Falls back to deterministic commentary when LLM is unavailable.
    """
    result = summary_df.copy()

    if result.empty:
        result[
            "ExecutiveCommentary"
        ] = pd.Series(
            dtype=str
        )

        result[
            "CommentarySource"
        ] = pd.Series(
            dtype=str
        )

        return result

    resolved_language = normalize_language(
        language
    )

    resolved_max_pages = (
        max_pages
        if max_pages is not None
        else SETTINGS.llm_max_pages
    )

    resolved_max_pages = max(
        0,
        int(
            resolved_max_pages
        ),
    )

    rows_to_process = result.head(
        resolved_max_pages
    )

    logger.info(
        "Preparing executive commentary for %d SEO pages.",
        len(
            rows_to_process
        ),
    )

    commentaries: List[str] = []
    commentary_sources: List[str] = []

    for _, row in rows_to_process.iterrows():
        fallback_commentary = (
            build_deterministic_page_commentary(
                row=row,
                language=resolved_language,
            )
        )

        prompt = build_page_commentary_prompt(
            row=row,
            language=resolved_language,
        )

        generated_commentary = generate_text(
            prompt=prompt,
            max_tokens=min(
                400,
                getattr(SETTINGS, "llm_max_tokens", 800),
            ),
            temperature=getattr(SETTINGS, "llm_temperature", 0.2),
        )

        if generated_commentary:
            commentary = generated_commentary
            commentary_source = "llm"
        else:
            commentary = fallback_commentary
            commentary_source = "deterministic"

        commentaries.append(
            commentary
        )

        commentary_sources.append(
            commentary_source
        )

    remaining_count = (
        len(
            result
        )
        - len(
            rows_to_process
        )
    )

    if remaining_count > 0:
        remaining_rows = result.iloc[
            len(
                rows_to_process
            ):
        ]

        for _, row in remaining_rows.iterrows():
            commentaries.append(
                build_deterministic_page_commentary(
                    row=row,
                    language=resolved_language,
                )
            )

            commentary_sources.append(
                "deterministic_limit"
            )

    result[
        "ExecutiveCommentary"
    ] = commentaries

    result[
        "CommentarySource"
    ] = commentary_sources

    return result


# ============================================================
# PORTFOLIO PROMPT
# ============================================================


def build_portfolio_commentary_prompt(
    summary_df: pd.DataFrame,
    intent_df: pd.DataFrame,
    language: str = "en",
) -> str:
    """
    Build a concise portfolio-level SEO commentary prompt.
    """
    language = normalize_language(
        language
    )

    language_instruction = (
        "Respond in Turkish."
        if language == "tr"
        else "Respond in English."
    )

    total_pages = len(
        summary_df
    )

    priority_series = summary_df.get(
        "PriorityTier",
        pd.Series(
            dtype=str
        ),
    )

    high_priority_pages = int(
        (
            priority_series
            == "High Priority"
        ).sum()
    )

    total_clicks = safe_float(
        summary_df.get(
            "CurrentClicks",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    total_incremental_value = safe_float(
        summary_df.get(
            "ExpectedIncrementalTrafficValue",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    total_net_value = safe_float(
        summary_df.get(
            "ExpectedNetValue",
            pd.Series(
                dtype=float
            ),
        ).sum()
    )

    roi_series = pd.to_numeric(
        summary_df.get(
            "EstimatedROI",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    ).dropna()

    average_roi = (
        float(
            roi_series.mean()
        )
        if not roi_series.empty
        else 0.0
    )

    recommended_actions = (
        summary_df.get(
            "RecommendedAction",
            pd.Series(
                dtype=str
            ),
        )
        .fillna("")
        .astype(str)
    )

    recommended_actions = (
        recommended_actions[
            recommended_actions.str.strip()
            != ""
        ]
    )

    top_action = (
        recommended_actions
        .value_counts()
        .idxmax()
        if not recommended_actions.empty
        else "N/A"
    )

    top_intent = "N/A"

    if not intent_df.empty:
        top_intent = str(
            intent_df.iloc[0].get(
                "keyword_intent",
                "N/A",
            )
            or "N/A"
        )

    return f"""
You are an experienced SEO, GEO and organic-growth director.

Write a concise portfolio-level executive summary.

Requirements:
- Maximum 5 sentences.
- Describe the overall SEO status.
- Highlight the highest-value opportunities.
- State the dominant strategic action.
- State the recommended implementation direction.
- Use only the data supplied below.
- Do not invent information.
- Treat forecasts and ROI values as estimates rather than guarantees.
{language_instruction}

Total pages analyzed: {total_pages}
High-priority pages: {high_priority_pages}
Total current clicks: {total_clicks:.0f}
Total expected incremental traffic value: {total_incremental_value:.2f}
Total expected net value: {total_net_value:.2f}
Average estimated ROI: {average_roi:.2f}
Most common recommended action: {top_action}
Dominant keyword intent: {top_intent}

Portfolio executive summary:
""".strip()


# ============================================================
# PORTFOLIO COMMENTARY GENERATION
# ============================================================


def generate_seo_portfolio_commentary(
    summary_df: pd.DataFrame,
    intent_df: pd.DataFrame,
    language: str | None = None,
) -> str:
    """
    Generate portfolio-level SEO executive commentary.

    Uses the common multi-provider LLM manager and falls back to
    deterministic commentary when no live provider is available.
    """
    resolved_language = normalize_language(
        language
    )

    if summary_df.empty:
        return (
            build_deterministic_portfolio_commentary(
                summary_df=summary_df,
                intent_df=intent_df,
                language=resolved_language,
            )
        )

    fallback_commentary = (
        build_deterministic_portfolio_commentary(
            summary_df=summary_df,
            intent_df=intent_df,
            language=resolved_language,
        )
    )

    prompt = build_portfolio_commentary_prompt(
        summary_df=summary_df,
        intent_df=intent_df,
        language=resolved_language,
    )

    generated_commentary = generate_text(
        prompt=prompt,
        max_tokens=min(
            500,
            getattr(SETTINGS, "llm_max_tokens", 800),
        ),
        temperature=getattr(SETTINGS, "llm_temperature", 0.2),
    )

    if generated_commentary:
        return generated_commentary

    return fallback_commentary


# ============================================================
# SAFE RUNTIME STATUS
# ============================================================


def get_commentary_runtime_info() -> dict:
    """
    Return safe runtime information for dashboard/system status.

    API keys and credentials are never exposed.
    """
    info = get_llm_runtime_info()

    return {
        "llm": info,
        "language": SETTINGS.llm_language,
        "max_pages": SETTINGS.llm_max_pages,
        "max_tokens": getattr(SETTINGS, "llm_max_tokens", 800),
        "temperature": getattr(SETTINGS, "llm_temperature", 0.2),
    }