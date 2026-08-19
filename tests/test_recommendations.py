from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.recommendations import (
    ACTION_MAP,
    CONFIDENCE_MULTIPLIERS,
    REASON_MAP,
    add_priority_tier,
    apply_confidence_guardrail,
    build_confidence_scores,
    build_page_commentary_prompt,
    build_portfolio_commentary_prompt,
    build_recommendations,
)


def build_recommendation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "page": "https://example.com/product/a",
                "Scenario": "content_refresh",
                "ExpectedNetValue": 120.0,
                "EstimatedROI": 1.20,
            }
        ]
    )


def test_recommendation_maps_exist() -> None:
    assert (
        ACTION_MAP["content_refresh"]
        == "Refresh Content"
    )

    assert "content_refresh" in REASON_MAP

    assert (
        CONFIDENCE_MULTIPLIERS["High"]
        == 1.0
    )


def test_build_recommendations() -> None:
    result = build_recommendations(
        build_recommendation_dataframe()
    )

    assert (
        result.iloc[0]["RecommendedAction"]
        == "Refresh Content"
    )

    assert (
        "content refresh opportunity"
        in result.iloc[0]["RecommendationReason"]
    )


def test_build_recommendations_requires_scenario() -> None:
    with pytest.raises(
        ValueError,
        match="Scenario column",
    ):
        build_recommendations(
            pd.DataFrame(
                [
                    {
                        "page": (
                            "https://example.com/a"
                        ),
                    }
                ]
            )
        )


def test_build_confidence_scores_high() -> None:
    recommendations = build_recommendations(
        build_recommendation_dataframe()
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": "Next_Clicks",
                "R2": 0.82,
            },
            {
                "Model": "Next_Impressions",
                "R2": 0.76,
            },
        ]
    )

    training = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/product/a"
                ),
            }
            for _ in range(25)
        ]
    )

    result = build_confidence_scores(
        recommendations,
        metrics,
        training,
    )

    assert (
        result.iloc[0]["HistoryRows"]
        == 25
    )

    assert (
        result.iloc[0]["ConfidenceLevel"]
        == "High"
    )

    assert (
        result.iloc[0]["AverageModelR2"]
        == pytest.approx(0.79)
    )


def test_build_confidence_scores_low() -> None:
    recommendations = build_recommendations(
        build_recommendation_dataframe()
    )

    metrics = pd.DataFrame(
        [
            {
                "Model": "Next_Clicks",
                "R2": 0.10,
            }
        ]
    )

    training = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/product/a"
                ),
            }
            for _ in range(3)
        ]
    )

    result = build_confidence_scores(
        recommendations,
        metrics,
        training,
    )

    assert (
        result.iloc[0]["ConfidenceLevel"]
        == "Low"
    )


def test_apply_confidence_guardrail() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "RecommendedAction": (
                    "Refresh Content"
                ),
                "RecommendationReason": (
                    "Sample reason"
                ),
                "ConfidenceLevel": "Low",
            }
        ]
    )

    result = apply_confidence_guardrail(
        dataframe
    )

    assert (
        result.iloc[0]["RecommendedAction"]
        == "Review"
    )

    assert (
        "Manual SEO validation"
        in result.iloc[0][
            "RecommendationReason"
        ]
    )


def test_add_priority_tier_high() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "ConfidenceLevel": "High",
                "ExpectedNetValue": 120.0,
                "EstimatedROI": 1.20,
            }
        ]
    )

    result = add_priority_tier(
        dataframe
    )

    assert (
        result.iloc[0]["ConfidenceMultiplier"]
        == 1.0
    )

    assert (
        result.iloc[0]["AdjustedNetValue"]
        == 120.0
    )

    assert (
        result.iloc[0]["PriorityTier"]
        == "High Priority"
    )


def test_add_priority_tier_medium() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "ConfidenceLevel": "Medium",
                "ExpectedNetValue": 20.0,
                "EstimatedROI": 0.20,
            }
        ]
    )

    result = add_priority_tier(
        dataframe
    )

    assert (
        result.iloc[0]["PriorityTier"]
        == "Medium Priority"
    )


def test_add_priority_tier_low() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "ConfidenceLevel": "Low",
                "ExpectedNetValue": -10.0,
                "EstimatedROI": -0.20,
            }
        ]
    )

    result = add_priority_tier(
        dataframe
    )

    assert (
        result.iloc[0]["PriorityTier"]
        == "Low Priority"
    )


def test_page_commentary_prompt() -> None:
    row = pd.Series(
        {
            "page": (
                "https://example.com/product/a"
            ),
            "page_type": "product",
            "keyword_intent": (
                "Transactional"
            ),
            "CurrentClicks": 100,
            "CurrentImpressions": 2000,
            "CurrentCTR": 0.05,
            "CurrentPosition": 7.2,
            "RecommendedAction": (
                "Refresh Content"
            ),
        }
    )

    prompt = build_page_commentary_prompt(
        row,
        language="en",
    )

    assert (
        "https://example.com/product/a"
        in prompt
    )

    assert "Refresh Content" in prompt
    assert "Respond in English." in prompt


def test_portfolio_commentary_prompt() -> None:
    summary = pd.DataFrame(
        [
            {
                "PriorityTier": (
                    "High Priority"
                ),
                "CurrentClicks": 100,
                "ExpectedIncrementalTrafficValue": 20,
                "ExpectedNetValue": 10,
                "EstimatedROI": 0.5,
                "RecommendedAction": (
                    "Refresh Content"
                ),
            }
        ]
    )

    intent = pd.DataFrame(
        [
            {
                "keyword_intent": (
                    "Transactional"
                ),
                "total_clicks": 100,
            }
        ]
    )

    prompt = build_portfolio_commentary_prompt(
        summary,
        intent,
        language="en",
    )

    assert (
        "Total pages analyzed: 1"
        in prompt
    )

    assert (
        "High-priority pages: 1"
        in prompt
    )

    assert "Transactional" in prompt


def test_llm_fallback_without_available_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the common LLM manager cannot return a response,
    page commentary must fall back to deterministic output.
    """
    import src.recommendations.llm_commentary as llm_module

    test_settings = SimpleNamespace(
        llm_language="en",
        llm_max_pages=30,
        llm_max_tokens=800,
        llm_temperature=0.2,
    )

    monkeypatch.setattr(
        llm_module,
        "SETTINGS",
        test_settings,
    )

    monkeypatch.setattr(
        llm_module,
        "generate_text",
        lambda *args, **kwargs: None,
    )

    summary = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/product/a"
                ),
                "page_type": "product",
                "CurrentClicks": 100,
                "ScenarioLabel": (
                    "Content Refresh"
                ),
                "RecommendedAction": (
                    "Refresh Content"
                ),
            }
        ]
    )

    result = (
        llm_module.generate_page_commentaries(
            summary
        )
    )

    assert (
        result.iloc[0][
            "CommentarySource"
        ]
        == "deterministic"
    )

    commentary = result.iloc[0][
        "ExecutiveCommentary"
    ]

    assert isinstance(
        commentary,
        str,
    )

    assert commentary

    assert (
        "100 organic clicks"
        in commentary
    )

    assert (
        "Refresh Content"
        in commentary
    )


def test_portfolio_fallback_without_available_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the common LLM manager cannot return a response,
    portfolio commentary must fall back to deterministic output.
    """
    import src.recommendations.llm_commentary as llm_module

    test_settings = SimpleNamespace(
        llm_language="en",
        llm_max_pages=30,
        llm_max_tokens=800,
        llm_temperature=0.2,
    )

    monkeypatch.setattr(
        llm_module,
        "SETTINGS",
        test_settings,
    )

    monkeypatch.setattr(
        llm_module,
        "generate_text",
        lambda *args, **kwargs: None,
    )

    summary = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/product/a"
                ),
                "PriorityTier": (
                    "High Priority"
                ),
                "CurrentClicks": 100,
                "ExpectedIncrementalTrafficValue": 20,
                "ExpectedNetValue": 10,
                "EstimatedROI": 0.5,
                "RecommendedAction": (
                    "Refresh Content"
                ),
            }
        ]
    )

    intent = pd.DataFrame(
        [
            {
                "keyword_intent": (
                    "Transactional"
                ),
                "total_clicks": 100,
            }
        ]
    )

    result = (
        llm_module
        .generate_seo_portfolio_commentary(
            summary,
            intent,
        )
    )

    assert isinstance(
        result,
        str,
    )

    assert result

    assert (
        "1 analyzed pages"
        in result
    )

    assert (
        "100"
        in result
    )

    assert (
        "Refresh Content"
        in result
    )

    assert (
        "Transactional"
        in result
    )