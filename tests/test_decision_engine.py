from __future__ import annotations

import pandas as pd

from dashboard.services.decision_engine import (
    build_decision_intelligence,
    build_page_change_table,
    build_period_comparison,
)


def _current() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-01"],
            "page": ["/a", "/b"],
            "clicks": [80, 120],
            "impressions": [1000, 1000],
            "position": [8.0, 5.0],
            "sessions": [100, 200],
            "conversions": [10, 20],
            "revenue": [1000, 2000],
        }
    )


def _previous() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-01"],
            "page": ["/a", "/b"],
            "clicks": [100, 100],
            "impressions": [900, 900],
            "position": [6.0, 6.0],
            "sessions": [90, 180],
            "conversions": [9, 18],
            "revenue": [900, 1800],
        }
    )


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "page": ["/a", "/b"],
            "RecommendedAction": [
                "Optimize Title and Meta",
                "Apply Full SEO and GEO Optimization",
            ],
            "RecommendationReason": [
                "CTR can be improved.",
                "Growth can be scaled.",
            ],
            "ConfidenceLevel": ["High", "Medium"],
            "PriorityTier": [
                "High Priority",
                "Medium Priority",
            ],
            "ClicksUplift": [15.0, 20.0],
            "ClicksUpliftPct": [18.0, 15.0],
            "ExpectedIncrementalTrafficValue": [7.5, 10.0],
            "ExpectedNetValue": [50.0, 75.0],
            "EstimatedROI": [2.0, 1.5],
            "PredictedNextClicks": [95.0, 140.0],
            "PredictedNextImpressions": [1100.0, 1200.0],
            "OpportunityScore": [90.0, 80.0],
        }
    )


def test_period_comparison_calculates_weighted_ctr_and_deltas():
    result = build_period_comparison(
        _current(),
        _previous(),
    )

    assert result.current["clicks"] == 200.0
    assert result.previous["clicks"] == 200.0
    assert result.current["ctr"] == 0.10
    assert result.previous["ctr"] == 200 / 1800
    assert result.deltas["clicks_pct"] == 0.0


def test_page_change_table_detects_page_level_changes():
    result = build_page_change_table(
        _current(),
        _previous(),
    )

    row_a = result.loc[
        result["page"].eq("/a")
    ].iloc[0]

    assert row_a["ClicksDelta"] == -20
    assert row_a["ClicksChangePct"] == -20.0
    assert row_a["PositionImprovement"] == -2.0


def test_decision_engine_returns_problem_action_and_expected_impact():
    result = build_decision_intelligence(
        current_period=_current(),
        comparison_period=_previous(),
        recommendations=_recommendations(),
        language="tr",
        forecast_horizon_days=7,
    )

    assert not result.decisions.empty

    row_a = result.decisions.loc[
        result.decisions["page"].eq("/a")
    ].iloc[0]

    assert row_a["ProblemOpportunity"] in {
        "Sıralama Kaybı",
        "Organik Trafik Kaybı",
        "CTR Kaybı",
    }
    assert row_a["Action"] == "Başlık ve Meta Optimizasyonu"
    assert row_a["Priority"] == "Yüksek Öncelik"
    assert row_a["Confidence"] == "Yüksek"
    assert "+15.0" in row_a["ExpectedImpact"]
    assert result.forecast_status == "one_step_model_available"


def test_missing_comparison_is_explicit_and_deltas_are_none():
    current = pd.DataFrame({
        "Date": ["2026-08-03"],
        "Page": ["/a"],
        "Clicks": [30],
        "Impressions": [300],
        "Position": [4],
    })
    result = build_period_comparison(current, pd.DataFrame(columns=current.columns))
    assert result.current_available is True
    assert result.comparison_available is False
    assert all(value is None for value in result.deltas.values())


def test_page_change_does_not_invent_delta_without_comparison():
    current = pd.DataFrame({
        "Page": ["/a"],
        "Clicks": [30],
        "Impressions": [300],
        "Position": [4],
    })
    result = build_page_change_table(current, pd.DataFrame(columns=current.columns))
    assert pd.isna(result.loc[0, "ClicksDelta"])
    assert pd.isna(result.loc[0, "ImpressionsDelta"])
    assert pd.isna(result.loc[0, "CTRDelta"])
