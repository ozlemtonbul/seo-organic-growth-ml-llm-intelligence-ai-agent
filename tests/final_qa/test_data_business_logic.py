from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.filters import (
    DateRange,
    comparison_has_same_length,
    filter_dataframe_by_date,
    get_forecast_range,
    resolve_date_range,
)
from dashboard.services.analysis_service import aggregate_seo_kpis
from dashboard.services.decision_engine import (
    aggregate_period_kpis,
    resolve_forecast_status,
)


def _column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def test_date_presets_are_inclusive_and_correct():
    ref = date(2026, 8, 15)

    s7, e7 = resolve_date_range("last_7_days", today=ref)
    s30, e30 = resolve_date_range("last_30_days", today=ref)
    s60, e60 = resolve_date_range("last_60_days", today=ref)

    assert (e7 - s7).days + 1 == 7
    assert (e30 - s30).days + 1 == 30
    assert (e60 - s60).days + 1 == 60
    assert e7 == e30 == e60 == ref


def test_forecast_range_starts_after_observed_period():
    ref = date(2026, 8, 15)
    start, end = get_forecast_range(ref, 30)

    assert start == date(2026, 8, 16)
    assert end == date(2026, 9, 14)
    assert (end - start).days + 1 == 30


def test_previous_period_length_contract():
    assert comparison_has_same_length(
        date(2026, 8, 9),
        date(2026, 8, 15),
        date(2026, 8, 2),
        date(2026, 8, 8),
    )


def test_shared_date_filter_is_inclusive():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=10, freq="D"),
        "clicks": range(10),
    })

    result = filter_dataframe_by_date(
        frame,
        DateRange(
            start_date=date(2026, 8, 4),
            end_date=date(2026, 8, 7),
        ),
    )

    assert len(result) == 4
    assert result["date"].min().date() == date(2026, 8, 4)
    assert result["date"].max().date() == date(2026, 8, 7)


def test_kpi_math_ctr_and_weighted_position():
    frame = pd.DataFrame({
        "clicks": [10, 20],
        "impressions": [100, 300],
        "position": [2.0, 6.0],
        "sessions": [8, 12],
        "users": [7, 10],
        "conversions": [1, 2],
        "revenue": [100.0, 300.0],
    })

    kpis = aggregate_seo_kpis(frame)

    assert kpis["clicks"] == pytest.approx(30.0)
    assert kpis["impressions"] == pytest.approx(400.0)
    assert kpis["ctr"] == pytest.approx(30.0 / 400.0)
    assert kpis["position"] == pytest.approx((2 * 100 + 6 * 300) / 400)

    decision_kpis = aggregate_period_kpis(frame)
    assert decision_kpis["ctr"] == pytest.approx(kpis["ctr"])
    assert decision_kpis["position"] == pytest.approx(kpis["position"])


def test_forecast_contract_does_not_fake_multi_horizon():
    sample = pd.DataFrame({
        "PredictedNextClicks": [10.0],
        "PredictedNextImpressions": [100.0],
    })

    assert resolve_forecast_status(sample, 1) == "ready"
    assert resolve_forecast_status(sample, 7) == "one_step_model_available"
    assert resolve_forecast_status(sample, 14) == "one_step_model_available"
    assert resolve_forecast_status(sample, 30) == "one_step_model_available"


def test_integrated_primary_contract(integrated):
    assert not integrated.empty

    date_col = _column(integrated, ["date", "Date"])
    page_col = _column(integrated, ["page", "Page", "url", "URL"])
    clicks_col = _column(integrated, ["clicks", "Clicks"])
    impressions_col = _column(integrated, ["impressions", "Impressions"])

    assert date_col is not None, "Integrated data has no date column."
    assert page_col is not None, "Integrated data has no page/URL column."
    assert clicks_col is not None, "Integrated data has no clicks column."
    assert impressions_col is not None, "Integrated data has no impressions column."

    dates = pd.to_datetime(integrated[date_col], errors="coerce")
    assert dates.notna().any(), "Integrated date column contains no valid dates."

    duplicate_count = (
        integrated.assign(_qa_date=dates.dt.date)
        .duplicated(subset=["_qa_date", page_col])
        .sum()
    )
    assert duplicate_count == 0, (
        f"Integrated data contains {duplicate_count} duplicate date+page rows."
    )


@pytest.mark.parametrize(
    "metric_candidates",
    [
        ["clicks", "Clicks"],
        ["impressions", "Impressions"],
        ["sessions", "Sessions"],
        ["conversions", "Conversions"],
        ["revenue", "Revenue"],
    ],
)
def test_integrated_additive_metrics_are_non_negative(integrated, metric_candidates):
    col = _column(integrated, metric_candidates)
    if col is None:
        pytest.skip(f"Metric not present: {metric_candidates}")

    values = _num(integrated, col).dropna()
    assert (values >= 0).all(), f"Negative values found in {col}."


def test_integrated_ctr_matches_clicks_over_impressions(integrated):
    clicks = _column(integrated, ["clicks", "Clicks"])
    impressions = _column(integrated, ["impressions", "Impressions"])
    ctr = _column(integrated, ["ctr", "CTR"])

    assert clicks and impressions and ctr

    work = integrated[[clicks, impressions, ctr]].copy()
    work[clicks] = _num(work, clicks).fillna(0)
    work[impressions] = _num(work, impressions).fillna(0)
    work[ctr] = _num(work, ctr).fillna(0)

    expected = np.where(
        work[impressions].to_numpy() > 0,
        work[clicks].to_numpy() / work[impressions].to_numpy(),
        0.0,
    )

    assert np.allclose(
        work[ctr].to_numpy(),
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    ), "Row-level CTR does not match clicks / impressions."


def test_engagement_rate_is_a_rate_not_a_sum(integrated):
    col = _column(integrated, ["engagement_rate", "EngagementRate"])
    if col is None:
        pytest.skip("No engagement_rate column.")

    values = _num(integrated, col).dropna()
    assert ((values >= 0) & (values <= 1)).all(), (
        "engagement_rate contains values outside 0..1."
    )


def test_real_7_30_60_day_windows_are_different(integrated):
    date_col = _column(integrated, ["date", "Date"])
    clicks_col = _column(integrated, ["clicks", "Clicks"])
    impressions_col = _column(integrated, ["impressions", "Impressions"])
    assert date_col and clicks_col and impressions_col

    dates = pd.to_datetime(integrated[date_col], errors="coerce")
    end = dates.max().normalize()

    results = {}
    for days in (7, 30, 60):
        start = end - pd.Timedelta(days=days - 1)
        mask = dates.between(start, end)
        part = integrated.loc[mask]

        results[days] = {
            "rows": len(part),
            "days": pd.to_datetime(part[date_col], errors="coerce").dt.date.nunique(),
            "clicks": _num(part, clicks_col).fillna(0).sum(),
            "impressions": _num(part, impressions_col).fillna(0).sum(),
        }

    assert results[7]["rows"] < results[30]["rows"] <= results[60]["rows"]
    assert results[7]["clicks"] <= results[30]["clicks"] <= results[60]["clicks"]
    assert (
        results[7]["impressions"]
        <= results[30]["impressions"]
        <= results[60]["impressions"]
    )

    # The latest produced dataset is expected to have daily coverage.
    assert results[7]["days"] == 7, f"7-day period has {results[7]['days']} data days."
    assert results[30]["days"] == 30, f"30-day period has {results[30]['days']} data days."


def test_recommendations_are_one_row_per_page(recommendations):
    assert not recommendations.empty
    page_col = _column(recommendations, ["page", "Page", "url", "URL"])
    assert page_col is not None

    duplicates = recommendations.duplicated(subset=[page_col]).sum()
    assert duplicates == 0, (
        f"Recommendations contain {duplicates} duplicate page rows."
    )


def test_scenario_output_has_real_variation(scenarios):
    assert not scenarios.empty

    page_col = _column(scenarios, ["page", "Page", "url", "URL"])
    scenario_col = _column(scenarios, ["Scenario", "scenario"])
    assert page_col and scenario_col

    analysis_date_col = _column(
        scenarios,
        ["AnalysisDate", "analysis_date", "date", "Date"],
    )

    subset = [page_col, scenario_col]
    if analysis_date_col:
        subset.insert(0, analysis_date_col)

    duplicate_count = scenarios.duplicated(subset=subset).sum()
    assert duplicate_count == 0, (
        f"Scenario output contains {duplicate_count} duplicate scenario keys."
    )

    predicted_clicks_col = _column(
        scenarios,
        ["PredictedNextClicks", "ProjectedIncrementalClicks", "ScenarioClicks"],
    )
    predicted_impressions_col = _column(
        scenarios,
        ["PredictedNextImpressions", "ScenarioImpressions"],
    )

    if predicted_clicks_col:
        values = _num(scenarios, predicted_clicks_col).dropna()
        assert (values >= 0).all(), f"Negative values in {predicted_clicks_col}."

    if predicted_impressions_col:
        values = _num(scenarios, predicted_impressions_col).dropna()
        assert (values >= 0).all(), f"Negative values in {predicted_impressions_col}."

    # At least some pages must have more than one scenario.
    counts = scenarios.groupby(page_col)[scenario_col].nunique()
    assert counts.max() > 1, "No page contains multiple SEO scenarios."

    variation_col = _column(
        scenarios,
        [
            "ExpectedNetValue",
            "EstimatedROI",
            "ProjectedIncrementalClicks",
            "ClicksUplift",
            "ScenarioCTR",
        ],
    )
    if variation_col:
        variation = (
            scenarios.groupby(page_col)[variation_col]
            .nunique(dropna=True)
        )
        assert variation.max() > 1, (
            f"{variation_col} is identical across all scenarios."
        )


def test_model_metrics_are_numerically_valid(model_metrics):
    assert not model_metrics.empty

    numeric_frame = model_metrics.apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )

    rmse_cols = [
        c for c in model_metrics.columns
        if "rmse" in c.lower()
    ]
    r2_cols = [
        c for c in model_metrics.columns
        if c.lower() in {"r2", "r2_score", "r²"}
        or "r2" in c.lower()
    ]

    assert rmse_cols, "No RMSE column found in model metrics."
    assert r2_cols, "No R2 column found in model metrics."

    for col in rmse_cols:
        values = numeric_frame[col].dropna()
        assert np.isfinite(values).all()
        assert (values >= 0).all(), f"RMSE must be >= 0: {col}"

    for col in r2_cols:
        values = numeric_frame[col].dropna()
        assert np.isfinite(values).all()
        assert (values <= 1).all(), f"R2 cannot be greater than 1: {col}"
