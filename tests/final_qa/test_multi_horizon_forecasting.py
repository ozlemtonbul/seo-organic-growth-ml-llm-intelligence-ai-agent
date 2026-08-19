from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.multi_horizon_forecasting import (
    build_daily_page_panel,
    build_horizon_forecast_summary,
    build_portfolio_horizon_summary,
    build_recursive_daily_forecast,
    prepare_daily_inference_state,
    prepare_daily_training_data,
)


class GrowthClicksModel:
    def predict(self, x):
        clicks = pd.to_numeric(x["clicks"], errors="coerce").fillna(0.0).to_numpy()
        return clicks * 1.05 + 1.0


class GrowthImpressionsModel:
    def predict(self, x):
        impressions = pd.to_numeric(x["impressions"], errors="coerce").fillna(0.0).to_numpy()
        return impressions * 1.03 + 10.0


def _synthetic_raw() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=40, freq="D")
    rows = []
    for page_index, page in enumerate(["/a", "/b"]):
        for day_index, date in enumerate(dates):
            # Deliberately omit some page/day rows so panel completion is tested.
            if page == "/b" and day_index % 6 == 0:
                continue
            clicks = 10 + page_index * 3 + day_index % 5
            impressions = 100 + page_index * 20 + day_index * 2
            rows.append(
                {
                    "date": date,
                    "page": page,
                    "clicks": clicks,
                    "impressions": impressions,
                    "position": 3.0 + page_index,
                    "ctr": clicks / impressions,
                    "sessions": clicks * 0.8,
                    "users": clicks * 0.7,
                    "engaged_sessions": clicks * 0.6,
                    "engagement_rate": 0.6,
                    "average_session_duration": 45.0,
                    "conversions": clicks * 0.02,
                    "revenue": clicks * 5.0,
                    "purchases": clicks * 0.01,
                    "add_to_carts": clicks * 0.03,
                    "checkouts": clicks * 0.02,
                    "page_type": "category",
                    "keyword_intent": "Transactional",
                }
            )
    return pd.DataFrame(rows)


def test_daily_panel_uses_real_calendar_days():
    raw = _synthetic_raw()
    panel = build_daily_page_panel(raw)

    assert panel["page"].nunique() == 2
    assert panel["date"].nunique() == 40
    assert len(panel) == 80
    assert not panel.duplicated(["page", "date"]).any()


def test_daily_training_target_is_next_calendar_day():
    panel = build_daily_page_panel(_synthetic_raw())
    training = prepare_daily_training_data(panel)

    counts = training.groupby("page").size().to_dict()
    assert counts == {"/a": 39, "/b": 39}
    assert "target_clicks_next" in training.columns
    assert "target_impressions_next" in training.columns


def test_recursive_forecast_generates_365_distinct_future_days():
    panel = build_daily_page_panel(_synthetic_raw())
    latest, histories = prepare_daily_inference_state(panel)

    features = [
        "clicks",
        "impressions",
        "position",
        "CTR",
        "TrafficValue",
        "RankStrength",
        "VisibilityScore",
        "Top3Flag",
        "Top10Flag",
        "Page2Flag",
        "day_of_week",
        "day_of_month",
        "month_num",
        "quarter",
        "is_weekend",
        "clicks_lag_1",
        "clicks_lag_7_avg",
        "impressions_lag_1",
        "impressions_lag_7_avg",
        "position_lag_1",
        "position_lag_7_avg",
        "CTR_lag_1",
        "CTR_lag_7_avg",
        "TrafficValue_lag_1",
        "TrafficValue_lag_7_avg",
        "clicks_change",
        "impressions_change",
        "position_change",
        "ctr_change",
    ]

    forecast = build_recursive_daily_forecast(
        latest_state=latest,
        histories=histories,
        model_clicks=GrowthClicksModel(),
        model_impressions=GrowthImpressionsModel(),
        feature_columns=features,
        max_horizon_days=365,
    )

    assert len(forecast) == 730
    assert forecast["HorizonDay"].min() == 1
    assert forecast["HorizonDay"].max() == 365
    assert forecast.groupby("page")["ForecastDate"].nunique().eq(365).all()
    assert (forecast["PredictedClicks"] >= 0).all()
    assert (forecast["PredictedImpressions"] >= forecast["PredictedClicks"]).all()


def test_horizon_totals_are_derived_from_recursive_daily_path_not_multiplication():
    panel = build_daily_page_panel(_synthetic_raw())
    latest, histories = prepare_daily_inference_state(panel)

    features = [
        "clicks",
        "impressions",
        "position",
        "CTR",
        "TrafficValue",
        "RankStrength",
        "VisibilityScore",
        "Top3Flag",
        "Top10Flag",
        "Page2Flag",
        "day_of_week",
        "day_of_month",
        "month_num",
        "quarter",
        "is_weekend",
        "clicks_lag_1",
        "clicks_lag_7_avg",
        "impressions_lag_1",
        "impressions_lag_7_avg",
        "position_lag_1",
        "position_lag_7_avg",
        "CTR_lag_1",
        "CTR_lag_7_avg",
        "TrafficValue_lag_1",
        "TrafficValue_lag_7_avg",
        "clicks_change",
        "impressions_change",
        "position_change",
        "ctr_change",
    ]

    daily = build_recursive_daily_forecast(
        latest_state=latest,
        histories=histories,
        model_clicks=GrowthClicksModel(),
        model_impressions=GrowthImpressionsModel(),
        feature_columns=features,
        max_horizon_days=365,
    )

    metrics = pd.DataFrame(
        {
            "Model": ["Next_Clicks", "Next_Impressions"],
            "R2": [0.90, 0.88],
        }
    )

    horizons = build_horizon_forecast_summary(
        daily_forecast=daily,
        daily_panel=panel,
        metrics=metrics,
        horizons=(7, 14, 30, 90, 180, 365),
    )

    assert set(horizons["HorizonDays"].unique()) == {7, 14, 30, 90, 180, 365}
    assert len(horizons) == 12

    page_a = horizons[horizons["page"] == "/a"].set_index("HorizonDays")
    assert not np.isclose(
        page_a.loc[14, "PredictedClicks"],
        page_a.loc[7, "PredictedClicks"] * 2,
    )

    portfolio = build_portfolio_horizon_summary(horizons)
    assert set(portfolio["HorizonDays"].tolist()) == {7, 14, 30, 90, 180, 365}
    assert portfolio["ForecastReliability"].between(0, 1).all()
    assert portfolio.sort_values("HorizonDays")["ForecastReliability"].is_monotonic_decreasing


def test_long_horizons_are_classified_as_strategic():
    panel = build_daily_page_panel(_synthetic_raw())
    latest, histories = prepare_daily_inference_state(panel)

    features = [
        "clicks",
        "impressions",
        "position",
        "CTR",
        "TrafficValue",
        "RankStrength",
        "VisibilityScore",
        "Top3Flag",
        "Top10Flag",
        "Page2Flag",
        "day_of_week",
        "day_of_month",
        "month_num",
        "quarter",
        "is_weekend",
        "clicks_lag_1",
        "clicks_lag_7_avg",
        "impressions_lag_1",
        "impressions_lag_7_avg",
        "position_lag_1",
        "position_lag_7_avg",
        "CTR_lag_1",
        "CTR_lag_7_avg",
        "TrafficValue_lag_1",
        "TrafficValue_lag_7_avg",
        "clicks_change",
        "impressions_change",
        "position_change",
        "ctr_change",
    ]

    daily = build_recursive_daily_forecast(
        latest_state=latest,
        histories=histories,
        model_clicks=GrowthClicksModel(),
        model_impressions=GrowthImpressionsModel(),
        feature_columns=features,
        max_horizon_days=365,
    )

    metrics = pd.DataFrame(
        {
            "Model": ["Next_Clicks", "Next_Impressions"],
            "R2": [0.90, 0.88],
        }
    )

    horizons = build_horizon_forecast_summary(
        daily_forecast=daily,
        daily_panel=panel,
        metrics=metrics,
        horizons=(7, 14, 30, 90, 180, 365),
    )

    portfolio = build_portfolio_horizon_summary(horizons)

    operational = portfolio.loc[
        portfolio["HorizonDays"].isin([7, 14, 30])
    ]
    strategic = portfolio.loc[
        portfolio["HorizonDays"].isin([90, 180, 365])
    ]

    assert operational["HorizonType"].eq("Operational").all()
    assert strategic["HorizonType"].eq("Strategic").all()
    assert len(portfolio) == 6

    annual = portfolio.loc[
        portfolio["HorizonDays"].eq(365)
    ].iloc[0]

    assert pd.to_datetime(annual["ForecastEndDate"]) > pd.to_datetime(
        annual["ForecastStartDate"]
    )
    assert float(annual["PredictedClicks"]) >= 0
    assert float(annual["PredictedImpressions"]) >= float(
        annual["PredictedClicks"]
    )
