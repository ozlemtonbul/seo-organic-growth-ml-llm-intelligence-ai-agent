from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.features.feature_engineering import (
    GA4_FEATURE_COLUMNS,
    add_lag_features,
    add_time_features,
    compute_kpis,
    get_feature_columns,
)
from src.models.traffic_forecasting import (
    get_last_model_benchmark,
    safe_prediction,
    train_and_validate_models,
)


logger = get_logger(__name__)


DEFAULT_HORIZONS: Tuple[int, ...] = (7, 14, 30, 90, 180, 365)


@dataclass(frozen=True)
class MultiHorizonForecastResult:
    """Container for the multi-horizon forecasting outputs."""

    training: pd.DataFrame
    latest_state: pd.DataFrame
    daily_forecast: pd.DataFrame
    horizon_forecast: pd.DataFrame
    portfolio_forecast: pd.DataFrame
    metrics: pd.DataFrame
    benchmark: pd.DataFrame
    feature_importance: pd.DataFrame


def _numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    return [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def build_daily_page_panel(
    seo_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert page observations into a complete page x calendar-day panel.

    The existing core model predicts the next observed row for each page.
    Multi-horizon forecasting needs a genuine daily step, so missing page/day
    observations are explicitly represented as zero-activity days.
    """
    if seo_raw is None or seo_raw.empty:
        return pd.DataFrame()

    required = {"page", "date", "clicks", "impressions", "position"}
    missing = sorted(required.difference(seo_raw.columns))
    if missing:
        raise ValueError(
            "Daily forecast panel is missing required columns: "
            f"{missing}"
        )

    source = seo_raw.copy()
    source["date"] = (
        pd.to_datetime(source["date"], errors="coerce")
        .dt.normalize()
    )
    source = source.dropna(subset=["date", "page"]).copy()

    if source.empty:
        return pd.DataFrame()

    if source.duplicated(["page", "date"]).any():
        raise ValueError(
            "Multi-horizon daily forecasting requires unique page + date rows."
        )

    min_date = source["date"].min().normalize()
    max_date = source["date"].max().normalize()
    dates = pd.date_range(min_date, max_date, freq="D")
    pages = source["page"].astype(str).drop_duplicates().tolist()

    complete_index = pd.MultiIndex.from_product(
        [pages, dates],
        names=["page", "date"],
    )

    source["page"] = source["page"].astype(str)
    panel = (
        source.set_index(["page", "date"])
        .reindex(complete_index)
        .reset_index()
    )

    static_candidates = [
        "page_type",
        "keyword_intent",
    ]

    for column in static_candidates:
        if column not in panel.columns:
            continue

        page_values = (
            source[["page", column]]
            .dropna(subset=[column])
            .drop_duplicates("page", keep="last")
            .set_index("page")[column]
        )
        panel[column] = panel[column].fillna(
            panel["page"].map(page_values)
        )

    numeric_columns = _numeric_columns(source)
    for column in numeric_columns:
        if column in panel.columns:
            panel[column] = pd.to_numeric(
                panel[column], errors="coerce"
            ).fillna(0.0)

    for column in panel.columns:
        if column in {"page", "date"} or column in numeric_columns:
            continue
        panel[column] = panel[column].fillna("")

    if "ctr" in panel.columns:
        panel["ctr"] = np.where(
            panel["impressions"] > 0,
            panel["clicks"] / panel["impressions"],
            0.0,
        )

    panel = panel.sort_values(["page", "date"]).reset_index(drop=True)

    logger.info(
        "Daily ML panel prepared | Pages: %d | Days: %d | Rows: %d",
        panel["page"].nunique(),
        panel["date"].nunique(),
        len(panel),
    )

    return panel


def prepare_daily_training_data(
    daily_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build genuine next-calendar-day targets for the recursive ML model."""
    if daily_panel is None or daily_panel.empty:
        return pd.DataFrame()

    result = compute_kpis(daily_panel)
    result = add_time_features(result)
    result = add_lag_features(result)
    result = result.sort_values(["page", "date"]).copy()

    grouped = result.groupby("page", group_keys=False)
    result["target_clicks_next"] = grouped["clicks"].shift(-1)
    result["target_impressions_next"] = grouped["impressions"].shift(-1)

    return (
        result.dropna(
            subset=["target_clicks_next", "target_impressions_next"]
        )
        .reset_index(drop=True)
    )


def prepare_daily_inference_state(
    daily_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """
    Build the latest daily feature state plus rolling metric histories.

    The latest row is retained even though it has no future target. This avoids
    the penultimate-observation limitation of target-bearing training frames.
    """
    if daily_panel is None or daily_panel.empty:
        return pd.DataFrame(), {}

    featured = compute_kpis(daily_panel)
    featured = add_time_features(featured)
    featured = add_lag_features(featured)
    featured = featured.sort_values(["page", "date"]).reset_index(drop=True)

    latest = (
        featured.groupby("page", as_index=False, group_keys=False)
        .tail(1)
        .copy()
        .reset_index(drop=True)
    )

    # Carry the most recent positive ranking position into the future baseline.
    last_positive_position = (
        featured.loc[pd.to_numeric(featured["position"], errors="coerce") > 0]
        .groupby("page")["position"]
        .last()
    )

    latest["position"] = np.where(
        pd.to_numeric(latest["position"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(latest["position"], errors="coerce").fillna(0),
        latest["page"].map(last_positive_position).fillna(0),
    )

    latest["RankStrength"] = np.where(
        latest["position"] > 0,
        1.0 / latest["position"],
        0.0,
    )
    latest["Top3Flag"] = ((latest["position"] > 0) & (latest["position"] <= 3)).astype(int)
    latest["Top10Flag"] = ((latest["position"] > 0) & (latest["position"] <= 10)).astype(int)
    latest["Page2Flag"] = ((latest["position"] > 10) & (latest["position"] <= 20)).astype(int)

    # Future GA4 values are unknown. Use the trailing 7-day page average as a
    # stable exogenous baseline instead of assuming the latest missing day is 0.
    for column in GA4_FEATURE_COLUMNS:
        if column not in featured.columns or column not in latest.columns:
            continue
        trailing = (
            featured.groupby("page")[column]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        featured[f"__mh_{column}_avg"] = trailing
        latest_map = (
            featured.groupby("page")[f"__mh_{column}_avg"]
            .last()
        )
        latest[column] = latest["page"].map(latest_map).fillna(0.0)

    histories: Dict[str, Dict[str, List[float]]] = {}
    metrics = ["clicks", "impressions", "position", "CTR", "TrafficValue"]

    for page, group in featured.groupby("page", sort=False):
        histories[str(page)] = {}
        tail = group.tail(7)
        for metric in metrics:
            values = pd.to_numeric(
                tail[metric], errors="coerce"
            ).fillna(0.0).astype(float).tolist()
            histories[str(page)][metric] = values

    return latest, histories


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def _base_model_r2(metrics: pd.DataFrame) -> float:
    if metrics is None or metrics.empty or "R2" not in metrics.columns:
        return 0.0
    values = pd.to_numeric(metrics["R2"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.min())


def build_recursive_daily_forecast(
    latest_state: pd.DataFrame,
    histories: Dict[str, Dict[str, List[float]]],
    model_clicks: object,
    model_impressions: object,
    feature_columns: List[str],
    max_horizon_days: int = 30,
) -> pd.DataFrame:
    """
    Forecast each future calendar day recursively with the trained daily ML model.

    This is a genuine multi-step ML forecast: each predicted day becomes part of
    the feature state used to predict the following day. It is not a simple
    multiplication of a one-step output.
    """
    if latest_state is None or latest_state.empty:
        return pd.DataFrame()

    horizon = max(1, int(max_horizon_days))
    state = latest_state.copy().reset_index(drop=True)
    state["date"] = pd.to_datetime(state["date"], errors="coerce")

    missing_features = [c for c in feature_columns if c not in state.columns]
    for column in missing_features:
        state[column] = 0.0

    rows: List[Dict[str, object]] = []

    for step in range(1, horizon + 1):
        x_input = (
            state.reindex(columns=feature_columns)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        raw_clicks = model_clicks.predict(x_input)
        raw_impressions = model_impressions.predict(x_input)

        predicted_clicks = np.array(
            [safe_prediction(value) for value in raw_clicks],
            dtype=float,
        )
        predicted_impressions = np.array(
            [safe_prediction(value) for value in raw_impressions],
            dtype=float,
        )

        predicted_impressions = np.maximum(
            predicted_impressions,
            predicted_clicks,
        )

        forecast_dates = state["date"] + pd.Timedelta(days=1)
        predicted_ctr = np.divide(
            predicted_clicks,
            predicted_impressions,
            out=np.zeros_like(predicted_clicks),
            where=predicted_impressions > 0,
        )

        for idx, page in enumerate(state["page"].astype(str)):
            rows.append(
                {
                    "page": page,
                    "ForecastDate": forecast_dates.iloc[idx],
                    "HorizonDay": step,
                    "PredictedClicks": round(float(predicted_clicks[idx]), 4),
                    "PredictedImpressions": round(float(predicted_impressions[idx]), 4),
                    "PredictedCTR": round(float(predicted_ctr[idx]), 6),
                    "PredictedTrafficValue": round(
                        float(predicted_clicks[idx]) * float(SETTINGS.value_per_click),
                        4,
                    ),
                    "ForecastMethod": "RecursiveDailyML",
                }
            )

        previous_clicks = pd.to_numeric(state["clicks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        previous_impressions = pd.to_numeric(state["impressions"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        previous_position = pd.to_numeric(state["position"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        previous_ctr = pd.to_numeric(state["CTR"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        previous_traffic_value = pd.to_numeric(state["TrafficValue"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        state["clicks_lag_1"] = previous_clicks
        state["impressions_lag_1"] = previous_impressions
        state["position_lag_1"] = previous_position
        state["CTR_lag_1"] = previous_ctr
        state["TrafficValue_lag_1"] = previous_traffic_value

        state["clicks_change"] = predicted_clicks - previous_clicks
        state["impressions_change"] = predicted_impressions - previous_impressions
        state["position_change"] = 0.0
        state["ctr_change"] = predicted_ctr - previous_ctr

        state["clicks"] = predicted_clicks
        state["impressions"] = predicted_impressions
        state["CTR"] = predicted_ctr
        state["TrafficValue"] = predicted_clicks * float(SETTINGS.value_per_click)
        state["VisibilityScore"] = predicted_impressions * predicted_ctr
        state["date"] = forecast_dates

        state["day_of_week"] = state["date"].dt.dayofweek
        state["day_of_month"] = state["date"].dt.day
        state["month_num"] = state["date"].dt.month
        state["quarter"] = state["date"].dt.quarter
        state["is_weekend"] = (state["date"].dt.dayofweek >= 5).astype(int)

        for idx, page in enumerate(state["page"].astype(str)):
            history = histories.setdefault(page, {})
            updates = {
                "clicks": float(predicted_clicks[idx]),
                "impressions": float(predicted_impressions[idx]),
                "position": float(previous_position[idx]),
                "CTR": float(predicted_ctr[idx]),
                "TrafficValue": float(predicted_clicks[idx]) * float(SETTINGS.value_per_click),
            }

            for metric, value in updates.items():
                values = history.setdefault(metric, [])
                values.append(value)
                if len(values) > 7:
                    del values[:-7]

            state.loc[idx, "clicks_lag_7_avg"] = float(np.mean(history["clicks"]))
            state.loc[idx, "impressions_lag_7_avg"] = float(np.mean(history["impressions"]))
            state.loc[idx, "position_lag_7_avg"] = float(np.mean(history["position"]))
            state.loc[idx, "CTR_lag_7_avg"] = float(np.mean(history["CTR"]))
            state.loc[idx, "TrafficValue_lag_7_avg"] = float(np.mean(history["TrafficValue"]))

    return pd.DataFrame(rows)


def build_horizon_forecast_summary(
    daily_forecast: pd.DataFrame,
    daily_panel: pd.DataFrame,
    metrics: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Aggregate recursive daily predictions into operational and strategic ML horizons."""
    if daily_forecast is None or daily_forecast.empty:
        return pd.DataFrame()

    base_r2 = _base_model_r2(metrics)
    latest_real_date = pd.to_datetime(daily_panel["date"], errors="coerce").max()

    rows: List[Dict[str, object]] = []
    panel = daily_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    for horizon in sorted({int(h) for h in horizons if int(h) > 0}):
        forecast_slice = daily_forecast.loc[
            pd.to_numeric(daily_forecast["HorizonDay"], errors="coerce") <= horizon
        ].copy()

        if forecast_slice.empty:
            continue

        reference_start = latest_real_date - pd.Timedelta(days=horizon - 1)
        reference = panel.loc[
            panel["date"].between(reference_start, latest_real_date)
        ].copy()
        reference_summary = (
            reference.groupby("page", as_index=True)[["clicks", "impressions"]]
            .sum()
        )

        grouped = forecast_slice.groupby("page", sort=False)

        for page, group in grouped:
            predicted_clicks = float(pd.to_numeric(group["PredictedClicks"], errors="coerce").fillna(0.0).sum())
            predicted_impressions = float(pd.to_numeric(group["PredictedImpressions"], errors="coerce").fillna(0.0).sum())
            predicted_ctr = predicted_clicks / predicted_impressions if predicted_impressions > 0 else 0.0

            if page in reference_summary.index:
                reference_clicks = float(reference_summary.loc[page, "clicks"])
                reference_impressions = float(reference_summary.loc[page, "impressions"])
            else:
                reference_clicks = 0.0
                reference_impressions = 0.0

            click_change_pct = (
                (predicted_clicks - reference_clicks) / reference_clicks * 100.0
                if reference_clicks > 0
                else 0.0
            )
            impression_change_pct = (
                (predicted_impressions - reference_impressions) / reference_impressions * 100.0
                if reference_impressions > 0
                else 0.0
            )

            # Recursive uncertainty increases with horizon; this score is a
            # reliability indicator, not a statistical prediction interval.
            reliability = float(
                np.clip(base_r2 * np.exp(-0.0125 * max(0, horizon - 1)), 0.0, 1.0)
            )

            rows.append(
                {
                    "page": page,
                    "HorizonDays": horizon,
                    "ForecastStartDate": pd.to_datetime(group["ForecastDate"]).min(),
                    "ForecastEndDate": pd.to_datetime(group["ForecastDate"]).max(),
                    "PredictedClicks": round(predicted_clicks, 2),
                    "PredictedImpressions": round(predicted_impressions, 2),
                    "PredictedCTR": round(predicted_ctr, 6),
                    "PredictedTrafficValue": round(predicted_clicks * float(SETTINGS.value_per_click), 2),
                    "ReferenceClicks": round(reference_clicks, 2),
                    "ReferenceImpressions": round(reference_impressions, 2),
                    "ClickChangePct": round(click_change_pct, 2),
                    "ImpressionChangePct": round(impression_change_pct, 2),
                    "ForecastReliability": round(reliability, 4),
                    "ConfidenceLevel": _confidence_label(reliability),
                    "HorizonType": (
                        "Operational"
                        if horizon <= 30
                        else "Strategic"
                    ),
                    "ForecastMethod": "RecursiveDailyML",
                }
            )

    return pd.DataFrame(rows)


def build_portfolio_horizon_summary(
    horizon_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate page-level horizon forecasts into portfolio-level KPIs."""
    if horizon_forecast is None or horizon_forecast.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for horizon, group in horizon_forecast.groupby("HorizonDays", sort=True):
        predicted_clicks = float(pd.to_numeric(group["PredictedClicks"], errors="coerce").fillna(0.0).sum())
        predicted_impressions = float(pd.to_numeric(group["PredictedImpressions"], errors="coerce").fillna(0.0).sum())
        reference_clicks = float(pd.to_numeric(group["ReferenceClicks"], errors="coerce").fillna(0.0).sum())
        reference_impressions = float(pd.to_numeric(group["ReferenceImpressions"], errors="coerce").fillna(0.0).sum())

        predicted_ctr = predicted_clicks / predicted_impressions if predicted_impressions > 0 else 0.0
        click_change_pct = (
            (predicted_clicks - reference_clicks) / reference_clicks * 100.0
            if reference_clicks > 0 else 0.0
        )
        impression_change_pct = (
            (predicted_impressions - reference_impressions) / reference_impressions * 100.0
            if reference_impressions > 0 else 0.0
        )

        reliability_values = pd.to_numeric(group["ForecastReliability"], errors="coerce").dropna()
        reliability = float(reliability_values.mean()) if not reliability_values.empty else 0.0

        rows.append(
            {
                "HorizonDays": int(horizon),
                "ForecastStartDate": pd.to_datetime(group["ForecastStartDate"]).min(),
                "ForecastEndDate": pd.to_datetime(group["ForecastEndDate"]).max(),
                "PredictedClicks": round(predicted_clicks, 2),
                "PredictedImpressions": round(predicted_impressions, 2),
                "PredictedCTR": round(predicted_ctr, 6),
                "PredictedTrafficValue": round(predicted_clicks * float(SETTINGS.value_per_click), 2),
                "ReferenceClicks": round(reference_clicks, 2),
                "ReferenceImpressions": round(reference_impressions, 2),
                "ClickChangePct": round(click_change_pct, 2),
                "ImpressionChangePct": round(impression_change_pct, 2),
                "ForecastReliability": round(reliability, 4),
                "ConfidenceLevel": _confidence_label(reliability),
                "HorizonType": (
                    "Operational"
                    if int(horizon) <= 30
                    else "Strategic"
                ),
                "ForecastMethod": "RecursiveDailyML",
                "PageCount": int(group["page"].nunique()),
            }
        )

    return pd.DataFrame(rows)


def run_multi_horizon_forecasting(
    seo_raw: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> MultiHorizonForecastResult:
    """
    Train the daily ML baseline and generate recursive 7/14/30/90/180/365-day forecasts.

    The daily model is benchmarked with the same Random Forest / XGBoost /
    LightGBM selection logic as the existing production forecasting layer.
    """
    horizon_values = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not horizon_values:
        raise ValueError("At least one positive forecast horizon is required.")

    daily_panel = build_daily_page_panel(seo_raw)
    training = prepare_daily_training_data(daily_panel)

    if training.empty:
        raise ValueError("No daily ML training rows were generated.")

    (
        model_clicks,
        model_impressions,
        feature_columns,
        metrics,
        feature_importance,
    ) = train_and_validate_models(
        train_df=training,
        with_holiday=False,
    )

    benchmark = get_last_model_benchmark()

    metrics = metrics.copy()
    metrics["ForecastFamily"] = "RecursiveDailyML"
    metrics["BaseStepDays"] = 1
    metrics["SupportedHorizons"] = ",".join(str(h) for h in horizon_values)

    benchmark = benchmark.copy()
    if not benchmark.empty:
        benchmark["ForecastFamily"] = "RecursiveDailyML"
        benchmark["BaseStepDays"] = 1

    feature_importance = feature_importance.copy()
    if not feature_importance.empty:
        feature_importance["ForecastFamily"] = "RecursiveDailyML"

    latest_state, histories = prepare_daily_inference_state(daily_panel)

    daily_forecast = build_recursive_daily_forecast(
        latest_state=latest_state,
        histories=histories,
        model_clicks=model_clicks,
        model_impressions=model_impressions,
        feature_columns=feature_columns,
        max_horizon_days=max(horizon_values),
    )

    horizon_forecast = build_horizon_forecast_summary(
        daily_forecast=daily_forecast,
        daily_panel=daily_panel,
        metrics=metrics,
        horizons=horizon_values,
    )

    portfolio_forecast = build_portfolio_horizon_summary(
        horizon_forecast
    )

    logger.info(
        "Multi-horizon daily ML forecast completed | Training rows: %d | "
        "Forecast rows: %d | Horizons: %s",
        len(training),
        len(daily_forecast),
        horizon_values,
    )

    return MultiHorizonForecastResult(
        training=training,
        latest_state=latest_state,
        daily_forecast=daily_forecast,
        horizon_forecast=horizon_forecast,
        portfolio_forecast=portfolio_forecast,
        metrics=metrics,
        benchmark=benchmark,
        feature_importance=feature_importance,
    )
