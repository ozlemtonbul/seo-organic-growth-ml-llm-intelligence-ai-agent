from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategicCalibrationResult:
    daily: pd.DataFrame
    summary: pd.DataFrame
    anchor_diagnostics: pd.DataFrame


def _safe_pct(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(
        numerator
        / denominator
        * 100.0
    )


def _wape(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_values = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0.0)

    predicted_values = pd.to_numeric(
        predicted,
        errors="coerce",
    ).fillna(0.0)

    denominator = float(
        np.abs(
            actual_values
        ).sum()
    )

    if denominator == 0:
        return 0.0

    return float(
        np.abs(
            predicted_values
            - actual_values
        ).sum()
        / denominator
        * 100.0
    )


def _portfolio_training_series(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    metric: str,
) -> pd.Series:
    source = historical.copy()

    source["date"] = pd.to_datetime(
        source["date"],
        errors="coerce",
    ).dt.normalize()

    source[metric] = pd.to_numeric(
        source[metric],
        errors="coerce",
    ).fillna(0.0)

    training = source.loc[
        source["date"] <= cutoff_date
    ].copy()

    if training.empty:
        raise ValueError(
            f"No historical training rows are available for {metric}."
        )

    daily = (
        training
        .groupby(
            "date",
            as_index=True,
        )[metric]
        .sum()
        .sort_index()
    )

    complete_dates = pd.date_range(
        daily.index.min(),
        cutoff_date,
        freq="D",
    )

    return (
        daily
        .reindex(
            complete_dates,
            fill_value=0.0,
        )
        .astype(float)
    )


def _build_damped_weekday_anchor(
    training_series: pd.Series,
    forecast_dates: pd.Series,
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Robust strategic anchor using only pre-cutoff actual history.

    Components:
    - 28-day rolling median level
    - weekday seasonality from recent history
    - damped log trend
    - conservative clipping of trend and weekday factors

    No holdout/future actual value is used.
    """
    values = pd.to_numeric(
        training_series,
        errors="coerce",
    ).fillna(0.0).astype(float)

    if values.empty:
        raise ValueError(
            "Strategic anchor training series is empty."
        )

    smooth = (
        values
        .rolling(
            28,
            min_periods=min(
                7,
                len(values),
            ),
        )
        .median()
    )

    smooth = smooth.bfill().ffill()

    recent_level = float(
        smooth.tail(
            min(
                14,
                len(smooth),
            )
        ).median()
    )

    recent_level = max(
        0.0,
        recent_level,
    )

    seasonal_window = min(
        182,
        len(values),
    )

    seasonal_values = values.tail(
        seasonal_window
    )

    seasonal_smooth = smooth.reindex(
        seasonal_values.index
    ).replace(
        0.0,
        np.nan,
    )

    seasonal_ratio = (
        seasonal_values
        / seasonal_smooth
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    weekday_factor: Dict[int, float] = {}

    for weekday in range(7):
        mask = (
            seasonal_values.index.dayofweek
            == weekday
        )

        weekday_values = (
            seasonal_ratio.loc[
                mask
            ]
            .dropna()
        )

        factor = (
            float(
                weekday_values.median()
            )
            if not weekday_values.empty
            else 1.0
        )

        weekday_factor[
            weekday
        ] = float(
            np.clip(
                factor,
                0.75,
                1.25,
            )
        )

    factor_mean = float(
        np.mean(
            list(
                weekday_factor.values()
            )
        )
    )

    if factor_mean <= 0:
        factor_mean = 1.0

    weekday_factor = {
        key: float(
            value
            / factor_mean
        )
        for key, value
        in weekday_factor.items()
    }

    trend_window = min(
        84,
        len(smooth),
    )

    trend_values = (
        smooth.tail(
            trend_window
        )
        .clip(
            lower=0.0,
        )
    )

    if len(
        trend_values
    ) >= 14:
        x = np.arange(
            len(
                trend_values
            ),
            dtype=float,
        )

        y = np.log1p(
            trend_values.to_numpy(
                dtype=float
            )
        )

        slope = float(
            np.polyfit(
                x,
                y,
                1,
            )[0]
        )
    else:
        slope = 0.0

    # Prevent recent short-lived spikes from becoming runaway strategic trend.
    slope = float(
        np.clip(
            slope,
            -0.003,
            0.003,
        )
    )

    phi = 0.97
    anchors = []

    for step, forecast_date in enumerate(
        pd.to_datetime(
            forecast_dates,
            errors="coerce",
        ),
        start=1,
    ):
        damped_step = (
            (
                1.0
                - phi ** step
            )
            / (
                1.0
                - phi
            )
        )

        trend_factor = float(
            np.exp(
                slope
                * damped_step
            )
        )

        weekday = int(
            forecast_date.dayofweek
        )

        seasonal_factor = float(
            weekday_factor.get(
                weekday,
                1.0,
            )
        )

        anchor = (
            recent_level
            * trend_factor
            * seasonal_factor
        )

        anchors.append(
            max(
                0.0,
                float(
                    anchor
                ),
            )
        )

    diagnostics = {
        "RecentLevel": recent_level,
        "DailyLogTrendSlope": slope,
        "TrendPhi": phi,
        "SeasonalWindowDays": seasonal_window,
        "TrendWindowDays": trend_window,
    }

    return (
        np.asarray(
            anchors,
            dtype=float,
        ),
        diagnostics,
    )


def _reliability_weights(
    horizon_days: int,
    base_r2: float,
) -> np.ndarray:
    """
    Convert the existing model reliability logic into a daily recursive weight.

    The recursive ML path dominates early strategic days, then gradually
    shrinks toward the robust historical anchor as horizon uncertainty grows.
    """
    horizon = int(
        horizon_days
    )

    steps = np.arange(
        1,
        horizon + 1,
        dtype=float,
    )

    weights = (
        float(
            np.clip(
                base_r2,
                0.0,
                1.0,
            )
        )
        * np.exp(
            -0.0125
            * (
                steps
                - 1.0
            )
        )
    )

    return np.clip(
        weights,
        0.15,
        0.90,
    )


def evaluate_strategic_stabilizer(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    horizon_days: int,
    cutoff_date: pd.Timestamp,
    base_r2: float,
) -> StrategicCalibrationResult:
    """
    Evaluate a deterministic leakage-safe strategic stabilizer.

    Candidate forecast:
        reliability_weight * RecursiveDailyML
        + (1 - reliability_weight) * robust historical anchor

    The historical anchor is built strictly from rows on/before cutoff.
    """
    horizon = int(
        horizon_days
    )

    daily = baseline_daily.copy()

    daily["ForecastDate"] = pd.to_datetime(
        daily["ForecastDate"],
        errors="coerce",
    ).dt.normalize()

    daily = (
        daily
        .sort_values(
            "ForecastDate"
        )
        .head(
            horizon
        )
        .reset_index(
            drop=True
        )
    )

    required = {
        "ForecastDate",
        "PredictedClicks",
        "PredictedImpressions",
        "ActualClicks",
        "ActualImpressions",
    }

    missing = sorted(
        required.difference(
            daily.columns
        )
    )

    if missing:
        raise ValueError(
            "Baseline strategic daily file is missing columns: "
            f"{missing}"
        )

    click_training = (
        _portfolio_training_series(
            historical=historical,
            cutoff_date=cutoff_date,
            metric="clicks",
        )
    )

    impression_training = (
        _portfolio_training_series(
            historical=historical,
            cutoff_date=cutoff_date,
            metric="impressions",
        )
    )

    click_anchor, click_diagnostics = (
        _build_damped_weekday_anchor(
            training_series=click_training,
            forecast_dates=daily[
                "ForecastDate"
            ],
        )
    )

    impression_anchor, impression_diagnostics = (
        _build_damped_weekday_anchor(
            training_series=impression_training,
            forecast_dates=daily[
                "ForecastDate"
            ],
        )
    )

    recursive_weight = (
        _reliability_weights(
            horizon_days=horizon,
            base_r2=base_r2,
        )
    )

    recursive_clicks = pd.to_numeric(
        daily["PredictedClicks"],
        errors="coerce",
    ).fillna(0.0).to_numpy(
        dtype=float
    )

    recursive_impressions = pd.to_numeric(
        daily["PredictedImpressions"],
        errors="coerce",
    ).fillna(0.0).to_numpy(
        dtype=float
    )

    stabilized_clicks = (
        recursive_weight
        * recursive_clicks
        + (
            1.0
            - recursive_weight
        )
        * click_anchor
    )

    stabilized_impressions = (
        recursive_weight
        * recursive_impressions
        + (
            1.0
            - recursive_weight
        )
        * impression_anchor
    )

    stabilized_impressions = np.maximum(
        stabilized_impressions,
        stabilized_clicks,
    )

    daily[
        "RecursiveWeight"
    ] = recursive_weight

    daily[
        "HistoricalAnchorWeight"
    ] = (
        1.0
        - recursive_weight
    )

    daily[
        "AnchorClicks"
    ] = click_anchor

    daily[
        "AnchorImpressions"
    ] = impression_anchor

    daily[
        "StabilizedPredictedClicks"
    ] = stabilized_clicks

    daily[
        "StabilizedPredictedImpressions"
    ] = stabilized_impressions

    actual_clicks = pd.to_numeric(
        daily["ActualClicks"],
        errors="coerce",
    ).fillna(0.0)

    actual_impressions = pd.to_numeric(
        daily["ActualImpressions"],
        errors="coerce",
    ).fillna(0.0)

    baseline_clicks = pd.to_numeric(
        daily["PredictedClicks"],
        errors="coerce",
    ).fillna(0.0)

    baseline_impressions = pd.to_numeric(
        daily["PredictedImpressions"],
        errors="coerce",
    ).fillna(0.0)

    candidate_clicks = pd.Series(
        stabilized_clicks
    )

    candidate_impressions = pd.Series(
        stabilized_impressions
    )

    baseline_click_bias = _safe_pct(
        float(
            baseline_clicks.sum()
            - actual_clicks.sum()
        ),
        float(
            actual_clicks.sum()
        ),
    )

    candidate_click_bias = _safe_pct(
        float(
            candidate_clicks.sum()
            - actual_clicks.sum()
        ),
        float(
            actual_clicks.sum()
        ),
    )

    baseline_impression_bias = _safe_pct(
        float(
            baseline_impressions.sum()
            - actual_impressions.sum()
        ),
        float(
            actual_impressions.sum()
        ),
    )

    candidate_impression_bias = _safe_pct(
        float(
            candidate_impressions.sum()
            - actual_impressions.sum()
        ),
        float(
            actual_impressions.sum()
        ),
    )

    summary = pd.DataFrame(
        [
            {
                "HorizonDays": horizon,
                "Method": "RecursiveDailyML",
                "ClickTotalErrorPct": round(
                    abs(
                        baseline_click_bias
                    ),
                    2,
                ),
                "ClickWAPE": round(
                    _wape(
                        actual_clicks,
                        baseline_clicks,
                    ),
                    2,
                ),
                "ClickBiasPct": round(
                    baseline_click_bias,
                    2,
                ),
                "ImpressionTotalErrorPct": round(
                    abs(
                        baseline_impression_bias
                    ),
                    2,
                ),
                "ImpressionWAPE": round(
                    _wape(
                        actual_impressions,
                        baseline_impressions,
                    ),
                    2,
                ),
                "ImpressionBiasPct": round(
                    baseline_impression_bias,
                    2,
                ),
            },
            {
                "HorizonDays": horizon,
                "Method": "StrategicDampedEnsembleML",
                "ClickTotalErrorPct": round(
                    abs(
                        candidate_click_bias
                    ),
                    2,
                ),
                "ClickWAPE": round(
                    _wape(
                        actual_clicks,
                        candidate_clicks,
                    ),
                    2,
                ),
                "ClickBiasPct": round(
                    candidate_click_bias,
                    2,
                ),
                "ImpressionTotalErrorPct": round(
                    abs(
                        candidate_impression_bias
                    ),
                    2,
                ),
                "ImpressionWAPE": round(
                    _wape(
                        actual_impressions,
                        candidate_impressions,
                    ),
                    2,
                ),
                "ImpressionBiasPct": round(
                    candidate_impression_bias,
                    2,
                ),
            },
        ]
    )

    diagnostics = pd.DataFrame(
        [
            {
                "HorizonDays": horizon,
                "Metric": "Clicks",
                **click_diagnostics,
                "BaseR2": float(
                    base_r2
                ),
                "MinRecursiveWeight": float(
                    recursive_weight.min()
                ),
                "MaxRecursiveWeight": float(
                    recursive_weight.max()
                ),
            },
            {
                "HorizonDays": horizon,
                "Metric": "Impressions",
                **impression_diagnostics,
                "BaseR2": float(
                    base_r2
                ),
                "MinRecursiveWeight": float(
                    recursive_weight.min()
                ),
                "MaxRecursiveWeight": float(
                    recursive_weight.max()
                ),
            },
        ]
    )

    return StrategicCalibrationResult(
        daily=daily,
        summary=summary,
        anchor_diagnostics=diagnostics,
    )
