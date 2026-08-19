from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


FEATURE_WINDOWS: Tuple[int, ...] = (7, 14, 28, 56)


@dataclass(frozen=True)
class DirectCalibrationResult:
    summary: pd.DataFrame
    daily: pd.DataFrame
    selection: pd.DataFrame
    samples: pd.DataFrame


def _aligned_arrays(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    actual_values = pd.to_numeric(
        pd.Series(actual).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    predicted_values = pd.to_numeric(
        pd.Series(predicted).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    if len(actual_values) != len(predicted_values):
        raise ValueError(
            f"Metric length mismatch: actual={len(actual_values)}, "
            f"predicted={len(predicted_values)}"
        )

    return actual_values, predicted_values


def _wape(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    actual_values, predicted_values = _aligned_arrays(
        actual,
        predicted,
    )

    denominator = float(
        np.abs(actual_values).sum()
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


def _bias_pct(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    actual_values, predicted_values = _aligned_arrays(
        actual,
        predicted,
    )

    denominator = float(
        actual_values.sum()
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            predicted_values.sum()
            - actual_values.sum()
        )
        / denominator
        * 100.0
    )


def _ape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    denominator = np.maximum(
        np.abs(actual),
        1e-9,
    )

    return (
        np.abs(
            predicted
            - actual
        )
        / denominator
        * 100.0
    )


def _prepare_history(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "date",
        "page",
        "clicks",
        "impressions",
    }

    missing = sorted(
        required.difference(
            historical.columns
        )
    )

    if missing:
        raise ValueError(
            f"Historical GSC source is missing columns: {missing}"
        )

    frame = historical.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()

    frame["page"] = (
        frame["page"]
        .astype(str)
    )

    for column in (
        "clicks",
        "impressions",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)

    return (
        frame
        .dropna(
            subset=[
                "date",
                "page",
            ]
        )
        .sort_values(
            [
                "date",
                "page",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _build_portfolio_matrix(
    historical: pd.DataFrame,
    metric: str,
    cutoff_date: pd.Timestamp,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    pd.Series,
]:
    frame = _prepare_history(
        historical
    )

    frame = frame.loc[
        frame["date"] <= cutoff_date
    ].copy()

    pages = (
        frame["page"]
        .drop_duplicates()
        .tolist()
    )

    dates = pd.date_range(
        frame["date"].min(),
        cutoff_date,
        freq="D",
    )

    page_index = {
        page: idx
        for idx, page
        in enumerate(pages)
    }

    date_index = {
        date: idx
        for idx, date
        in enumerate(dates)
    }

    matrix = np.zeros(
        (
            len(dates),
            len(pages),
        ),
        dtype=np.float64,
    )

    for row in frame[
        [
            "date",
            "page",
            metric,
        ]
    ].itertuples(
        index=False
    ):
        matrix[
            date_index[
                pd.Timestamp(
                    row.date
                )
            ],
            page_index[
                str(
                    row.page
                )
            ],
        ] += float(
            getattr(
                row,
                metric
            )
        )

    first_seen = (
        frame.groupby(
            "page"
        )[
            "date"
        ]
        .min()
    )

    first_seen_idx = np.asarray(
        [
            int(
                (
                    pd.Timestamp(
                        first_seen.loc[
                            page
                        ]
                    )
                    - dates[0]
                ).days
            )
            for page in pages
        ],
        dtype=int,
    )

    portfolio_series = pd.Series(
        matrix.sum(
            axis=1
        ),
        index=dates,
        dtype=float,
    )

    return (
        dates,
        matrix,
        first_seen_idx,
        portfolio_series,
    )


def _known_page_future_target(
    matrix: np.ndarray,
    first_seen_idx: np.ndarray,
    origin_idx: int,
    horizon_days: int,
) -> float:
    start = int(
        origin_idx
        + 1
    )

    end = int(
        origin_idx
        + horizon_days
        + 1
    )

    if end > matrix.shape[0]:
        raise ValueError(
            "Future target window exceeds available pre-cutoff matrix."
        )

    known_mask = (
        first_seen_idx
        <= origin_idx
    )

    if not np.any(
        known_mask
    ):
        return 0.0

    return float(
        matrix[
            start:end,
            :,
        ][
            :,
            known_mask,
        ]
        .sum()
    )


def _feature_values(
    portfolio_series: pd.Series,
    origin_idx: int,
) -> Dict[str, float]:
    values = portfolio_series.to_numpy(
        dtype=float
    )

    features: Dict[str, float] = {}

    for window in FEATURE_WINDOWS:
        window_values = values[
            origin_idx
            - window
            + 1:
            origin_idx
            + 1
        ]

        features[
            f"sum_{window}"
        ] = float(
            window_values.sum()
        )

        features[
            f"mean_{window}"
        ] = float(
            window_values.mean()
        )

        features[
            f"median_{window}"
        ] = float(
            np.median(
                window_values
            )
        )

        features[
            f"std_{window}"
        ] = float(
            np.std(
                window_values
            )
        )

    recent28 = values[
        origin_idx
        - 27:
        origin_idx
        + 1
    ]

    previous28 = values[
        origin_idx
        - 55:
        origin_idx
        - 27
    ]

    recent_sum = float(
        recent28.sum()
    )

    previous_sum = float(
        previous28.sum()
    )

    features[
        "momentum_28_vs_prev28"
    ] = (
        recent_sum
        / previous_sum
        if previous_sum > 0
        else 1.0
    )

    features[
        "last_value"
    ] = float(
        values[
            origin_idx
        ]
    )

    features[
        "trend_7_vs_28"
    ] = (
        features[
            "mean_7"
        ]
        / features[
            "mean_28"
        ]
        if features[
            "mean_28"
        ] > 0
        else 1.0
    )

    origin_date = pd.Timestamp(
        portfolio_series.index[
            origin_idx
        ]
    )

    features[
        "origin_month"
    ] = float(
        origin_date.month
    )

    features[
        "origin_dow"
    ] = float(
        origin_date.dayofweek
    )

    features[
        "month_sin"
    ] = float(
        np.sin(
            2
            * np.pi
            * origin_date.month
            / 12.0
        )
    )

    features[
        "month_cos"
    ] = float(
        np.cos(
            2
            * np.pi
            * origin_date.month
            / 12.0
        )
    )

    features[
        "doy_sin"
    ] = float(
        np.sin(
            2
            * np.pi
            * origin_date.dayofyear
            / 365.25
        )
    )

    features[
        "doy_cos"
    ] = float(
        np.cos(
            2
            * np.pi
            * origin_date.dayofyear
            / 365.25
        )
    )

    return features


def build_direct_training_samples(
    historical: pd.DataFrame,
    metric: str,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Build rolling-origin direct-horizon samples using only data on/before cutoff.

    Each target is the next N-day total for pages already known at that origin.
    This mirrors the existing backtest's known-page evaluation contract.
    """
    (
        dates,
        matrix,
        first_seen_idx,
        portfolio_series,
    ) = _build_portfolio_matrix(
        historical=historical,
        metric=metric,
        cutoff_date=cutoff_date,
    )

    max_window = max(
        FEATURE_WINDOWS
    )

    horizon = int(
        horizon_days
    )

    rows = []

    first_origin = (
        max_window
        - 1
    )

    last_origin = (
        len(
            dates
        )
        - horizon
        - 1
    )

    for origin_idx in range(
        first_origin,
        last_origin + 1,
    ):
        features = _feature_values(
            portfolio_series=portfolio_series,
            origin_idx=origin_idx,
        )

        target = _known_page_future_target(
            matrix=matrix,
            first_seen_idx=first_seen_idx,
            origin_idx=origin_idx,
            horizon_days=horizon,
        )

        rows.append(
            {
                "OriginDate": dates[
                    origin_idx
                ],
                **features,
                "TargetTotal": float(
                    target
                ),
                "Recent28RunRate": float(
                    features[
                        "mean_28"
                    ]
                    * horizon
                ),
                "Recent56RunRate": float(
                    features[
                        "mean_56"
                    ]
                    * horizon
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _models() -> Dict[str, object]:
    return {
        "DirectHGBR": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=42,
        ),
        "DirectRandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "DirectGradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        ),
    }


def _model_feature_columns(
    samples: pd.DataFrame,
) -> list[str]:
    excluded = {
        "OriginDate",
        "TargetTotal",
        "Recent28RunRate",
        "Recent56RunRate",
    }

    return [
        str(
            column
        )
        for column in samples.columns
        if column not in excluded
    ]


def _inner_select_candidate(
    samples: pd.DataFrame,
) -> tuple[
    str,
    pd.DataFrame,
]:
    if samples is None or samples.empty:
        raise ValueError(
            "No direct-horizon training samples."
        )

    minimum_training = 30

    validation_rows = min(
        45,
        max(
            15,
            int(
                len(
                    samples
                )
                * 0.25
            ),
        ),
    )

    if (
        len(
            samples
        )
        - validation_rows
        < minimum_training
    ):
        validation_rows = max(
            10,
            len(
                samples
            )
            - minimum_training
        )

    if validation_rows <= 0:
        raise ValueError(
            f"Not enough direct samples for inner validation: {len(samples)}"
        )

    train = samples.iloc[
        :-validation_rows
    ].copy()

    valid = samples.iloc[
        -validation_rows:
    ].copy()

    feature_columns = _model_feature_columns(
        samples
    )

    X_train = train[
        feature_columns
    ].astype(float)

    X_valid = valid[
        feature_columns
    ].astype(float)

    y_train = np.log1p(
        pd.to_numeric(
            train[
                "TargetTotal"
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(
            dtype=float
        )
    )

    y_valid = pd.to_numeric(
        valid[
            "TargetTotal"
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy(
        dtype=float
    )

    rows = []

    for name, model in _models().items():
        model.fit(
            X_train,
            y_train,
        )

        prediction = np.expm1(
            model.predict(
                X_valid
            )
        )

        prediction = np.maximum(
            0.0,
            prediction,
        )

        rows.append(
            {
                "Candidate": name,
                "InnerValidationRows": int(
                    validation_rows
                ),
                "MAPE": round(
                    float(
                        _ape(
                            y_valid,
                            prediction,
                        ).mean()
                    ),
                    4,
                ),
                "MedianAPE": round(
                    float(
                        np.median(
                            _ape(
                                y_valid,
                                prediction,
                            )
                        )
                    ),
                    4,
                ),
                "BiasPct": round(
                    _bias_pct(
                        y_valid,
                        prediction,
                    ),
                    4,
                ),
            }
        )

    for candidate_column in (
        "Recent28RunRate",
        "Recent56RunRate",
    ):
        prediction = pd.to_numeric(
            valid[
                candidate_column
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(
            dtype=float
        )

        rows.append(
            {
                "Candidate": candidate_column,
                "InnerValidationRows": int(
                    validation_rows
                ),
                "MAPE": round(
                    float(
                        _ape(
                            y_valid,
                            prediction,
                        ).mean()
                    ),
                    4,
                ),
                "MedianAPE": round(
                    float(
                        np.median(
                            _ape(
                                y_valid,
                                prediction,
                            )
                        )
                    ),
                    4,
                ),
                "BiasPct": round(
                    _bias_pct(
                        y_valid,
                        prediction,
                    ),
                    4,
                ),
            }
        )

    ranking = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "MAPE",
                "MedianAPE",
                "Candidate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        str(
            ranking.iloc[
                0
            ][
                "Candidate"
            ]
        ),
        ranking,
    )


def _final_direct_total(
    samples: pd.DataFrame,
    historical: pd.DataFrame,
    metric: str,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
    candidate: str,
) -> float:
    frame = _prepare_history(
        historical
    )

    frame = frame.loc[
        frame[
            "date"
        ] <= cutoff_date
    ].copy()

    daily = (
        frame.groupby(
            "date"
        )[
            metric
        ]
        .sum()
        .sort_index()
    )

    full_dates = pd.date_range(
        daily.index.min(),
        cutoff_date,
        freq="D",
    )

    daily = daily.reindex(
        full_dates,
        fill_value=0.0,
    )

    origin_features = _feature_values(
        portfolio_series=daily,
        origin_idx=len(
            daily
        ) - 1,
    )

    horizon = int(
        horizon_days
    )

    if candidate == "Recent28RunRate":
        return max(
            0.0,
            float(
                origin_features[
                    "mean_28"
                ]
                * horizon
            ),
        )

    if candidate == "Recent56RunRate":
        return max(
            0.0,
            float(
                origin_features[
                    "mean_56"
                ]
                * horizon
            ),
        )

    models = _models()

    if candidate not in models:
        raise ValueError(
            f"Unknown direct candidate: {candidate}"
        )

    feature_columns = _model_feature_columns(
        samples
    )

    model = models[
        candidate
    ]

    X_train = samples[
        feature_columns
    ].astype(float)

    y_train = np.log1p(
        pd.to_numeric(
            samples[
                "TargetTotal"
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(
            dtype=float
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    row = pd.DataFrame(
        [
            origin_features
        ]
    )[
        feature_columns
    ].astype(float)

    prediction = float(
        np.expm1(
            model.predict(
                row
            )[0]
        )
    )

    return max(
        0.0,
        prediction,
    )


def evaluate_direct_horizon_calibration(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
) -> DirectCalibrationResult:
    """
    Select a direct N-day portfolio total model using only pre-cutoff samples,
    then use that total to calibrate the existing recursive daily shape.

    The final 90/180 holdout is untouched until the final comparison.
    """
    horizon = int(
        horizon_days
    )

    baseline = baseline_daily.copy()

    baseline["ForecastDate"] = pd.to_datetime(
        baseline[
            "ForecastDate"
        ],
        errors="coerce",
    ).dt.normalize()

    baseline = (
        baseline
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

    all_summaries = []
    all_selections = []
    all_samples = []
    output = baseline.copy()

    for (
        metric,
        baseline_column,
        actual_column,
    ) in (
        (
            "clicks",
            "PredictedClicks",
            "ActualClicks",
        ),
        (
            "impressions",
            "PredictedImpressions",
            "ActualImpressions",
        ),
    ):
        samples = build_direct_training_samples(
            historical=historical,
            metric=metric,
            cutoff_date=cutoff_date,
            horizon_days=horizon,
        )

        if len(
            samples
        ) < 40:
            raise ValueError(
                f"Insufficient direct samples for {horizon}d {metric}: "
                f"{len(samples)}"
            )

        winner, ranking = _inner_select_candidate(
            samples
        )

        ranking.insert(
            0,
            "Metric",
            metric,
        )

        ranking.insert(
            0,
            "HorizonDays",
            horizon,
        )

        ranking[
            "Selected"
        ] = ranking[
            "Candidate"
        ].eq(
            winner
        )

        all_selections.append(
            ranking
        )

        sample_copy = samples.copy()
        sample_copy.insert(
            0,
            "Metric",
            metric,
        )
        sample_copy.insert(
            0,
            "HorizonDays",
            horizon,
        )
        all_samples.append(
            sample_copy
        )

        direct_total = _final_direct_total(
            samples=samples,
            historical=historical,
            metric=metric,
            cutoff_date=cutoff_date,
            horizon_days=horizon,
            candidate=winner,
        )

        baseline_values = pd.to_numeric(
            baseline[
                baseline_column
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(
            dtype=float
        )

        baseline_total = float(
            baseline_values.sum()
        )

        scale_factor = (
            direct_total
            / baseline_total
            if baseline_total > 0
            else 1.0
        )

        # Prevent pathological inner-model output from exploding the path.
        scale_factor = float(
            np.clip(
                scale_factor,
                0.40,
                1.60,
            )
        )

        calibrated_values = (
            baseline_values
            * scale_factor
        )

        output[
            f"DirectCalibrated{metric.title()}"
        ] = calibrated_values

        actual = pd.to_numeric(
            baseline[
                actual_column
            ],
            errors="coerce",
        ).fillna(0.0).to_numpy(
            dtype=float
        )

        baseline_bias = _bias_pct(
            actual,
            baseline_values,
        )

        calibrated_bias = _bias_pct(
            actual,
            calibrated_values,
        )

        all_summaries.extend(
            [
                {
                    "HorizonDays": horizon,
                    "Metric": metric,
                    "Method": "RecursiveDailyML",
                    "SelectedCandidate": "",
                    "DirectPredictedTotal": np.nan,
                    "ScaleFactor": 1.0,
                    "TotalErrorPct": round(
                        abs(
                            baseline_bias
                        ),
                        2,
                    ),
                    "WAPE": round(
                        _wape(
                            actual,
                            baseline_values,
                        ),
                        2,
                    ),
                    "BiasPct": round(
                        baseline_bias,
                        2,
                    ),
                },
                {
                    "HorizonDays": horizon,
                    "Metric": metric,
                    "Method": "DirectHorizonCalibratedML",
                    "SelectedCandidate": winner,
                    "DirectPredictedTotal": round(
                        direct_total,
                        2,
                    ),
                    "ScaleFactor": round(
                        scale_factor,
                        6,
                    ),
                    "TotalErrorPct": round(
                        abs(
                            calibrated_bias
                        ),
                        2,
                    ),
                    "WAPE": round(
                        _wape(
                            actual,
                            calibrated_values,
                        ),
                        2,
                    ),
                    "BiasPct": round(
                        calibrated_bias,
                        2,
                    ),
                },
            ]
        )

    output[
        "DirectCalibratedImpressions"
    ] = np.maximum(
        pd.to_numeric(
            output[
                "DirectCalibratedImpressions"
            ],
            errors="coerce",
        ).fillna(0.0),
        pd.to_numeric(
            output[
                "DirectCalibratedClicks"
            ],
            errors="coerce",
        ).fillna(0.0),
    )

    return DirectCalibrationResult(
        summary=pd.DataFrame(
            all_summaries
        ),
        daily=output,
        selection=pd.concat(
            all_selections,
            ignore_index=True,
        ),
        samples=pd.concat(
            all_samples,
            ignore_index=True,
        ),
    )
