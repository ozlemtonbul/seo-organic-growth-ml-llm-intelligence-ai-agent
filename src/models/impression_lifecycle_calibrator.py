from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


HORIZON_DAYS = 90
WINDOWS: Tuple[int, ...] = (7, 14, 28, 56, 90)


@dataclass(frozen=True)
class LifecycleCalibrationResult:
    summary: pd.DataFrame
    selection: pd.DataFrame
    samples: pd.DataFrame
    daily: pd.DataFrame


def _wape(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    den = float(np.abs(a).sum())
    if den == 0:
        return 0.0
    return float(np.abs(p - a).sum() / den * 100.0)


def _bias_pct(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    den = float(a.sum())
    if den == 0:
        return 0.0
    return float((p.sum() - a.sum()) / den * 100.0)


def _prepare_history(historical: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "page", "impressions"}
    missing = sorted(required.difference(historical.columns))
    if missing:
        raise ValueError(f"Historical source missing: {missing}")

    frame = historical.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()
    frame["page"] = frame["page"].astype(str)
    frame["impressions"] = pd.to_numeric(
        frame["impressions"],
        errors="coerce",
    ).fillna(0.0)

    return (
        frame.dropna(subset=["date", "page"])
        .groupby(["date", "page"], as_index=False)
        .agg(impressions=("impressions", "sum"))
        .sort_values(["date", "page"])
        .reset_index(drop=True)
    )


def _daily_page_panel(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
):
    frame = _prepare_history(historical)
    frame = frame.loc[frame["date"] <= cutoff_date].copy()

    pages = frame["page"].drop_duplicates().tolist()
    dates = pd.date_range(frame["date"].min(), cutoff_date, freq="D")

    page_to_idx = {p: i for i, p in enumerate(pages)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    matrix = np.zeros((len(dates), len(pages)), dtype=np.float32)

    for row in frame.itertuples(index=False):
        matrix[
            date_to_idx[pd.Timestamp(row.date)],
            page_to_idx[str(row.page)],
        ] += float(row.impressions)

    first_seen = (
        frame.groupby("page")["date"]
        .min()
        .reindex(pages)
    )
    first_seen_idx = np.asarray(
        [
            int((pd.Timestamp(value) - dates[0]).days)
            for value in first_seen
        ],
        dtype=int,
    )

    return dates, pages, matrix, first_seen_idx


def _origin_features(
    dates: pd.DatetimeIndex,
    matrix: np.ndarray,
    first_seen_idx: np.ndarray,
    origin_idx: int,
) -> Dict[str, float]:
    totals = matrix.sum(axis=1).astype(float)
    active_counts = (matrix > 0).sum(axis=1).astype(float)

    features: Dict[str, float] = {}

    for window in WINDOWS:
        start = origin_idx - window + 1
        values = totals[start:origin_idx + 1]
        active = active_counts[start:origin_idx + 1]

        features[f"imp_sum_{window}"] = float(values.sum())
        features[f"imp_mean_{window}"] = float(values.mean())
        features[f"imp_median_{window}"] = float(np.median(values))
        features[f"imp_std_{window}"] = float(np.std(values))
        features[f"active_mean_{window}"] = float(active.mean())
        features[f"active_last_{window}"] = float(active[-1])

    recent28_active_pages = (
        matrix[origin_idx - 27:origin_idx + 1, :].sum(axis=0) > 0
    )
    prev28_active_pages = (
        matrix[origin_idx - 55:origin_idx - 27, :].sum(axis=0) > 0
    )

    recent28_count = float(recent28_active_pages.sum())
    prev28_count = float(prev28_active_pages.sum())

    features["active_page_growth_28"] = (
        recent28_count / prev28_count
        if prev28_count > 0
        else 1.0
    )

    features["recent28_active_pages"] = recent28_count
    features["prev28_active_pages"] = prev28_count

    recent28_total = float(
        matrix[origin_idx - 27:origin_idx + 1, :].sum()
    )
    prev28_total = float(
        matrix[origin_idx - 55:origin_idx - 27, :].sum()
    )

    features["impression_momentum_28"] = (
        recent28_total / prev28_total
        if prev28_total > 0
        else 1.0
    )

    features["imp_per_active_page_28"] = (
        recent28_total / recent28_count
        if recent28_count > 0
        else 0.0
    )

    recent56 = matrix[origin_idx - 55:origin_idx + 1, :]
    page_totals56 = recent56.sum(axis=0).astype(float)
    positive = page_totals56[page_totals56 > 0]

    if len(positive):
        sorted_values = np.sort(positive)[::-1]
        total_positive = float(sorted_values.sum())
        top10_count = min(10, len(sorted_values))
        features["top10_share_56"] = (
            float(sorted_values[:top10_count].sum()) / total_positive
            if total_positive > 0
            else 0.0
        )
        features["page_median_imp_56"] = float(np.median(positive))
        features["page_p90_imp_56"] = float(np.quantile(positive, 0.90))
    else:
        features["top10_share_56"] = 0.0
        features["page_median_imp_56"] = 0.0
        features["page_p90_imp_56"] = 0.0

    known_pages = first_seen_idx <= origin_idx
    features["known_pages"] = float(known_pages.sum())

    new_28 = (
        (first_seen_idx > origin_idx - 28)
        & (first_seen_idx <= origin_idx)
    )
    new_56 = (
        (first_seen_idx > origin_idx - 56)
        & (first_seen_idx <= origin_idx)
    )

    features["new_pages_28"] = float(new_28.sum())
    features["new_pages_56"] = float(new_56.sum())

    origin_date = pd.Timestamp(dates[origin_idx])
    features["month"] = float(origin_date.month)
    features["dow"] = float(origin_date.dayofweek)
    features["month_sin"] = float(
        np.sin(2 * np.pi * origin_date.month / 12.0)
    )
    features["month_cos"] = float(
        np.cos(2 * np.pi * origin_date.month / 12.0)
    )
    features["doy_sin"] = float(
        np.sin(2 * np.pi * origin_date.dayofyear / 365.25)
    )
    features["doy_cos"] = float(
        np.cos(2 * np.pi * origin_date.dayofyear / 365.25)
    )

    return features


def _known_page_target(
    matrix: np.ndarray,
    first_seen_idx: np.ndarray,
    origin_idx: int,
    horizon_days: int,
) -> float:
    known = first_seen_idx <= origin_idx
    future = matrix[
        origin_idx + 1:
        origin_idx + horizon_days + 1,
        :,
    ]
    return float(
        future[:, known].sum()
    )


def build_samples(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    dates, pages, matrix, first_seen_idx = _daily_page_panel(
        historical,
        cutoff_date,
    )

    max_window = max(WINDOWS)
    horizon = int(horizon_days)

    first_origin = max_window - 1
    last_origin = len(dates) - horizon - 1

    rows = []

    for origin_idx in range(first_origin, last_origin + 1, 7):
        features = _origin_features(
            dates,
            matrix,
            first_seen_idx,
            origin_idx,
        )

        target = _known_page_target(
            matrix,
            first_seen_idx,
            origin_idx,
            horizon,
        )

        recent28_run_rate = (
            features["imp_mean_28"] * horizon
        )
        recent56_run_rate = (
            features["imp_mean_56"] * horizon
        )
        recent90_run_rate = (
            features["imp_mean_90"] * horizon
        )

        rows.append(
            {
                "OriginDate": dates[origin_idx],
                **features,
                "TargetTotal": target,
                "Recent28RunRate": recent28_run_rate,
                "Recent56RunRate": recent56_run_rate,
                "Recent90RunRate": recent90_run_rate,
                "TargetVsRecent28": (
                    target / recent28_run_rate
                    if recent28_run_rate > 0
                    else 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def _feature_columns(samples: pd.DataFrame) -> list[str]:
    excluded = {
        "OriginDate",
        "TargetTotal",
        "Recent28RunRate",
        "Recent56RunRate",
        "Recent90RunRate",
        "TargetVsRecent28",
    }
    return [
        str(column)
        for column in samples.columns
        if column not in excluded
    ]


def _models():
    return {
        "LifecycleHGBR": HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=12,
            l2_regularization=3.0,
            random_state=42,
        ),
        "LifecycleRF": RandomForestRegressor(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "LifecycleExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "LifecycleGBR": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=42,
        ),
        "LifecycleRidge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        ),
    }


def _walk_forward_rank(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    sample_count = len(samples)

    # With the currently available 499-day GSC history and a 90-day target,
    # weekly rolling origins yield about 33 leakage-safe samples.
    # This is sufficient for a cautious development evaluation, but not for
    # a high-confidence production claim. Use smaller expanding validation
    # folds and require at least 30 samples.
    if sample_count < 30:
        raise ValueError(
            f"Not enough lifecycle samples: {sample_count}. "
            "At least 30 leakage-safe rolling-origin samples are required."
        )

    features = _feature_columns(samples)

    # Multiple expanding pre-cutoff folds. Final holdout is never used.
    # For 30-39 samples, use 5-row validation windows and 3 expanding folds.
    # Larger histories can still use up to 4 folds.
    validation_size = 5 if sample_count < 40 else 6
    minimum_train_rows = 18 if sample_count < 40 else 20

    possible_starts = list(
        range(
            minimum_train_rows,
            sample_count - validation_size + 1,
            validation_size,
        )
    )

    fold_starts = possible_starts[-4:]

    if len(fold_starts) < 2:
        raise ValueError(
            "Not enough lifecycle samples for multi-fold selection. "
            f"Samples={sample_count}, validation_size={validation_size}, "
            f"minimum_train_rows={minimum_train_rows}."
        )

    records = []

    for candidate_name, model_template in _models().items():
        fold_apes = []
        fold_biases = []

        for start in fold_starts:
            train = samples.iloc[:start].copy()
            valid = samples.iloc[start:start + validation_size].copy()

            X_train = train[features].astype(float)
            X_valid = valid[features].astype(float)

            y_train_ratio = np.log(
                pd.to_numeric(
                    train["TargetVsRecent28"],
                    errors="coerce",
                )
                .clip(lower=0.35, upper=2.50)
                .to_numpy(dtype=float)
            )

            actual = pd.to_numeric(
                valid["TargetTotal"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            baseline = pd.to_numeric(
                valid["Recent28RunRate"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            import copy
            model = copy.deepcopy(model_template)
            model.fit(X_train, y_train_ratio)

            ratio = np.exp(model.predict(X_valid))
            ratio = np.clip(ratio, 0.50, 1.50)
            prediction = baseline * ratio

            ape = np.abs(prediction - actual) / np.maximum(
                np.abs(actual),
                1e-9,
            ) * 100.0

            fold_apes.extend(ape.tolist())
            fold_biases.append(
                _bias_pct(actual, prediction)
            )

        records.append(
            {
                "Candidate": candidate_name,
                "FoldCount": len(fold_starts),
                "MeanAPE": round(float(np.mean(fold_apes)), 4),
                "MedianAPE": round(float(np.median(fold_apes)), 4),
                "P90APE": round(float(np.quantile(fold_apes, 0.90)), 4),
                "MeanAbsBiasPct": round(
                    float(np.mean(np.abs(fold_biases))),
                    4,
                ),
            }
        )

    # Robust non-ML baselines.
    for candidate in (
        "Recent28RunRate",
        "Recent56RunRate",
        "Recent90RunRate",
    ):
        fold_apes = []
        fold_biases = []

        for start in fold_starts:
            valid = samples.iloc[start:start + validation_size].copy()
            actual = pd.to_numeric(
                valid["TargetTotal"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            prediction = pd.to_numeric(
                valid[candidate],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            ape = np.abs(prediction - actual) / np.maximum(
                np.abs(actual),
                1e-9,
            ) * 100.0

            fold_apes.extend(ape.tolist())
            fold_biases.append(
                _bias_pct(actual, prediction)
            )

        records.append(
            {
                "Candidate": candidate,
                "FoldCount": len(fold_starts),
                "MeanAPE": round(float(np.mean(fold_apes)), 4),
                "MedianAPE": round(float(np.median(fold_apes)), 4),
                "P90APE": round(float(np.quantile(fold_apes, 0.90)), 4),
                "MeanAbsBiasPct": round(
                    float(np.mean(np.abs(fold_biases))),
                    4,
                ),
            }
        )

    ranking = (
        pd.DataFrame(records)
        .sort_values(
            [
                "MeanAPE",
                "P90APE",
                "MeanAbsBiasPct",
                "Candidate",
            ]
        )
        .reset_index(drop=True)
    )

    ranking["Selected"] = False
    ranking.loc[0, "Selected"] = True

    return ranking


def _final_total(
    samples: pd.DataFrame,
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    candidate: str,
) -> tuple[float, float]:
    dates, pages, matrix, first_seen_idx = _daily_page_panel(
        historical,
        cutoff_date,
    )

    origin_idx = len(dates) - 1
    features = _origin_features(
        dates,
        matrix,
        first_seen_idx,
        origin_idx,
    )

    recent28 = features["imp_mean_28"] * HORIZON_DAYS
    recent56 = features["imp_mean_56"] * HORIZON_DAYS
    recent90 = features["imp_mean_90"] * HORIZON_DAYS

    if candidate == "Recent28RunRate":
        return float(recent28), 1.0
    if candidate == "Recent56RunRate":
        return float(recent56), 1.0
    if candidate == "Recent90RunRate":
        return float(recent90), 1.0

    models = _models()
    if candidate not in models:
        raise ValueError(
            f"Unknown candidate: {candidate}"
        )

    features_cols = _feature_columns(samples)

    model = models[candidate]
    X_train = samples[features_cols].astype(float)

    y_ratio = np.log(
        pd.to_numeric(
            samples["TargetVsRecent28"],
            errors="coerce",
        )
        .clip(lower=0.35, upper=2.50)
        .to_numpy(dtype=float)
    )

    model.fit(
        X_train,
        y_ratio,
    )

    row = pd.DataFrame([features])[features_cols].astype(float)
    ratio = float(
        np.exp(
            model.predict(row)[0]
        )
    )
    ratio = float(
        np.clip(
            ratio,
            0.50,
            1.50,
        )
    )

    return float(recent28 * ratio), ratio


def evaluate_90d_impression_lifecycle(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> LifecycleCalibrationResult:
    samples = build_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=HORIZON_DAYS,
    )

    ranking = _walk_forward_rank(
        samples
    )

    winner = str(
        ranking.iloc[0]["Candidate"]
    )

    direct_total, ratio = _final_total(
        samples=samples,
        historical=historical,
        cutoff_date=cutoff_date,
        candidate=winner,
    )

    daily = baseline_daily.copy()
    daily["ForecastDate"] = pd.to_datetime(
        daily["ForecastDate"],
        errors="coerce",
    ).dt.normalize()
    daily = (
        daily.sort_values("ForecastDate")
        .head(HORIZON_DAYS)
        .reset_index(drop=True)
    )

    baseline = pd.to_numeric(
        daily["PredictedImpressions"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    actual = pd.to_numeric(
        daily["ActualImpressions"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    baseline_total = float(baseline.sum())

    scale_factor = (
        direct_total / baseline_total
        if baseline_total > 0
        else 1.0
    )

    # Safety bound; prevents a one-fold outlier from over-correcting production path.
    scale_factor = float(
        np.clip(
            scale_factor,
            0.45,
            1.10,
        )
    )

    calibrated = baseline * scale_factor

    daily["LifecycleCalibratedImpressions"] = calibrated

    baseline_bias = _bias_pct(
        actual,
        baseline,
    )
    calibrated_bias = _bias_pct(
        actual,
        calibrated,
    )

    summary = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "RecursiveDailyML",
                "SelectedCandidate": "",
                "DirectPredictedTotal": np.nan,
                "ScaleFactor": 1.0,
                "TotalErrorPct": round(abs(baseline_bias), 2),
                "WAPE": round(_wape(actual, baseline), 2),
                "BiasPct": round(baseline_bias, 2),
            },
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "LifecycleCalibratedML",
                "SelectedCandidate": winner,
                "DirectPredictedTotal": round(direct_total, 2),
                "ScaleFactor": round(scale_factor, 6),
                "TotalErrorPct": round(abs(calibrated_bias), 2),
                "WAPE": round(_wape(actual, calibrated), 2),
                "BiasPct": round(calibrated_bias, 2),
            },
        ]
    )

    ranking.insert(0, "Metric", "impressions")
    ranking.insert(0, "HorizonDays", 90)

    return LifecycleCalibrationResult(
        summary=summary,
        selection=ranking,
        samples=samples,
        daily=daily,
    )
