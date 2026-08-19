from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HORIZON_DAYS = 90
WINDOWS: Tuple[int, ...] = (7, 14, 28, 56, 90)


@dataclass(frozen=True)
class CTRImpliedResult:
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
    required = {"date", "page", "clicks", "impressions"}
    missing = sorted(required.difference(historical.columns))
    if missing:
        raise ValueError(f"Historical source missing: {missing}")

    frame = historical.copy()
    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce"
    ).dt.normalize()
    frame["page"] = frame["page"].astype(str)

    for col in ("clicks", "impressions"):
        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).fillna(0.0)

    return (
        frame.dropna(subset=["date", "page"])
        .groupby(["date", "page"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .sort_values(["date", "page"])
        .reset_index(drop=True)
    )


def _panel(historical: pd.DataFrame, cutoff_date: pd.Timestamp):
    frame = _prepare_history(historical)
    frame = frame.loc[frame["date"] <= cutoff_date].copy()

    pages = frame["page"].drop_duplicates().tolist()
    dates = pd.date_range(frame["date"].min(), cutoff_date, freq="D")

    page_idx = {p: i for i, p in enumerate(pages)}
    date_idx = {d: i for i, d in enumerate(dates)}

    clicks = np.zeros((len(dates), len(pages)), dtype=np.float32)
    impressions = np.zeros((len(dates), len(pages)), dtype=np.float32)

    for row in frame.itertuples(index=False):
        di = date_idx[pd.Timestamp(row.date)]
        pi = page_idx[str(row.page)]
        clicks[di, pi] += float(row.clicks)
        impressions[di, pi] += float(row.impressions)

    first_seen = (
        frame.groupby("page")["date"]
        .min()
        .reindex(pages)
    )

    first_seen_idx = np.asarray(
        [
            int((pd.Timestamp(v) - dates[0]).days)
            for v in first_seen
        ],
        dtype=int,
    )

    return dates, clicks, impressions, first_seen_idx


def _aggregate_ctr(clicks: np.ndarray, impressions: np.ndarray) -> float:
    c = float(np.sum(clicks))
    i = float(np.sum(impressions))
    if i <= 0:
        return 0.0
    return float(c / i)


def _origin_features(
    dates: pd.DatetimeIndex,
    clicks: np.ndarray,
    impressions: np.ndarray,
    first_seen_idx: np.ndarray,
    origin_idx: int,
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    click_totals = clicks.sum(axis=1).astype(float)
    imp_totals = impressions.sum(axis=1).astype(float)

    for window in WINDOWS:
        start = origin_idx - window + 1
        c = click_totals[start:origin_idx + 1]
        i = imp_totals[start:origin_idx + 1]

        features[f"click_sum_{window}"] = float(c.sum())
        features[f"imp_sum_{window}"] = float(i.sum())
        features[f"ctr_{window}"] = (
            float(c.sum() / i.sum()) if i.sum() > 0 else 0.0
        )
        daily_ctr = np.divide(
            c,
            i,
            out=np.zeros_like(c, dtype=float),
            where=i > 0,
        )
        features[f"ctr_mean_daily_{window}"] = float(np.mean(daily_ctr))
        features[f"ctr_std_daily_{window}"] = float(np.std(daily_ctr))

    features["ctr_momentum_7_vs_28"] = (
        features["ctr_7"] / features["ctr_28"]
        if features["ctr_28"] > 0
        else 1.0
    )

    features["ctr_momentum_28_vs_56"] = (
        features["ctr_28"] / features["ctr_56"]
        if features["ctr_56"] > 0
        else 1.0
    )

    recent28_active = (
        impressions[origin_idx - 27:origin_idx + 1, :].sum(axis=0) > 0
    )
    prev28_active = (
        impressions[origin_idx - 55:origin_idx - 27, :].sum(axis=0) > 0
    )

    recent_count = float(recent28_active.sum())
    prev_count = float(prev28_active.sum())

    features["active_pages_28"] = recent_count
    features["active_page_growth_28"] = (
        recent_count / prev_count if prev_count > 0 else 1.0
    )

    known_pages = first_seen_idx <= origin_idx
    features["known_pages"] = float(known_pages.sum())

    new28 = (
        (first_seen_idx > origin_idx - 28)
        & (first_seen_idx <= origin_idx)
    )
    features["new_pages_28"] = float(new28.sum())

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


def _future_known_page_totals(
    clicks: np.ndarray,
    impressions: np.ndarray,
    first_seen_idx: np.ndarray,
    origin_idx: int,
    horizon_days: int,
) -> tuple[float, float, float]:
    known = first_seen_idx <= origin_idx

    future_clicks = float(
        clicks[
            origin_idx + 1:origin_idx + horizon_days + 1,
            :,
        ][:, known].sum()
    )

    future_impressions = float(
        impressions[
            origin_idx + 1:origin_idx + horizon_days + 1,
            :,
        ][:, known].sum()
    )

    future_ctr = (
        future_clicks / future_impressions
        if future_impressions > 0
        else 0.0
    )

    return future_clicks, future_impressions, future_ctr


def build_ctr_samples(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    dates, clicks, impressions, first_seen_idx = _panel(
        historical,
        cutoff_date,
    )

    first_origin = max(WINDOWS) - 1
    last_origin = len(dates) - horizon_days - 1

    rows = []

    # Weekly origins preserve independence better than daily overlapping targets.
    for origin_idx in range(first_origin, last_origin + 1, 7):
        features = _origin_features(
            dates,
            clicks,
            impressions,
            first_seen_idx,
            origin_idx,
        )

        future_clicks, future_impressions, future_ctr = (
            _future_known_page_totals(
                clicks,
                impressions,
                first_seen_idx,
                origin_idx,
                horizon_days,
            )
        )

        rows.append(
            {
                "OriginDate": dates[origin_idx],
                **features,
                "FutureClicks": future_clicks,
                "FutureImpressions": future_impressions,
                "TargetCTR": future_ctr,
            }
        )

    return pd.DataFrame(rows)


def _feature_columns(samples: pd.DataFrame) -> list[str]:
    excluded = {
        "OriginDate",
        "FutureClicks",
        "FutureImpressions",
        "TargetCTR",
    }

    return [
        str(col)
        for col in samples.columns
        if col not in excluded
    ]


def _logit(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 1e-5, 0.25)
    return np.log(x / (1.0 - x))


def _inv_logit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _models():
    return {
        "CTRRidge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        ),
        "CTRHGBR": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=12,
            l2_regularization=3.0,
            random_state=42,
        ),
        "CTRRandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "CTRGradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=42,
        ),
    }


def _walk_forward_select(samples: pd.DataFrame) -> pd.DataFrame:
    if len(samples) < 30:
        raise ValueError(
            f"Not enough CTR samples: {len(samples)}"
        )

    features = _feature_columns(samples)

    validation_size = 5
    starts = list(
        range(
            18,
            len(samples) - validation_size + 1,
            validation_size,
        )
    )[-3:]

    if len(starts) < 2:
        raise ValueError(
            "Not enough CTR samples for walk-forward validation."
        )

    records = []

    for name, model_template in _models().items():
        ctr_apes = []
        imp_apes = []
        imp_biases = []

        for start in starts:
            train = samples.iloc[:start].copy()
            valid = samples.iloc[start:start + validation_size].copy()

            X_train = train[features].astype(float)
            X_valid = valid[features].astype(float)

            y_train = _logit(
                pd.to_numeric(
                    train["TargetCTR"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(dtype=float)
            )

            import copy
            model = copy.deepcopy(model_template)
            model.fit(X_train, y_train)

            pred_ctr = _inv_logit(
                model.predict(X_valid)
            )
            pred_ctr = np.clip(pred_ctr, 0.005, 0.10)

            actual_ctr = pd.to_numeric(
                valid["TargetCTR"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            future_clicks = pd.to_numeric(
                valid["FutureClicks"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            actual_impressions = pd.to_numeric(
                valid["FutureImpressions"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            implied_impressions = np.divide(
                future_clicks,
                pred_ctr,
                out=np.zeros_like(future_clicks, dtype=float),
                where=pred_ctr > 0,
            )

            ctr_ape = (
                np.abs(pred_ctr - actual_ctr)
                / np.maximum(np.abs(actual_ctr), 1e-9)
                * 100.0
            )

            imp_ape = (
                np.abs(implied_impressions - actual_impressions)
                / np.maximum(np.abs(actual_impressions), 1e-9)
                * 100.0
            )

            ctr_apes.extend(ctr_ape.tolist())
            imp_apes.extend(imp_ape.tolist())
            imp_biases.append(
                _bias_pct(
                    actual_impressions,
                    implied_impressions,
                )
            )

        records.append(
            {
                "Candidate": name,
                "FoldCount": len(starts),
                "CTRMeanAPE": round(float(np.mean(ctr_apes)), 4),
                "CTRMedianAPE": round(float(np.median(ctr_apes)), 4),
                "ImpressionMeanAPE": round(float(np.mean(imp_apes)), 4),
                "ImpressionMedianAPE": round(float(np.median(imp_apes)), 4),
                "MeanAbsImpressionBiasPct": round(
                    float(np.mean(np.abs(imp_biases))),
                    4,
                ),
            }
        )

    # Robust recent CTR baselines.
    for candidate, col in (
        ("Recent28CTR", "ctr_28"),
        ("Recent56CTR", "ctr_56"),
        ("Recent90CTR", "ctr_90"),
    ):
        ctr_apes = []
        imp_apes = []
        imp_biases = []

        for start in starts:
            valid = samples.iloc[start:start + validation_size].copy()

            pred_ctr = pd.to_numeric(
                valid[col],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            pred_ctr = np.clip(pred_ctr, 0.005, 0.10)

            actual_ctr = pd.to_numeric(
                valid["TargetCTR"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            future_clicks = pd.to_numeric(
                valid["FutureClicks"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            actual_impressions = pd.to_numeric(
                valid["FutureImpressions"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            implied_impressions = np.divide(
                future_clicks,
                pred_ctr,
                out=np.zeros_like(future_clicks, dtype=float),
                where=pred_ctr > 0,
            )

            ctr_ape = (
                np.abs(pred_ctr - actual_ctr)
                / np.maximum(np.abs(actual_ctr), 1e-9)
                * 100.0
            )

            imp_ape = (
                np.abs(implied_impressions - actual_impressions)
                / np.maximum(np.abs(actual_impressions), 1e-9)
                * 100.0
            )

            ctr_apes.extend(ctr_ape.tolist())
            imp_apes.extend(imp_ape.tolist())
            imp_biases.append(
                _bias_pct(
                    actual_impressions,
                    implied_impressions,
                )
            )

        records.append(
            {
                "Candidate": candidate,
                "FoldCount": len(starts),
                "CTRMeanAPE": round(float(np.mean(ctr_apes)), 4),
                "CTRMedianAPE": round(float(np.median(ctr_apes)), 4),
                "ImpressionMeanAPE": round(float(np.mean(imp_apes)), 4),
                "ImpressionMedianAPE": round(float(np.median(imp_apes)), 4),
                "MeanAbsImpressionBiasPct": round(
                    float(np.mean(np.abs(imp_biases))),
                    4,
                ),
            }
        )

    ranking = (
        pd.DataFrame(records)
        .sort_values(
            [
                "ImpressionMeanAPE",
                "CTRMeanAPE",
                "MeanAbsImpressionBiasPct",
                "Candidate",
            ]
        )
        .reset_index(drop=True)
    )

    ranking["Selected"] = False
    ranking.loc[0, "Selected"] = True

    return ranking


def _final_ctr(
    samples: pd.DataFrame,
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    candidate: str,
) -> float:
    dates, clicks, impressions, first_seen_idx = _panel(
        historical,
        cutoff_date,
    )

    origin_idx = len(dates) - 1
    features = _origin_features(
        dates,
        clicks,
        impressions,
        first_seen_idx,
        origin_idx,
    )

    if candidate == "Recent28CTR":
        return float(np.clip(features["ctr_28"], 0.005, 0.10))
    if candidate == "Recent56CTR":
        return float(np.clip(features["ctr_56"], 0.005, 0.10))
    if candidate == "Recent90CTR":
        return float(np.clip(features["ctr_90"], 0.005, 0.10))

    models = _models()
    if candidate not in models:
        raise ValueError(f"Unknown CTR candidate: {candidate}")

    feature_cols = _feature_columns(samples)
    model = models[candidate]

    X_train = samples[feature_cols].astype(float)
    y_train = _logit(
        pd.to_numeric(
            samples["TargetCTR"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
    )

    model.fit(X_train, y_train)

    row = pd.DataFrame([features])[feature_cols].astype(float)
    predicted_ctr = float(
        _inv_logit(
            np.asarray([model.predict(row)[0]], dtype=float)
        )[0]
    )

    return float(np.clip(predicted_ctr, 0.005, 0.10))


def evaluate_ctr_implied_impressions(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    predicted_90d_click_total: float,
) -> CTRImpliedResult:
    samples = build_ctr_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=HORIZON_DAYS,
    )

    ranking = _walk_forward_select(samples)
    winner = str(ranking.iloc[0]["Candidate"])

    predicted_ctr = _final_ctr(
        samples=samples,
        historical=historical,
        cutoff_date=cutoff_date,
        candidate=winner,
    )

    implied_total = (
        float(predicted_90d_click_total) / predicted_ctr
        if predicted_ctr > 0
        else 0.0
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

    baseline_imp = pd.to_numeric(
        daily["PredictedImpressions"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    actual_imp = pd.to_numeric(
        daily["ActualImpressions"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    baseline_total = float(baseline_imp.sum())

    scale_factor = (
        implied_total / baseline_total
        if baseline_total > 0
        else 1.0
    )

    # Conservative operational safety bound.
    scale_factor = float(
        np.clip(scale_factor, 0.40, 1.10)
    )

    calibrated = baseline_imp * scale_factor
    daily["CTRImpliedPredictedImpressions"] = calibrated

    baseline_bias = _bias_pct(actual_imp, baseline_imp)
    calibrated_bias = _bias_pct(actual_imp, calibrated)

    summary = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "RecursiveDailyML",
                "SelectedCandidate": "",
                "Predicted90DClicks": np.nan,
                "PredictedCTR": np.nan,
                "ImpliedImpressionTotal": np.nan,
                "ScaleFactor": 1.0,
                "TotalErrorPct": round(abs(baseline_bias), 2),
                "WAPE": round(_wape(actual_imp, baseline_imp), 2),
                "BiasPct": round(baseline_bias, 2),
            },
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "CTRImpliedImpressionsML",
                "SelectedCandidate": winner,
                "Predicted90DClicks": round(float(predicted_90d_click_total), 2),
                "PredictedCTR": round(predicted_ctr, 6),
                "ImpliedImpressionTotal": round(implied_total, 2),
                "ScaleFactor": round(scale_factor, 6),
                "TotalErrorPct": round(abs(calibrated_bias), 2),
                "WAPE": round(_wape(actual_imp, calibrated), 2),
                "BiasPct": round(calibrated_bias, 2),
            },
        ]
    )

    ranking.insert(0, "Metric", "impressions")
    ranking.insert(0, "HorizonDays", 90)

    return CTRImpliedResult(
        summary=summary,
        selection=ranking,
        samples=samples,
        daily=daily,
    )
