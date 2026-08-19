from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import copy

import numpy as np
import pandas as pd

from src.models.ctr_implied_impressions import (
    HORIZON_DAYS,
    _feature_columns,
    _final_ctr,
    _inv_logit,
    _logit,
    _models,
    _wape,
    _bias_pct,
    build_ctr_samples,
)


ML_CANDIDATES = (
    "CTRRidge",
    "CTRGradientBoosting",
    "CTRRandomForest",
    "CTRHGBR",
)


@dataclass(frozen=True)
class MLOnlyCTREnsembleResult:
    summary: pd.DataFrame
    base_ranking: pd.DataFrame
    ensemble_ranking: pd.DataFrame
    final_predictions: pd.DataFrame
    daily: pd.DataFrame


def _fold_starts(sample_count: int) -> List[int]:
    validation_size = 5
    starts = list(
        range(
            18,
            sample_count - validation_size + 1,
            validation_size,
        )
    )[-3:]

    if len(starts) < 2:
        raise ValueError(
            f"Not enough CTR samples for ML-only walk-forward: {sample_count}"
        )

    return starts


def _ml_oof_predictions(
    samples: pd.DataFrame,
    starts: List[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_columns(samples)
    validation_size = 5

    rows = []
    metrics = []

    all_models = _models()

    for candidate_name in ML_CANDIDATES:
        model_template = all_models[candidate_name]
        candidate_rows = []

        for fold_no, start in enumerate(starts, start=1):
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

            model = copy.deepcopy(model_template)
            model.fit(X_train, y_train)

            predicted_ctr = _inv_logit(
                model.predict(X_valid)
            )
            predicted_ctr = np.clip(
                predicted_ctr,
                0.005,
                0.10,
            )

            for local_idx, (_, row) in enumerate(valid.iterrows()):
                candidate_rows.append(
                    {
                        "Candidate": candidate_name,
                        "Fold": fold_no,
                        "OriginDate": pd.Timestamp(row["OriginDate"]),
                        "PredictedCTR": float(predicted_ctr[local_idx]),
                        "ActualCTR": float(row["TargetCTR"]),
                        "FutureClicks": float(row["FutureClicks"]),
                        "FutureImpressions": float(row["FutureImpressions"]),
                    }
                )

        candidate_df = pd.DataFrame(candidate_rows)

        implied_impressions = (
            candidate_df["FutureClicks"].to_numpy(dtype=float)
            / candidate_df["PredictedCTR"].to_numpy(dtype=float)
        )
        actual_impressions = (
            candidate_df["FutureImpressions"].to_numpy(dtype=float)
        )

        ape = (
            np.abs(implied_impressions - actual_impressions)
            / np.maximum(np.abs(actual_impressions), 1e-9)
            * 100.0
        )

        metrics.append(
            {
                "Candidate": candidate_name,
                "ImpressionMeanAPE": round(float(np.mean(ape)), 4),
                "ImpressionMedianAPE": round(float(np.median(ape)), 4),
                "P90APE": round(float(np.quantile(ape, 0.90)), 4),
                "ImpressionBiasPct": round(
                    _bias_pct(actual_impressions, implied_impressions),
                    4,
                ),
            }
        )

        rows.extend(candidate_rows)

    ranking = (
        pd.DataFrame(metrics)
        .sort_values(
            [
                "ImpressionMeanAPE",
                "P90APE",
                "Candidate",
            ]
        )
        .reset_index(drop=True)
    )

    return pd.DataFrame(rows), ranking


def _wide_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "Fold",
        "OriginDate",
        "ActualCTR",
        "FutureClicks",
        "FutureImpressions",
    ]

    wide = (
        predictions.pivot_table(
            index=keys,
            columns="Candidate",
            values="PredictedCTR",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide


def _ensemble_rank(
    wide: pd.DataFrame,
    base_ranking: pd.DataFrame,
) -> pd.DataFrame:
    ordered = (
        base_ranking["Candidate"]
        .astype(str)
        .tolist()
    )

    errors = (
        base_ranking
        .set_index("Candidate")["ImpressionMeanAPE"]
        .to_dict()
    )

    def weighted(row):
        weights = np.asarray(
            [
                1.0 / max(float(errors[name]), 0.25) ** 2
                for name in ordered
            ],
            dtype=float,
        )
        values = np.asarray(
            [float(row[name]) for name in ordered],
            dtype=float,
        )
        return float(
            np.sum(values * weights)
            / np.sum(weights)
        )

    aggregators = {
        "MLWeightedCTR": lambda row: weighted(row),
        "MLMedianCTR": lambda row: float(
            np.median([float(row[name]) for name in ordered])
        ),
        "MLQ60CTR": lambda row: float(
            np.quantile([float(row[name]) for name in ordered], 0.60)
        ),
        "MLQ70CTR": lambda row: float(
            np.quantile([float(row[name]) for name in ordered], 0.70)
        ),
        "MLQ80CTR": lambda row: float(
            np.quantile([float(row[name]) for name in ordered], 0.80)
        ),
    }

    actual_impressions = wide["FutureImpressions"].to_numpy(dtype=float)
    future_clicks = wide["FutureClicks"].to_numpy(dtype=float)

    rows = []

    for ensemble_name, aggregator in aggregators.items():
        ctr = np.asarray(
            [
                np.clip(
                    aggregator(row),
                    0.005,
                    0.10,
                )
                for _, row in wide.iterrows()
            ],
            dtype=float,
        )

        implied = future_clicks / ctr

        # Bias correction is learned strictly from OOF pre-cutoff observations.
        correction = float(
            np.median(
                np.divide(
                    actual_impressions,
                    implied,
                    out=np.ones_like(actual_impressions),
                    where=implied > 0,
                )
            )
        )
        correction = float(
            np.clip(
                correction,
                0.85,
                1.15,
            )
        )

        corrected = implied * correction

        ape = (
            np.abs(corrected - actual_impressions)
            / np.maximum(np.abs(actual_impressions), 1e-9)
            * 100.0
        )

        rows.append(
            {
                "Ensemble": ensemble_name,
                "Members": ",".join(ordered),
                "BiasCorrection": round(correction, 6),
                "ImpressionMeanAPE": round(float(np.mean(ape)), 4),
                "ImpressionMedianAPE": round(float(np.median(ape)), 4),
                "P90APE": round(float(np.quantile(ape, 0.90)), 4),
                "ImpressionBiasPct": round(
                    _bias_pct(actual_impressions, corrected),
                    4,
                ),
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "ImpressionMeanAPE",
                "P90APE",
                "Ensemble",
            ]
        )
        .reset_index(drop=True)
    )

    ranking["Selected"] = False
    ranking.loc[0, "Selected"] = True

    return ranking


def _final_ml_ctrs(
    samples: pd.DataFrame,
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Dict[str, float]:
    return {
        name: float(
            _final_ctr(
                samples=samples,
                historical=historical,
                cutoff_date=cutoff_date,
                candidate=name,
            )
        )
        for name in ML_CANDIDATES
    }


def _final_ensemble_ctr(
    final_ctrs: Dict[str, float],
    base_ranking: pd.DataFrame,
    ensemble_name: str,
) -> float:
    ordered = (
        base_ranking["Candidate"]
        .astype(str)
        .tolist()
    )

    values = np.asarray(
        [float(final_ctrs[name]) for name in ordered],
        dtype=float,
    )

    errors = (
        base_ranking
        .set_index("Candidate")["ImpressionMeanAPE"]
        .to_dict()
    )

    if ensemble_name == "MLWeightedCTR":
        weights = np.asarray(
            [
                1.0 / max(float(errors[name]), 0.25) ** 2
                for name in ordered
            ],
            dtype=float,
        )
        result = float(
            np.sum(values * weights)
            / np.sum(weights)
        )
    elif ensemble_name == "MLMedianCTR":
        result = float(np.median(values))
    elif ensemble_name == "MLQ60CTR":
        result = float(np.quantile(values, 0.60))
    elif ensemble_name == "MLQ70CTR":
        result = float(np.quantile(values, 0.70))
    elif ensemble_name == "MLQ80CTR":
        result = float(np.quantile(values, 0.80))
    else:
        raise ValueError(
            f"Unknown ML ensemble: {ensemble_name}"
        )

    return float(
        np.clip(result, 0.005, 0.10)
    )


def evaluate_ml_only_ctr_ensemble(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    predicted_90d_click_total: float,
) -> MLOnlyCTREnsembleResult:
    samples = build_ctr_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=HORIZON_DAYS,
    )

    starts = _fold_starts(len(samples))

    predictions, base_ranking = _ml_oof_predictions(
        samples,
        starts,
    )

    wide = _wide_predictions(predictions)

    ensemble_ranking = _ensemble_rank(
        wide,
        base_ranking,
    )

    winner = str(
        ensemble_ranking.iloc[0]["Ensemble"]
    )

    correction = float(
        ensemble_ranking.iloc[0]["BiasCorrection"]
    )

    final_ctrs = _final_ml_ctrs(
        samples=samples,
        historical=historical,
        cutoff_date=cutoff_date,
    )

    ensemble_ctr = _final_ensemble_ctr(
        final_ctrs=final_ctrs,
        base_ranking=base_ranking,
        ensemble_name=winner,
    )

    implied_total = (
        float(predicted_90d_click_total)
        / ensemble_ctr
        * correction
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

    scale_factor = float(
        np.clip(scale_factor, 0.40, 1.10)
    )

    calibrated = baseline_imp * scale_factor
    daily["MLOnlyCTRPredictedImpressions"] = calibrated

    baseline_bias = _bias_pct(
        actual_imp,
        baseline_imp,
    )
    candidate_bias = _bias_pct(
        actual_imp,
        calibrated,
    )

    summary = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "RecursiveDailyML",
                "SelectedEnsemble": "",
                "Predicted90DClicks": np.nan,
                "PredictedCTR": np.nan,
                "BiasCorrection": 1.0,
                "ImpliedImpressionTotal": np.nan,
                "ScaleFactor": 1.0,
                "TotalErrorPct": round(abs(baseline_bias), 2),
                "WAPE": round(_wape(actual_imp, baseline_imp), 2),
                "BiasPct": round(baseline_bias, 2),
            },
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "Method": "MLOnlyCTREnsembleImpressions",
                "SelectedEnsemble": winner,
                "Predicted90DClicks": round(
                    float(predicted_90d_click_total),
                    2,
                ),
                "PredictedCTR": round(ensemble_ctr, 6),
                "BiasCorrection": round(correction, 6),
                "ImpliedImpressionTotal": round(implied_total, 2),
                "ScaleFactor": round(scale_factor, 6),
                "TotalErrorPct": round(abs(candidate_bias), 2),
                "WAPE": round(_wape(actual_imp, calibrated), 2),
                "BiasPct": round(candidate_bias, 2),
            },
        ]
    )

    final_predictions = pd.DataFrame(
        [
            {
                "Candidate": name,
                "FinalPredictedCTR": value,
            }
            for name, value in final_ctrs.items()
        ]
    )

    return MLOnlyCTREnsembleResult(
        summary=summary,
        base_ranking=base_ranking,
        ensemble_ranking=ensemble_ranking,
        final_predictions=final_predictions,
        daily=daily,
    )
