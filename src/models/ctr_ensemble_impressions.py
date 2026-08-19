from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
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
    _panel,
    _origin_features,
    _wape,
    _bias_pct,
    build_ctr_samples,
)


@dataclass(frozen=True)
class CTREnsembleResult:
    summary: pd.DataFrame
    base_ranking: pd.DataFrame
    ensemble_ranking: pd.DataFrame
    final_predictions: pd.DataFrame
    daily: pd.DataFrame


RECENT_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("Recent28CTR", "ctr_28"),
    ("Recent56CTR", "ctr_56"),
    ("Recent90CTR", "ctr_90"),
)


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
            f"Not enough samples for ensemble walk-forward: {sample_count}"
        )

    return starts


def _ml_fold_predictions(
    samples: pd.DataFrame,
    starts: List[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_columns(samples)
    validation_size = 5

    prediction_rows = []
    metric_rows = []

    for candidate_name, model_template in _models().items():
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

            pred_ctr = _inv_logit(
                model.predict(X_valid)
            )
            pred_ctr = np.clip(pred_ctr, 0.005, 0.10)

            for local_idx, (_, row) in enumerate(valid.iterrows()):
                candidate_rows.append(
                    {
                        "Candidate": candidate_name,
                        "Fold": fold_no,
                        "OriginDate": pd.Timestamp(row["OriginDate"]),
                        "PredictedCTR": float(pred_ctr[local_idx]),
                        "ActualCTR": float(row["TargetCTR"]),
                        "FutureClicks": float(row["FutureClicks"]),
                        "FutureImpressions": float(row["FutureImpressions"]),
                    }
                )

        candidate_df = pd.DataFrame(candidate_rows)
        implied = (
            candidate_df["FutureClicks"].to_numpy(dtype=float)
            / candidate_df["PredictedCTR"].to_numpy(dtype=float)
        )
        actual_imp = candidate_df["FutureImpressions"].to_numpy(dtype=float)

        metric_rows.append(
            {
                "Candidate": candidate_name,
                "ImpressionMeanAPE": round(
                    float(
                        np.mean(
                            np.abs(implied - actual_imp)
                            / np.maximum(np.abs(actual_imp), 1e-9)
                            * 100.0
                        )
                    ),
                    4,
                ),
                "ImpressionMedianAPE": round(
                    float(
                        np.median(
                            np.abs(implied - actual_imp)
                            / np.maximum(np.abs(actual_imp), 1e-9)
                            * 100.0
                        )
                    ),
                    4,
                ),
                "ImpressionBiasPct": round(
                    _bias_pct(actual_imp, implied),
                    4,
                ),
            }
        )

        prediction_rows.extend(candidate_rows)

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _recent_fold_predictions(
    samples: pd.DataFrame,
    starts: List[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_size = 5
    prediction_rows = []
    metric_rows = []

    for candidate_name, column in RECENT_CANDIDATES:
        candidate_rows = []

        for fold_no, start in enumerate(starts, start=1):
            valid = samples.iloc[start:start + validation_size].copy()

            pred_ctr = pd.to_numeric(
                valid[column],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            pred_ctr = np.clip(pred_ctr, 0.005, 0.10)

            for local_idx, (_, row) in enumerate(valid.iterrows()):
                candidate_rows.append(
                    {
                        "Candidate": candidate_name,
                        "Fold": fold_no,
                        "OriginDate": pd.Timestamp(row["OriginDate"]),
                        "PredictedCTR": float(pred_ctr[local_idx]),
                        "ActualCTR": float(row["TargetCTR"]),
                        "FutureClicks": float(row["FutureClicks"]),
                        "FutureImpressions": float(row["FutureImpressions"]),
                    }
                )

        candidate_df = pd.DataFrame(candidate_rows)
        implied = (
            candidate_df["FutureClicks"].to_numpy(dtype=float)
            / candidate_df["PredictedCTR"].to_numpy(dtype=float)
        )
        actual_imp = candidate_df["FutureImpressions"].to_numpy(dtype=float)

        metric_rows.append(
            {
                "Candidate": candidate_name,
                "ImpressionMeanAPE": round(
                    float(
                        np.mean(
                            np.abs(implied - actual_imp)
                            / np.maximum(np.abs(actual_imp), 1e-9)
                            * 100.0
                        )
                    ),
                    4,
                ),
                "ImpressionMedianAPE": round(
                    float(
                        np.median(
                            np.abs(implied - actual_imp)
                            / np.maximum(np.abs(actual_imp), 1e-9)
                            * 100.0
                        )
                    ),
                    4,
                ),
                "ImpressionBiasPct": round(
                    _bias_pct(actual_imp, implied),
                    4,
                ),
            }
        )

        prediction_rows.extend(candidate_rows)

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _wide_oof(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = [
        "Fold",
        "OriginDate",
        "ActualCTR",
        "FutureClicks",
        "FutureImpressions",
    ]

    wide = (
        predictions
        .pivot_table(
            index=key_cols,
            columns="Candidate",
            values="PredictedCTR",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide


def _ensemble_candidates(
    wide: pd.DataFrame,
    base_ranking: pd.DataFrame,
) -> pd.DataFrame:
    candidate_order = (
        base_ranking
        .sort_values(
            [
                "ImpressionMeanAPE",
                "Candidate",
            ]
        )["Candidate"]
        .tolist()
    )

    top3 = candidate_order[:3]
    all_candidates = candidate_order

    errors = (
        base_ranking
        .set_index("Candidate")["ImpressionMeanAPE"]
        .to_dict()
    )

    def weighted_mean(row, candidates):
        weights = np.asarray(
            [
                1.0 / max(float(errors[name]), 0.25) ** 2
                for name in candidates
            ],
            dtype=float,
        )
        values = np.asarray(
            [
                float(row[name])
                for name in candidates
            ],
            dtype=float,
        )
        return float(np.sum(values * weights) / np.sum(weights))

    records = []

    aggregators = {
        "Top3WeightedCTR": lambda row: weighted_mean(row, top3),
        "Top3MedianCTR": lambda row: float(
            np.median([float(row[name]) for name in top3])
        ),
        "AllWeightedCTR": lambda row: weighted_mean(row, all_candidates),
        "AllMedianCTR": lambda row: float(
            np.median([float(row[name]) for name in all_candidates])
        ),
        "AllQ60CTR": lambda row: float(
            np.quantile(
                [float(row[name]) for name in all_candidates],
                0.60,
            )
        ),
        "AllQ70CTR": lambda row: float(
            np.quantile(
                [float(row[name]) for name in all_candidates],
                0.70,
            )
        ),
    }

    actual_imp = wide["FutureImpressions"].to_numpy(dtype=float)
    future_clicks = wide["FutureClicks"].to_numpy(dtype=float)

    for name, aggregator in aggregators.items():
        ctr = np.asarray(
            [
                np.clip(aggregator(row), 0.005, 0.10)
                for _, row in wide.iterrows()
            ],
            dtype=float,
        )

        implied = future_clicks / ctr

        # Pre-cutoff OOF bias correction only.
        correction = float(
            np.median(
                np.divide(
                    actual_imp,
                    implied,
                    out=np.ones_like(actual_imp),
                    where=implied > 0,
                )
            )
        )
        correction = float(np.clip(correction, 0.85, 1.15))
        corrected = implied * correction

        ape = (
            np.abs(corrected - actual_imp)
            / np.maximum(np.abs(actual_imp), 1e-9)
            * 100.0
        )

        records.append(
            {
                "Ensemble": name,
                "Members": ",".join(
                    top3
                    if name.startswith("Top3")
                    else all_candidates
                ),
                "BiasCorrection": round(correction, 6),
                "ImpressionMeanAPE": round(float(np.mean(ape)), 4),
                "ImpressionMedianAPE": round(float(np.median(ape)), 4),
                "P90APE": round(float(np.quantile(ape, 0.90)), 4),
                "ImpressionBiasPct": round(
                    _bias_pct(actual_imp, corrected),
                    4,
                ),
            }
        )

    ranking = (
        pd.DataFrame(records)
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


def _final_base_ctrs(
    samples: pd.DataFrame,
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    candidate_names: List[str],
) -> Dict[str, float]:
    output: Dict[str, float] = {}

    for candidate in candidate_names:
        output[candidate] = float(
            _final_ctr(
                samples=samples,
                historical=historical,
                cutoff_date=cutoff_date,
                candidate=candidate,
            )
        )

    return output


def _aggregate_final_ctr(
    final_ctrs: Dict[str, float],
    base_ranking: pd.DataFrame,
    ensemble_name: str,
) -> float:
    candidate_order = (
        base_ranking
        .sort_values(
            [
                "ImpressionMeanAPE",
                "Candidate",
            ]
        )["Candidate"]
        .tolist()
    )

    top3 = candidate_order[:3]
    all_candidates = candidate_order

    errors = (
        base_ranking
        .set_index("Candidate")["ImpressionMeanAPE"]
        .to_dict()
    )

    def weighted(candidates):
        weights = np.asarray(
            [
                1.0 / max(float(errors[name]), 0.25) ** 2
                for name in candidates
            ],
            dtype=float,
        )
        values = np.asarray(
            [
                float(final_ctrs[name])
                for name in candidates
            ],
            dtype=float,
        )
        return float(np.sum(values * weights) / np.sum(weights))

    if ensemble_name == "Top3WeightedCTR":
        value = weighted(top3)
    elif ensemble_name == "Top3MedianCTR":
        value = float(np.median([final_ctrs[name] for name in top3]))
    elif ensemble_name == "AllWeightedCTR":
        value = weighted(all_candidates)
    elif ensemble_name == "AllMedianCTR":
        value = float(
            np.median([final_ctrs[name] for name in all_candidates])
        )
    elif ensemble_name == "AllQ60CTR":
        value = float(
            np.quantile(
                [final_ctrs[name] for name in all_candidates],
                0.60,
            )
        )
    elif ensemble_name == "AllQ70CTR":
        value = float(
            np.quantile(
                [final_ctrs[name] for name in all_candidates],
                0.70,
            )
        )
    else:
        raise ValueError(f"Unknown ensemble: {ensemble_name}")

    return float(np.clip(value, 0.005, 0.10))


def evaluate_ctr_ensemble(
    historical: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    predicted_90d_click_total: float,
) -> CTREnsembleResult:
    samples = build_ctr_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=HORIZON_DAYS,
    )

    starts = _fold_starts(len(samples))

    ml_predictions, ml_metrics = _ml_fold_predictions(samples, starts)
    recent_predictions, recent_metrics = _recent_fold_predictions(samples, starts)

    predictions = pd.concat(
        [ml_predictions, recent_predictions],
        ignore_index=True,
    )

    base_ranking = (
        pd.concat(
            [ml_metrics, recent_metrics],
            ignore_index=True,
        )
        .sort_values(
            [
                "ImpressionMeanAPE",
                "ImpressionMedianAPE",
                "Candidate",
            ]
        )
        .reset_index(drop=True)
    )

    wide = _wide_oof(predictions)

    ensemble_ranking = _ensemble_candidates(
        wide=wide,
        base_ranking=base_ranking,
    )

    winner = str(
        ensemble_ranking.iloc[0]["Ensemble"]
    )

    correction = float(
        ensemble_ranking.iloc[0]["BiasCorrection"]
    )

    candidate_names = base_ranking["Candidate"].tolist()

    final_ctrs = _final_base_ctrs(
        samples=samples,
        historical=historical,
        cutoff_date=cutoff_date,
        candidate_names=candidate_names,
    )

    ensemble_ctr = _aggregate_final_ctr(
        final_ctrs=final_ctrs,
        base_ranking=base_ranking,
        ensemble_name=winner,
    )

    implied_total = (
        float(predicted_90d_click_total) / ensemble_ctr
        if ensemble_ctr > 0
        else 0.0
    )

    corrected_total = implied_total * correction

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
        corrected_total / baseline_total
        if baseline_total > 0
        else 1.0
    )

    scale_factor = float(
        np.clip(scale_factor, 0.40, 1.10)
    )

    calibrated = baseline_imp * scale_factor

    daily["CTREnsemblePredictedImpressions"] = calibrated

    baseline_bias = _bias_pct(actual_imp, baseline_imp)
    candidate_bias = _bias_pct(actual_imp, calibrated)

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
                "Method": "CTREnsembleImpressionsML",
                "SelectedEnsemble": winner,
                "Predicted90DClicks": round(
                    float(predicted_90d_click_total), 2
                ),
                "PredictedCTR": round(ensemble_ctr, 6),
                "BiasCorrection": round(correction, 6),
                "ImpliedImpressionTotal": round(corrected_total, 2),
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

    return CTREnsembleResult(
        summary=summary,
        base_ranking=base_ranking,
        ensemble_ranking=ensemble_ranking,
        final_predictions=final_predictions,
        daily=daily,
    )
