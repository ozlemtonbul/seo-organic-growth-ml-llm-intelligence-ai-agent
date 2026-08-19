
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.strategic_direct_calibrator import (
    build_direct_training_samples,
    _inner_select_candidate,
    _final_direct_total,
)
from src.models.ml_only_ctr_ensemble import (
    _fold_starts as _ml_ctr_fold_starts,
    _ml_oof_predictions,
    _wide_predictions,
    _ensemble_rank,
    _final_ml_ctrs,
    _final_ensemble_ctr,
)
from src.models.ctr_implied_impressions import build_ctr_samples


HORIZONS = (7, 14, 30, 90, 180, 365)
MIN_SCALE = 0.40
MAX_SCALE = 1.60


@dataclass(frozen=True)
class StrategicMLOnlyCandidate:
    daily: pd.DataFrame
    portfolio: pd.DataFrame
    report: pd.DataFrame
    qa: pd.DataFrame


def _resolve_column(
    frame: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> str | None:
    lower_map = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Required column not found. Tried={candidates}. "
            f"Available={list(frame.columns)}"
        )

    return None


def _prepare_history(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    frame = historical.copy()

    date_col = _resolve_column(frame, ["date", "Date"])
    page_col = _resolve_column(frame, ["page", "Page", "url", "URL"])
    clicks_col = _resolve_column(frame, ["clicks", "Clicks"])
    impressions_col = _resolve_column(frame, ["impressions", "Impressions"])

    frame = frame.rename(
        columns={
            date_col: "date",
            page_col: "page",
            clicks_col: "clicks",
            impressions_col: "impressions",
        }
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()

    frame["page"] = frame["page"].astype(str)

    for column in ("clicks", "impressions"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)

    return (
        frame.dropna(subset=["date", "page"])
        .sort_values(["date", "page"])
        .reset_index(drop=True)
    )


def _prepare_forecast(
    forecast_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    frame = forecast_daily.copy()

    date_col = _resolve_column(
        frame,
        ["ForecastDate", "forecast_date", "Date", "date"],
    )

    page_col = _resolve_column(
        frame,
        ["Page", "page", "URL", "url", "LandingPage"],
        required=False,
    )

    clicks_col = _resolve_column(
        frame,
        [
            "PredictedClicks",
            "predicted_clicks",
            "ForecastClicks",
            "forecast_clicks",
        ],
    )

    impressions_col = _resolve_column(
        frame,
        [
            "PredictedImpressions",
            "predicted_impressions",
            "ForecastImpressions",
            "forecast_impressions",
        ],
    )

    frame[date_col] = pd.to_datetime(
        frame[date_col],
        errors="coerce",
    ).dt.normalize()

    frame[clicks_col] = pd.to_numeric(
        frame[clicks_col],
        errors="coerce",
    ).fillna(0.0)

    frame[impressions_col] = pd.to_numeric(
        frame[impressions_col],
        errors="coerce",
    ).fillna(0.0)

    frame = (
        frame.dropna(subset=[date_col])
        .sort_values([date_col])
        .reset_index(drop=True)
    )

    return frame, {
        "date": date_col,
        "page": page_col or "",
        "clicks": clicks_col,
        "impressions": impressions_col,
    }


def _aggregate_daily(
    frame: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    return (
        frame.groupby(columns["date"], as_index=False)
        .agg(
            PredictedClicks=(columns["clicks"], "sum"),
            PredictedImpressions=(columns["impressions"], "sum"),
        )
        .rename(columns={columns["date"]: "ForecastDate"})
        .sort_values("ForecastDate")
        .reset_index(drop=True)
    )


def _current_total(
    frame: pd.DataFrame,
    columns: Dict[str, str],
    value_column: str,
    dates: pd.DatetimeIndex,
) -> float:
    mask = frame[columns["date"]].isin(dates)
    return float(
        pd.to_numeric(
            frame.loc[mask, columns[value_column]],
            errors="coerce",
        ).fillna(0.0).sum()
    )


def _apply_cumulative_target_to_tail(
    frame: pd.DataFrame,
    columns: Dict[str, str],
    value_column: str,
    preserved_dates: pd.DatetimeIndex,
    adjustable_dates: pd.DatetimeIndex,
    target_cumulative_total: float,
) -> Dict[str, float | bool | str]:
    preserved_total = _current_total(
        frame,
        columns,
        value_column,
        preserved_dates,
    )

    adjustable_total = _current_total(
        frame,
        columns,
        value_column,
        adjustable_dates,
    )

    required_tail_total = max(
        0.0,
        float(target_cumulative_total) - preserved_total,
    )

    required_scale = (
        required_tail_total / adjustable_total
        if adjustable_total > 0
        else 1.0
    )

    guardrail_hit = not (
        MIN_SCALE <= required_scale <= MAX_SCALE
    )

    if guardrail_hit:
        return {
            "PreservedTotal": preserved_total,
            "AdjustableBaselineTotal": adjustable_total,
            "RequestedCumulativeTarget": float(target_cumulative_total),
            "RequiredTailTotal": required_tail_total,
            "RequiredTailScale": float(required_scale),
            "AppliedTailScale": 1.0,
            "GuardrailHit": True,
            "Applied": False,
            "Status": "ML_FALLBACK_RECURSIVE",
        }

    mask = frame[columns["date"]].isin(adjustable_dates)

    frame.loc[mask, columns[value_column]] = (
        pd.to_numeric(
            frame.loc[mask, columns[value_column]],
            errors="coerce",
        ).fillna(0.0)
        * float(required_scale)
    )

    return {
        "PreservedTotal": preserved_total,
        "AdjustableBaselineTotal": adjustable_total,
        "RequestedCumulativeTarget": float(target_cumulative_total),
        "RequiredTailTotal": required_tail_total,
        "RequiredTailScale": float(required_scale),
        "AppliedTailScale": float(required_scale),
        "GuardrailHit": False,
        "Applied": True,
        "Status": "APPLIED",
    }


def _direct_ml_total(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    metric: str,
    horizon_days: int,
) -> tuple[float, str, int]:
    samples = build_direct_training_samples(
        historical=historical,
        metric=metric,
        cutoff_date=cutoff_date,
        horizon_days=horizon_days,
    )

    winner, _ = _inner_select_candidate(samples)

    total = _final_direct_total(
        samples=samples,
        historical=historical,
        metric=metric,
        cutoff_date=cutoff_date,
        horizon_days=horizon_days,
        candidate=winner,
    )

    return float(total), str(winner), int(len(samples))


def _ml_only_ctr_total(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    predicted_90d_click_total: float,
) -> tuple[float, str, float, float, str]:
    samples = build_ctr_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=90,
    )

    starts = _ml_ctr_fold_starts(len(samples))

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
        / float(ensemble_ctr)
        * correction
    )

    members = ",".join(
        base_ranking["Candidate"]
        .astype(str)
        .tolist()
    )

    return (
        float(implied_total),
        winner,
        float(ensemble_ctr),
        correction,
        members,
    )


def build_strategic_ml_only_candidate(
    historical: pd.DataFrame,
    forecast_daily: pd.DataFrame,
) -> StrategicMLOnlyCandidate:
    history = _prepare_history(historical)

    cutoff = pd.Timestamp(
        history["date"].max()
    ).normalize()

    frame, columns = _prepare_forecast(
        forecast_daily
    )

    baseline = _aggregate_daily(
        frame.copy(),
        columns,
    )

    forecast_dates = pd.DatetimeIndex(
        baseline["ForecastDate"]
        .drop_duplicates()
        .sort_values()
        .head(365)
    )

    if len(forecast_dates) < 180:
        raise ValueError(
            f"At least 180 forecast dates are required. "
            f"Found={len(forecast_dates)}"
        )

    dates_1_30 = forecast_dates[:30]
    dates_31_90 = forecast_dates[30:90]
    dates_1_90 = forecast_dates[:90]
    dates_91_180 = forecast_dates[90:180]

    report_rows = []

    # 90-day clicks: direct ML if safe; otherwise RecursiveDailyML.
    click90_target, click90_model, click90_samples = _direct_ml_total(
        historical=history,
        cutoff_date=cutoff,
        metric="clicks",
        horizon_days=90,
    )

    click90_apply = _apply_cumulative_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="clicks",
        preserved_dates=dates_1_30,
        adjustable_dates=dates_31_90,
        target_cumulative_total=click90_target,
    )

    click90_method = (
        "DirectHorizonCalibratedML"
        if bool(click90_apply["Applied"])
        else "RecursiveDailyML"
    )

    report_rows.append(
        {
            "HorizonDays": 90,
            "Metric": "clicks",
            "PrimaryMLMethod": "DirectHorizonCalibratedML",
            "AppliedMethod": click90_method,
            "SelectedMLModel": click90_model,
            "TrainingSamples": click90_samples,
            "RequestedTargetTotal": click90_target,
            "StatisticalPrimaryUsed": False,
            **click90_apply,
        }
    )

    current90 = _aggregate_daily(
        frame,
        columns,
    )

    applied_click90_total = float(
        current90.iloc[:90]["PredictedClicks"].sum()
    )

    # 90-day impressions: PURE ML CTR ensemble only.
    (
        imp90_target,
        imp90_ensemble,
        imp90_ctr,
        imp90_correction,
        imp90_members,
    ) = _ml_only_ctr_total(
        historical=history,
        cutoff_date=cutoff,
        predicted_90d_click_total=applied_click90_total,
    )

    imp90_apply = _apply_cumulative_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="impressions",
        preserved_dates=dates_1_30,
        adjustable_dates=dates_31_90,
        target_cumulative_total=imp90_target,
    )

    imp90_method = (
        "MLOnlyCTREnsembleImpressions"
        if bool(imp90_apply["Applied"])
        else "RecursiveDailyML"
    )

    report_rows.append(
        {
            "HorizonDays": 90,
            "Metric": "impressions",
            "PrimaryMLMethod": "MLOnlyCTREnsembleImpressions",
            "AppliedMethod": imp90_method,
            "SelectedMLModel": imp90_ensemble,
            "MLMembers": imp90_members,
            "PredictedCTR": imp90_ctr,
            "BiasCorrection": imp90_correction,
            "TrainingSamples": np.nan,
            "RequestedTargetTotal": imp90_target,
            "StatisticalPrimaryUsed": False,
            **imp90_apply,
        }
    )

    # 180-day clicks: direct ML if safe; otherwise validated RecursiveDailyML.
    click180_target, click180_model, click180_samples = _direct_ml_total(
        historical=history,
        cutoff_date=cutoff,
        metric="clicks",
        horizon_days=180,
    )

    click180_apply = _apply_cumulative_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="clicks",
        preserved_dates=dates_1_90,
        adjustable_dates=dates_91_180,
        target_cumulative_total=click180_target,
    )

    click180_method = (
        "DirectHorizonCalibratedML"
        if bool(click180_apply["Applied"])
        else "RecursiveDailyML"
    )

    report_rows.append(
        {
            "HorizonDays": 180,
            "Metric": "clicks",
            "PrimaryMLMethod": "DirectHorizonCalibratedML",
            "AppliedMethod": click180_method,
            "SelectedMLModel": click180_model,
            "TrainingSamples": click180_samples,
            "RequestedTargetTotal": click180_target,
            "StatisticalPrimaryUsed": False,
            **click180_apply,
        }
    )

    # 180-day impressions:
    # RecursiveDailyML is already a validated ML model on this horizon and
    # has materially better historical accuracy than the experimental direct
    # calibrator. Keep it as the production ML method.
    current180 = _aggregate_daily(
        frame,
        columns,
    )

    imp180_total = float(
        current180.iloc[:180]["PredictedImpressions"].sum()
    )

    report_rows.append(
        {
            "HorizonDays": 180,
            "Metric": "impressions",
            "PrimaryMLMethod": "RecursiveDailyML",
            "AppliedMethod": "RecursiveDailyML",
            "SelectedMLModel": "RecursiveDailyML",
            "TrainingSamples": np.nan,
            "RequestedTargetTotal": imp180_total,
            "StatisticalPrimaryUsed": False,
            "GuardrailHit": False,
            "Applied": True,
            "Status": "VALIDATED_ML_BASELINE",
        }
    )

    # Metadata. No statistical method is allowed in primary production path.
    first_date = forecast_dates[0]
    day_number = (
        (frame[columns["date"]] - first_date).dt.days + 1
    )

    frame["StrategicMLAdjusted"] = day_number.between(31, 180)

    frame["StrategicMLPrimary"] = True

    frame[columns["impressions"]] = np.maximum(
        pd.to_numeric(
            frame[columns["impressions"]],
            errors="coerce",
        ).fillna(0.0),
        pd.to_numeric(
            frame[columns["clicks"]],
            errors="coerce",
        ).fillna(0.0),
    )

    candidate_daily = _aggregate_daily(
        frame,
        columns,
    )

    report = pd.DataFrame(report_rows)

    applied_method = {
        (int(row.HorizonDays), str(row.Metric)): str(row.AppliedMethod)
        for row in report.itertuples(index=False)
    }

    portfolio_rows = []

    for horizon in HORIZONS:
        if len(candidate_daily) < horizon:
            continue

        subset = candidate_daily.iloc[:horizon]

        if horizon <= 30:
            click_method = "RecursiveDailyML"
            impression_method = "RecursiveDailyML"
        elif horizon == 90:
            click_method = applied_method[(90, "clicks")]
            impression_method = applied_method[(90, "impressions")]
        elif horizon == 180:
            click_method = applied_method[(180, "clicks")]
            impression_method = applied_method[(180, "impressions")]
        else:
            click_method = "MLHybridFirst180+RecursiveTail-Unvalidated365"
            impression_method = "MLHybridFirst180+RecursiveTail-Unvalidated365"

        portfolio_rows.append(
            {
                "HorizonDays": horizon,
                "HorizonType": (
                    "Operational" if horizon <= 30 else "Strategic"
                ),
                "PredictedClicks": float(
                    subset["PredictedClicks"].sum()
                ),
                "PredictedImpressions": float(
                    subset["PredictedImpressions"].sum()
                ),
                "ClickMethod": click_method,
                "ImpressionMethod": impression_method,
                "PrimaryForecastType": "ML",
            }
        )

    portfolio = pd.DataFrame(portfolio_rows)

    baseline_rows = []

    for horizon in HORIZONS:
        if len(baseline) < horizon:
            continue

        subset = baseline.iloc[:horizon]

        baseline_rows.append(
            {
                "HorizonDays": horizon,
                "BaselineClicks": float(
                    subset["PredictedClicks"].sum()
                ),
                "BaselineImpressions": float(
                    subset["PredictedImpressions"].sum()
                ),
            }
        )

    baseline_horizons = pd.DataFrame(
        baseline_rows
    )

    qa = portfolio.merge(
        baseline_horizons,
        on="HorizonDays",
        how="left",
    )

    qa["ClickDeltaPct"] = np.where(
        qa["BaselineClicks"] > 0,
        (
            qa["PredictedClicks"] / qa["BaselineClicks"] - 1.0
        ) * 100.0,
        0.0,
    )

    qa["ImpressionDeltaPct"] = np.where(
        qa["BaselineImpressions"] > 0,
        (
            qa["PredictedImpressions"]
            / qa["BaselineImpressions"]
            - 1.0
        ) * 100.0,
        0.0,
    )

    qa["OperationalUnchanged"] = np.where(
        qa["HorizonDays"].le(30),
        (
            np.isclose(
                qa["PredictedClicks"],
                qa["BaselineClicks"],
                rtol=1e-10,
                atol=1e-6,
            )
            & np.isclose(
                qa["PredictedImpressions"],
                qa["BaselineImpressions"],
                rtol=1e-10,
                atol=1e-6,
            )
        ),
        True,
    )

    qa["NonNegative"] = (
        qa["PredictedClicks"].ge(0.0)
        & qa["PredictedImpressions"].ge(0.0)
    )

    qa["ImpressionsGTEClicks"] = (
        qa["PredictedImpressions"]
        >= qa["PredictedClicks"]
    )

    qa["MLPrimaryOnly"] = (
        qa["PrimaryForecastType"].eq("ML")
        & ~qa["ClickMethod"].str.contains(
            "WeeklyRepeat|Recent28|Recent56|Recent90|Champion",
            case=False,
            regex=True,
        )
        & ~qa["ImpressionMethod"].str.contains(
            "WeeklyRepeat|Recent28|Recent56|Recent90|Champion",
            case=False,
            regex=True,
        )
    )

    return StrategicMLOnlyCandidate(
        daily=frame,
        portfolio=portfolio,
        report=report,
        qa=qa,
    )
