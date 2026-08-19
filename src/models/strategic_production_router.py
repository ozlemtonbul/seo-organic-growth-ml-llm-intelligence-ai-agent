
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
from src.models.strategic_champion_selector import (
    _portfolio_series,
    _select_candidate,
    _candidate_forecasts,
)
from src.models.ctr_ensemble_impressions import (
    _fold_starts,
    _ml_fold_predictions,
    _recent_fold_predictions,
    _wide_oof,
    _ensemble_candidates,
    _final_base_ctrs,
    _aggregate_final_ctr,
)
from src.models.ctr_implied_impressions import build_ctr_samples


HORIZONS = (7, 14, 30, 90, 180, 365)
MIN_SCALE = 0.40
MAX_SCALE = 1.60


@dataclass(frozen=True)
class StrategicProductionCandidate:
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


def _prepare_history(historical: pd.DataFrame) -> pd.DataFrame:
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
        frame["date"], errors="coerce"
    ).dt.normalize()
    frame["page"] = frame["page"].astype(str)

    for column in ("clicks", "impressions"):
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
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
        frame[date_col], errors="coerce"
    ).dt.normalize()
    frame[clicks_col] = pd.to_numeric(
        frame[clicks_col], errors="coerce"
    ).fillna(0.0)
    frame[impressions_col] = pd.to_numeric(
        frame[impressions_col], errors="coerce"
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


def _current_total_for_dates(
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


def _apply_target_to_tail(
    frame: pd.DataFrame,
    columns: Dict[str, str],
    value_column: str,
    preserved_dates: pd.DatetimeIndex,
    adjustable_dates: pd.DatetimeIndex,
    target_cumulative_total: float,
    *,
    allow_guardrail_fallback: bool = True,
) -> Dict[str, float | bool | str]:
    """
    Preserve earlier dates exactly and achieve a cumulative target only by
    scaling the later adjustable tail.

    If the required tail scale is outside the safety envelope, do not modify
    the tail when allow_guardrail_fallback=True.
    """
    preserved_total = _current_total_for_dates(
        frame,
        columns,
        value_column,
        preserved_dates,
    )
    adjustable_total = _current_total_for_dates(
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

    if guardrail_hit and allow_guardrail_fallback:
        return {
            "PreservedTotal": preserved_total,
            "AdjustableBaselineTotal": adjustable_total,
            "RequestedCumulativeTarget": float(target_cumulative_total),
            "RequiredTailTotal": required_tail_total,
            "RequiredTailScale": float(required_scale),
            "AppliedTailScale": 1.0,
            "GuardrailHit": True,
            "Applied": False,
            "Status": "FALLBACK_BASELINE",
        }

    applied_scale = float(
        np.clip(required_scale, MIN_SCALE, MAX_SCALE)
    )

    mask = frame[columns["date"]].isin(adjustable_dates)

    frame.loc[mask, columns[value_column]] = (
        pd.to_numeric(
            frame.loc[mask, columns[value_column]],
            errors="coerce",
        ).fillna(0.0)
        * applied_scale
    )

    return {
        "PreservedTotal": preserved_total,
        "AdjustableBaselineTotal": adjustable_total,
        "RequestedCumulativeTarget": float(target_cumulative_total),
        "RequiredTailTotal": required_tail_total,
        "RequiredTailScale": float(required_scale),
        "AppliedTailScale": applied_scale,
        "GuardrailHit": bool(guardrail_hit),
        "Applied": True,
        "Status": "APPLIED" if not guardrail_hit else "CAPPED",
    }


def _apply_daily_tail_targets(
    frame: pd.DataFrame,
    columns: Dict[str, str],
    value_column: str,
    target_series: pd.Series,
    *,
    allow_guardrail_fallback: bool = True,
) -> Dict[str, float | bool | str]:
    """
    Apply portfolio daily targets only for the supplied dates.
    Earlier dates remain untouched.
    """
    ratios = []

    for date, target in target_series.items():
        mask = frame[columns["date"]].eq(pd.Timestamp(date))
        current = float(
            pd.to_numeric(
                frame.loc[mask, columns[value_column]],
                errors="coerce",
            ).fillna(0.0).sum()
        )

        if current <= 0:
            continue

        ratios.append(float(target) / current)

    if not ratios:
        return {
            "RequiredMinDailyScale": 1.0,
            "RequiredMaxDailyScale": 1.0,
            "GuardrailHit": False,
            "Applied": False,
            "Status": "NO_TARGETS",
        }

    min_ratio = float(np.min(ratios))
    max_ratio = float(np.max(ratios))
    guardrail_hit = (
        min_ratio < MIN_SCALE
        or max_ratio > MAX_SCALE
    )

    if guardrail_hit and allow_guardrail_fallback:
        return {
            "RequiredMinDailyScale": min_ratio,
            "RequiredMaxDailyScale": max_ratio,
            "GuardrailHit": True,
            "Applied": False,
            "Status": "FALLBACK_BASELINE",
        }

    for date, target in target_series.items():
        mask = frame[columns["date"]].eq(pd.Timestamp(date))
        current = float(
            pd.to_numeric(
                frame.loc[mask, columns[value_column]],
                errors="coerce",
            ).fillna(0.0).sum()
        )
        if current <= 0:
            continue

        ratio = float(target) / current
        ratio = float(np.clip(ratio, MIN_SCALE, MAX_SCALE))

        frame.loc[mask, columns[value_column]] = (
            pd.to_numeric(
                frame.loc[mask, columns[value_column]],
                errors="coerce",
            ).fillna(0.0)
            * ratio
        )

    return {
        "RequiredMinDailyScale": min_ratio,
        "RequiredMaxDailyScale": max_ratio,
        "GuardrailHit": bool(guardrail_hit),
        "Applied": True,
        "Status": "APPLIED" if not guardrail_hit else "CAPPED",
    }


def _direct_total(
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

    target_total = _final_direct_total(
        samples=samples,
        historical=historical,
        metric=metric,
        cutoff_date=cutoff_date,
        horizon_days=horizon_days,
        candidate=winner,
    )

    return float(target_total), str(winner), int(len(samples))


def _ctr_ensemble_total(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    predicted_90d_click_total: float,
) -> tuple[float, str, float, float]:
    samples = build_ctr_samples(
        historical=historical,
        cutoff_date=cutoff_date,
        horizon_days=90,
    )
    starts = _fold_starts(len(samples))

    ml_predictions, ml_metrics = _ml_fold_predictions(samples, starts)
    recent_predictions, recent_metrics = _recent_fold_predictions(
        samples, starts
    )

    predictions = pd.concat(
        [ml_predictions, recent_predictions],
        ignore_index=True,
    )

    base_ranking = (
        pd.concat([ml_metrics, recent_metrics], ignore_index=True)
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

    winner = str(ensemble_ranking.iloc[0]["Ensemble"])
    correction = float(ensemble_ranking.iloc[0]["BiasCorrection"])

    candidate_names = base_ranking["Candidate"].astype(str).tolist()

    final_ctrs = _final_base_ctrs(
        samples=samples,
        historical=historical,
        cutoff_date=cutoff_date,
        candidate_names=candidate_names,
    )

    predicted_ctr = _aggregate_final_ctr(
        final_ctrs=final_ctrs,
        base_ranking=base_ranking,
        ensemble_name=winner,
    )

    implied_total = (
        float(predicted_90d_click_total)
        / float(predicted_ctr)
        * correction
    )

    return (
        float(implied_total),
        winner,
        float(predicted_ctr),
        correction,
    )


def _champion_path(
    historical: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    metric: str,
    horizon_days: int,
    forecast_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, str]:
    series = _portfolio_series(
        historical=historical,
        cutoff=cutoff_date,
        metric=metric,
    )

    winner, _ = _select_candidate(
        full_pre_cutoff_series=series,
        horizon_days=horizon_days,
    )

    candidates = _candidate_forecasts(
        train_series=series,
        future_dates=forecast_dates,
    )

    prediction = pd.Series(
        np.asarray(candidates[winner], dtype=float),
        index=forecast_dates,
        dtype=float,
    )

    return prediction, winner


def build_strategic_production_candidate(
    historical: pd.DataFrame,
    forecast_daily: pd.DataFrame,
) -> StrategicProductionCandidate:
    history = _prepare_history(historical)
    cutoff = pd.Timestamp(history["date"].max()).normalize()

    frame, columns = _prepare_forecast(forecast_daily)

    # Immutable baseline snapshot. 7/14/30 MUST match this exactly.
    baseline = _aggregate_daily(frame.copy(), columns)

    forecast_dates = pd.DatetimeIndex(
        baseline["ForecastDate"]
        .drop_duplicates()
        .sort_values()
        .head(365)
    )

    if len(forecast_dates) < 180:
        raise ValueError(
            f"At least 180 forecast dates required. Found={len(forecast_dates)}"
        )

    dates_1_30 = forecast_dates[:30]
    dates_31_90 = forecast_dates[30:90]
    dates_1_90 = forecast_dates[:90]
    dates_91_180 = forecast_dates[90:180]

    report_rows = []

    # ------------------------------------------------------------------
    # 90d clicks: direct model, but NEVER touch days 1-30.
    # If current production distribution requires an unsafe tail scale,
    # preserve RecursiveDailyML rather than forcing an extreme adjustment.
    # ------------------------------------------------------------------
    click90_target, click90_winner, click90_samples = _direct_total(
        history, cutoff, "clicks", 90
    )

    click90_apply = _apply_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="clicks",
        preserved_dates=dates_1_30,
        adjustable_dates=dates_31_90,
        target_cumulative_total=click90_target,
        allow_guardrail_fallback=True,
    )

    click90_method = (
        "DirectHorizonCalibratedML"
        if bool(click90_apply["Applied"])
        else "RecursiveDailyML-FallbackGuardrail"
    )

    report_rows.append(
        {
            "HorizonDays": 90,
            "Metric": "clicks",
            "Method": click90_method,
            "SelectedCandidate": click90_winner,
            "TrainingSamples": click90_samples,
            "RequestedTargetTotal": click90_target,
            **click90_apply,
        }
    )

    current90 = _aggregate_daily(frame, columns)
    applied_click90_total = float(
        current90.iloc[:90]["PredictedClicks"].sum()
    )

    # ------------------------------------------------------------------
    # 90d impressions: CTR ensemble. Preserve days 1-30 exactly.
    # ------------------------------------------------------------------
    imp90_target, ctr_ensemble, predicted_ctr, ctr_correction = (
        _ctr_ensemble_total(
            historical=history,
            cutoff_date=cutoff,
            predicted_90d_click_total=applied_click90_total,
        )
    )

    imp90_apply = _apply_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="impressions",
        preserved_dates=dates_1_30,
        adjustable_dates=dates_31_90,
        target_cumulative_total=imp90_target,
        allow_guardrail_fallback=True,
    )

    imp90_method = (
        "CTREnsembleImpressionsML"
        if bool(imp90_apply["Applied"])
        else "RecursiveDailyML-FallbackGuardrail"
    )

    report_rows.append(
        {
            "HorizonDays": 90,
            "Metric": "impressions",
            "Method": imp90_method,
            "SelectedCandidate": ctr_ensemble,
            "TrainingSamples": np.nan,
            "RequestedTargetTotal": imp90_target,
            "PredictedCTR": predicted_ctr,
            "BiasCorrection": ctr_correction,
            **imp90_apply,
        }
    )

    # ------------------------------------------------------------------
    # 180d clicks:
    # Primary = direct horizon calibrator on days 91-180 only.
    # If unsafe, secondary = validated StrategicChampionPortfolio.
    # If that is also unsafe, keep baseline tail.
    # ------------------------------------------------------------------
    click180_target, click180_winner, click180_samples = _direct_total(
        history, cutoff, "clicks", 180
    )

    click180_apply = _apply_target_to_tail(
        frame=frame,
        columns=columns,
        value_column="clicks",
        preserved_dates=dates_1_90,
        adjustable_dates=dates_91_180,
        target_cumulative_total=click180_target,
        allow_guardrail_fallback=True,
    )

    click180_method = "DirectHorizonCalibratedML"
    selected_click180 = click180_winner
    secondary_used = False

    if not bool(click180_apply["Applied"]):
        champion_click_path, champion_click_winner = _champion_path(
            historical=history,
            cutoff_date=cutoff,
            metric="clicks",
            horizon_days=180,
            forecast_dates=forecast_dates[:180],
        )

        current_after_90 = _aggregate_daily(frame, columns)
        first90_clicks = float(
            current_after_90.iloc[:90]["PredictedClicks"].sum()
        )

        champion_total = float(champion_click_path.sum())
        remaining_target = max(0.0, champion_total - first90_clicks)
        champion_tail = champion_click_path.iloc[90:180].copy()
        champion_tail_sum = float(champion_tail.sum())

        if champion_tail_sum > 0:
            champion_tail = (
                champion_tail * (remaining_target / champion_tail_sum)
            )

        secondary_apply = _apply_daily_tail_targets(
            frame=frame,
            columns=columns,
            value_column="clicks",
            target_series=champion_tail,
            allow_guardrail_fallback=True,
        )

        if bool(secondary_apply["Applied"]):
            click180_method = "StrategicChampionPortfolio"
            selected_click180 = champion_click_winner
            secondary_used = True
            click180_apply = {
                **click180_apply,
                "SecondaryMethod": "StrategicChampionPortfolio",
                "SecondarySelectedCandidate": champion_click_winner,
                "SecondaryApplied": True,
                **{
                    f"Secondary{k}": v
                    for k, v in secondary_apply.items()
                },
            }
        else:
            click180_method = "RecursiveDailyML-FallbackGuardrail"
            selected_click180 = champion_click_winner
            click180_apply = {
                **click180_apply,
                "SecondaryMethod": "StrategicChampionPortfolio",
                "SecondarySelectedCandidate": champion_click_winner,
                "SecondaryApplied": False,
                **{
                    f"Secondary{k}": v
                    for k, v in secondary_apply.items()
                },
            }

    report_rows.append(
        {
            "HorizonDays": 180,
            "Metric": "clicks",
            "Method": click180_method,
            "SelectedCandidate": selected_click180,
            "TrainingSamples": click180_samples,
            "RequestedTargetTotal": click180_target,
            "SecondaryUsed": secondary_used,
            **click180_apply,
        }
    )

    # ------------------------------------------------------------------
    # 180d impressions: validated champion on days 91-180 only.
    # Preserve the accepted 90d cumulative result.
    # ------------------------------------------------------------------
    champion_imp_path, champion_imp_winner = _champion_path(
        historical=history,
        cutoff_date=cutoff,
        metric="impressions",
        horizon_days=180,
        forecast_dates=forecast_dates[:180],
    )

    current_after_90_imp = _aggregate_daily(frame, columns)
    first90_imp = float(
        current_after_90_imp.iloc[:90]["PredictedImpressions"].sum()
    )

    champion_imp_total = float(champion_imp_path.sum())
    remaining_imp_target = max(0.0, champion_imp_total - first90_imp)

    champion_imp_tail = champion_imp_path.iloc[90:180].copy()
    champion_imp_tail_sum = float(champion_imp_tail.sum())

    if champion_imp_tail_sum > 0:
        champion_imp_tail = (
            champion_imp_tail
            * (remaining_imp_target / champion_imp_tail_sum)
        )

    imp180_apply = _apply_daily_tail_targets(
        frame=frame,
        columns=columns,
        value_column="impressions",
        target_series=champion_imp_tail,
        allow_guardrail_fallback=True,
    )

    imp180_method = (
        "StrategicChampionPortfolio"
        if bool(imp180_apply["Applied"])
        else "RecursiveDailyML-FallbackGuardrail"
    )

    report_rows.append(
        {
            "HorizonDays": 180,
            "Metric": "impressions",
            "Method": imp180_method,
            "SelectedCandidate": champion_imp_winner,
            "TrainingSamples": np.nan,
            "RequestedTargetTotal": champion_imp_total,
            **imp180_apply,
        }
    )

    # Final metadata
    first_date = forecast_dates[0]
    day_number = (
        (frame[columns["date"]] - first_date).dt.days + 1
    )

    frame["StrategicAdjusted"] = day_number.between(31, 180)

    # Enforce the physical invariant; this should normally be a no-op.
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

    candidate_daily = _aggregate_daily(frame, columns)

    portfolio_rows = []

    report = pd.DataFrame(report_rows)

    method_lookup = {
        (int(row.HorizonDays), str(row.Metric)): str(row.Method)
        for row in report.itertuples(index=False)
    }

    for horizon in HORIZONS:
        if len(candidate_daily) < horizon:
            continue

        subset = candidate_daily.iloc[:horizon]

        if horizon <= 30:
            click_method = "RecursiveDailyML"
            impression_method = "RecursiveDailyML"
        elif horizon == 90:
            click_method = method_lookup[(90, "clicks")]
            impression_method = method_lookup[(90, "impressions")]
        elif horizon == 180:
            click_method = method_lookup[(180, "clicks")]
            impression_method = method_lookup[(180, "impressions")]
        else:
            # 365 necessarily contains the adjusted first 180 days, so it is
            # hybrid, not an untouched RecursiveDailyML forecast.
            click_method = "HybridFirst180+RecursiveTail-Unvalidated365"
            impression_method = "HybridFirst180+RecursiveTail-Unvalidated365"

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

    baseline_horizons = pd.DataFrame(baseline_rows)

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

    return StrategicProductionCandidate(
        daily=frame,
        portfolio=portfolio,
        report=report,
        qa=qa,
    )
