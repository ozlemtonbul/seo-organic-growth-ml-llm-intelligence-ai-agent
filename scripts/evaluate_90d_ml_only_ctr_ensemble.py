from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ml_only_ctr_ensemble import (
    evaluate_ml_only_ctr_ensemble,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = PROJECT_ROOT / "data" / "historical"


def _history_file() -> Path:
    files = sorted(
        HISTORY_DIR.glob("gsc_page_daily_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("Historical GSC CSV bulunamadı.")
    return files[0]


def _passed_click_total() -> float:
    path = OUTPUT_DIR / "seo_ml_direct_calibrator_90d_daily.csv"

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    if "DirectCalibratedClicks" not in frame.columns:
        raise ValueError(
            "DirectCalibratedClicks kolonu bulunamadı."
        )

    return float(
        pd.to_numeric(
            frame["DirectCalibratedClicks"],
            errors="coerce",
        ).fillna(0.0).head(90).sum()
    )


def main() -> int:
    print("=" * 120)
    print("SEO 90-DAY ML-ONLY CTR ENSEMBLE - DEVELOPMENT EVALUATION")
    print("=" * 120)
    print("[INFO] Statistical baselines are EXCLUDED from the ensemble.")
    print("[INFO] Members: CTRRidge, CTRGradientBoosting, CTRRandomForest, CTRHGBR.")
    print("[INFO] OOF ensemble selection and bias correction are pre-cutoff only.")
    print("[INFO] Final 90-day holdout is used only for the final comparison.")
    print("[INFO] Production forecast files are NOT modified.")

    historical = pd.read_csv(
        _history_file(),
        low_memory=False,
    )

    summary = pd.read_csv(
        OUTPUT_DIR / "seo_ml_backtest_strategic_90d_summary.csv",
        low_memory=False,
    )

    cutoff = pd.to_datetime(
        summary.iloc[0]["CutoffDate"],
        errors="raise",
    ).normalize()

    baseline_daily = pd.read_csv(
        OUTPUT_DIR / "seo_ml_backtest_strategic_90d_daily.csv",
        low_memory=False,
    )

    click_total = _passed_click_total()

    print()
    print(
        f"[INFO] Passed 90-day calibrated click total: {click_total:.2f}"
    )

    result = evaluate_ml_only_ctr_ensemble(
        historical=historical,
        baseline_daily=baseline_daily,
        cutoff_date=cutoff,
        predicted_90d_click_total=click_total,
    )

    result.base_ranking.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_base_ranking.csv",
        index=False,
    )
    result.ensemble_ranking.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_ensemble_ranking.csv",
        index=False,
    )
    result.final_predictions.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_final_predictions.csv",
        index=False,
    )
    result.summary.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_summary.csv",
        index=False,
    )
    result.daily.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_daily.csv",
        index=False,
    )

    print()
    print("PRE-CUTOFF ML BASE RANKING")
    print(result.base_ranking.to_string(index=False))

    print()
    print("PRE-CUTOFF ML-ONLY ENSEMBLE RANKING")
    print(result.ensemble_ranking.to_string(index=False))

    print()
    print("FINAL ML CTR PREDICTIONS")
    print(result.final_predictions.to_string(index=False))

    print()
    print("FINAL HOLDOUT COMPARISON")
    print(result.summary.to_string(index=False))

    baseline = result.summary.loc[
        result.summary["Method"].eq("RecursiveDailyML")
    ].iloc[0]

    candidate = result.summary.loc[
        result.summary["Method"].eq(
            "MLOnlyCTREnsembleImpressions"
        )
    ].iloc[0]

    accepted = (
        float(candidate["TotalErrorPct"])
        < float(baseline["TotalErrorPct"])
        and float(candidate["WAPE"])
        < float(baseline["WAPE"])
        and float(candidate["TotalErrorPct"]) <= 30.0
    )

    gate = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "PureMLPrimary": True,
                "BaselineTotalErrorPct": float(
                    baseline["TotalErrorPct"]
                ),
                "CandidateTotalErrorPct": float(
                    candidate["TotalErrorPct"]
                ),
                "BaselineWAPE": float(
                    baseline["WAPE"]
                ),
                "CandidateWAPE": float(
                    candidate["WAPE"]
                ),
                "AbsoluteErrorThresholdPct": 30.0,
                "PromoteCandidate": bool(accepted),
            }
        ]
    )

    gate.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ml_only_ctr_gate.csv",
        index=False,
    )

    print()
    print("ML-ONLY PRODUCTION PROMOTION GATE")
    print(gate.to_string(index=False))

    print()
    print("[IMPORTANT] Production remains unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
