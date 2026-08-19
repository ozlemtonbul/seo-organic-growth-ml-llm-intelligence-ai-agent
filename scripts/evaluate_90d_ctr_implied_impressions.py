from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ctr_implied_impressions import (
    evaluate_ctr_implied_impressions,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = PROJECT_ROOT / "data" / "historical"


def _history_file() -> Path:
    candidates = sorted(
        HISTORY_DIR.glob("gsc_page_daily_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Historical GSC CSV bulunamadı.")
    return candidates[0]


def _click_total() -> float:
    path = OUTPUT_DIR / "seo_ml_direct_calibrator_90d_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            "seo_ml_direct_calibrator_90d_summary.csv bulunamadı."
        )

    frame = pd.read_csv(path, low_memory=False)

    row = frame.loc[
        (frame["Metric"].eq("clicks"))
        & (frame["Method"].eq("DirectHorizonCalibratedML"))
    ]

    if row.empty:
        raise ValueError(
            "90d DirectHorizonCalibratedML clicks satırı bulunamadı."
        )

    # The calibrated daily total, not the unconstrained direct model target.
    daily_path = OUTPUT_DIR / "seo_ml_direct_calibrator_90d_daily.csv"
    daily = pd.read_csv(daily_path, low_memory=False)

    if "DirectCalibratedClicks" not in daily.columns:
        raise ValueError("DirectCalibratedClicks kolonu bulunamadı.")

    return float(
        pd.to_numeric(
            daily["DirectCalibratedClicks"],
            errors="coerce",
        ).fillna(0.0).head(90).sum()
    )


def main() -> int:
    print("=" * 112)
    print("SEO 90-DAY CTR-IMPLIED IMPRESSION FORECAST - DEVELOPMENT EVALUATION")
    print("=" * 112)
    print("[INFO] Uses the already-passed 90-day direct click forecast.")
    print("[INFO] Forecasts aggregate 90-day CTR from pre-cutoff history only.")
    print("[INFO] Impressions = predicted clicks / predicted CTR.")
    print("[INFO] Final 90-day holdout is used only for final comparison.")
    print("[INFO] Production forecast files are NOT modified.")

    historical = pd.read_csv(
        _history_file(),
        low_memory=False,
    )

    summary_path = OUTPUT_DIR / "seo_ml_backtest_strategic_90d_summary.csv"
    daily_path = OUTPUT_DIR / "seo_ml_backtest_strategic_90d_daily.csv"

    summary = pd.read_csv(summary_path, low_memory=False)
    cutoff = pd.to_datetime(
        summary.iloc[0]["CutoffDate"],
        errors="raise",
    ).normalize()

    baseline_daily = pd.read_csv(
        daily_path,
        low_memory=False,
    )

    predicted_click_total = _click_total()

    print()
    print(
        f"[INFO] Passed 90-day calibrated click total used: "
        f"{predicted_click_total:.2f}"
    )

    result = evaluate_ctr_implied_impressions(
        historical=historical,
        baseline_daily=baseline_daily,
        cutoff_date=cutoff,
        predicted_90d_click_total=predicted_click_total,
    )

    result.summary.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ctr_implied_summary.csv",
        index=False,
    )
    result.selection.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ctr_implied_selection.csv",
        index=False,
    )
    result.samples.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ctr_implied_samples.csv",
        index=False,
    )
    result.daily.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ctr_implied_daily.csv",
        index=False,
    )

    print()
    print("PRE-CUTOFF CTR MODEL SELECTION")
    print(result.selection.to_string(index=False))

    print()
    print("FINAL HOLDOUT COMPARISON")
    print(result.summary.to_string(index=False))

    baseline = result.summary.loc[
        result.summary["Method"].eq("RecursiveDailyML")
    ].iloc[0]

    candidate = result.summary.loc[
        result.summary["Method"].eq("CTRImpliedImpressionsML")
    ].iloc[0]

    accepted = (
        float(candidate["TotalErrorPct"]) < float(baseline["TotalErrorPct"])
        and float(candidate["WAPE"]) < float(baseline["WAPE"])
        and float(candidate["TotalErrorPct"]) <= 30.0
    )

    gate = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "BaselineTotalErrorPct": float(baseline["TotalErrorPct"]),
                "CandidateTotalErrorPct": float(candidate["TotalErrorPct"]),
                "BaselineWAPE": float(baseline["WAPE"]),
                "CandidateWAPE": float(candidate["WAPE"]),
                "AbsoluteErrorThresholdPct": 30.0,
                "PromoteCandidate": bool(accepted),
            }
        ]
    )

    gate.to_csv(
        OUTPUT_DIR / "seo_ml_90d_ctr_implied_gate.csv",
        index=False,
    )

    print()
    print("PRODUCTION PROMOTION GATE")
    print(gate.to_string(index=False))

    print()
    print("[OUTPUT] outputs/seo_ml_90d_ctr_implied_summary.csv")
    print("[IMPORTANT] Production remains unchanged.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
