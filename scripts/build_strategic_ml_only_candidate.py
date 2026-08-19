from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.strategic_ml_only_router import (
    build_strategic_ml_only_candidate,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = PROJECT_ROOT / "data" / "historical"


def _latest_history() -> Path:
    files = sorted(
        HISTORY_DIR.glob("gsc_page_daily_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "Historical GSC file not found."
        )

    return files[0]


def main() -> int:
    print("=" * 124)
    print("SEO STRATEGIC ML-ONLY PRODUCTION CANDIDATE - GUARDED DRY RUN")
    print("=" * 124)
    print("[INFO] 7/14/30 -> RecursiveDailyML.")
    print("[INFO] 90 clicks -> Direct ML when safe, otherwise RecursiveDailyML.")
    print("[INFO] 90 impressions -> PURE ML CTR ensemble only.")
    print("[INFO] 180 clicks -> Direct ML when safe, otherwise RecursiveDailyML.")
    print("[INFO] 180 impressions -> validated RecursiveDailyML.")
    print("[INFO] 365 -> ML hybrid, unvalidated because 730-day history is unavailable.")
    print("[INFO] Statistical baselines/champion methods are NOT primary production forecasts.")
    print("[INFO] Existing production outputs are NOT overwritten.")

    historical = pd.read_csv(
        _latest_history(),
        low_memory=False,
    )

    forecast_path = (
        OUTPUT_DIR / "seo_ml_forecast_daily.csv"
    )

    if not forecast_path.exists():
        raise FileNotFoundError(str(forecast_path))

    forecast = pd.read_csv(
        forecast_path,
        low_memory=False,
    )

    result = build_strategic_ml_only_candidate(
        historical=historical,
        forecast_daily=forecast,
    )

    result.daily.to_csv(
        OUTPUT_DIR / "seo_ml_forecast_daily_ml_only_candidate.csv",
        index=False,
    )

    result.portfolio.to_csv(
        OUTPUT_DIR / "seo_ml_forecast_portfolio_ml_only_candidate.csv",
        index=False,
    )

    result.report.to_csv(
        OUTPUT_DIR / "seo_strategic_ml_only_candidate_report.csv",
        index=False,
    )

    result.qa.to_csv(
        OUTPUT_DIR / "seo_strategic_ml_only_candidate_qa.csv",
        index=False,
    )

    print()
    print("ML-ONLY METHOD REPORT")
    print(result.report.to_string(index=False))

    print()
    print("PORTFOLIO ML CANDIDATE")
    print(result.portfolio.to_string(index=False))

    print()
    print("QA")
    print(result.qa.to_string(index=False))

    operational_ok = bool(
        result.qa.loc[
            result.qa["HorizonDays"].le(30),
            "OperationalUnchanged",
        ].all()
    )

    nonnegative_ok = bool(
        result.qa["NonNegative"].all()
    )

    invariant_ok = bool(
        result.qa["ImpressionsGTEClicks"].all()
    )

    ml_only_ok = bool(
        result.qa["MLPrimaryOnly"].all()
    )

    if (
        operational_ok
        and nonnegative_ok
        and invariant_ok
        and ml_only_ok
    ):
        print()
        print(
            "[PASS] Strategic ML-only production candidate dry-run checks passed."
        )
        print(
            "[IMPORTANT] All primary 7/14/30/90/180/365 forecast routes are ML-based."
        )
        print(
            "[IMPORTANT] 365 remains unvalidated, not because it is non-ML, "
            "but because current history is too short for a 365+365 backtest."
        )
    else:
        print()
        print(
            "[FAIL] Strategic ML-only production candidate dry-run checks failed."
        )
        return 1

    print(
        "[IMPORTANT] No original production output was overwritten."
    )
    print(
        "[NEXT] After review, create the final apply/dashboard integration package."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
