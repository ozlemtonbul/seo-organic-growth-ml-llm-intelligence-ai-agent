
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.strategic_production_router import (
    build_strategic_production_candidate,
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
        raise FileNotFoundError("Historical GSC file not found.")
    return files[0]


def main() -> int:
    print("=" * 120)
    print("SEO STRATEGIC PRODUCTION CANDIDATE V2 - GUARDED DRY RUN")
    print("=" * 120)
    print("[INFO] Days 1-30 are immutable; 7/14/30 must remain EXACTLY baseline.")
    print("[INFO] 90-day strategic adjustments may change only days 31-90.")
    print("[INFO] 180-day strategic adjustments may change only days 91-180.")
    print("[INFO] Unsafe production scaling triggers fallback instead of forced adjustment.")
    print("[INFO] 365 is labelled HybridFirst180+RecursiveTail-Unvalidated365.")
    print("[INFO] Existing production outputs are NOT overwritten.")

    history = pd.read_csv(
        _latest_history(),
        low_memory=False,
    )

    forecast_path = OUTPUT_DIR / "seo_ml_forecast_daily.csv"

    if not forecast_path.exists():
        raise FileNotFoundError(str(forecast_path))

    forecast = pd.read_csv(
        forecast_path,
        low_memory=False,
    )

    result = build_strategic_production_candidate(
        historical=history,
        forecast_daily=forecast,
    )

    result.daily.to_csv(
        OUTPUT_DIR / "seo_ml_forecast_daily_strategic_candidate_v2.csv",
        index=False,
    )
    result.portfolio.to_csv(
        OUTPUT_DIR / "seo_ml_forecast_portfolio_strategic_candidate_v2.csv",
        index=False,
    )
    result.report.to_csv(
        OUTPUT_DIR / "seo_strategic_production_candidate_v2_report.csv",
        index=False,
    )
    result.qa.to_csv(
        OUTPUT_DIR / "seo_strategic_production_candidate_v2_qa.csv",
        index=False,
    )

    print()
    print("STRATEGIC METHOD REPORT")
    print(result.report.to_string(index=False))

    print()
    print("PORTFOLIO CANDIDATE")
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

    if operational_ok and nonnegative_ok and invariant_ok:
        print()
        print("[PASS] Strategic production candidate V2 dry-run checks passed.")
        print("[IMPORTANT] PASS means structural/guardrail QA passed.")
        print("[IMPORTANT] It does NOT mean 365-day accuracy was backtested.")
    else:
        print()
        print("[FAIL] Strategic production candidate V2 dry-run checks failed.")
        return 1

    print("[IMPORTANT] No original production output was overwritten.")
    print("[NEXT] Review V2 methods/guardrail fallbacks before final apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
