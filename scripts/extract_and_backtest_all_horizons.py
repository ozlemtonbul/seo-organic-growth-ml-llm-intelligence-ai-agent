from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_gsc_historical_page_daily import (
    DEFAULT_START_DATE,
    _resolve_end_date,
    extract_history,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
    )
    parser.add_argument(
        "--end-date",
        default="",
    )
    parser.add_argument(
        "--reset-parts",
        action="store_true",
    )
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date).date()
    end_date = (
        pd.Timestamp(args.end_date).date()
        if args.end_date.strip()
        else _resolve_end_date()
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "historical"
        / f"gsc_page_daily_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
    )

    print("=" * 92)
    print("SEO HISTORICAL GSC + FULL-HORIZON BACKTEST PIPELINE")
    print("=" * 92)
    print("[1/2] Historical page-date GSC extraction")

    try:
        result = extract_history(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            reset_parts=bool(args.reset_parts),
        )
    except Exception as exc:
        print(f"[FAIL] Extraction stopped: {exc}")
        return 1

    print()
    print("[2/2] Leakage-safe backtest for all feasible horizons")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_ml_backtest.py"),
        "--strategic-source",
        str(result.output_path),
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if completed.returncode == 0:
        print()
        print("[PASS] Historical extraction + all feasible horizon backtests completed.")
        return 0

    if completed.returncode == 2:
        print()
        print("[WARN] Extraction completed, but not every strategic horizon had enough history.")
        print(
            "[INFO] Bu durum API'nin gercekte dondurdugu tarih araligina gore "
            "normal olabilir. Uretilen source ve calisabilen backtest sonuclari korunur."
        )
        return 2

    print()
    print(f"[FAIL] Full backtest exited with code {completed.returncode}.")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
