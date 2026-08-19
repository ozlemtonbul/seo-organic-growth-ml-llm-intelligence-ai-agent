from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.models.ml_backtesting import (
    run_holdout_backtest,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)


def _format_pct(
    value: object,
) -> str:
    try:
        return f"{float(value):.2f}%"
    except (
        TypeError,
        ValueError,
    ):
        return "-"


def main() -> int:
    source_path = (
        OUTPUT_DIR
        / "seo_integrated_data.csv"
    )

    if not source_path.exists():
        print(
            "[FAIL] outputs/seo_integrated_data.csv bulunamadı."
        )
        print(
            "Önce python main.py çalıştırılmalı."
        )
        return 1

    print(
        "=" * 76
    )
    print(
        "SEO MULTI-HORIZON ML - LEAKAGE-SAFE BACKTEST"
    )
    print(
        "=" * 76
    )

    integrated = pd.read_csv(
        source_path,
        low_memory=False,
    )

    result = (
        run_holdout_backtest(
            integrated_data=integrated,
            horizons=(
                7,
                14,
                30,
            ),
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "seo_ml_backtest_summary.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "seo_ml_backtest_daily.csv"
    )

    metrics_path = (
        OUTPUT_DIR
        / "seo_ml_backtest_model_metrics.csv"
    )

    result.summary.to_csv(
        summary_path,
        index=False,
    )

    result.daily.to_csv(
        daily_path,
        index=False,
    )

    result.model_metrics.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"[INFO] Cutoff date: "
        f"{result.cutoff_date.date()}"
    )
    print(
        f"[INFO] Unseen actual end: "
        f"{result.actual_end_date.date()}"
    )
    print()

    display_columns = [
        "HorizonDays",
        "PredictedClicks",
        "ActualClicks",
        "ClickTotalErrorPct",
        "ClickWAPE",
        "PredictedImpressions",
        "ActualImpressions",
        "ImpressionTotalErrorPct",
        "ImpressionWAPE",
        "ClickBiasPct",
    ]

    display = (
        result.summary[
            display_columns
        ]
        .copy()
    )

    for column in (
        "ClickTotalErrorPct",
        "ClickWAPE",
        "ImpressionTotalErrorPct",
        "ImpressionWAPE",
        "ClickBiasPct",
    ):
        display[column] = (
            display[column]
            .map(
                _format_pct
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Backtest completed without future-data leakage."
    )
    print(
        "[PASS] 7 / 14 / 30-day operational forecasts "
        "were compared with unseen real data."
    )
    print()
    print(
        f"[OUTPUT] {summary_path}"
    )
    print(
        f"[OUTPUT] {daily_path}"
    )
    print(
        f"[OUTPUT] {metrics_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
