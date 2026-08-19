from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ml_backtesting import (
    OPERATIONAL_HORIZONS,
    STRATEGIC_HORIZONS,
    minimum_coverage_days,
    run_holdout_backtest,
    source_coverage,
    standardize_gsc_page_daily,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _print_summary(
    title: str,
    summary: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    columns = [
        "BacktestClass",
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

    print(
        summary[columns].to_string(
            index=False
        )
    )


def _discover_best_source() -> Path | None:
    candidates = []

    for root in (
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "outputs",
    ):
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            lowered = path.name.lower()

            if any(
                token in lowered
                for token in (
                    "forecast",
                    "backtest",
                    "scenario",
                    "recommend",
                    "shap",
                    "benchmark",
                    "feature_importance",
                )
            ):
                continue

            try:
                frame = pd.read_csv(
                    path,
                    low_memory=False,
                )

                standardized = standardize_gsc_page_daily(
                    frame
                )

                coverage = source_coverage(
                    standardized
                )

                if int(
                    coverage["calendar_days"]
                ) >= minimum_coverage_days(90):
                    candidates.append(
                        (
                            int(
                                coverage["calendar_days"]
                            ),
                            path,
                        )
                    )
            except Exception:
                continue

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )[0][1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategic-source",
        default="",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrated_path = (
        OUTPUT_DIR
        / "seo_integrated_data.csv"
    )

    if not integrated_path.exists():
        print(
            "[FAIL] outputs/seo_integrated_data.csv bulunamadı."
        )
        return 1

    print("=" * 100)
    print("SEO FULL-HORIZON ML BACKTEST V2")
    print("=" * 100)

    print(
        "[1/2] Operational: 7 / 14 / 30 days "
        "(production integrated feature set preserved)"
    )

    integrated = pd.read_csv(
        integrated_path,
        low_memory=False,
    )

    operational = run_holdout_backtest(
        source=integrated,
        horizons=OPERATIONAL_HORIZONS,
        holdout_days=30,
        lookback_days=None,
        backtest_class="Operational",
    )

    operational.summary.to_csv(
        OUTPUT_DIR
        / "seo_ml_backtest_operational_summary.csv",
        index=False,
    )

    operational.daily.to_csv(
        OUTPUT_DIR
        / "seo_ml_backtest_operational_daily.csv",
        index=False,
    )

    operational.model_metrics.to_csv(
        OUTPUT_DIR
        / "seo_ml_backtest_operational_model_metrics.csv",
        index=False,
    )

    _print_summary(
        "OPERATIONAL BACKTEST RESULTS",
        operational.summary,
    )

    print()
    print(
        "[2/2] Strategic: 90 / 180 / 365 days "
        "(each horizon validated independently when coverage permits)"
    )

    if args.strategic_source.strip():
        strategic_path = Path(
            args.strategic_source.strip()
        )

        if not strategic_path.is_absolute():
            strategic_path = (
                PROJECT_ROOT
                / strategic_path
            )
    else:
        strategic_path = (
            _discover_best_source()
        )

    if strategic_path is None:
        print()
        print(
            "[WARN] Strategic historical source not found."
        )
        print(
            "[INFO] Operational backtest completed and was saved."
        )
        print(
            "[NEXT] Find a longer historical GSC CSV with:"
        )
        print(
            'python scripts\\audit_ml_backtest_sources.py '
            '--search-root "C:\\Users\\DEMO STORE\\Desktop"'
        )
        return 2

    print(
        f"[INFO] Strategic source: {strategic_path}"
    )

    strategic_raw = pd.read_csv(
        strategic_path,
        low_memory=False,
    )

    strategic_daily = standardize_gsc_page_daily(
        strategic_raw
    )

    coverage = source_coverage(
        strategic_daily
    )

    calendar_days = int(
        coverage["calendar_days"]
    )

    strategic_summaries = []

    for horizon in STRATEGIC_HORIZONS:
        required = minimum_coverage_days(
            horizon
        )

        if calendar_days < required:
            print(
                f"[SKIP] {horizon}-day strategic backtest: "
                f"coverage={calendar_days}, required={required}."
            )
            continue

        result = run_holdout_backtest(
            source=strategic_raw,
            horizons=(horizon,),
            holdout_days=horizon,
            lookback_days=max(
                90,
                horizon,
            ),
            backtest_class="Strategic",
        )

        result.summary.to_csv(
            OUTPUT_DIR
            / f"seo_ml_backtest_strategic_{horizon}d_summary.csv",
            index=False,
        )

        result.daily.to_csv(
            OUTPUT_DIR
            / f"seo_ml_backtest_strategic_{horizon}d_daily.csv",
            index=False,
        )

        result.model_metrics.to_csv(
            OUTPUT_DIR
            / f"seo_ml_backtest_strategic_{horizon}d_model_metrics.csv",
            index=False,
        )

        strategic_summaries.append(
            result.summary
        )

        _print_summary(
            f"STRATEGIC {horizon}-DAY BACKTEST",
            result.summary,
        )

    combined_parts = [
        operational.summary,
    ]

    if strategic_summaries:
        combined_parts.extend(
            strategic_summaries
        )

    combined = (
        pd.concat(
            combined_parts,
            ignore_index=True,
        )
        .sort_values(
            "HorizonDays"
        )
    )

    combined.to_csv(
        OUTPUT_DIR
        / "seo_ml_backtest_all_available_horizons_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("BACKTEST COMPLETE FOR ALL AVAILABLE HORIZONS")
    print("=" * 100)
    print(
        combined[
            [
                "HorizonDays",
                "BacktestClass",
                "ClickTotalErrorPct",
                "ClickWAPE",
                "ImpressionTotalErrorPct",
                "ImpressionWAPE",
            ]
        ].to_string(index=False)
    )

    if not strategic_summaries:
        print(
            "[WARN] No strategic horizon had sufficient historical coverage."
        )
        return 2

    print(
        "[PASS] All feasible horizons were tested without future-data leakage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
