from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )

from src.models.strategic_forecast_stabilizer import (
    evaluate_strategic_stabilizer,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical"
)


def _find_history() -> Path:
    candidates = sorted(
        HISTORY_DIR.glob(
            "gsc_page_daily_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "data/historical/gsc_page_daily_*.csv bulunamadı."
        )

    return candidates[
        0
    ]


def _base_r2(
    horizon: int,
) -> float:
    metrics_path = (
        OUTPUT_DIR
        / f"seo_ml_backtest_strategic_{horizon}d_model_metrics.csv"
    )

    if not metrics_path.exists():
        return 0.80

    metrics = pd.read_csv(
        metrics_path,
        low_memory=False,
    )

    if (
        metrics.empty
        or "R2" not in metrics.columns
    ):
        return 0.80

    values = pd.to_numeric(
        metrics[
            "R2"
        ],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.80

    return float(
        values.min()
    )


def _load_cutoff(
    horizon: int,
) -> pd.Timestamp:
    summary_path = (
        OUTPUT_DIR
        / f"seo_ml_backtest_strategic_{horizon}d_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            str(
                summary_path
            )
        )

    summary = pd.read_csv(
        summary_path,
        low_memory=False,
    )

    if (
        summary.empty
        or "CutoffDate" not in summary.columns
    ):
        raise ValueError(
            f"CutoffDate bulunamadı: {summary_path}"
        )

    return pd.to_datetime(
        summary.iloc[
            0
        ][
            "CutoffDate"
        ],
        errors="raise",
    ).normalize()


def main() -> int:
    print(
        "=" * 100
    )
    print(
        "SEO STRATEGIC FORECAST STABILIZER - DEVELOPMENT EVALUATION"
    )
    print(
        "=" * 100
    )

    history_path = (
        _find_history()
    )

    print(
        f"[INFO] Historical source: {history_path}"
    )
    print(
        "[INFO] Production forecast files are NOT modified by this evaluation."
    )

    historical = pd.read_csv(
        history_path,
        low_memory=False,
    )

    all_summaries = []

    for horizon in (
        90,
        180,
    ):
        daily_path = (
            OUTPUT_DIR
            / f"seo_ml_backtest_strategic_{horizon}d_daily.csv"
        )

        if not daily_path.exists():
            print(
                f"[SKIP] {horizon}d baseline daily file bulunamadı: "
                f"{daily_path}"
            )
            continue

        baseline_daily = pd.read_csv(
            daily_path,
            low_memory=False,
        )

        cutoff = (
            _load_cutoff(
                horizon
            )
        )

        base_r2 = (
            _base_r2(
                horizon
            )
        )

        print()
        print(
            f"[RUN] {horizon}-day | cutoff={cutoff.date()} "
            f"| base_r2={base_r2:.4f}"
        )

        result = (
            evaluate_strategic_stabilizer(
                historical=historical,
                baseline_daily=baseline_daily,
                horizon_days=horizon,
                cutoff_date=cutoff,
                base_r2=base_r2,
            )
        )

        result.daily.to_csv(
            OUTPUT_DIR
            / f"seo_ml_strategic_stabilizer_{horizon}d_daily.csv",
            index=False,
        )

        result.summary.to_csv(
            OUTPUT_DIR
            / f"seo_ml_strategic_stabilizer_{horizon}d_summary.csv",
            index=False,
        )

        result.anchor_diagnostics.to_csv(
            OUTPUT_DIR
            / f"seo_ml_strategic_stabilizer_{horizon}d_diagnostics.csv",
            index=False,
        )

        print(
            result.summary.to_string(
                index=False
            )
        )

        baseline = (
            result.summary.loc[
                result.summary[
                    "Method"
                ].eq(
                    "RecursiveDailyML"
                )
            ]
            .iloc[
                0
            ]
        )

        candidate = (
            result.summary.loc[
                result.summary[
                    "Method"
                ].eq(
                    "StrategicDampedEnsembleML"
                )
            ]
            .iloc[
                0
            ]
        )

        click_delta = (
            float(
                baseline[
                    "ClickTotalErrorPct"
                ]
            )
            - float(
                candidate[
                    "ClickTotalErrorPct"
                ]
            )
        )

        impression_delta = (
            float(
                baseline[
                    "ImpressionTotalErrorPct"
                ]
            )
            - float(
                candidate[
                    "ImpressionTotalErrorPct"
                ]
            )
        )

        print(
            f"[DELTA] {horizon}d click total-error improvement: "
            f"{click_delta:+.2f} percentage points"
        )

        print(
            f"[DELTA] {horizon}d impression total-error improvement: "
            f"{impression_delta:+.2f} percentage points"
        )

        all_summaries.append(
            result.summary
        )

    if not all_summaries:
        print(
            "[FAIL] Değerlendirilecek 90/180 backtest output bulunamadı."
        )
        return 1

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "seo_ml_strategic_stabilizer_evaluation.csv",
        index=False,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "STRATEGIC STABILIZER EVALUATION COMPLETE"
    )
    print(
        "=" * 100
    )

    print(
        combined.to_string(
            index=False
        )
    )

    print()
    print(
        "[OUTPUT] outputs/seo_ml_strategic_stabilizer_evaluation.csv"
    )
    print(
        "[IMPORTANT] Bu sadece development evaluation'dır; "
        "production forecast henüz değiştirilmedi."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
