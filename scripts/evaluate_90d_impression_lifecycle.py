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
        str(
            PROJECT_ROOT
        ),
    )

from src.models.impression_lifecycle_calibrator import (
    evaluate_90d_impression_lifecycle,
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


def _history_file() -> Path:
    candidates = sorted(
        HISTORY_DIR.glob(
            "gsc_page_daily_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "Historical GSC CSV bulunamadı."
        )

    return candidates[
        0
    ]


def main() -> int:
    print(
        "=" * 112
    )
    print(
        "SEO 90-DAY IMPRESSION LIFECYCLE CALIBRATOR - DEVELOPMENT EVALUATION"
    )
    print(
        "=" * 112
    )
    print(
        "[INFO] 90-day impressions only."
    )
    print(
        "[INFO] Candidate selection uses multiple pre-cutoff walk-forward folds."
    )
    print(
        "[INFO] Page lifecycle / active-page / concentration features are included."
    )
    print(
        "[INFO] Final 90-day holdout is used only for the final comparison."
    )
    print(
        "[INFO] Production forecast files are NOT modified."
    )

    history = pd.read_csv(
        _history_file(),
        low_memory=False,
    )

    summary_path = (
        OUTPUT_DIR
        / "seo_ml_backtest_strategic_90d_summary.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "seo_ml_backtest_strategic_90d_daily.csv"
    )

    summary = pd.read_csv(
        summary_path,
        low_memory=False,
    )

    cutoff = pd.to_datetime(
        summary.iloc[
            0
        ][
            "CutoffDate"
        ],
        errors="raise",
    ).normalize()

    baseline_daily = pd.read_csv(
        daily_path,
        low_memory=False,
    )

    result = (
        evaluate_90d_impression_lifecycle(
            historical=history,
            baseline_daily=baseline_daily,
            cutoff_date=cutoff,
        )
    )

    result.summary.to_csv(
        OUTPUT_DIR
        / "seo_ml_90d_impression_lifecycle_summary.csv",
        index=False,
    )

    result.selection.to_csv(
        OUTPUT_DIR
        / "seo_ml_90d_impression_lifecycle_selection.csv",
        index=False,
    )

    result.samples.to_csv(
        OUTPUT_DIR
        / "seo_ml_90d_impression_lifecycle_samples.csv",
        index=False,
    )

    result.daily.to_csv(
        OUTPUT_DIR
        / "seo_ml_90d_impression_lifecycle_daily.csv",
        index=False,
    )

    print()
    print(
        "PRE-CUTOFF WALK-FORWARD MODEL SELECTION"
    )
    print(
        result.selection.to_string(
            index=False
        )
    )

    print()
    print(
        "FINAL HOLDOUT COMPARISON"
    )
    print(
        result.summary.to_string(
            index=False
        )
    )

    baseline = result.summary.loc[
        result.summary[
            "Method"
        ].eq(
            "RecursiveDailyML"
        )
    ].iloc[
        0
    ]

    candidate = result.summary.loc[
        result.summary[
            "Method"
        ].eq(
            "LifecycleCalibratedML"
        )
    ].iloc[
        0
    ]

    accepted = (
        float(
            candidate[
                "TotalErrorPct"
            ]
        )
        < float(
            baseline[
                "TotalErrorPct"
            ]
        )
        and float(
            candidate[
                "WAPE"
            ]
        )
        < float(
            baseline[
                "WAPE"
            ]
        )
    )

    gate = pd.DataFrame(
        [
            {
                "HorizonDays": 90,
                "Metric": "impressions",
                "BaselineTotalErrorPct": float(
                    baseline[
                        "TotalErrorPct"
                    ]
                ),
                "CandidateTotalErrorPct": float(
                    candidate[
                        "TotalErrorPct"
                    ]
                ),
                "BaselineWAPE": float(
                    baseline[
                        "WAPE"
                    ]
                ),
                "CandidateWAPE": float(
                    candidate[
                        "WAPE"
                    ]
                ),
                "PromoteCandidate": bool(
                    accepted
                ),
            }
        ]
    )

    gate.to_csv(
        OUTPUT_DIR
        / "seo_ml_90d_impression_lifecycle_gate.csv",
        index=False,
    )

    print()
    print(
        "PRODUCTION PROMOTION GATE"
    )
    print(
        gate.to_string(
            index=False
        )
    )

    print()
    print(
        "[OUTPUT] outputs/seo_ml_90d_impression_lifecycle_summary.csv"
    )
    print(
        "[IMPORTANT] Production remains unchanged."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
