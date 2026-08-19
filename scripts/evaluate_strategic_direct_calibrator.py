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

from src.models.strategic_direct_calibrator import (
    evaluate_direct_horizon_calibration,
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


def _cutoff(
    horizon: int,
) -> pd.Timestamp:
    path = (
        OUTPUT_DIR
        / f"seo_ml_backtest_strategic_{horizon}d_summary.csv"
    )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    return pd.to_datetime(
        frame.iloc[
            0
        ][
            "CutoffDate"
        ],
        errors="raise",
    ).normalize()


def main() -> int:
    print(
        "=" * 108
    )
    print(
        "SEO STRATEGIC DIRECT-HORIZON CALIBRATOR - DEVELOPMENT EVALUATION"
    )
    print(
        "=" * 108
    )
    print(
        "[INFO] Direct model candidate selection uses only pre-cutoff rolling-origin samples."
    )
    print(
        "[INFO] Final 90/180 holdout is used only for the final comparison."
    )
    print(
        "[INFO] Existing RecursiveDailyML daily shape is retained and only its strategic total is calibrated."
    )
    print(
        "[INFO] Production forecast files are NOT modified."
    )

    historical = pd.read_csv(
        _history_file(),
        low_memory=False,
    )

    all_summary = []
    all_selection = []

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
                f"[SKIP] Baseline daily file missing: {daily_path}"
            )
            continue

        baseline = pd.read_csv(
            daily_path,
            low_memory=False,
        )

        cutoff = _cutoff(
            horizon
        )

        print()
        print(
            f"[RUN] {horizon}-day | cutoff={cutoff.date()}"
        )

        result = (
            evaluate_direct_horizon_calibration(
                historical=historical,
                baseline_daily=baseline,
                cutoff_date=cutoff,
                horizon_days=horizon,
            )
        )

        result.summary.to_csv(
            OUTPUT_DIR
            / f"seo_ml_direct_calibrator_{horizon}d_summary.csv",
            index=False,
        )

        result.daily.to_csv(
            OUTPUT_DIR
            / f"seo_ml_direct_calibrator_{horizon}d_daily.csv",
            index=False,
        )

        result.selection.to_csv(
            OUTPUT_DIR
            / f"seo_ml_direct_calibrator_{horizon}d_selection.csv",
            index=False,
        )

        result.samples.to_csv(
            OUTPUT_DIR
            / f"seo_ml_direct_calibrator_{horizon}d_samples.csv",
            index=False,
        )

        print()
        print(
            "PRE-CUTOFF DIRECT MODEL SELECTION"
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

        gate_rows = []

        for metric in (
            "clicks",
            "impressions",
        ):
            subset = result.summary.loc[
                result.summary[
                    "Metric"
                ].eq(
                    metric
                )
            ]

            baseline_row = subset.loc[
                subset[
                    "Method"
                ].eq(
                    "RecursiveDailyML"
                )
            ].iloc[
                0
            ]

            candidate_row = subset.loc[
                subset[
                    "Method"
                ].eq(
                    "DirectHorizonCalibratedML"
                )
            ].iloc[
                0
            ]

            accepted = (
                float(
                    candidate_row[
                        "TotalErrorPct"
                    ]
                )
                < float(
                    baseline_row[
                        "TotalErrorPct"
                    ]
                )
                and float(
                    candidate_row[
                        "WAPE"
                    ]
                )
                < float(
                    baseline_row[
                        "WAPE"
                    ]
                )
            )

            gate_rows.append(
                {
                    "HorizonDays": horizon,
                    "Metric": metric,
                    "BaselineTotalErrorPct": float(
                        baseline_row[
                            "TotalErrorPct"
                        ]
                    ),
                    "CandidateTotalErrorPct": float(
                        candidate_row[
                            "TotalErrorPct"
                        ]
                    ),
                    "BaselineWAPE": float(
                        baseline_row[
                            "WAPE"
                        ]
                    ),
                    "CandidateWAPE": float(
                        candidate_row[
                            "WAPE"
                        ]
                    ),
                    "PromoteCandidate": bool(
                        accepted
                    ),
                }
            )

        gate = pd.DataFrame(
            gate_rows
        )

        gate.to_csv(
            OUTPUT_DIR
            / f"seo_ml_direct_calibrator_{horizon}d_gate.csv",
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

        all_summary.append(
            result.summary
        )

        all_selection.append(
            result.selection
        )

    if not all_summary:
        print(
            "[FAIL] No strategic backtest output found."
        )
        return 1

    pd.concat(
        all_summary,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "seo_ml_direct_calibrator_evaluation.csv",
        index=False,
    )

    pd.concat(
        all_selection,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "seo_ml_direct_calibrator_selection.csv",
        index=False,
    )

    print()
    print(
        "=" * 108
    )
    print(
        "DIRECT-HORIZON CALIBRATOR EVALUATION COMPLETE"
    )
    print(
        "=" * 108
    )
    print(
        "[OUTPUT] outputs/seo_ml_direct_calibrator_evaluation.csv"
    )
    print(
        "[IMPORTANT] Production remains unchanged."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
