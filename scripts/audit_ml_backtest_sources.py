from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ml_backtesting import (
    minimum_coverage_days,
    standardize_gsc_page_daily,
    source_coverage,
    _resolve_column,
)


def inspect_roots(
    roots: list[Path],
) -> pd.DataFrame:
    rows = []

    for root in roots:
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
                header = pd.read_csv(
                    path,
                    nrows=5,
                    low_memory=False,
                )

                aliases = (
                    ("date", "day"),
                    ("page", "landing_page", "landingpage", "url"),
                    ("clicks", "click"),
                    ("impressions", "impression"),
                    ("position", "avg_position", "average_position"),
                )

                if not all(
                    _resolve_column(
                        header,
                        candidate_aliases,
                    ) is not None
                    for candidate_aliases in aliases
                ):
                    continue

                full = pd.read_csv(
                    path,
                    low_memory=False,
                )

                standardized = standardize_gsc_page_daily(
                    full
                )

                coverage = source_coverage(
                    standardized
                )

                calendar_days = int(
                    coverage["calendar_days"]
                )

                rows.append(
                    {
                        "path": str(path),
                        **coverage,
                        "Backtest90": (
                            "YES"
                            if calendar_days >= minimum_coverage_days(90)
                            else "NO"
                        ),
                        "Backtest180": (
                            "YES"
                            if calendar_days >= minimum_coverage_days(180)
                            else "NO"
                        ),
                        "Backtest365": (
                            "YES"
                            if calendar_days >= minimum_coverage_days(365)
                            else "NO"
                        ),
                    }
                )
            except Exception:
                continue

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result
        .drop_duplicates(
            subset=["path"]
        )
        .sort_values(
            [
                "calendar_days",
                "observed_days",
                "rows",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help=(
            "Extra folder to search recursively for historical GSC CSV files. "
            "Can be supplied more than once."
        ),
    )
    args = parser.parse_args()

    roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "outputs",
    ]

    for value in args.search_root:
        roots.append(
            Path(value)
        )

    print("=" * 100)
    print("SEO ML BACKTEST - HISTORICAL SOURCE AUDIT V2")
    print("=" * 100)

    discovery = inspect_roots(
        roots
    )

    if discovery.empty:
        print(
            "[FAIL] Uygun date/page/clicks/impressions/position CSV bulunamadı."
        )
        return 1

    display = discovery.copy()

    for column in (
        "min_date",
        "max_date",
    ):
        display[column] = display[column].astype(str)

    print(
        display[
            [
                "path",
                "min_date",
                "max_date",
                "calendar_days",
                "observed_days",
                "pages",
                "rows",
                "Backtest90",
                "Backtest180",
                "Backtest365",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"[INFO] 90-day minimum coverage: "
        f"{minimum_coverage_days(90)} calendar days."
    )
    print(
        f"[INFO] 180-day minimum coverage: "
        f"{minimum_coverage_days(180)} calendar days."
    )
    print(
        f"[INFO] 365-day minimum coverage: "
        f"{minimum_coverage_days(365)} calendar days."
    )

    feasible = discovery.loc[
        discovery["Backtest90"].eq("YES")
    ]

    if feasible.empty:
        print()
        print(
            "[WARN] Project içindeki dosyalar 90/180/365 strategic backtest "
            "için henüz yeterli değil."
        )
        return 2

    print()
    print(
        "[PASS] En az bir strategic horizon için uygun historical source bulundu:"
    )
    print(
        feasible.iloc[0]["path"]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
