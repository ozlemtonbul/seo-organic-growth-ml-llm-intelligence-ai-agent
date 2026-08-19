from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

EXPECTED = {7, 14, 30, 90, 180, 365}


def _load(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)


def main() -> int:
    print("=" * 104)
    print("SEO STRATEGIC ML DASHBOARD INTEGRATION VALIDATOR")
    print("=" * 104)

    daily = _load("seo_ml_forecast_daily.csv")
    horizons = _load("seo_ml_forecast_horizons.csv")
    portfolio = _load("seo_ml_forecast_portfolio.csv")
    metadata = _load("seo_ml_forecast_validation_metadata.csv")

    checks = []

    daily_horizon = pd.to_numeric(
        daily["HorizonDay"],
        errors="coerce",
    )

    checks.append(
        (
            "Daily covers 1..365",
            daily_horizon.min() == 1
            and daily_horizon.max() == 365,
        )
    )

    checks.append(
        (
            "Daily page+ForecastDate unique",
            daily.duplicated(
                subset=["page", "ForecastDate"]
            ).sum()
            == 0,
        )
    )

    page_horizons = set(
        pd.to_numeric(
            horizons["HorizonDays"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    checks.append(
        (
            "Page horizons 7/14/30/90/180/365",
            page_horizons == EXPECTED,
        )
    )

    portfolio_horizons = set(
        pd.to_numeric(
            portfolio["HorizonDays"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    checks.append(
        (
            "Portfolio has six horizons",
            portfolio_horizons == EXPECTED
            and len(portfolio) == 6,
        )
    )

    checks.append(
        (
            "Primary route ML only",
            "PrimaryForecastType" in portfolio.columns
            and portfolio["PrimaryForecastType"].eq("ML").all(),
        )
    )

    row365 = portfolio.loc[
        pd.to_numeric(
            portfolio["HorizonDays"],
            errors="coerce",
        ).eq(365)
    ]

    checks.append(
        (
            "365 unvalidated label",
            not row365.empty
            and str(
                row365.iloc[0].get("ValidationStatus", "")
            )
            == "Unvalidated-HistoryTooShort",
        )
    )

    metadata_365 = metadata.loc[
        pd.to_numeric(
            metadata["HorizonDays"],
            errors="coerce",
        ).eq(365)
    ]

    checks.append(
        (
            "365 validation metrics intentionally blank",
            not metadata_365.empty
            and pd.isna(
                metadata_365.iloc[0]["ClickTotalErrorPct"]
            )
            and pd.isna(
                metadata_365.iloc[0]["ImpressionTotalErrorPct"]
            ),
        )
    )

    filters_path = DASHBOARD_DIR / "filters.py"
    layout_path = DASHBOARD_DIR / "layout.py"
    ai_path = DASHBOARD_DIR / "pages" / "4_AI_Insights.py"

    filters = (
        filters_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if filters_path.exists()
        else ""
    )

    checks.append(
        (
            "Dashboard selector contains 90/180/365",
            all(
                token in filters
                for token in ("90,", "180,", "365,")
            ),
        )
    )

    layout = (
        layout_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if layout_path.exists()
        else ""
    )

    checks.append(
        (
            "Dashboard labels 3/6/12 month ML",
            all(
                token in layout
                for token in (
                    "3 Aylık ML Tahmini",
                    "6 Aylık ML Tahmini",
                    "1 Yıllık ML Tahmini",
                )
            ),
        )
    )

    ai = (
        ai_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if ai_path.exists()
        else ""
    )

    checks.append(
        (
            "AI Insights consumes multi-horizon CSVs",
            all(
                token in ai
                for token in (
                    "ml_forecast_portfolio",
                    "ml_forecast_daily",
                    "Multi-Horizon ML Forecast Center",
                )
            ),
        )
    )

    failed = False

    for label, passed in checks:
        print(
            f"[{'PASS' if passed else 'FAIL'}] {label}"
        )
        failed = failed or not passed

    print("=" * 104)

    if failed:
        print("[FAIL] Strategic ML dashboard integration validation failed.")
        return 1

    print("[PASS] Strategic ML dashboard integration validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
