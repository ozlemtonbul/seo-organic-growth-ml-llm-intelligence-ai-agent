from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

EXPECTED_HORIZONS = (7, 14, 30, 90, 180, 365)


def _load(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def _resolve(frame: pd.DataFrame, names: list[str]) -> str:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(
        f"Column not found. Tried={names}, available={list(frame.columns)}"
    )


def _check(label: str, passed: bool, detail: str = "") -> tuple[str, bool, str]:
    return label, bool(passed), detail


def main() -> int:
    print("=" * 112)
    print("SEO ML HORIZON AUTO SMOKE CHECK")
    print("=" * 112)

    daily = _load("seo_ml_forecast_daily.csv")
    portfolio = _load("seo_ml_forecast_portfolio.csv")

    date_col = _resolve(
        daily,
        ["ForecastDate", "forecast_date", "date", "Date"],
    )
    horizon_day_col = _resolve(
        daily,
        ["HorizonDay", "horizon_day"],
    )
    clicks_col = _resolve(
        daily,
        ["PredictedClicks", "predicted_clicks"],
    )
    impressions_col = _resolve(
        daily,
        ["PredictedImpressions", "predicted_impressions"],
    )

    daily[date_col] = pd.to_datetime(
        daily[date_col],
        errors="coerce",
    )
    daily[horizon_day_col] = pd.to_numeric(
        daily[horizon_day_col],
        errors="coerce",
    )
    daily[clicks_col] = pd.to_numeric(
        daily[clicks_col],
        errors="coerce",
    ).fillna(0.0)
    daily[impressions_col] = pd.to_numeric(
        daily[impressions_col],
        errors="coerce",
    ).fillna(0.0)

    portfolio["HorizonDays"] = pd.to_numeric(
        portfolio["HorizonDays"],
        errors="coerce",
    )

    checks: list[tuple[str, bool, str]] = []

    observed = tuple(
        sorted(
            portfolio["HorizonDays"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    )

    checks.append(
        _check(
            "Six horizon rows exist",
            observed == EXPECTED_HORIZONS and len(portfolio) == 6,
            f"observed={observed}",
        )
    )

    checks.append(
        _check(
            "Daily forecast covers day 1..365",
            int(daily[horizon_day_col].min()) == 1
            and int(daily[horizon_day_col].max()) == 365,
            (
                f"min={int(daily[horizon_day_col].min())}, "
                f"max={int(daily[horizon_day_col].max())}"
            ),
        )
    )

    # Verify each portfolio row equals the actual cumulative daily output.
    reconciliation_rows = []

    for horizon in EXPECTED_HORIZONS:
        p = portfolio.loc[
            portfolio["HorizonDays"].eq(horizon)
        ]

        if p.empty:
            checks.append(
                _check(
                    f"{horizon}d portfolio row",
                    False,
                    "missing",
                )
            )
            continue

        p = p.iloc[0]

        subset = daily.loc[
            daily[horizon_day_col].le(horizon)
        ]

        daily_clicks = float(
            subset[clicks_col].sum()
        )
        daily_impressions = float(
            subset[impressions_col].sum()
        )

        portfolio_clicks = float(
            p["PredictedClicks"]
        )
        portfolio_impressions = float(
            p["PredictedImpressions"]
        )

        clicks_match = np.isclose(
            daily_clicks,
            portfolio_clicks,
            rtol=1e-9,
            atol=1e-4,
        )
        impressions_match = np.isclose(
            daily_impressions,
            portfolio_impressions,
            rtol=1e-9,
            atol=1e-4,
        )

        checks.append(
            _check(
                f"{horizon}d daily -> portfolio reconciliation",
                clicks_match and impressions_match,
                (
                    f"clicks={portfolio_clicks:.2f}, "
                    f"impressions={portfolio_impressions:.2f}"
                ),
            )
        )

        if "ForecastStartDate" in portfolio.columns and "ForecastEndDate" in portfolio.columns:
            start = pd.to_datetime(
                p["ForecastStartDate"],
                errors="coerce",
            )
            end = pd.to_datetime(
                p["ForecastEndDate"],
                errors="coerce",
            )
            span = (
                int((end - start).days) + 1
                if pd.notna(start) and pd.notna(end)
                else -1
            )
            checks.append(
                _check(
                    f"{horizon}d date span",
                    span == horizon,
                    f"span={span}",
                )
            )

        reconciliation_rows.append(
            {
                "HorizonDays": horizon,
                "PredictedClicks": portfolio_clicks,
                "PredictedImpressions": portfolio_impressions,
                "ClickMethod": p.get("ClickMethod", p.get("ForecastMethod", "")),
                "ImpressionMethod": p.get(
                    "ImpressionMethod",
                    p.get("ForecastMethod", ""),
                ),
                "ValidationStatus": p.get("ValidationStatus", ""),
            }
        )

    reconciliation = pd.DataFrame(reconciliation_rows)

    # Cumulative forecasts should grow with horizon.
    checks.append(
        _check(
            "Clicks increase with horizon",
            reconciliation["PredictedClicks"].is_monotonic_increasing,
            reconciliation[
                ["HorizonDays", "PredictedClicks"]
            ].to_dict("records").__repr__(),
        )
    )

    checks.append(
        _check(
            "Impressions increase with horizon",
            reconciliation["PredictedImpressions"].is_monotonic_increasing,
            reconciliation[
                ["HorizonDays", "PredictedImpressions"]
            ].to_dict("records").__repr__(),
        )
    )

    # Ensure strategic horizons are not accidentally showing the same totals.
    for left, right in ((30, 90), (90, 180), (180, 365)):
        a = reconciliation.loc[
            reconciliation["HorizonDays"].eq(left)
        ].iloc[0]
        b = reconciliation.loc[
            reconciliation["HorizonDays"].eq(right)
        ].iloc[0]

        changed = (
            not np.isclose(
                float(a["PredictedClicks"]),
                float(b["PredictedClicks"]),
                rtol=1e-8,
                atol=1e-4,
            )
            or not np.isclose(
                float(a["PredictedImpressions"]),
                float(b["PredictedImpressions"]),
                rtol=1e-8,
                atol=1e-4,
            )
        )

        checks.append(
            _check(
                f"{left}d -> {right}d forecast actually changes",
                changed,
            )
        )

    if "PrimaryForecastType" in portfolio.columns:
        checks.append(
            _check(
                "All primary routes marked ML",
                portfolio["PrimaryForecastType"].astype(str).eq("ML").all(),
            )
        )

    if "ValidationStatus" in portfolio.columns:
        row365 = portfolio.loc[
            portfolio["HorizonDays"].eq(365)
        ].iloc[0]
        checks.append(
            _check(
                "365 remains explicitly unvalidated",
                str(row365["ValidationStatus"])
                == "Unvalidated-HistoryTooShort",
                str(row365["ValidationStatus"]),
            )
        )

    # Static dashboard contract checks.
    filters_path = DASHBOARD_DIR / "filters.py"
    layout_path = DASHBOARD_DIR / "layout.py"
    ai_path = DASHBOARD_DIR / "pages" / "4_AI_Insights.py"

    filters_text = (
        filters_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if filters_path.exists()
        else ""
    )
    layout_text = (
        layout_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if layout_path.exists()
        else ""
    )
    ai_text = (
        ai_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if ai_path.exists()
        else ""
    )

    checks.append(
        _check(
            "Dashboard selector contains 7/14/30/90/180/365",
            all(
                token in filters_text
                for token in ("7,", "14,", "30,", "90,", "180,", "365,")
            ),
        )
    )

    checks.append(
        _check(
            "Dashboard labels contain 3/6/12 month ML",
            all(
                token in layout_text
                for token in (
                    "3 Aylık ML Tahmini",
                    "6 Aylık ML Tahmini",
                    "1 Yıllık ML Tahmini",
                )
            ),
        )
    )

    checks.append(
        _check(
            "AI Insights reads multi-horizon forecast outputs",
            all(
                token in ai_text
                for token in (
                    "ml_forecast_portfolio",
                    "ml_forecast_daily",
                )
            ),
        )
    )

    result = pd.DataFrame(
        checks,
        columns=["Check", "Passed", "Detail"],
    )

    print()
    print("AUTO CHECK RESULTS")
    print(result.to_string(index=False))

    print()
    print("HORIZON VALUES")
    print(reconciliation.to_string(index=False))

    failed = result.loc[
        ~result["Passed"]
    ]

    print()
    print("=" * 112)

    if failed.empty:
        print("[PASS] All ML horizon automatic smoke checks passed.")
        print(
            "[INFO] Platform Durumu kartlarinin horizon degismesiyle "
            "degismemesi normaldir; onlar veri/sistem durumunu temsil eder."
        )
        print(
            "[INFO] Horizon degisiminin otomatik teyidi daily->portfolio "
            "reconciliation ve 30/90/180/365 deger farklariyla yapildi."
        )
        return 0

    print("[FAIL] One or more ML horizon automatic smoke checks failed.")
    print(failed[["Check", "Detail"]].to_string(index=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
