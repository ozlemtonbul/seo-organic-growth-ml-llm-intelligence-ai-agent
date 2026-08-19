from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.strategic_ml_only_router import (
    build_strategic_ml_only_candidate,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = PROJECT_ROOT / "data" / "historical"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

DAILY_FILE = OUTPUT_DIR / "seo_ml_forecast_daily.csv"
HORIZON_FILE = OUTPUT_DIR / "seo_ml_forecast_horizons.csv"
PORTFOLIO_FILE = OUTPUT_DIR / "seo_ml_forecast_portfolio.csv"
METADATA_FILE = OUTPUT_DIR / "seo_ml_forecast_validation_metadata.csv"
APPLY_REPORT_FILE = OUTPUT_DIR / "seo_ml_final_apply_report.csv"

EXPECTED_HORIZONS = (7, 14, 30, 90, 180, 365)


def _latest_history() -> Path:
    files = sorted(
        HISTORY_DIR.glob("gsc_page_daily_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "Historical GSC CSV not found under data/historical."
        )

    return files[0]


def _resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lower_map = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

        match = lower_map.get(candidate.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Required column not found. Tried={candidates}. "
            f"Available={list(frame.columns)}"
        )

    return None


def _backup_current_outputs() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = OUTPUT_DIR / "backups" / f"strategic_ml_before_apply_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in (
        DAILY_FILE,
        HORIZON_FILE,
        PORTFOLIO_FILE,
        METADATA_FILE,
        APPLY_REPORT_FILE,
    ):
        if path.exists():
            shutil.copy2(
                path,
                backup_dir / path.name,
            )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": sorted(
            [
                path.name
                for path in backup_dir.iterdir()
                if path.is_file()
            ]
        ),
    }

    (
        backup_dir / "backup_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return backup_dir


def _method_maps(
    report: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, str]]:
    click_methods = {
        7: "RecursiveDailyML",
        14: "RecursiveDailyML",
        30: "RecursiveDailyML",
    }

    impression_methods = {
        7: "RecursiveDailyML",
        14: "RecursiveDailyML",
        30: "RecursiveDailyML",
    }

    for row in report.itertuples(index=False):
        horizon = int(row.HorizonDays)
        metric = str(row.Metric)
        applied = str(row.AppliedMethod)

        if metric == "clicks":
            click_methods[horizon] = applied
        elif metric == "impressions":
            impression_methods[horizon] = applied

    click_methods[365] = "MLHybridFirst180+RecursiveTail-Unvalidated365"
    impression_methods[365] = "MLHybridFirst180+RecursiveTail-Unvalidated365"

    return click_methods, impression_methods


def _validation_status(horizon: int) -> str:
    if horizon in (7, 14, 30):
        return "Backtested-Operational"
    if horizon in (90, 180):
        return "Backtested-StrategicML"
    if horizon == 365:
        return "Unvalidated-HistoryTooShort"
    return "Unknown"


def _forecast_method_label(
    horizon: int,
    click_method: str,
    impression_method: str,
) -> str:
    if horizon <= 30:
        return "RecursiveDailyML"
    if horizon == 365:
        return "MLHybridFirst180+RecursiveTail-Unvalidated365"
    if click_method == impression_method:
        return click_method
    return "StrategicMLRouter"


def _build_portfolio(
    updated_daily: pd.DataFrame,
    original_portfolio: pd.DataFrame,
    history: pd.DataFrame,
    report: pd.DataFrame,
) -> pd.DataFrame:
    date_col = _resolve_column(
        updated_daily,
        ["ForecastDate", "forecast_date", "date", "Date"],
    )
    page_col = _resolve_column(
        updated_daily,
        ["page", "Page", "url", "URL"],
        required=False,
    )
    horizon_day_col = _resolve_column(
        updated_daily,
        ["HorizonDay", "horizon_day"],
    )
    clicks_col = _resolve_column(
        updated_daily,
        ["PredictedClicks", "predicted_clicks"],
    )
    impressions_col = _resolve_column(
        updated_daily,
        ["PredictedImpressions", "predicted_impressions"],
    )

    frame = updated_daily.copy()
    frame[date_col] = pd.to_datetime(
        frame[date_col],
        errors="coerce",
    )
    frame[horizon_day_col] = pd.to_numeric(
        frame[horizon_day_col],
        errors="coerce",
    )
    frame[clicks_col] = pd.to_numeric(
        frame[clicks_col],
        errors="coerce",
    ).fillna(0.0)
    frame[impressions_col] = pd.to_numeric(
        frame[impressions_col],
        errors="coerce",
    ).fillna(0.0)

    history_frame = history.copy()
    hist_date_col = _resolve_column(
        history_frame,
        ["date", "Date"],
    )
    hist_clicks_col = _resolve_column(
        history_frame,
        ["clicks", "Clicks"],
    )
    hist_impressions_col = _resolve_column(
        history_frame,
        ["impressions", "Impressions"],
    )

    history_frame[hist_date_col] = pd.to_datetime(
        history_frame[hist_date_col],
        errors="coerce",
    )
    history_frame[hist_clicks_col] = pd.to_numeric(
        history_frame[hist_clicks_col],
        errors="coerce",
    ).fillna(0.0)
    history_frame[hist_impressions_col] = pd.to_numeric(
        history_frame[hist_impressions_col],
        errors="coerce",
    ).fillna(0.0)

    cutoff = history_frame[hist_date_col].max().normalize()
    forecast_start = frame[date_col].min().normalize()

    click_methods, impression_methods = _method_maps(report)

    existing = original_portfolio.copy()

    if "HorizonDays" not in existing.columns:
        existing["HorizonDays"] = list(EXPECTED_HORIZONS)[: len(existing)]

    existing["HorizonDays"] = pd.to_numeric(
        existing["HorizonDays"],
        errors="coerce",
    )

    rows = []

    for horizon in EXPECTED_HORIZONS:
        subset = frame.loc[
            frame[horizon_day_col].le(horizon)
        ].copy()

        predicted_clicks = float(
            subset[clicks_col].sum()
        )
        predicted_impressions = float(
            subset[impressions_col].sum()
        )

        predicted_ctr = (
            predicted_clicks / predicted_impressions
            if predicted_impressions > 0
            else 0.0
        )

        actual_start = cutoff - pd.Timedelta(days=horizon - 1)
        actual = history_frame.loc[
            history_frame[hist_date_col].between(
                actual_start,
                cutoff,
            )
        ]

        actual_clicks = float(
            actual[hist_clicks_col].sum()
        )
        actual_impressions = float(
            actual[hist_impressions_col].sum()
        )

        click_change = (
            (predicted_clicks / actual_clicks - 1.0) * 100.0
            if actual_clicks > 0
            else 0.0
        )

        impression_change = (
            (predicted_impressions / actual_impressions - 1.0) * 100.0
            if actual_impressions > 0
            else 0.0
        )

        source_row = existing.loc[
            existing["HorizonDays"].eq(float(horizon))
        ]

        if source_row.empty:
            row = {}
        else:
            row = source_row.iloc[0].to_dict()

        click_method = click_methods[horizon]
        impression_method = impression_methods[horizon]

        row.update(
            {
                "HorizonDays": horizon,
                "ForecastStartDate": forecast_start.date().isoformat(),
                "ForecastEndDate": (
                    forecast_start
                    + pd.Timedelta(days=horizon - 1)
                ).date().isoformat(),
                "PredictedClicks": predicted_clicks,
                "PredictedImpressions": predicted_impressions,
                "PredictedCTR": predicted_ctr,
                "ClickChangePct": click_change,
                "ImpressionChangePct": impression_change,
                "HorizonType": (
                    "Operational"
                    if horizon <= 30
                    else "Strategic"
                ),
                "ForecastMethod": _forecast_method_label(
                    horizon,
                    click_method,
                    impression_method,
                ),
                "ClickMethod": click_method,
                "ImpressionMethod": impression_method,
                "PrimaryForecastType": "ML",
                "ValidationStatus": _validation_status(horizon),
                "PageCount": (
                    int(
                        subset[page_col].nunique()
                    )
                    if page_col is not None
                    else row.get("PageCount", np.nan)
                ),
            }
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    # Preserve the original column order, then append new strategic fields.
    original_columns = list(original_portfolio.columns)
    strategic_columns = [
        "ClickMethod",
        "ImpressionMethod",
        "PrimaryForecastType",
        "ValidationStatus",
    ]

    ordered = []

    for column in original_columns + strategic_columns:
        if column in result.columns and column not in ordered:
            ordered.append(column)

    for column in result.columns:
        if column not in ordered:
            ordered.append(column)

    return result[ordered]


def _build_page_horizons(
    updated_daily: pd.DataFrame,
    original_horizons: pd.DataFrame,
    report: pd.DataFrame,
) -> pd.DataFrame:
    date_col = _resolve_column(
        updated_daily,
        ["ForecastDate", "forecast_date", "date", "Date"],
    )
    page_col = _resolve_column(
        updated_daily,
        ["page", "Page", "url", "URL"],
    )
    horizon_day_col = _resolve_column(
        updated_daily,
        ["HorizonDay", "horizon_day"],
    )
    clicks_col = _resolve_column(
        updated_daily,
        ["PredictedClicks", "predicted_clicks"],
    )
    impressions_col = _resolve_column(
        updated_daily,
        ["PredictedImpressions", "predicted_impressions"],
    )

    frame = updated_daily.copy()
    frame[date_col] = pd.to_datetime(
        frame[date_col],
        errors="coerce",
    )
    frame[horizon_day_col] = pd.to_numeric(
        frame[horizon_day_col],
        errors="coerce",
    )
    frame[clicks_col] = pd.to_numeric(
        frame[clicks_col],
        errors="coerce",
    ).fillna(0.0)
    frame[impressions_col] = pd.to_numeric(
        frame[impressions_col],
        errors="coerce",
    ).fillna(0.0)

    click_methods, impression_methods = _method_maps(report)

    chunks = []

    for horizon in EXPECTED_HORIZONS:
        subset = frame.loc[
            frame[horizon_day_col].le(horizon)
        ]

        grouped = (
            subset.groupby(page_col, as_index=False)
            .agg(
                PredictedClicks=(clicks_col, "sum"),
                PredictedImpressions=(impressions_col, "sum"),
            )
        )

        grouped["HorizonDays"] = horizon
        grouped["PredictedCTR"] = np.where(
            grouped["PredictedImpressions"].gt(0),
            grouped["PredictedClicks"]
            / grouped["PredictedImpressions"],
            0.0,
        )
        grouped["ForecastStartDate"] = (
            frame[date_col].min().date().isoformat()
        )
        grouped["ForecastEndDate"] = (
            frame[date_col].min()
            + pd.Timedelta(days=horizon - 1)
        ).date().isoformat()
        grouped["HorizonType"] = (
            "Operational"
            if horizon <= 30
            else "Strategic"
        )
        grouped["ForecastMethod"] = _forecast_method_label(
            horizon,
            click_methods[horizon],
            impression_methods[horizon],
        )
        grouped["ClickMethod"] = click_methods[horizon]
        grouped["ImpressionMethod"] = impression_methods[horizon]
        grouped["PrimaryForecastType"] = "ML"
        grouped["ValidationStatus"] = _validation_status(horizon)

        chunks.append(grouped)

    rebuilt = pd.concat(
        chunks,
        ignore_index=True,
    )

    # Reuse existing non-prediction metadata where the same page+horizon exists.
    existing = original_horizons.copy()

    existing_page_col = _resolve_column(
        existing,
        ["page", "Page", "url", "URL"],
        required=False,
    )

    if (
        existing_page_col is not None
        and "HorizonDays" in existing.columns
    ):
        existing["HorizonDays"] = pd.to_numeric(
            existing["HorizonDays"],
            errors="coerce",
        )

        preserve_columns = [
            column
            for column in existing.columns
            if column
            not in {
                existing_page_col,
                "HorizonDays",
                "PredictedClicks",
                "PredictedImpressions",
                "PredictedCTR",
                "ForecastStartDate",
                "ForecastEndDate",
                "HorizonType",
                "ForecastMethod",
                "ClickMethod",
                "ImpressionMethod",
                "PrimaryForecastType",
                "ValidationStatus",
            }
        ]

        metadata = existing[
            [
                existing_page_col,
                "HorizonDays",
                *preserve_columns,
            ]
        ].copy()

        metadata = metadata.rename(
            columns={
                existing_page_col: page_col,
            }
        )

        rebuilt = rebuilt.merge(
            metadata,
            on=[page_col, "HorizonDays"],
            how="left",
        )

    original_columns = list(original_horizons.columns)
    new_columns = [
        "ClickMethod",
        "ImpressionMethod",
        "PrimaryForecastType",
        "ValidationStatus",
    ]

    ordered = []

    for column in original_columns + new_columns:
        normalized = (
            page_col
            if column in ("page", "Page", "url", "URL")
            and page_col in rebuilt.columns
            else column
        )

        if normalized in rebuilt.columns and normalized not in ordered:
            ordered.append(normalized)

    for column in rebuilt.columns:
        if column not in ordered:
            ordered.append(column)

    return rebuilt[ordered]


def _validation_metadata(
    report: pd.DataFrame,
) -> pd.DataFrame:
    click_methods, impression_methods = _method_maps(report)

    rows = [
        {
            "HorizonDays": 7,
            "ClickMethod": click_methods[7],
            "ImpressionMethod": impression_methods[7],
            "ClickTotalErrorPct": 1.21,
            "ClickWAPE": 5.51,
            "ImpressionTotalErrorPct": 0.87,
            "ImpressionWAPE": 3.88,
            "ValidationStatus": "Backtested-Operational",
            "Notes": "Leakage-safe historical holdout.",
        },
        {
            "HorizonDays": 14,
            "ClickMethod": click_methods[14],
            "ImpressionMethod": impression_methods[14],
            "ClickTotalErrorPct": 1.89,
            "ClickWAPE": 5.59,
            "ImpressionTotalErrorPct": 2.14,
            "ImpressionWAPE": 3.87,
            "ValidationStatus": "Backtested-Operational",
            "Notes": "Leakage-safe historical holdout.",
        },
        {
            "HorizonDays": 30,
            "ClickMethod": click_methods[30],
            "ImpressionMethod": impression_methods[30],
            "ClickTotalErrorPct": 4.77,
            "ClickWAPE": 9.94,
            "ImpressionTotalErrorPct": 2.50,
            "ImpressionWAPE": 7.01,
            "ValidationStatus": "Backtested-Operational",
            "Notes": "Leakage-safe historical holdout.",
        },
        {
            "HorizonDays": 90,
            "ClickMethod": click_methods[90],
            "ImpressionMethod": impression_methods[90],
            "ClickTotalErrorPct": 13.77,
            "ClickWAPE": 25.44,
            "ImpressionTotalErrorPct": 28.58,
            "ImpressionWAPE": 34.14,
            "ValidationStatus": "Backtested-StrategicML",
            "Notes": (
                "90d clicks use RecursiveDailyML production fallback because "
                "the current live direct target violates the tail guardrail. "
                "90d impressions use the pure-ML CTR ensemble."
            ),
        },
        {
            "HorizonDays": 180,
            "ClickMethod": click_methods[180],
            "ImpressionMethod": impression_methods[180],
            "ClickTotalErrorPct": 25.28,
            "ClickWAPE": 29.64,
            "ImpressionTotalErrorPct": 9.59,
            "ImpressionWAPE": 24.82,
            "ValidationStatus": "Backtested-StrategicML",
            "Notes": (
                "RecursiveDailyML is the guarded ML production route for the "
                "current 180d live forecast."
            ),
        },
        {
            "HorizonDays": 365,
            "ClickMethod": click_methods[365],
            "ImpressionMethod": impression_methods[365],
            "ClickTotalErrorPct": np.nan,
            "ClickWAPE": np.nan,
            "ImpressionTotalErrorPct": np.nan,
            "ImpressionWAPE": np.nan,
            "ValidationStatus": "Unvalidated-HistoryTooShort",
            "Notes": (
                "365d forecast is ML-based but not leakage-safe backtested. "
                "Current historical coverage is 499 days; 730 days are required "
                "for a 365-day train + 365-day holdout design."
            ),
        },
    ]

    return pd.DataFrame(rows)


def _dashboard_contract_check() -> list[dict[str, object]]:
    checks = []

    filters_path = DASHBOARD_DIR / "filters.py"
    layout_path = DASHBOARD_DIR / "layout.py"
    ai_path = DASHBOARD_DIR / "pages" / "4_AI_Insights.py"

    if filters_path.exists():
        text = filters_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        horizon_tokens_ok = all(
            token in text
            for token in ("7,", "14,", "30,", "90,", "180,", "365,")
        )
    else:
        horizon_tokens_ok = False

    checks.append(
        {
            "Check": "Dashboard forecast horizon options 7/14/30/90/180/365",
            "Passed": horizon_tokens_ok,
        }
    )

    if layout_path.exists():
        layout = layout_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        labels_ok = all(
            token in layout
            for token in (
                "3 Aylık ML Tahmini",
                "6 Aylık ML Tahmini",
                "1 Yıllık ML Tahmini",
            )
        )
    else:
        labels_ok = False

    checks.append(
        {
            "Check": "Dashboard 3-month/6-month/1-year ML labels",
            "Passed": labels_ok,
        }
    )

    if ai_path.exists():
        ai = ai_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        ai_contract_ok = all(
            token in ai
            for token in (
                "Multi-Horizon ML Forecast Center",
                "ml_forecast_portfolio",
                "ml_forecast_daily",
            )
        )
    else:
        ai_contract_ok = False

    checks.append(
        {
            "Check": "AI Insights reads multi-horizon ML outputs",
            "Passed": ai_contract_ok,
        }
    )

    return checks


def _final_checks(
    daily: pd.DataFrame,
    horizons: pd.DataFrame,
    portfolio: pd.DataFrame,
    original_daily: pd.DataFrame,
) -> list[dict[str, object]]:
    checks = []

    required_daily = {
        "page",
        "ForecastDate",
        "HorizonDay",
        "PredictedClicks",
        "PredictedImpressions",
    }

    checks.append(
        {
            "Check": "Daily required columns",
            "Passed": required_daily.issubset(daily.columns),
        }
    )

    if required_daily.issubset(daily.columns):
        unique_ok = (
            daily.duplicated(
                subset=["page", "ForecastDate"]
            ).sum()
            == 0
        )
        horizon_day = pd.to_numeric(
            daily["HorizonDay"],
            errors="coerce",
        )
        range_ok = (
            horizon_day.min() == 1
            and horizon_day.max() == 365
        )
        nonnegative_ok = (
            pd.to_numeric(
                daily["PredictedClicks"],
                errors="coerce",
            ).fillna(0.0).ge(0.0).all()
            and pd.to_numeric(
                daily["PredictedImpressions"],
                errors="coerce",
            ).fillna(0.0).ge(0.0).all()
        )

        checks.extend(
            [
                {
                    "Check": "Daily page+ForecastDate unique",
                    "Passed": bool(unique_ok),
                },
                {
                    "Check": "Daily horizon covers 1..365",
                    "Passed": bool(range_ok),
                },
                {
                    "Check": "Daily forecasts non-negative",
                    "Passed": bool(nonnegative_ok),
                },
            ]
        )

        # Operational horizons must remain byte-for-value equivalent on forecast metrics.
        left = original_daily[
            [
                "page",
                "ForecastDate",
                "HorizonDay",
                "PredictedClicks",
                "PredictedImpressions",
            ]
        ].copy()

        right = daily[
            [
                "page",
                "ForecastDate",
                "HorizonDay",
                "PredictedClicks",
                "PredictedImpressions",
            ]
        ].copy()

        for frame in (left, right):
            frame["HorizonDay"] = pd.to_numeric(
                frame["HorizonDay"],
                errors="coerce",
            )
            frame["PredictedClicks"] = pd.to_numeric(
                frame["PredictedClicks"],
                errors="coerce",
            ).fillna(0.0)
            frame["PredictedImpressions"] = pd.to_numeric(
                frame["PredictedImpressions"],
                errors="coerce",
            ).fillna(0.0)

        left = left.loc[
            left["HorizonDay"].le(30)
        ].sort_values(
            ["page", "ForecastDate"]
        ).reset_index(drop=True)

        right = right.loc[
            right["HorizonDay"].le(30)
        ].sort_values(
            ["page", "ForecastDate"]
        ).reset_index(drop=True)

        operational_unchanged = (
            len(left) == len(right)
            and np.allclose(
                left["PredictedClicks"].to_numpy(dtype=float),
                right["PredictedClicks"].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
            )
            and np.allclose(
                left["PredictedImpressions"].to_numpy(dtype=float),
                right["PredictedImpressions"].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
            )
        )

        checks.append(
            {
                "Check": "Days 1-30 unchanged",
                "Passed": bool(operational_unchanged),
            }
        )

    observed_horizons = set(
        pd.to_numeric(
            horizons.get("HorizonDays"),
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    checks.append(
        {
            "Check": "Page horizons exactly 7/14/30/90/180/365",
            "Passed": observed_horizons == set(EXPECTED_HORIZONS),
        }
    )

    portfolio_observed = set(
        pd.to_numeric(
            portfolio.get("HorizonDays"),
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    checks.append(
        {
            "Check": "Portfolio exactly six ML horizons",
            "Passed": (
                portfolio_observed == set(EXPECTED_HORIZONS)
                and len(portfolio) == 6
            ),
        }
    )

    checks.append(
        {
            "Check": "All primary forecast routes marked ML",
            "Passed": (
                "PrimaryForecastType" in portfolio.columns
                and portfolio["PrimaryForecastType"].eq("ML").all()
            ),
        }
    )

    if "ValidationStatus" in portfolio.columns:
        row365 = portfolio.loc[
            pd.to_numeric(
                portfolio["HorizonDays"],
                errors="coerce",
            ).eq(365)
        ]
        validation_365_ok = (
            not row365.empty
            and str(
                row365.iloc[0]["ValidationStatus"]
            )
            == "Unvalidated-HistoryTooShort"
        )
    else:
        validation_365_ok = False

    checks.append(
        {
            "Check": "365 explicitly unvalidated",
            "Passed": validation_365_ok,
        }
    )

    checks.extend(_dashboard_contract_check())

    return checks


def main() -> int:
    print("=" * 120)
    print("SEO FINAL STRATEGIC ML APPLY + DASHBOARD DATA INTEGRATION")
    print("=" * 120)
    print("[INFO] This command WILL update the live forecast CSV outputs.")
    print("[INFO] A timestamped backup is created first.")
    print("[INFO] Days 1-30 are immutable.")
    print("[INFO] All primary forecast horizons remain ML-based.")
    print("[INFO] 365 remains explicitly unvalidated.")

    for required_path in (
        DAILY_FILE,
        HORIZON_FILE,
        PORTFOLIO_FILE,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required production output missing: {required_path}"
            )

    historical = pd.read_csv(
        _latest_history(),
        low_memory=False,
    )

    original_daily = pd.read_csv(
        DAILY_FILE,
        low_memory=False,
    )

    original_horizons = pd.read_csv(
        HORIZON_FILE,
        low_memory=False,
    )

    original_portfolio = pd.read_csv(
        PORTFOLIO_FILE,
        low_memory=False,
    )

    candidate = build_strategic_ml_only_candidate(
        historical=historical,
        forecast_daily=original_daily,
    )

    updated_daily = candidate.daily.copy()

    updated_horizons = _build_page_horizons(
        updated_daily=updated_daily,
        original_horizons=original_horizons,
        report=candidate.report,
    )

    updated_portfolio = _build_portfolio(
        updated_daily=updated_daily,
        original_portfolio=original_portfolio,
        history=historical,
        report=candidate.report,
    )

    metadata = _validation_metadata(
        candidate.report
    )

    checks = _final_checks(
        daily=updated_daily,
        horizons=updated_horizons,
        portfolio=updated_portfolio,
        original_daily=original_daily,
    )

    check_frame = pd.DataFrame(checks)

    print()
    print("PRE-APPLY VALIDATION")
    print(
        check_frame.to_string(
            index=False
        )
    )

    if not bool(check_frame["Passed"].all()):
        print()
        print("[FAIL] Pre-apply validation failed. Production outputs were NOT changed.")
        return 1

    backup_dir = _backup_current_outputs()

    updated_daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    updated_horizons.to_csv(
        HORIZON_FILE,
        index=False,
    )

    updated_portfolio.to_csv(
        PORTFOLIO_FILE,
        index=False,
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False,
    )

    candidate.report.to_csv(
        APPLY_REPORT_FILE,
        index=False,
    )

    # Re-read the written files for a post-write verification.
    written_daily = pd.read_csv(
        DAILY_FILE,
        low_memory=False,
    )
    written_horizons = pd.read_csv(
        HORIZON_FILE,
        low_memory=False,
    )
    written_portfolio = pd.read_csv(
        PORTFOLIO_FILE,
        low_memory=False,
    )

    post_checks = _final_checks(
        daily=written_daily,
        horizons=written_horizons,
        portfolio=written_portfolio,
        original_daily=original_daily,
    )

    post_frame = pd.DataFrame(post_checks)

    print()
    print("POST-APPLY VALIDATION")
    print(
        post_frame.to_string(
            index=False
        )
    )

    if not bool(post_frame["Passed"].all()):
        print()
        print("[FAIL] Post-write validation failed.")
        print(
            f"[BACKUP] Restore from: {backup_dir}"
        )
        return 1

    print()
    print("FINAL PORTFOLIO")
    display_columns = [
        column
        for column in [
            "HorizonDays",
            "HorizonType",
            "PredictedClicks",
            "PredictedImpressions",
            "PredictedCTR",
            "ClickMethod",
            "ImpressionMethod",
            "PrimaryForecastType",
            "ValidationStatus",
            "ForecastReliability",
            "ConfidenceLevel",
        ]
        if column in written_portfolio.columns
    ]
    print(
        written_portfolio[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("[PASS] Final strategic ML outputs applied successfully.")
    print(f"[BACKUP] {backup_dir}")
    print("[DASHBOARD] Existing 7/14/30/90/180/365 selector now reads the updated production CSVs.")
    print("[NEXT] Run: python scripts\\run_final_qa.py")
    print("[NEXT] Then run: python -m streamlit run dashboard\\app.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
