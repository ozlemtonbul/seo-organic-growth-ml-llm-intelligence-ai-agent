from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception as exc:
        print(
            f"[FAIL] {name}: okunamadı / could not be read: {exc}"
        )
        return pd.DataFrame()


def _numeric(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def _first(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _weighted_position(
    dataframe: pd.DataFrame,
) -> float | None:
    position_col = _first(
        dataframe,
        (
            "position",
            "Position",
            "CurrentPosition",
        ),
    )

    impressions_col = _first(
        dataframe,
        (
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ),
    )

    if position_col is None:
        return None

    position = _numeric(
        dataframe,
        position_col,
    )

    if impressions_col is None:
        clean = position.dropna()
        return (
            float(clean.mean())
            if not clean.empty
            else None
        )

    impressions = _numeric(
        dataframe,
        impressions_col,
    )

    valid = (
        position.notna()
        & impressions.notna()
        & impressions.gt(0)
    )

    if valid.any():
        return float(
            (
                position.loc[valid]
                * impressions.loc[valid]
            ).sum()
            / impressions.loc[valid].sum()
        )

    clean = position.dropna()

    return (
        float(clean.mean())
        if not clean.empty
        else None
    )


def check_integrated() -> None:
    name = "seo_integrated_data.csv"
    data = _load(name)

    if data.empty:
        print(
            f"[FAIL] {name}: boş veya bulunamadı / empty or missing"
        )
        return

    date_col = _first(
        data,
        (
            "date",
            "Date",
        ),
    )

    page_col = _first(
        data,
        (
            "page",
            "Page",
            "url",
            "URL",
        ),
    )

    print(
        f"[INFO] {name}: rows={len(data):,}"
    )

    if date_col is not None:
        parsed = pd.to_datetime(
            data[date_col],
            errors="coerce",
        )

        valid_dates = parsed.dropna()

        if not valid_dates.empty:
            print(
                "[INFO] Integrated date range: "
                f"{valid_dates.min().date()} -> "
                f"{valid_dates.max().date()}"
            )

    if (
        date_col is not None
        and page_col is not None
    ):
        duplicates = int(
            data.duplicated(
                subset=[
                    date_col,
                    page_col,
                ]
            ).sum()
        )

        if duplicates == 0:
            print(
                "[PASS] date + page key is unique."
            )
        else:
            print(
                "[WARN] Duplicate date + page rows: "
                f"{duplicates:,}"
            )

    for column in (
        "clicks",
        "impressions",
        "sessions",
        "conversions",
        "revenue",
    ):
        if column not in data.columns:
            continue

        values = _numeric(
            data,
            column,
        )

        negatives = int(
            values.lt(0).sum()
        )

        if negatives == 0:
            print(
                f"[PASS] {column}: negative values not found."
            )
        else:
            print(
                f"[WARN] {column}: {negatives:,} negative rows."
            )

    if (
        "clicks" in data.columns
        and "impressions" in data.columns
    ):
        clicks = _numeric(
            data,
            "clicks",
        ).fillna(0)

        impressions = _numeric(
            data,
            "impressions",
        ).fillna(0)

        expected_ctr = (
            clicks
            / impressions.replace(
                0,
                pd.NA,
            )
        ).fillna(0)

        ctr_col = _first(
            data,
            (
                "ctr",
                "CTR",
            ),
        )

        if ctr_col is not None:
            actual_ctr = _numeric(
                data,
                ctr_col,
            ).fillna(0)

            maximum_error = float(
                (
                    actual_ctr
                    - expected_ctr
                ).abs().max()
            )

            if maximum_error <= 1e-6:
                print(
                    "[PASS] CTR matches clicks / impressions."
                )
            else:
                print(
                    "[WARN] CTR mismatch. Maximum absolute error: "
                    f"{maximum_error:.8f}"
                )

    weighted_position = _weighted_position(
        data
    )

    if weighted_position is not None:
        print(
            "[INFO] Impression-weighted average position: "
            f"{weighted_position:.4f}"
        )

    if "users" in data.columns:
        print(
            "[INFO] GA4 users is a non-additive metric across pages/dates. "
            "Do not interpret a simple sum of page-level users as exact "
            "period-level unique users."
        )

    if (
        "engagement_rate" in data.columns
    ):
        engagement_rate = _numeric(
            data,
            "engagement_rate",
        )

        invalid = int(
            (
                engagement_rate.lt(0)
                | engagement_rate.gt(1)
            ).sum()
        )

        if invalid == 0:
            print(
                "[PASS] engagement_rate is within 0-1."
            )
        else:
            print(
                "[WARN] engagement_rate outside 0-1: "
                f"{invalid:,} rows."
            )


def check_model_metrics() -> None:
    name = "seo_model_metrics.csv"
    data = _load(name)

    if data.empty:
        print(
            f"[WARN] {name}: empty or missing."
        )
        return

    print(
        f"[INFO] {name}: rows={len(data):,}"
    )

    for metric in (
        "MAE",
        "RMSE",
        "R2",
    ):
        if metric not in data.columns:
            print(
                f"[WARN] Missing model metric: {metric}"
            )

    if "RMSE" in data.columns:
        rmse = _numeric(
            data,
            "RMSE",
        )

        if rmse.notna().all():
            print(
                "[PASS] RMSE values are numeric."
            )

    if "R2" in data.columns:
        r2 = _numeric(
            data,
            "R2",
        )

        print(
            "[INFO] R2 range: "
            f"{r2.min():.4f} -> {r2.max():.4f}"
        )


def check_scenarios() -> None:
    name = "seo_scenario_simulation.csv"
    data = _load(name)

    if data.empty:
        print(
            f"[WARN] {name}: empty or missing."
        )
        return

    print(
        f"[INFO] {name}: rows={len(data):,}"
    )

    page_col = _first(
        data,
        (
            "page",
            "Page",
        ),
    )

    scenario_col = _first(
        data,
        (
            "Scenario",
            "scenario",
        ),
    )

    if (
        page_col is not None
        and scenario_col is not None
    ):
        duplicates = int(
            data.duplicated(
                subset=[
                    page_col,
                    scenario_col,
                ]
            ).sum()
        )

        if duplicates == 0:
            print(
                "[PASS] page + scenario rows are unique."
            )
        else:
            print(
                "[WARN] Duplicate page + scenario rows: "
                f"{duplicates:,}"
            )

    for column in (
        "PredictedNextClicks",
        "PredictedNextImpressions",
        "ExpectedNetValue",
    ):
        if column not in data.columns:
            continue

        values = _numeric(
            data,
            column,
        )

        if column.startswith(
            "Predicted"
        ):
            negative = int(
                values.lt(0).sum()
            )

            if negative == 0:
                print(
                    f"[PASS] {column}: no negative predictions."
                )
            else:
                print(
                    f"[WARN] {column}: {negative:,} negative rows."
                )


def check_recommendations() -> None:
    name = "seo_recommendations.csv"
    data = _load(name)

    if data.empty:
        print(
            f"[WARN] {name}: empty or missing."
        )
        return

    print(
        f"[INFO] {name}: rows={len(data):,}"
    )

    page_col = _first(
        data,
        (
            "page",
            "Page",
        ),
    )

    if page_col is not None:
        duplicate_pages = int(
            data.duplicated(
                subset=[
                    page_col
                ]
            ).sum()
        )

        if duplicate_pages == 0:
            print(
                "[PASS] Recommendation output has one row per page."
            )
        else:
            print(
                "[WARN] Duplicate recommendation pages: "
                f"{duplicate_pages:,}"
            )

    required = (
        "RecommendedAction",
        "PriorityTier",
        "ConfidenceLevel",
    )

    for column in required:
        if column not in data.columns:
            print(
                f"[WARN] Recommendation column missing: {column}"
            )



def check_multi_horizon_forecast() -> None:
    daily_name = "seo_ml_forecast_daily.csv"
    horizon_name = "seo_ml_forecast_horizons.csv"
    portfolio_name = "seo_ml_forecast_portfolio.csv"

    daily = _load(daily_name)
    horizons = _load(horizon_name)
    portfolio = _load(portfolio_name)

    if daily.empty:
        print(f"[WARN] {daily_name}: empty or missing.")
        return

    print(f"[INFO] {daily_name}: rows={len(daily):,}")

    required_daily = {
        "page",
        "ForecastDate",
        "HorizonDay",
        "PredictedClicks",
        "PredictedImpressions",
    }
    missing_daily = sorted(required_daily.difference(daily.columns))
    if missing_daily:
        print(f"[WARN] Multi-horizon daily columns missing: {missing_daily}")
    else:
        duplicate_count = int(
            daily.duplicated(subset=["page", "ForecastDate"]).sum()
        )
        if duplicate_count == 0:
            print("[PASS] ML daily forecast page + date rows are unique.")
        else:
            print(
                "[WARN] Duplicate ML daily page + date rows: "
                f"{duplicate_count:,}"
            )

        horizon_day = _numeric(daily, "HorizonDay")
        if horizon_day.min() == 1 and horizon_day.max() == 365:
            print("[PASS] ML daily forecast covers day 1 through day 365.")
        else:
            print(
                "[WARN] ML daily forecast horizon range: "
                f"{horizon_day.min()} -> {horizon_day.max()}"
            )

        for column in ("PredictedClicks", "PredictedImpressions"):
            values = _numeric(daily, column)
            negative = int(values.lt(0).sum())
            if negative == 0:
                print(f"[PASS] {column}: no negative ML forecasts.")
            else:
                print(f"[WARN] {column}: {negative:,} negative ML forecasts.")

    if horizons.empty:
        print(f"[WARN] {horizon_name}: empty or missing.")
    else:
        observed = set(
            pd.to_numeric(horizons.get("HorizonDays"), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if observed == {7, 14, 30, 90, 180, 365}:
            print("[PASS] Page-level ML horizons are exactly 7/14/30/90/180/365 days.")
        else:
            print(f"[WARN] Page-level ML horizons found: {sorted(observed)}")

    if portfolio.empty:
        print(f"[WARN] {portfolio_name}: empty or missing.")
    else:
        observed = set(
            pd.to_numeric(portfolio.get("HorizonDays"), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if observed == {7, 14, 30, 90, 180, 365} and len(portfolio) == 6:
            print("[PASS] Portfolio ML forecast has one row for each 7/14/30/90/180/365-day horizon.")
        else:
            print(
                "[WARN] Portfolio ML forecast horizon structure is unexpected: "
                f"rows={len(portfolio):,}, horizons={sorted(observed)}"
            )

def main() -> None:
    print(
        "=" * 72
    )
    print(
        "SEO ORGANIC GROWTH INTELLIGENCE - DATA VALIDATION"
    )
    print(
        "=" * 72
    )

    check_integrated()

    print(
        "-" * 72
    )

    check_model_metrics()

    print(
        "-" * 72
    )

    check_scenarios()

    print(
        "-" * 72
    )

    check_recommendations()

    print(
        "-" * 72
    )

    check_multi_horizon_forecast()

    print(
        "=" * 72
    )
    print(
        "Validation finished."
    )


if __name__ == "__main__":
    main()
