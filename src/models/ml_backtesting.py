from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.multi_horizon_forecasting import (
    run_multi_horizon_forecasting,
)


OPERATIONAL_HORIZONS: Tuple[int, ...] = (7, 14, 30)
STRATEGIC_HORIZONS: Tuple[int, ...] = (90, 180, 365)
ALL_HORIZONS: Tuple[int, ...] = (
    7,
    14,
    30,
    90,
    180,
    365,
)


@dataclass(frozen=True)
class BacktestResult:
    summary: pd.DataFrame
    daily: pd.DataFrame
    model_metrics: pd.DataFrame
    cutoff_date: pd.Timestamp
    actual_end_date: pd.Timestamp
    source_min_date: pd.Timestamp
    source_max_date: pd.Timestamp


def _safe_pct(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(
        numerator
        / denominator
        * 100.0
    )


def _mae(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_values = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0.0)

    predicted_values = pd.to_numeric(
        predicted,
        errors="coerce",
    ).fillna(0.0)

    return float(
        np.mean(
            np.abs(
                predicted_values
                - actual_values
            )
        )
    )


def _rmse(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_values = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0.0)

    predicted_values = pd.to_numeric(
        predicted,
        errors="coerce",
    ).fillna(0.0)

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    predicted_values
                    - actual_values
                )
            )
        )
    )


def _wape(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_values = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0.0)

    predicted_values = pd.to_numeric(
        predicted,
        errors="coerce",
    ).fillna(0.0)

    denominator = float(
        np.abs(
            actual_values
        ).sum()
    )

    if denominator == 0:
        return 0.0

    return float(
        np.abs(
            predicted_values
            - actual_values
        ).sum()
        / denominator
        * 100.0
    )


def _normalized_column_map(
    columns: Sequence[str],
) -> dict[str, str]:
    return {
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): str(column)
        for column in columns
    }


def _resolve_column(
    dataframe: pd.DataFrame,
    aliases: Sequence[str],
) -> Optional[str]:
    mapping = _normalized_column_map(
        dataframe.columns
    )

    for alias in aliases:
        key = (
            str(alias)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if key in mapping:
            return mapping[key]

    return None


def standardize_gsc_page_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize raw or integrated GSC-like data to one page x calendar-day row.

    Raw Search Console files may contain query/device/country dimensions.
    Those duplicate page/date rows are aggregated before backtesting.
    Position is aggregated with impression weighting.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError(
            "Historical SEO/GSC source is empty."
        )

    date_column = _resolve_column(
        dataframe,
        (
            "date",
            "day",
        ),
    )

    page_column = _resolve_column(
        dataframe,
        (
            "page",
            "landing_page",
            "landingpage",
            "url",
        ),
    )

    clicks_column = _resolve_column(
        dataframe,
        (
            "clicks",
            "click",
        ),
    )

    impressions_column = _resolve_column(
        dataframe,
        (
            "impressions",
            "impression",
        ),
    )

    position_column = _resolve_column(
        dataframe,
        (
            "position",
            "avg_position",
            "average_position",
        ),
    )

    missing = []

    if date_column is None:
        missing.append("date")

    if page_column is None:
        missing.append("page")

    if clicks_column is None:
        missing.append("clicks")

    if impressions_column is None:
        missing.append("impressions")

    if position_column is None:
        missing.append("position")

    if missing:
        raise ValueError(
            "Historical source is missing required GSC fields: "
            f"{missing}"
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(
                dataframe[date_column],
                errors="coerce",
            ).dt.normalize(),
            "page": (
                dataframe[page_column]
                .astype(str)
                .str.strip()
            ),
            "clicks": pd.to_numeric(
                dataframe[clicks_column],
                errors="coerce",
            ).fillna(0.0),
            "impressions": pd.to_numeric(
                dataframe[impressions_column],
                errors="coerce",
            ).fillna(0.0),
            "position": pd.to_numeric(
                dataframe[position_column],
                errors="coerce",
            ).fillna(0.0),
        }
    )

    result = (
        result
        .dropna(
            subset=[
                "date",
            ]
        )
        .loc[
            lambda frame:
            frame["page"].ne("")
            & frame["page"].ne("nan")
        ]
        .copy()
    )

    result[
        "__position_weight"
    ] = (
        result["position"]
        * result["impressions"]
    )

    grouped = (
        result
        .groupby(
            [
                "date",
                "page",
            ],
            as_index=False,
        )
        .agg(
            clicks=(
                "clicks",
                "sum",
            ),
            impressions=(
                "impressions",
                "sum",
            ),
            position_weight=(
                "__position_weight",
                "sum",
            ),
            fallback_position=(
                "position",
                "mean",
            ),
        )
    )

    grouped[
        "position"
    ] = np.where(
        grouped[
            "impressions"
        ] > 0,
        grouped[
            "position_weight"
        ]
        / grouped[
            "impressions"
        ],
        grouped[
            "fallback_position"
        ],
    )

    grouped[
        "ctr"
    ] = np.where(
        grouped[
            "impressions"
        ] > 0,
        grouped[
            "clicks"
        ]
        / grouped[
            "impressions"
        ],
        0.0,
    )

    grouped = (
        grouped[
            [
                "date",
                "page",
                "clicks",
                "impressions",
                "position",
                "ctr",
            ]
        ]
        .sort_values(
            [
                "date",
                "page",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return grouped



def prepare_model_source(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve production feature columns when the source is already one
    page x calendar-day row. Raw multi-dimensional GSC data is aggregated.

    This prevents operational backtests from silently dropping GA4/exogenous
    features that are present in seo_integrated_data.csv.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError(
            "Backtest model source is empty."
        )

    required = {
        "date",
        "page",
        "clicks",
        "impressions",
        "position",
    }

    if not required.issubset(
        set(
            dataframe.columns
        )
    ):
        return standardize_gsc_page_daily(
            dataframe
        )

    result = dataframe.copy()

    result[
        "date"
    ] = pd.to_datetime(
        result[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    result[
        "page"
    ] = (
        result[
            "page"
        ]
        .astype(str)
        .str.strip()
    )

    result = (
        result
        .dropna(
            subset=[
                "date",
            ]
        )
        .loc[
            lambda frame:
            frame[
                "page"
            ].ne("")
            & frame[
                "page"
            ].ne("nan")
        ]
        .copy()
    )

    if result.duplicated(
        [
            "page",
            "date",
        ]
    ).any():
        return standardize_gsc_page_daily(
            result
        )

    for column in (
        "clicks",
        "impressions",
        "position",
    ):
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        ).fillna(0.0)

    if "ctr" not in result.columns:
        result[
            "ctr"
        ] = np.where(
            result[
                "impressions"
            ] > 0,
            result[
                "clicks"
            ]
            / result[
                "impressions"
            ],
            0.0,
        )

    return (
        result
        .sort_values(
            [
                "date",
                "page",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def minimum_coverage_days(
    horizon: int,
) -> int:
    """
    Minimum calendar coverage for an independent leakage-safe backtest.

    The holdout itself equals the horizon. Training lookback is at least
    90 days for strategic tests, and at least as long as the horizon.
    """
    horizon_days = int(
        horizon
    )

    training_days = max(
        90,
        horizon_days,
    )

    return (
        training_days
        + horizon_days
    )

def load_and_standardize_source(
    file_path: Path | str,
) -> pd.DataFrame:
    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            str(
                path
            )
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    return standardize_gsc_page_daily(
        dataframe
    )


def source_coverage(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    if dataframe is None or dataframe.empty:
        return {
            "min_date": None,
            "max_date": None,
            "calendar_days": 0,
            "observed_days": 0,
            "pages": 0,
            "rows": 0,
        }

    dates = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="coerce",
    ).dropna()

    min_date = dates.min()
    max_date = dates.max()

    calendar_days = (
        int(
            (
                max_date
                - min_date
            ).days
        )
        + 1
        if not dates.empty
        else 0
    )

    return {
        "min_date": min_date,
        "max_date": max_date,
        "calendar_days": calendar_days,
        "observed_days": int(
            dates.nunique()
        ),
        "pages": int(
            dataframe[
                "page"
            ].nunique()
        ),
        "rows": int(
            len(
                dataframe
            )
        ),
    }


def run_holdout_backtest(
    source: pd.DataFrame,
    horizons: Iterable[int],
    holdout_days: int,
    lookback_days: Optional[int] = None,
    backtest_class: str = "Operational",
) -> BacktestResult:
    """
    Leakage-safe holdout backtest of the production RecursiveDailyML engine.

    The model is trained only on dates at or before cutoff. Future actuals are
    kept completely outside the training data and compared after forecasting.
    """
    model_source = prepare_model_source(
        source
    )

    # Evaluation always uses one page x day GSC totals, while model training
    # retains any production feature columns available in the source.
    daily_source = standardize_gsc_page_daily(
        source
    )

    horizon_values = tuple(
        sorted(
            {
                int(value)
                for value in horizons
                if int(value) > 0
            }
        )
    )

    if not horizon_values:
        raise ValueError(
            "At least one positive horizon is required."
        )

    maximum_horizon = max(
        horizon_values
    )

    if int(
        holdout_days
    ) < maximum_horizon:
        raise ValueError(
            "holdout_days must be at least the maximum horizon."
        )

    source_min_date = pd.to_datetime(
        daily_source[
            "date"
        ]
    ).min()

    source_max_date = pd.to_datetime(
        daily_source[
            "date"
        ]
    ).max()

    cutoff_date = (
        source_max_date
        - pd.Timedelta(
            days=int(
                holdout_days
            )
        )
    )

    training = model_source.loc[
        model_source[
            "date"
        ] <= cutoff_date
    ].copy()

    if lookback_days is not None:
        training_start = (
            cutoff_date
            - pd.Timedelta(
                days=int(
                    lookback_days
                )
                - 1
            )
        )

        training = training.loc[
            training[
                "date"
            ] >= training_start
        ].copy()

    training_calendar_days = (
        int(
            (
                cutoff_date
                - pd.to_datetime(
                    training[
                        "date"
                    ]
                ).min()
            ).days
        )
        + 1
        if not training.empty
        else 0
    )

    if training_calendar_days < 21:
        raise ValueError(
            "Insufficient historical training coverage before cutoff. "
            f"Training calendar days: {training_calendar_days}."
        )

    known_pages = set(
        training[
            "page"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if not known_pages:
        raise ValueError(
            "No pages are available at the backtest cutoff."
        )

    actual_end_date = (
        cutoff_date
        + pd.Timedelta(
            days=maximum_horizon
        )
    )

    if actual_end_date > source_max_date:
        raise ValueError(
            "Historical source does not contain enough unseen actual days. "
            f"Need data through {actual_end_date.date()}, "
            f"but source ends at {source_max_date.date()}."
        )

    forecast_result = (
        run_multi_horizon_forecasting(
            seo_raw=training,
            horizons=horizon_values,
        )
    )

    forecast_daily = (
        forecast_result
        .daily_forecast
        .copy()
    )

    forecast_daily[
        "ForecastDate"
    ] = pd.to_datetime(
        forecast_daily[
            "ForecastDate"
        ],
        errors="coerce",
    ).dt.normalize()

    predicted_daily = (
        forecast_daily
        .groupby(
            "ForecastDate",
            as_index=False,
        )
        .agg(
            PredictedClicks=(
                "PredictedClicks",
                "sum",
            ),
            PredictedImpressions=(
                "PredictedImpressions",
                "sum",
            ),
            ForecastPageCount=(
                "page",
                "nunique",
            ),
        )
    )

    actual_window = daily_source.loc[
        (
            daily_source[
                "date"
            ] > cutoff_date
        )
        & (
            daily_source[
                "date"
            ] <= actual_end_date
        )
    ].copy()

    actual_known = actual_window.loc[
        actual_window[
            "page"
        ]
        .astype(str)
        .isin(
            known_pages
        )
    ].copy()

    actual_daily = (
        actual_known
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            ActualClicks=(
                "clicks",
                "sum",
            ),
            ActualImpressions=(
                "impressions",
                "sum",
            ),
            ActualKnownPageCount=(
                "page",
                "nunique",
            ),
        )
        .rename(
            columns={
                "date": "ForecastDate",
            }
        )
    )

    expected_dates = pd.DataFrame(
        {
            "ForecastDate": pd.date_range(
                cutoff_date
                + pd.Timedelta(
                    days=1
                ),
                actual_end_date,
                freq="D",
            )
        }
    )

    daily = (
        expected_dates
        .merge(
            predicted_daily,
            on="ForecastDate",
            how="left",
        )
        .merge(
            actual_daily,
            on="ForecastDate",
            how="left",
        )
        .sort_values(
            "ForecastDate"
        )
        .reset_index(
            drop=True
        )
    )

    numeric_columns = [
        "PredictedClicks",
        "PredictedImpressions",
        "ForecastPageCount",
        "ActualClicks",
        "ActualImpressions",
        "ActualKnownPageCount",
    ]

    for column in numeric_columns:
        if column not in daily.columns:
            daily[
                column
            ] = 0.0

        daily[
            column
        ] = pd.to_numeric(
            daily[
                column
            ],
            errors="coerce",
        ).fillna(0.0)

    daily[
        "ClickError"
    ] = (
        daily[
            "PredictedClicks"
        ]
        - daily[
            "ActualClicks"
        ]
    )

    daily[
        "ImpressionError"
    ] = (
        daily[
            "PredictedImpressions"
        ]
        - daily[
            "ActualImpressions"
        ]
    )

    new_pages = actual_window.loc[
        ~actual_window[
            "page"
        ]
        .astype(str)
        .isin(
            known_pages
        )
    ].copy()

    new_page_count = int(
        new_pages[
            "page"
        ]
        .nunique()
    )

    new_page_clicks = float(
        pd.to_numeric(
            new_pages[
                "clicks"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    new_page_impressions = float(
        pd.to_numeric(
            new_pages[
                "impressions"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    summary_rows = []

    for horizon in horizon_values:
        horizon_daily = daily.head(
            horizon
        ).copy()

        predicted_clicks = float(
            horizon_daily[
                "PredictedClicks"
            ].sum()
        )

        actual_clicks = float(
            horizon_daily[
                "ActualClicks"
            ].sum()
        )

        predicted_impressions = float(
            horizon_daily[
                "PredictedImpressions"
            ].sum()
        )

        actual_impressions = float(
            horizon_daily[
                "ActualImpressions"
            ].sum()
        )

        predicted_ctr = (
            predicted_clicks
            / predicted_impressions
            if predicted_impressions > 0
            else 0.0
        )

        actual_ctr = (
            actual_clicks
            / actual_impressions
            if actual_impressions > 0
            else 0.0
        )

        click_bias_pct = _safe_pct(
            predicted_clicks
            - actual_clicks,
            actual_clicks,
        )

        impression_bias_pct = _safe_pct(
            predicted_impressions
            - actual_impressions,
            actual_impressions,
        )

        summary_rows.append(
            {
                "BacktestClass": str(
                    backtest_class
                ),
                "HorizonDays": int(
                    horizon
                ),
                "CutoffDate": cutoff_date,
                "ActualStartDate": (
                    cutoff_date
                    + pd.Timedelta(
                        days=1
                    )
                ),
                "ActualEndDate": (
                    cutoff_date
                    + pd.Timedelta(
                        days=horizon
                    )
                ),
                "SourceMinDate": source_min_date,
                "SourceMaxDate": source_max_date,
                "TrainingCalendarDays": int(
                    training_calendar_days
                ),
                "TrainingRows": int(
                    len(
                        training
                    )
                ),
                "KnownPagesAtCutoff": int(
                    len(
                        known_pages
                    )
                ),
                "PredictedClicks": round(
                    predicted_clicks,
                    2,
                ),
                "ActualClicks": round(
                    actual_clicks,
                    2,
                ),
                "ClickTotalErrorPct": round(
                    abs(
                        click_bias_pct
                    ),
                    2,
                ),
                "ClickBiasPct": round(
                    click_bias_pct,
                    2,
                ),
                "ClickDailyMAE": round(
                    _mae(
                        horizon_daily[
                            "ActualClicks"
                        ],
                        horizon_daily[
                            "PredictedClicks"
                        ],
                    ),
                    2,
                ),
                "ClickDailyRMSE": round(
                    _rmse(
                        horizon_daily[
                            "ActualClicks"
                        ],
                        horizon_daily[
                            "PredictedClicks"
                        ],
                    ),
                    2,
                ),
                "ClickWAPE": round(
                    _wape(
                        horizon_daily[
                            "ActualClicks"
                        ],
                        horizon_daily[
                            "PredictedClicks"
                        ],
                    ),
                    2,
                ),
                "PredictedImpressions": round(
                    predicted_impressions,
                    2,
                ),
                "ActualImpressions": round(
                    actual_impressions,
                    2,
                ),
                "ImpressionTotalErrorPct": round(
                    abs(
                        impression_bias_pct
                    ),
                    2,
                ),
                "ImpressionBiasPct": round(
                    impression_bias_pct,
                    2,
                ),
                "ImpressionDailyMAE": round(
                    _mae(
                        horizon_daily[
                            "ActualImpressions"
                        ],
                        horizon_daily[
                            "PredictedImpressions"
                        ],
                    ),
                    2,
                ),
                "ImpressionDailyRMSE": round(
                    _rmse(
                        horizon_daily[
                            "ActualImpressions"
                        ],
                        horizon_daily[
                            "PredictedImpressions"
                        ],
                    ),
                    2,
                ),
                "ImpressionWAPE": round(
                    _wape(
                        horizon_daily[
                            "ActualImpressions"
                        ],
                        horizon_daily[
                            "PredictedImpressions"
                        ],
                    ),
                    2,
                ),
                "PredictedCTR": round(
                    predicted_ctr,
                    6,
                ),
                "ActualCTR": round(
                    actual_ctr,
                    6,
                ),
                "CTRAbsoluteError": round(
                    abs(
                        predicted_ctr
                        - actual_ctr
                    ),
                    6,
                ),
                "NewFuturePageCount": (
                    new_page_count
                ),
                "NewFuturePageActualClicks": round(
                    new_page_clicks,
                    2,
                ),
                "NewFuturePageActualImpressions": round(
                    new_page_impressions,
                    2,
                ),
                "BacktestMethod": (
                    "LeakageSafeCalendarHoldout"
                ),
                "ForecastMethod": (
                    "RecursiveDailyML"
                ),
            }
        )

    return BacktestResult(
        summary=pd.DataFrame(
            summary_rows
        ),
        daily=daily,
        model_metrics=(
            forecast_result
            .metrics
            .copy()
        ),
        cutoff_date=cutoff_date,
        actual_end_date=actual_end_date,
        source_min_date=source_min_date,
        source_max_date=source_max_date,
    )


def discover_historical_sources(
    project_root: Path | str,
) -> pd.DataFrame:
    """
    Inspect plausible CSV files and return coverage metadata.

    Files are not selected just by filename: they must contain recognizable
    date/page/clicks/impressions/position columns.
    """
    root = Path(
        project_root
    )

    search_roots = [
        root
        / "data",
        root
        / "outputs",
    ]

    candidate_paths: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob(
            "*.csv"
        ):
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

            candidate_paths.append(
                path
            )

    rows = []

    for path in sorted(
        set(
            candidate_paths
        )
    ):
        try:
            header = pd.read_csv(
                path,
                nrows=5,
                low_memory=False,
            )

            required_resolved = all(
                (
                    _resolve_column(
                        header,
                        aliases,
                    )
                    is not None
                )
                for aliases in (
                    (
                        "date",
                        "day",
                    ),
                    (
                        "page",
                        "landing_page",
                        "landingpage",
                        "url",
                    ),
                    (
                        "clicks",
                        "click",
                    ),
                    (
                        "impressions",
                        "impression",
                    ),
                    (
                        "position",
                        "avg_position",
                        "average_position",
                    ),
                )
            )

            if not required_resolved:
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

            rows.append(
                {
                    "path": str(
                        path
                    ),
                    **coverage,
                }
            )
        except Exception:
            continue

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    return (
        result
        .sort_values(
            [
                "calendar_days",
                "observed_days",
                "rows",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def choose_strategic_source(
    discovery: pd.DataFrame,
    minimum_calendar_days: int = 730,
) -> Optional[str]:
    if discovery is None or discovery.empty:
        return None

    eligible = discovery.loc[
        pd.to_numeric(
            discovery[
                "calendar_days"
            ],
            errors="coerce",
        ).fillna(0)
        >= int(
            minimum_calendar_days
        )
    ].copy()

    if eligible.empty:
        return None

    return str(
        eligible.iloc[
            0
        ][
            "path"
        ]
    )
