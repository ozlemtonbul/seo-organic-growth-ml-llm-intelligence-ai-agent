from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.features.feature_engineering import (
    get_feature_columns,
    prepare_training_data,
)


logger = get_logger(__name__)


# Stores the latest complete benchmark table produced during
# train_and_validate_models().
#
# This keeps the existing five-value return contract intact so
# downstream pipeline code is not broken. The benchmark table can
# later be exported separately through get_last_model_benchmark().
_LAST_MODEL_BENCHMARK = pd.DataFrame()


SUPPORTED_ALGORITHMS = (
    "RandomForest",
    "XGBoost",
    "LightGBM",
)


def regression_metrics(
    model_name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    train_rows: int,
    test_rows: int,
    algorithm: str | None = None,
    validation_method: str = "time_aware_holdout",
    selected: bool | None = None,
) -> Dict[str, Any]:
    """
    Calculate regression evaluation metrics.

    Parameters
    ----------
    model_name
        Business target name such as Next_Clicks.
    actual
        Actual holdout values.
    predicted
        Model predictions.
    train_rows
        Number of rows used for holdout training.
    test_rows
        Number of rows used for holdout validation.
    algorithm
        Algorithm name.
    validation_method
        Validation strategy used.
    selected
        Whether the candidate became the production winner.
    """
    metrics: Dict[str, Any] = {
        "Model": model_name,
        "MAE": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "R2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
        "TrainRows": int(
            train_rows
        ),
        "TestRows": int(
            test_rows
        ),
        "ValidationMethod": (
            validation_method
        ),
    }

    if algorithm is not None:
        metrics[
            "Algorithm"
        ] = str(
            algorithm
        )

    if selected is not None:
        metrics[
            "Selected"
        ] = bool(
            selected
        )

    return metrics


def validate_training_dataframe(
    train_df: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """
    Validate the training DataFrame before model fitting.
    """
    required_targets = [
        "target_clicks_next",
        "target_impressions_next",
    ]

    missing_targets = [
        column
        for column in required_targets
        if column not in train_df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: "
            f"{missing_targets}"
        )

    if len(
        train_df
    ) < SETTINGS.min_ml_rows:
        raise ValueError(
            "Insufficient rows for model training. "
            f"Required: "
            f"{SETTINGS.min_ml_rows}, "
            f"received: "
            f"{len(train_df)}."
        )

    if not feature_columns:
        raise ValueError(
            "No usable machine-learning "
            "feature columns were found."
        )

    if "date" not in train_df.columns:
        raise ValueError(
            "Time-aware validation requires "
            "a date column."
        )

    parsed_dates = pd.to_datetime(
        train_df["date"],
        errors="coerce",
    )

    valid_dates = (
        parsed_dates
        .dropna()
        .drop_duplicates()
    )

    if len(
        valid_dates
    ) < 2:
        raise ValueError(
            "Time-aware validation requires "
            "at least two distinct dates."
        )


def build_random_forest_model(
) -> RandomForestRegressor:
    """
    Build a configured Random Forest regression model.
    """
    return RandomForestRegressor(
        n_estimators=(
            SETTINGS.n_estimators
        ),
        max_depth=(
            SETTINGS.max_depth
        ),
        min_samples_leaf=(
            SETTINGS.min_samples_leaf
        ),
        random_state=(
            SETTINGS.random_state
        ),
        n_jobs=-1,
    )


def build_xgboost_model(
) -> XGBRegressor:
    """
    Build a configured XGBoost regression model.
    """
    return XGBRegressor(
        n_estimators=int(
            getattr(
                SETTINGS,
                "xgb_n_estimators",
                SETTINGS.n_estimators,
            )
        ),
        max_depth=int(
            getattr(
                SETTINGS,
                "xgb_max_depth",
                6,
            )
        ),
        learning_rate=float(
            getattr(
                SETTINGS,
                "xgb_learning_rate",
                0.05,
            )
        ),
        subsample=float(
            getattr(
                SETTINGS,
                "xgb_subsample",
                0.90,
            )
        ),
        colsample_bytree=float(
            getattr(
                SETTINGS,
                "xgb_colsample_bytree",
                0.90,
            )
        ),
        objective="reg:squarederror",
        random_state=(
            SETTINGS.random_state
        ),
        n_jobs=-1,
        verbosity=0,
    )


def build_lightgbm_model(
) -> LGBMRegressor:
    """
    Build a configured LightGBM regression model.
    """
    return LGBMRegressor(
        n_estimators=int(
            getattr(
                SETTINGS,
                "lgbm_n_estimators",
                SETTINGS.n_estimators,
            )
        ),
        max_depth=int(
            getattr(
                SETTINGS,
                "lgbm_max_depth",
                -1,
            )
        ),
        learning_rate=float(
            getattr(
                SETTINGS,
                "lgbm_learning_rate",
                0.05,
            )
        ),
        num_leaves=int(
            getattr(
                SETTINGS,
                "lgbm_num_leaves",
                31,
            )
        ),
        subsample=float(
            getattr(
                SETTINGS,
                "lgbm_subsample",
                0.90,
            )
        ),
        colsample_bytree=float(
            getattr(
                SETTINGS,
                "lgbm_colsample_bytree",
                0.90,
            )
        ),
        random_state=(
            SETTINGS.random_state
        ),
        n_jobs=-1,
        verbosity=-1,
    )


def build_candidate_models(
) -> Dict[str, Any]:
    """
    Build one fresh instance of every benchmark candidate.
    """
    return {
        "RandomForest": (
            build_random_forest_model()
        ),
        "XGBoost": (
            build_xgboost_model()
        ),
        "LightGBM": (
            build_lightgbm_model()
        ),
    }


def build_model_by_name(
    algorithm: str,
) -> Any:
    """
    Build one model from its registered algorithm name.
    """
    normalized = str(
        algorithm or ""
    ).strip()

    builders = {
        "RandomForest": (
            build_random_forest_model
        ),
        "XGBoost": (
            build_xgboost_model
        ),
        "LightGBM": (
            build_lightgbm_model
        ),
    }

    builder = builders.get(
        normalized
    )

    if builder is None:
        raise ValueError(
            "Unsupported forecasting algorithm: "
            f"{algorithm}. "
            "Supported algorithms: "
            f"{', '.join(SUPPORTED_ALGORITHMS)}"
        )

    return builder()


def build_time_aware_split(
    train_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.Timestamp,
]:
    """
    Build a chronological train/test split.

    The earliest dates are used for training and the most recent
    dates are reserved for validation.

    Unlike a random split, future observations cannot be used as
    model features for predicting earlier observations.

    The split is performed by date rather than arbitrary row
    position so all rows from the same day remain on the same
    side of the holdout boundary.
    """
    dates = pd.to_datetime(
        train_df["date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Time-aware validation cannot use "
            "invalid or missing date values."
        )

    unique_dates = (
        pd.Series(
            dates.unique()
        )
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if len(
        unique_dates
    ) < 2:
        raise ValueError(
            "Time-aware validation requires "
            "at least two distinct dates."
        )

    requested_test_dates = int(
        np.ceil(
            len(
                unique_dates
            )
            * float(
                SETTINGS.test_size
            )
        )
    )

    number_of_test_dates = min(
        max(
            requested_test_dates,
            1,
        ),
        len(
            unique_dates
        )
        - 1,
    )

    first_test_position = (
        len(
            unique_dates
        )
        - number_of_test_dates
    )

    first_test_date = pd.Timestamp(
        unique_dates.iloc[
            first_test_position
        ]
    )

    train_mask = (
        dates
        < first_test_date
    )

    test_mask = (
        dates
        >= first_test_date
    )

    train_idx = np.flatnonzero(
        train_mask.to_numpy()
    )

    test_idx = np.flatnonzero(
        test_mask.to_numpy()
    )

    if (
        len(
            train_idx
        )
        == 0
        or len(
            test_idx
        )
        == 0
    ):
        raise ValueError(
            "Time-aware split produced an "
            "empty train or test partition."
        )

    return (
        train_idx,
        test_idx,
        first_test_date,
    )


def _selection_sort_key(
    metrics: Dict[str, Any],
) -> Tuple[
    float,
    float,
    float,
]:
    """
    Return the deterministic model-selection sort key.

    Selection order:
    1. Lowest RMSE
    2. Lowest MAE
    3. Highest R2
    """
    rmse = float(
        metrics.get(
            "RMSE",
            np.inf,
        )
    )

    mae = float(
        metrics.get(
            "MAE",
            np.inf,
        )
    )

    r2 = float(
        metrics.get(
            "R2",
            -np.inf,
        )
    )

    return (
        rmse,
        mae,
        -r2,
    )


def benchmark_target_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> Tuple[
    str,
    pd.DataFrame,
]:
    """
    Benchmark Random Forest, XGBoost and LightGBM for one target.

    Returns
    -------
    winner_algorithm
        Best algorithm based primarily on holdout RMSE.
    benchmark_df
        Complete metrics for every candidate.
    """
    benchmark_rows: List[
        Dict[str, Any]
    ] = []

    candidate_metrics: Dict[
        str,
        Dict[str, Any],
    ] = {}

    candidate_models = (
        build_candidate_models()
    )

    for (
        algorithm,
        model,
    ) in candidate_models.items():

        try:
            model.fit(
                x_train,
                y_train,
            )

            predictions = model.predict(
                x_test
            )

            metrics = regression_metrics(
                model_name=model_name,
                actual=y_test,
                predicted=np.asarray(
                    predictions
                ),
                train_rows=len(
                    x_train
                ),
                test_rows=len(
                    x_test
                ),
                algorithm=algorithm,
                validation_method=(
                    "time_aware_holdout"
                ),
                selected=False,
            )

            candidate_metrics[
                algorithm
            ] = metrics

            benchmark_rows.append(
                metrics
            )

        except Exception as exc:
            logger.exception(
                "Model benchmark failed | "
                "Target: %s | Algorithm: %s",
                model_name,
                algorithm,
            )

            benchmark_rows.append(
                {
                    "Model": model_name,
                    "Algorithm": algorithm,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "TrainRows": int(
                        len(
                            x_train
                        )
                    ),
                    "TestRows": int(
                        len(
                            x_test
                        )
                    ),
                    "ValidationMethod": (
                        "time_aware_holdout"
                    ),
                    "Selected": False,
                    "Status": "failed",
                    "Error": str(
                        exc
                    )[
                        :500
                    ],
                }
            )

    if not candidate_metrics:
        raise RuntimeError(
            "All forecasting benchmark "
            f"models failed for {model_name}."
        )

    winner_algorithm = min(
        candidate_metrics,
        key=lambda algorithm: (
            _selection_sort_key(
                candidate_metrics[
                    algorithm
                ]
            )
        ),
    )

    for row in benchmark_rows:
        algorithm = str(
            row.get(
                "Algorithm",
                "",
            )
        )

        row[
            "Selected"
        ] = (
            algorithm
            == winner_algorithm
        )

        if "Status" not in row:
            row[
                "Status"
            ] = "success"

        if "Error" not in row:
            row[
                "Error"
            ] = ""

    benchmark_df = (
        pd.DataFrame(
            benchmark_rows
        )
        .sort_values(
            [
                "Selected",
                "RMSE",
                "MAE",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Model benchmark completed | "
        "Target: %s | Winner: %s",
        model_name,
        winner_algorithm,
    )

    return (
        winner_algorithm,
        benchmark_df,
    )


def _selected_metric_row(
    benchmark_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """
    Convert the selected benchmark row into the backward-compatible
    production metrics structure.

    model_metrics continues to contain only:
    - Next_Clicks
    - Next_Impressions

    This prevents recommendation confidence calculations from
    accidentally averaging all benchmark candidates.
    """
    selected_rows = benchmark_df[
        benchmark_df[
            "Selected"
        ].astype(
            bool
        )
    ]

    if selected_rows.empty:
        raise RuntimeError(
            "No selected benchmark model "
            f"was found for {model_name}."
        )

    selected = (
        selected_rows
        .iloc[0]
    )

    return {
        "Model": model_name,
        "Algorithm": str(
            selected[
                "Algorithm"
            ]
        ),
        "MAE": float(
            selected[
                "MAE"
            ]
        ),
        "RMSE": float(
            selected[
                "RMSE"
            ]
        ),
        "R2": float(
            selected[
                "R2"
            ]
        ),
        "TrainRows": int(
            selected[
                "TrainRows"
            ]
        ),
        "TestRows": int(
            selected[
                "TestRows"
            ]
        ),
        "ValidationMethod": str(
            selected[
                "ValidationMethod"
            ]
        ),
        "Selected": True,
    }


def _extract_feature_importance(
    model: Any,
    feature_columns: List[str],
    model_name: str,
    algorithm: str,
) -> pd.DataFrame:
    """
    Extract native tree feature importance from the winner model.
    """
    importance_values = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importance_values is None:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
                "Model",
                "Algorithm",
            ]
        )

    values = np.asarray(
        importance_values,
        dtype=float,
    )

    if len(
        values
    ) != len(
        feature_columns
    ):
        raise ValueError(
            "Feature importance length does "
            "not match feature column count."
        )

    return pd.DataFrame(
        {
            "Feature": (
                feature_columns
            ),
            "Importance": values,
            "Model": model_name,
            "Algorithm": algorithm,
        }
    )


def get_last_model_benchmark(
) -> pd.DataFrame:
    """
    Return the latest complete model benchmark table.

    The returned DataFrame is a copy so callers cannot mutate
    internal runtime state accidentally.
    """
    return (
        _LAST_MODEL_BENCHMARK
        .copy()
    )


def train_and_validate_models(
    train_df: pd.DataFrame,
    with_holiday: bool = False,
) -> Tuple[
    Any,
    Any,
    List[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Train, benchmark and select production forecasting models.

    Candidate algorithms
    --------------------
    - Random Forest
    - XGBoost
    - LightGBM

    Validation
    ----------
    Uses the most recent dates as a chronological holdout rather
    than random train/test splitting.

    Model selection
    ---------------
    Clicks and impressions are selected independently.

    Primary selection metric:
        Lowest RMSE

    Tie breakers:
        Lowest MAE
        Highest R2

    Production training
    -------------------
    After the winner is selected using the chronological holdout,
    a fresh instance of that winning algorithm is retrained using
    the complete available training dataset.

    Returns
    -------
    model_clicks
        Selected and fully retrained next-click model.
    model_impressions
        Selected and fully retrained next-impression model.
    feature_columns
        Feature columns used during training.
    metrics_df
        Holdout metrics for the two selected production models.
        This intentionally remains one row per target for backward
        compatibility with recommendation confidence logic.
    feature_importance_df
        Native feature importance from the selected models.

    Complete benchmark metrics are available separately through
    get_last_model_benchmark().
    """
    global _LAST_MODEL_BENCHMARK

    feature_columns = [
        column
        for column in get_feature_columns(
            with_holiday=(
                with_holiday
            ),
        )
        if column
        in train_df.columns
    ]

    validate_training_dataframe(
        train_df,
        feature_columns,
    )

    working_df = (
        train_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    working_df[
        "date"
    ] = pd.to_datetime(
        working_df[
            "date"
        ],
        errors="coerce",
    )

    (
        train_idx,
        test_idx,
        first_test_date,
    ) = build_time_aware_split(
        working_df
    )

    x = working_df[
        feature_columns
    ].copy()

    y_clicks = working_df[
        "target_clicks_next"
    ].copy()

    y_impressions = working_df[
        "target_impressions_next"
    ].copy()

    x_train = x.iloc[
        train_idx
    ].copy()

    x_test = x.iloc[
        test_idx
    ].copy()

    clicks_y_train = (
        y_clicks.iloc[
            train_idx
        ]
    )

    clicks_y_test = (
        y_clicks.iloc[
            test_idx
        ]
    )

    impressions_y_train = (
        y_impressions.iloc[
            train_idx
        ]
    )

    impressions_y_test = (
        y_impressions.iloc[
            test_idx
        ]
    )

    logger.info(
        "Time-aware model validation prepared | "
        "Train rows: %d | Test rows: %d | "
        "First test date: %s",
        len(
            train_idx
        ),
        len(
            test_idx
        ),
        first_test_date.date(),
    )

    (
        clicks_winner,
        clicks_benchmark,
    ) = benchmark_target_models(
        x_train=x_train,
        x_test=x_test,
        y_train=clicks_y_train,
        y_test=clicks_y_test,
        model_name="Next_Clicks",
    )

    (
        impressions_winner,
        impressions_benchmark,
    ) = benchmark_target_models(
        x_train=x_train,
        x_test=x_test,
        y_train=(
            impressions_y_train
        ),
        y_test=(
            impressions_y_test
        ),
        model_name=(
            "Next_Impressions"
        ),
    )

    benchmark_df = (
        pd.concat(
            [
                clicks_benchmark,
                impressions_benchmark,
            ],
            ignore_index=True,
        )
        .reset_index(
            drop=True
        )
    )

    benchmark_df[
        "FirstTestDate"
    ] = first_test_date

    _LAST_MODEL_BENCHMARK = (
        benchmark_df.copy()
    )

    # Preserve only selected metrics for downstream recommendation
    # confidence calculations.
    metrics_df = pd.DataFrame(
        [
            _selected_metric_row(
                benchmark_df=(
                    clicks_benchmark
                ),
                model_name=(
                    "Next_Clicks"
                ),
            ),
            _selected_metric_row(
                benchmark_df=(
                    impressions_benchmark
                ),
                model_name=(
                    "Next_Impressions"
                ),
            ),
        ]
    )

    # Rebuild fresh winning models and train them on ALL available
    # observations for production inference.
    model_clicks = (
        build_model_by_name(
            clicks_winner
        )
    )

    model_impressions = (
        build_model_by_name(
            impressions_winner
        )
    )

    model_clicks.fit(
        x,
        y_clicks,
    )

    model_impressions.fit(
        x,
        y_impressions,
    )

    importance_clicks = (
        _extract_feature_importance(
            model=model_clicks,
            feature_columns=(
                feature_columns
            ),
            model_name=(
                "Next_Clicks"
            ),
            algorithm=(
                clicks_winner
            ),
        )
    )

    importance_impressions = (
        _extract_feature_importance(
            model=(
                model_impressions
            ),
            feature_columns=(
                feature_columns
            ),
            model_name=(
                "Next_Impressions"
            ),
            algorithm=(
                impressions_winner
            ),
        )
    )

    feature_importance_df = (
        pd.concat(
            [
                importance_clicks,
                importance_impressions,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "Model",
                "Importance",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Production forecasting models selected | "
        "Clicks: %s | Impressions: %s",
        clicks_winner,
        impressions_winner,
    )

    return (
        model_clicks,
        model_impressions,
        feature_columns,
        metrics_df,
        feature_importance_df,
    )


def get_latest_page_state(
    seo_raw: pd.DataFrame,
    holiday_map: Optional[
        Dict[str, str]
    ] = None,
) -> pd.DataFrame:
    """
    Return the latest model-ready state for each page.
    """
    featured = prepare_training_data(
        seo_raw,
        holiday_map,
    )

    if featured.empty:
        return featured

    return (
        featured
        .sort_values(
            [
                "page",
                "date",
            ]
        )
        .groupby(
            "page",
            as_index=False,
        )
        .tail(1)
        .reset_index(
            drop=True
        )
    )


def safe_prediction(
    value: Any,
) -> float:
    """
    Convert a model prediction into a finite non-negative float.
    """
    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if (
        np.isnan(
            number
        )
        or np.isinf(
            number
        )
    ):
        return 0.0

    return max(
        0.0,
        number,
    )