from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import shap

from config.logging_config import get_logger


logger = get_logger(__name__)


SHAP_DETAIL_COLUMNS = [
    "Model",
    "Algorithm",
    "RowIndex",
    "Feature",
    "FeatureValue",
    "SHAPValue",
    "AbsSHAPValue",
    "Direction",
    "BaseValue",
    "Prediction",
]


SHAP_SUMMARY_COLUMNS = [
    "Model",
    "Algorithm",
    "Feature",
    "MeanAbsSHAP",
    "MeanSHAP",
    "PositiveImpactRows",
    "NegativeImpactRows",
    "ZeroImpactRows",
    "ImportanceRank",
]


def _empty_shap_detail() -> pd.DataFrame:
    """
    Return an empty row-level SHAP dataframe.
    """
    return pd.DataFrame(
        columns=SHAP_DETAIL_COLUMNS
    )


def _empty_shap_summary() -> pd.DataFrame:
    """
    Return an empty global SHAP summary dataframe.
    """
    return pd.DataFrame(
        columns=SHAP_SUMMARY_COLUMNS
    )


def _safe_float(
    value: Any,
) -> float:
    """
    Convert a value into a finite float.
    """
    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if (
        np.isnan(number)
        or np.isinf(number)
    ):
        return 0.0

    return number


def _normalize_base_value(
    base_value: Any,
) -> float:
    """
    Normalize SHAP expected/base value into one scalar.
    """
    values = np.asarray(
        base_value
    ).reshape(-1)

    if len(values) == 0:
        return 0.0

    return _safe_float(
        values[0]
    )


def _normalize_shap_values(
    shap_values: Any,
    row_count: int,
    feature_count: int,
) -> np.ndarray:
    """
    Normalize SHAP output into:

        rows x features

    SHAP can return slightly different shapes depending
    on model and library version.
    """
    values = np.asarray(
        shap_values
    )

    if values.ndim == 1:
        values = values.reshape(
            1,
            -1,
        )

    if values.ndim == 3:
        if values.shape[-1] == 1:
            values = values[
                :,
                :,
                0,
            ]

        elif values.shape[0] == 1:
            values = values[
                0
            ]

    if values.ndim != 2:
        raise ValueError(
            "Unsupported SHAP value shape: "
            f"{values.shape}"
        )

    if values.shape == (
        feature_count,
        row_count,
    ):
        values = values.T

    if values.shape != (
        row_count,
        feature_count,
    ):
        raise ValueError(
            "SHAP values do not match "
            "the explanation dataset. "
            f"Expected: "
            f"({row_count}, {feature_count}), "
            f"received: {values.shape}."
        )

    return values.astype(
        float
    )


def _direction_from_shap(
    shap_value: float,
) -> str:
    """
    Convert signed SHAP contribution into
    a deterministic business-readable direction.
    """
    if shap_value > 0:
        return "increase"

    if shap_value < 0:
        return "decrease"

    return "neutral"


def select_explanation_sample(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    max_rows: int = 200,
) -> pd.DataFrame:
    """
    Select a deterministic explanation sample.

    The most recent rows are preferred when a date
    column exists. Otherwise the latest dataframe
    positions are used.

    SHAP is intentionally calculated on a bounded
    sample to keep scheduled production runs efficient.
    """
    if dataframe is None:
        raise ValueError(
            "Explanation dataframe cannot be None."
        )

    if max_rows <= 0:
        raise ValueError(
            "max_rows must be greater than zero."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing SHAP feature columns: "
            f"{missing_features}"
        )

    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    if "date" in result.columns:
        result[
            "__explanation_date"
        ] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

        result = (
            result
            .sort_values(
                "__explanation_date",
                ascending=False,
                na_position="last",
            )
            .drop(
                columns=[
                    "__explanation_date"
                ]
            )
        )

    result = (
        result
        .head(
            max_rows
        )
        .copy()
    )

    return result


def build_shap_explanations(
    model: Any,
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    model_name: str,
    algorithm: str,
    max_rows: int = 200,
    top_features_per_row: int = 10,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build row-level and global SHAP explanations
    for a selected production forecasting model.

    Parameters
    ----------
    model
        Fully trained production model.

    dataframe
        Model-ready observations.

    feature_columns
        Exact model feature order.

    model_name
        Business model target such as Next_Clicks.

    algorithm
        Selected production algorithm name.

    max_rows
        Maximum observations to explain.

    top_features_per_row
        Number of strongest SHAP features stored
        for each observation.

    Returns
    -------
    detail_df
        Row-level top feature contributions.

    summary_df
        Global mean absolute SHAP importance and
        impact direction statistics.
    """
    if model is None:
        raise ValueError(
            "SHAP model cannot be None."
        )

    if not feature_columns:
        raise ValueError(
            "SHAP feature columns cannot be empty."
        )

    if top_features_per_row <= 0:
        raise ValueError(
            "top_features_per_row must be "
            "greater than zero."
        )

    explanation_df = (
        select_explanation_sample(
            dataframe=dataframe,
            feature_columns=feature_columns,
            max_rows=max_rows,
        )
    )

    if explanation_df.empty:
        return (
            _empty_shap_detail(),
            _empty_shap_summary(),
        )

    x_explain = (
        explanation_df[
            feature_columns
        ]
        .copy()
    )

    x_explain = (
        x_explain
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
    )

    predictions = np.asarray(
        model.predict(
            x_explain
        )
    ).reshape(-1)

    try:
        explainer = shap.TreeExplainer(
            model
        )

        raw_shap_values = (
            explainer.shap_values(
                x_explain
            )
        )

        base_value = (
            _normalize_base_value(
                explainer.expected_value
            )
        )

    except Exception:
        logger.exception(
            "TreeExplainer failed | "
            "Model: %s | Algorithm: %s",
            model_name,
            algorithm,
        )

        raise

    shap_values = (
        _normalize_shap_values(
            shap_values=raw_shap_values,
            row_count=len(
                x_explain
            ),
            feature_count=len(
                feature_columns
            ),
        )
    )

    detail_rows = []

    for row_position in range(
        len(
            x_explain
        )
    ):
        row_values = shap_values[
            row_position
        ]

        strongest_positions = (
            np.argsort(
                np.abs(
                    row_values
                )
            )[
                ::-1
            ][
                :top_features_per_row
            ]
        )

        source_index = (
            explanation_df.index[
                row_position
            ]
        )

        prediction = _safe_float(
            predictions[
                row_position
            ]
        )

        for feature_position in (
            strongest_positions
        ):
            feature_name = (
                feature_columns[
                    int(
                        feature_position
                    )
                ]
            )

            feature_value = (
                x_explain.iloc[
                    row_position
                ][
                    feature_name
                ]
            )

            shap_value = (
                _safe_float(
                    row_values[
                        feature_position
                    ]
                )
            )

            detail_rows.append(
                {
                    "Model": (
                        str(
                            model_name
                        )
                    ),
                    "Algorithm": (
                        str(
                            algorithm
                        )
                    ),
                    "RowIndex": (
                        str(
                            source_index
                        )
                    ),
                    "Feature": (
                        feature_name
                    ),
                    "FeatureValue": (
                        _safe_float(
                            feature_value
                        )
                    ),
                    "SHAPValue": (
                        shap_value
                    ),
                    "AbsSHAPValue": (
                        abs(
                            shap_value
                        )
                    ),
                    "Direction": (
                        _direction_from_shap(
                            shap_value
                        )
                    ),
                    "BaseValue": (
                        base_value
                    ),
                    "Prediction": (
                        prediction
                    ),
                }
            )

    detail_df = pd.DataFrame(
        detail_rows,
        columns=SHAP_DETAIL_COLUMNS,
    )

    mean_abs_shap = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    mean_shap = np.mean(
        shap_values,
        axis=0,
    )

    positive_counts = np.sum(
        shap_values > 0,
        axis=0,
    )

    negative_counts = np.sum(
        shap_values < 0,
        axis=0,
    )

    zero_counts = np.sum(
        shap_values == 0,
        axis=0,
    )

    summary_df = pd.DataFrame(
        {
            "Model": [
                str(
                    model_name
                )
            ]
            * len(
                feature_columns
            ),
            "Algorithm": [
                str(
                    algorithm
                )
            ]
            * len(
                feature_columns
            ),
            "Feature": (
                feature_columns
            ),
            "MeanAbsSHAP": (
                mean_abs_shap.astype(
                    float
                )
            ),
            "MeanSHAP": (
                mean_shap.astype(
                    float
                )
            ),
            "PositiveImpactRows": (
                positive_counts.astype(
                    int
                )
            ),
            "NegativeImpactRows": (
                negative_counts.astype(
                    int
                )
            ),
            "ZeroImpactRows": (
                zero_counts.astype(
                    int
                )
            ),
        }
    )

    summary_df = (
        summary_df
        .sort_values(
            "MeanAbsSHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    summary_df[
        "ImportanceRank"
    ] = np.arange(
        1,
        len(
            summary_df
        )
        + 1,
    )

    summary_df = summary_df[
        SHAP_SUMMARY_COLUMNS
    ]

    logger.info(
        "SHAP explanations completed "
        "| Model: %s "
        "| Algorithm: %s "
        "| Explained rows: %d "
        "| Detail rows: %d.",
        model_name,
        algorithm,
        len(
            explanation_df
        ),
        len(
            detail_df
        ),
    )

    return (
        detail_df,
        summary_df,
    )


def combine_shap_explanations(
    clicks_model: Any,
    impressions_model: Any,
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    clicks_algorithm: str,
    impressions_algorithm: str,
    max_rows: int = 200,
    top_features_per_row: int = 10,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Explain both selected production forecasting models.

    Clicks and impressions may use different algorithms,
    therefore each model is explained independently and
    the outputs are combined afterwards.
    """
    (
        clicks_detail,
        clicks_summary,
    ) = build_shap_explanations(
        model=clicks_model,
        dataframe=dataframe,
        feature_columns=feature_columns,
        model_name="Next_Clicks",
        algorithm=clicks_algorithm,
        max_rows=max_rows,
        top_features_per_row=(
            top_features_per_row
        ),
    )

    (
        impressions_detail,
        impressions_summary,
    ) = build_shap_explanations(
        model=impressions_model,
        dataframe=dataframe,
        feature_columns=feature_columns,
        model_name="Next_Impressions",
        algorithm=impressions_algorithm,
        max_rows=max_rows,
        top_features_per_row=(
            top_features_per_row
        ),
    )

    detail_df = pd.concat(
        [
            clicks_detail,
            impressions_detail,
        ],
        ignore_index=True,
    )

    summary_df = pd.concat(
        [
            clicks_summary,
            impressions_summary,
        ],
        ignore_index=True,
    )

    return (
        detail_df,
        summary_df,
    )