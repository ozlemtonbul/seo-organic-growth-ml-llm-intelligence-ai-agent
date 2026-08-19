from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config.logging_config import get_logger


logger = get_logger(__name__)


SEO_REQUIRED_COLUMNS = [
    "date",
    "page",
    "clicks",
    "impressions",
    "position",
]


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """
    Validate that a DataFrame contains all required columns.
    """
    required = list(required_columns)

    missing_columns = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def normalize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize DataFrame column names into snake_case.
    """
    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in result.columns
    ]

    return result


def load_csv_file(
    file_path: str,
    dataset_name: str = "CSV dataset",
) -> pd.DataFrame:
    """
    Load and normalize a UTF-8 compatible CSV file.
    """
    if not file_path:
        raise ValueError(
            f"{dataset_name} file path is required."
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file was not found: "
            f"{path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"{dataset_name} path must point to a file."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"{dataset_name} must be a CSV file."
        )

    try:
        dataframe = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            path,
            encoding="utf-8",
        )

    dataframe = normalize_column_names(
        dataframe
    )

    logger.info(
        "%s loaded: %s | Rows: %d",
        dataset_name,
        path,
        len(dataframe),
    )

    return dataframe


def standardize_seo_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize Search Console or SEO CSV data.
    """
    result = normalize_column_names(
        dataframe
    )

    validate_required_columns(
        result,
        SEO_REQUIRED_COLUMNS,
        "SEO dataset",
    )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    numeric_columns: List[str] = [
        "clicks",
        "impressions",
        "position",
        "ctr",
    ]

    for column in numeric_columns:
        if column not in result.columns:
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0)

    text_columns = [
        "page",
        "query",
        "device",
        "country",
        "page_type",
        "keyword_intent",
        "title",
        "meta_description",
        "h1",
        "content",
        "schema_type",
        "brand",
        "product_name",
        "category_name",
    ]

    for column in text_columns:
        if column not in result.columns:
            result[column] = ""

        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    result = result.dropna(
        subset=["date"],
    )

    result = result[
        result["page"] != ""
    ]

    return result.reset_index(
        drop=True
    )


def load_seo_csv(
    file_path: str,
) -> pd.DataFrame:
    """
    Load and standardize the main SEO input CSV.
    """
    dataframe = load_csv_file(
        file_path=file_path,
        dataset_name="SEO input dataset",
    )

    return standardize_seo_dataframe(
        dataframe
    )


def load_optional_csv(
    file_path: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load an optional CSV file.

    An empty DataFrame is returned when no file path is configured.
    """
    if not file_path:
        logger.info(
            "%s is not configured. Skipping.",
            dataset_name,
        )

        return pd.DataFrame()

    return load_csv_file(
        file_path=file_path,
        dataset_name=dataset_name,
    )