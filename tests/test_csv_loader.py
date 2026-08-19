from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.extract.csv_loader import (
    load_csv_file,
    load_optional_csv,
    load_seo_csv,
    normalize_column_names,
    standardize_seo_dataframe,
    validate_required_columns,
)


def build_valid_seo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "page": "https://example.com/product/a",
                "clicks": 10,
                "impressions": 100,
                "position": 7,
                "ctr": 0.10,
            }
        ]
    )


def test_normalize_column_names() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "Page Type",
            "Meta-Description",
        ]
    )

    result = normalize_column_names(
        dataframe
    )

    assert list(
        result.columns
    ) == [
        "page_type",
        "meta_description",
    ]


def test_validate_required_columns() -> None:
    validate_required_columns(
        build_valid_seo_dataframe(),
        [
            "date",
            "page",
        ],
        "SEO dataset",
    )


def test_validate_required_columns_rejects_missing() -> None:
    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_required_columns(
            pd.DataFrame(
                [
                    {
                        "page": "https://example.com/a",
                    }
                ]
            ),
            [
                "date",
                "page",
            ],
            "SEO dataset",
        )


def test_load_csv_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"

    pd.DataFrame(
        [
            {
                "Page": "https://example.com/a",
                "Clicks": 10,
            }
        ]
    ).to_csv(
        path,
        index=False,
    )

    result = load_csv_file(
        str(path),
        "Sample dataset",
    )

    assert len(result) == 1
    assert "page" in result.columns
    assert "clicks" in result.columns


def test_load_csv_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        load_csv_file(
            str(tmp_path / "missing.csv")
        )


def test_standardize_seo_dataframe() -> None:
    result = standardize_seo_dataframe(
        build_valid_seo_dataframe()
    )

    assert len(result) == 1
    assert pd.api.types.is_datetime64_any_dtype(
        result["date"]
    )

    assert result.iloc[0]["clicks"] == 10
    assert "query" in result.columns
    assert "page_type" in result.columns


def test_load_seo_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seo.csv"

    build_valid_seo_dataframe().to_csv(
        path,
        index=False,
    )

    result = load_seo_csv(
        str(path)
    )

    assert len(result) == 1
    assert result.iloc[0]["page"] == (
        "https://example.com/product/a"
    )


def test_load_optional_csv_without_path() -> None:
    result = load_optional_csv(
        "",
        "Optional dataset",
    )

    assert result.empty