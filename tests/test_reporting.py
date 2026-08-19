from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features import build_holiday_map
from src.reporting import (
    build_daily_weekly_monthly_outputs,
    build_keyword_intent_summary,
    build_page_type_summary,
    build_recommendation_summary,
    build_run_manifest,
    build_seo_holiday_impact,
    ensure_output_directory,
    export_outputs,
    export_run_manifest,
    export_text_report,
    validate_output_name,
)


def build_reporting_dataframe() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [
            {
                "date": "2026-07-10",
                "page": "https://example.com/product/a",
                "page_type": "product",
                "keyword_intent": "Transactional",
                "clicks": 10,
                "impressions": 100,
                "position": 8,
            },
            {
                "date": "2026-07-11",
                "page": "https://example.com/product/a",
                "page_type": "product",
                "keyword_intent": "Transactional",
                "clicks": 12,
                "impressions": 120,
                "position": 7,
            },
            {
                "date": "2026-07-12",
                "page": "https://example.com/blog/a",
                "page_type": "blog",
                "keyword_intent": "Informational",
                "clicks": 8,
                "impressions": 160,
                "position": 11,
            },
        ]
    )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"]
    )

    return dataframe


def test_validate_output_name() -> None:
    assert validate_output_name(
        "seo_summary"
    ) == "seo_summary"


def test_ensure_output_directory(
    tmp_path: Path,
) -> None:
    target = ensure_output_directory(
        str(tmp_path / "outputs")
    )

    assert target.exists()
    assert target.is_dir()


def test_export_outputs(
    tmp_path: Path,
) -> None:
    outputs = {
        "seo_summary": pd.DataFrame(
            [
                {
                    "page": "https://example.com/a",
                    "clicks": 10,
                }
            ]
        )
    }

    written = export_outputs(
        outputs,
        str(tmp_path),
    )

    assert "seo_summary" in written
    assert written["seo_summary"].exists()


def test_build_run_manifest(
    tmp_path: Path,
) -> None:
    outputs = {
        "seo_summary": pd.DataFrame(
            [
                {
                    "clicks": 10,
                }
            ]
        )
    }

    manifest = build_run_manifest(
        output_dir=str(tmp_path),
        outputs=outputs,
        input_file="./data/raw/seo_data.csv",
    )

    assert manifest["tables"]["seo_summary"] == 1
    assert "pipeline" in manifest
    assert "run_timestamp" in manifest


def test_export_run_manifest(
    tmp_path: Path,
) -> None:
    outputs = {
        "seo_summary": pd.DataFrame(
            [
                {
                    "clicks": 10,
                }
            ]
        )
    }

    path = export_run_manifest(
        output_dir=str(tmp_path),
        outputs=outputs,
        input_file="./data/raw/seo_data.csv",
    )

    assert path.exists()

    content = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert content["tables"]["seo_summary"] == 1


def test_export_text_report(
    tmp_path: Path,
) -> None:
    path = export_text_report(
        content="Sample executive report",
        output_dir=str(tmp_path),
        filename="executive_report",
    )

    assert path.exists()

    assert (
        path.read_text(
            encoding="utf-8"
        )
        == "Sample executive report"
    )


def test_keyword_intent_summary() -> None:
    result = build_keyword_intent_summary(
        build_reporting_dataframe()
    )

    assert len(result) == 2

    assert result["total_clicks"].sum() == 30


def test_page_type_summary() -> None:
    result = build_page_type_summary(
        build_reporting_dataframe()
    )

    assert len(result) == 2
    assert result["total_clicks"].sum() == 30


def test_holiday_impact() -> None:
    dataframe = build_reporting_dataframe()

    holidays = build_holiday_map(
        "2026-07-10",
        "2026-07-15",
    )

    result = build_seo_holiday_impact(
        dataframe,
        holidays,
    )

    assert not result.empty
    assert "period_label" in result.columns


def test_daily_weekly_monthly_outputs() -> None:
    daily, weekly, monthly = (
        build_daily_weekly_monthly_outputs(
            build_reporting_dataframe()
        )
    )

    assert len(daily) == 3
    assert len(weekly) == 2
    assert len(monthly) == 2


def test_build_recommendation_summary() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "page": "https://example.com/a",
                "Scenario": "content_refresh",
                "RecommendedAction": "Refresh Content",
                "ExpectedNetValue": 20,
                "UnusedColumn": "remove me",
            }
        ]
    )

    result = build_recommendation_summary(
        dataframe
    )

    assert "page" in result.columns
    assert "RecommendedAction" in result.columns
    assert "UnusedColumn" not in result.columns