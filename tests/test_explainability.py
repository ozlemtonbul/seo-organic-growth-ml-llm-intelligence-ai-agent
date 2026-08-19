from __future__ import annotations

import pandas as pd

from src.models import (
    build_shap_explanations,
    select_explanation_sample,
    train_and_validate_models,
)

from src.features import (
    build_holiday_map,
    prepare_training_data,
)


def build_source_dataframe(
    number_of_days: int = 40,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2026-05-01",
        periods=number_of_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "page": [
                "https://example.com/a"
            ] * number_of_days,
            "date": dates,
            "clicks": [
                10 + index % 8
                for index in range(
                    number_of_days
                )
            ],
            "impressions": [
                100 + index * 3
                for index in range(
                    number_of_days
                )
            ],
            "position": [
                max(
                    1,
                    12 - index * 0.15,
                )
                for index in range(
                    number_of_days
                )
            ],
            "ctr": [
                0.10
            ] * number_of_days,
            "page_type": [
                "product"
            ] * number_of_days,
            "keyword_intent": [
                "Transactional"
            ] * number_of_days,
            "title": [
                "Sample Product"
            ] * number_of_days,
            "meta_description": [
                "Sample description"
            ] * number_of_days,
            "h1": [
                "Sample Product"
            ] * number_of_days,
            "content": [
                "sample product content " * 300
            ] * number_of_days,
            "schema_type": [
                "Product"
            ] * number_of_days,
            "brand": [
                "Example Brand"
            ] * number_of_days,
        }
    )


def build_training_dataframe() -> pd.DataFrame:
    source = build_source_dataframe()

    holidays = build_holiday_map(
        "2026-05-01",
        "2026-06-09",
    )

    return prepare_training_data(
        source,
        holidays,
    )


def test_select_explanation_sample() -> None:
    training = build_training_dataframe()

    feature_columns = [
        "clicks",
        "impressions",
    ]

    result = select_explanation_sample(
        dataframe=training,
        feature_columns=feature_columns,
        max_rows=5,
    )

    assert len(
        result
    ) == 5

    assert (
        result["date"].max()
        == training["date"].max()
    )


def test_build_shap_explanations() -> None:
    training = build_training_dataframe()

    (
        clicks_model,
        _,
        feature_columns,
        metrics,
        _,
    ) = train_and_validate_models(
        train_df=training,
        with_holiday=True,
    )

    clicks_algorithm = (
        metrics[
            metrics["Model"]
            == "Next_Clicks"
        ]["Algorithm"]
        .iloc[0]
    )

    detail, summary = build_shap_explanations(
        model=clicks_model,
        dataframe=training,
        feature_columns=feature_columns,
        model_name="Next_Clicks",
        algorithm=clicks_algorithm,
        max_rows=5,
        top_features_per_row=3,
    )

    assert not detail.empty
    assert not summary.empty

    assert len(
        detail
    ) == 15

    assert set(
        detail["Direction"]
    ).issubset(
        {
            "increase",
            "decrease",
            "neutral",
        }
    )

    assert (
        detail["Model"]
        == "Next_Clicks"
    ).all()

    assert (
        summary["ImportanceRank"]
        .min()
        == 1
    )

    assert (
        summary["MeanAbsSHAP"]
        >= 0
    ).all()