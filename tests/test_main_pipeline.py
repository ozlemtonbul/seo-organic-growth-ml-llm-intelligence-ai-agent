from __future__ import annotations

import pandas as pd

import main as pipeline_main


def build_main_test_dataframe(
    number_of_days: int = 40,
) -> pd.DataFrame:
    """
    Build deterministic page-level SEO observations
    for main pipeline tests.
    """
    dates = pd.date_range(
        "2026-05-01",
        periods=number_of_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "page": [
                "https://example.com/product/a"
            ]
            * number_of_days,
            "query": [
                "buy sample product"
            ]
            * number_of_days,
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
            ]
            * number_of_days,
            "page_type": [
                "product"
            ]
            * number_of_days,
            "keyword_intent": [
                "Transactional"
            ]
            * number_of_days,
            "title": [
                "Sample Product"
            ]
            * number_of_days,
            "meta_description": [
                "Sample description"
            ]
            * number_of_days,
            "h1": [
                "Sample Product"
            ]
            * number_of_days,
            "content": [
                "sample product content " * 300
            ]
            * number_of_days,
            "schema_type": [
                "Product"
            ]
            * number_of_days,
            "brand": [
                "Example Brand"
            ]
            * number_of_days,
        }
    )


def test_prepare_integrated_dataset() -> None:
    seo = build_main_test_dataframe(
        number_of_days=3
    )

    result = pipeline_main.prepare_integrated_dataset(
        seo_dataframe=seo,
        ga4_dataframe=pd.DataFrame(),
    )

    assert len(
        result
    ) == 3

    assert (
        "page_type"
        in result.columns
    )

    assert (
        "keyword_intent"
        in result.columns
    )

    assert (
        "sessions"
        in result.columns
    )


def test_train_pipeline_models() -> None:
    seo = build_main_test_dataframe()

    integrated = (
        pipeline_main.prepare_integrated_dataset(
            seo_dataframe=seo,
            ga4_dataframe=pd.DataFrame(),
        )
    )

    holidays = pipeline_main.build_holiday_map(
        "2026-05-01",
        "2026-06-09",
    )

    (
        training,
        clicks_model,
        impressions_model,
        features,
        metrics,
        importance,
        benchmark,
    ) = pipeline_main.train_pipeline_models(
        integrated,
        holidays,
    )

    assert not training.empty

    assert len(
        features
    ) == 46

    assert len(
        metrics
    ) == 2

    assert len(
        importance
    ) == 92

    assert clicks_model is not None
    assert impressions_model is not None

    assert hasattr(
        clicks_model,
        "predict",
    )

    assert hasattr(
        impressions_model,
        "predict",
    )

    # Full benchmark:
    # 3 algorithms x 2 targets.
    assert len(
        benchmark
    ) == 6

    assert set(
        benchmark[
            "Algorithm"
        ]
    ) == {
        "RandomForest",
        "XGBoost",
        "LightGBM",
    }

    assert set(
        benchmark[
            "Model"
        ]
    ) == {
        "Next_Clicks",
        "Next_Impressions",
    }

    assert (
        benchmark[
            "ValidationMethod"
        ]
        == "time_aware_holdout"
    ).all()

    assert (
        benchmark[
            "Status"
        ]
        == "success"
    ).all()

    assert (
        benchmark[
            "FirstTestDate"
        ]
        .notna()
        .all()
    )

    selected_counts = (
        benchmark
        .groupby(
            "Model"
        )[
            "Selected"
        ]
        .sum()
    )

    assert (
        selected_counts[
            "Next_Clicks"
        ]
        == 1
    )

    assert (
        selected_counts[
            "Next_Impressions"
        ]
        == 1
    )


def test_build_pipeline_recommendations() -> None:
    seo = build_main_test_dataframe()

    integrated = (
        pipeline_main.prepare_integrated_dataset(
            seo_dataframe=seo,
            ga4_dataframe=pd.DataFrame(),
        )
    )

    holidays = pipeline_main.build_holiday_map(
        "2026-05-01",
        "2026-06-09",
    )

    (
        training,
        clicks_model,
        impressions_model,
        features,
        metrics,
        _,
        _,
    ) = pipeline_main.train_pipeline_models(
        integrated,
        holidays,
    )

    (
        latest,
        simulation,
        recommendations,
    ) = pipeline_main.build_pipeline_recommendations(
        integrated_dataframe=integrated,
        training_dataframe=training,
        holiday_map=holidays,
        clicks_model=clicks_model,
        impressions_model=impressions_model,
        feature_columns=features,
        model_metrics=metrics,
    )

    assert len(
        latest
    ) == 1

    assert not simulation.empty

    assert len(
        recommendations
    ) == 1

    assert (
        "Scenario"
        in simulation.columns
    )

    assert (
        "BusinessDecisionScore"
        in simulation.columns
    )

    assert (
        "RecommendedAction"
        in recommendations.columns
    )

    assert (
        "PriorityTier"
        in recommendations.columns
    )


def test_main_returns_zero_when_pipeline_succeeds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline_main,
        "run_pipeline",
        lambda: {},
    )

    assert (
        pipeline_main.main()
        == 0
    )


def test_main_returns_one_when_pipeline_fails(
    monkeypatch,
) -> None:
    def raise_error():
        raise RuntimeError(
            "Test pipeline failure"
        )

    monkeypatch.setattr(
        pipeline_main,
        "run_pipeline",
        raise_error,
    )

    assert (
        pipeline_main.main()
        == 1
    )