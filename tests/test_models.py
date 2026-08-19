from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.features import (
    build_holiday_map,
    prepare_training_data,
)

from src.models import (
    SEO_SCENARIOS,
    SUPPORTED_ALGORITHMS,
    add_baseline_uplift,
    add_business_decision_score,
    add_business_value_layers,
    add_opportunity_score,
    build_lightgbm_model,
    build_random_forest_model,
    build_time_aware_split,
    build_xgboost_model,
    choose_best_scenario,
    estimate_current_content_score,
    estimate_current_geo_score,
    get_last_model_benchmark,
    regression_metrics,
    safe_prediction,
    scenario_cost,
    scenario_is_applicable,
    train_and_validate_models,
    validate_training_dataframe,
)


def build_model_source_data(
    number_of_days: int = 40,
) -> pd.DataFrame:
    """
    Build deterministic SEO observations for model tests.
    """
    dates = pd.date_range(
        "2026-05-01",
        periods=number_of_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "page": [
                "https://example.com/product/a"
            ]
            * number_of_days,
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
                "Sample product description"
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


def build_training_dataframe() -> pd.DataFrame:
    """
    Build the feature-engineered ML training dataset.
    """
    source = build_model_source_data()

    holidays = build_holiday_map(
        "2026-05-01",
        "2026-06-09",
    )

    return prepare_training_data(
        source,
        holidays,
    )


def test_build_random_forest_model() -> None:
    model = build_random_forest_model()

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators > 0
    assert model.n_jobs == -1


def test_build_xgboost_model() -> None:
    model = build_xgboost_model()

    assert isinstance(
        model,
        XGBRegressor,
    )

    assert model.n_estimators > 0
    assert model.n_jobs == -1


def test_build_lightgbm_model() -> None:
    model = build_lightgbm_model()

    assert isinstance(
        model,
        LGBMRegressor,
    )

    assert model.n_estimators > 0
    assert model.n_jobs == -1


def test_safe_prediction() -> None:
    assert safe_prediction(-10) == 0.0

    assert safe_prediction(
        "12.5"
    ) == 12.5

    assert safe_prediction(
        np.nan
    ) == 0.0

    assert safe_prediction(
        np.inf
    ) == 0.0

    assert safe_prediction(
        "invalid"
    ) == 0.0


def test_regression_metrics() -> None:
    actual = pd.Series(
        [
            10,
            20,
            30,
        ]
    )

    predicted = np.array(
        [
            11,
            19,
            31,
        ]
    )

    result = regression_metrics(
        model_name="Test_Model",
        actual=actual,
        predicted=predicted,
        train_rows=10,
        test_rows=3,
    )

    assert (
        result["Model"]
        == "Test_Model"
    )

    assert result[
        "MAE"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "RMSE"
    ] == pytest.approx(
        1.0
    )

    assert (
        result["TrainRows"]
        == 10
    )

    assert (
        result["TestRows"]
        == 3
    )

    assert (
        result[
            "ValidationMethod"
        ]
        == "time_aware_holdout"
    )


def test_validate_training_dataframe() -> None:
    training_dataframe = (
        build_training_dataframe()
    )

    validate_training_dataframe(
        training_dataframe,
        [
            "clicks",
            "impressions",
        ],
    )


def test_validate_training_dataframe_rejects_missing_targets(
) -> None:
    dataframe = pd.DataFrame(
        {
            "clicks": [
                10,
                20,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing target columns",
    ):
        validate_training_dataframe(
            dataframe,
            [
                "clicks",
            ],
        )


def test_build_time_aware_split() -> None:
    """
    Ensure validation always happens after training chronologically.
    """
    training_dataframe = (
        build_training_dataframe()
        .reset_index(
            drop=True
        )
    )

    (
        train_idx,
        test_idx,
        first_test_date,
    ) = build_time_aware_split(
        training_dataframe
    )

    assert len(
        train_idx
    ) > 0

    assert len(
        test_idx
    ) > 0

    dates = pd.to_datetime(
        training_dataframe[
            "date"
        ]
    )

    train_dates = dates.iloc[
        train_idx
    ]

    test_dates = dates.iloc[
        test_idx
    ]

    # No future leakage.
    assert (
        train_dates.max()
        < test_dates.min()
    )

    assert (
        test_dates.min()
        == first_test_date
    )

    assert (
        train_dates
        < first_test_date
    ).all()

    assert (
        test_dates
        >= first_test_date
    ).all()


def test_train_and_validate_models() -> None:
    """
    Test Ads-style multi-model benchmarking and winner selection.
    """
    training_dataframe = (
        build_training_dataframe()
    )

    (
        clicks_model,
        impressions_model,
        features,
        metrics,
        importance,
    ) = train_and_validate_models(
        train_df=training_dataframe,
        with_holiday=True,
    )

    # Production model can be RF, XGBoost or LightGBM.
    assert hasattr(
        clicks_model,
        "predict",
    )

    assert hasattr(
        impressions_model,
        "predict",
    )

    assert len(
        features
    ) == 46

    # Only winner metrics are passed downstream.
    assert len(
        metrics
    ) == 2

    assert set(
        metrics[
            "Model"
        ]
    ) == {
        "Next_Clicks",
        "Next_Impressions",
    }

    assert set(
        metrics[
            "Algorithm"
        ]
    ).issubset(
        set(
            SUPPORTED_ALGORITHMS
        )
    )

    assert (
        metrics[
            "ValidationMethod"
        ]
        == "time_aware_holdout"
    ).all()

    assert (
        metrics[
            "Selected"
        ]
        .astype(bool)
        .all()
    )

    # 46 features x 2 selected models.
    assert len(
        importance
    ) == 92

    assert set(
        importance[
            "Model"
        ]
    ) == {
        "Next_Clicks",
        "Next_Impressions",
    }

    assert set(
        importance[
            "Algorithm"
        ]
    ).issubset(
        set(
            SUPPORTED_ALGORITHMS
        )
    )

    benchmark = (
        get_last_model_benchmark()
    )

    # 3 algorithms x 2 targets.
    assert len(
        benchmark
    ) == 6

    assert set(
        benchmark[
            "Model"
        ]
    ) == {
        "Next_Clicks",
        "Next_Impressions",
    }

    assert set(
        benchmark[
            "Algorithm"
        ]
    ) == {
        "RandomForest",
        "XGBoost",
        "LightGBM",
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

    # Each target must have exactly one winner.
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

    expected_classes = {
        "RandomForest": (
            RandomForestRegressor
        ),
        "XGBoost": (
            XGBRegressor
        ),
        "LightGBM": (
            LGBMRegressor
        ),
    }

    clicks_algorithm = (
        metrics[
            metrics[
                "Model"
            ]
            == "Next_Clicks"
        ][
            "Algorithm"
        ]
        .iloc[0]
    )

    impressions_algorithm = (
        metrics[
            metrics[
                "Model"
            ]
            == "Next_Impressions"
        ][
            "Algorithm"
        ]
        .iloc[0]
    )

    assert isinstance(
        clicks_model,
        expected_classes[
            clicks_algorithm
        ],
    )

    assert isinstance(
        impressions_model,
        expected_classes[
            impressions_algorithm
        ],
    )


def test_seo_scenario_count() -> None:
    assert len(
        SEO_SCENARIOS
    ) == 10


def test_scenario_applicability() -> None:
    category_scenario = next(
        scenario
        for scenario in SEO_SCENARIOS
        if (
            scenario[
                "Scenario"
            ]
            == "category_expansion"
        )
    )

    assert scenario_is_applicable(
        "category",
        category_scenario,
    )

    assert not scenario_is_applicable(
        "product",
        category_scenario,
    )


def test_content_score() -> None:
    row = pd.Series(
        {
            "title": (
                "Sample Title"
            ),
            "meta_description": (
                "Sample description"
            ),
            "h1": (
                "Sample H1"
            ),
            "content": (
                "sample content "
                * 300
            ),
            "schema_type": (
                "Product"
            ),
        }
    )

    assert (
        estimate_current_content_score(
            row
        )
        == 100.0
    )


def test_geo_score() -> None:
    row = pd.Series(
        {
            "h1": (
                "Sample H1"
            ),
            "content": (
                "what is this product "
                + "sample content "
                * 300
            ),
            "schema_type": (
                "Product"
            ),
            "brand": (
                "Example Brand"
            ),
            "meta_description": (
                "Sample description"
            ),
        }
    )

    score = (
        estimate_current_geo_score(
            row
        )
    )

    assert score > 50
    assert score <= 100


def test_scenario_cost() -> None:
    effort, cost = scenario_cost(
        "content_refresh"
    )

    assert effort == 5

    assert cost == pytest.approx(
        120.0
    )


def test_choose_best_scenario() -> None:
    simulation = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "maintain"
                ),
                "PredictedNextClicks": (
                    10
                ),
                "PredictedNextImpressions": (
                    100
                ),
                "EstimatedPositionGain": (
                    0
                ),
                "EstimatedGeoScoreGain": (
                    0
                ),
                "EstimatedContentScoreGain": (
                    0
                ),
                "ScenarioCTR": (
                    0.10
                ),
            },
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "content_refresh"
                ),
                "PredictedNextClicks": (
                    20
                ),
                "PredictedNextImpressions": (
                    200
                ),
                "EstimatedPositionGain": (
                    2
                ),
                "EstimatedGeoScoreGain": (
                    10
                ),
                "EstimatedContentScoreGain": (
                    15
                ),
                "ScenarioCTR": (
                    0.15
                ),
            },
        ]
    )

    result = choose_best_scenario(
        simulation
    )

    assert len(
        result
    ) == 1

    assert (
        result.iloc[
            0
        ][
            "Scenario"
        ]
        == "content_refresh"
    )


def test_add_baseline_uplift() -> None:
    simulation = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "maintain"
                ),
                "PredictedNextClicks": (
                    10
                ),
                "PredictedNextImpressions": (
                    100
                ),
            },
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "content_refresh"
                ),
                "PredictedNextClicks": (
                    15
                ),
                "PredictedNextImpressions": (
                    120
                ),
            },
        ]
    )

    best = simulation[
        simulation[
            "Scenario"
        ]
        == "content_refresh"
    ].copy()

    result = add_baseline_uplift(
        best,
        simulation,
    )

    assert (
        result.iloc[
            0
        ][
            "ClicksUplift"
        ]
        == 5
    )

    assert (
        result.iloc[
            0
        ][
            "ImpressionsUplift"
        ]
        == 20
    )

    assert (
        result.iloc[
            0
        ][
            "ClicksUpliftPct"
        ]
        == pytest.approx(
            50.0
        )
    )


def test_add_business_value_layers() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "Scenario": (
                    "content_refresh"
                ),
                "ClicksUplift": (
                    300
                ),
            }
        ]
    )

    result = add_business_value_layers(
        dataframe
    )

    row = result.iloc[
        0
    ]

    assert (
        row[
            "EffortScore"
        ]
        == 5
    )

    assert (
        row[
            "EstimatedImplementationCost"
        ]
        == pytest.approx(
            120.0
        )
    )

    assert (
        row[
            "ExpectedIncrementalTrafficValue"
        ]
        == pytest.approx(
            150.0
        )
    )

    assert (
        row[
            "BusinessValueHorizonDays"
        ]
        == 30
    )

    assert (
        row[
            "ProjectedIncrementalClicks"
        ]
        == pytest.approx(
            9000.0
        )
    )

    assert (
        row[
            "ProjectedIncrementalTrafficValue"
        ]
        == pytest.approx(
            4500.0
        )
    )

    assert (
        row[
            "ExpectedNetValue"
        ]
        == pytest.approx(
            4380.0
        )
    )

    assert (
        row[
            "EstimatedROI"
        ]
        == pytest.approx(
            36.5
        )
    )

    assert (
        row[
            "PaybackPeriod"
        ]
        == pytest.approx(
            0.8
        )
    )


def test_add_opportunity_score_preserves_all_scenarios(
) -> None:
    simulation = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "maintain"
                ),
                "PredictedNextClicks": (
                    10
                ),
                "PredictedNextImpressions": (
                    100
                ),
                "EstimatedPositionGain": (
                    0
                ),
                "EstimatedGeoScoreGain": (
                    0
                ),
                "EstimatedContentScoreGain": (
                    0
                ),
                "ScenarioCTR": (
                    0.10
                ),
            },
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "content_refresh"
                ),
                "PredictedNextClicks": (
                    20
                ),
                "PredictedNextImpressions": (
                    200
                ),
                "EstimatedPositionGain": (
                    2
                ),
                "EstimatedGeoScoreGain": (
                    10
                ),
                "EstimatedContentScoreGain": (
                    15
                ),
                "ScenarioCTR": (
                    0.15
                ),
            },
        ]
    )

    scored = add_opportunity_score(
        simulation
    )

    assert len(
        scored
    ) == 2

    assert (
        "OpportunityScore"
        in scored.columns
    )

    # Use iloc for positional comparison.
    assert (
        scored.iloc[
            1
        ][
            "OpportunityScore"
        ]
        > scored.iloc[
            0
        ][
            "OpportunityScore"
        ]
    )


def test_add_business_decision_score() -> None:
    """
    Validate business-aware scenario scoring with all required fields.
    """
    dataframe = pd.DataFrame(
        [
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "maintain"
                ),
                "OpportunityScore": (
                    5.0
                ),
                "ExpectedNetValue": (
                    0.0
                ),
                "EstimatedROI": (
                    0.0
                ),
                "EstimatedImplementationCost": (
                    0.0
                ),
                "ConfidenceLevel": (
                    "High"
                ),
            },
            {
                "page": (
                    "https://example.com/a"
                ),
                "Scenario": (
                    "content_refresh"
                ),
                "OpportunityScore": (
                    50.0
                ),
                "ExpectedNetValue": (
                    500.0
                ),
                "EstimatedROI": (
                    4.0
                ),
                "EstimatedImplementationCost": (
                    120.0
                ),
                "ConfidenceLevel": (
                    "High"
                ),
            },
        ]
    )

    result = add_business_decision_score(
        dataframe
    )

    assert len(
        result
    ) == 2

    assert (
        "BusinessDecisionScore"
        in result.columns
    )

    assert (
        "DecisionEligible"
        in result.columns
    )

    # Use iloc instead of loc so VS Code/Pylance
    # does not show the previous strike-through warning.
    assert (
        result.iloc[
            1
        ][
            "BusinessDecisionScore"
        ]
        > result.iloc[
            0
        ][
            "BusinessDecisionScore"
        ]
    )

    assert bool(
        result.iloc[
            1
        ][
            "DecisionEligible"
        ]
    )

    assert bool(
        result.iloc[
            0
        ][
            "DecisionEligible"
        ]
    )