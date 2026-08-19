from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.settings import SETTINGS
from src.models.traffic_forecasting import safe_prediction
from src.utils.text_utils import clean_text


SEO_SCENARIOS: List[Dict[str, Any]] = [
    {
        "Scenario": "maintain",
        "label": "Maintain Current Setup",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.00,
        "position_delta": 0.0,
        "impressions_mult": 1.00,
        "content_score_delta": 0,
        "geo_score_delta": 0,
        "conversion_intent_mult": 1.00,
        "explanation": (
            "Maintain the current SEO and GEO setup without "
            "applying additional optimization."
        ),
    },
    {
        "Scenario": "title_meta_optimization",
        "label": "Title and Meta Optimization",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.15,
        "position_delta": -0.3,
        "impressions_mult": 1.03,
        "content_score_delta": 6,
        "geo_score_delta": 3,
        "conversion_intent_mult": 1.02,
        "explanation": (
            "Optimized titles and meta descriptions may improve "
            "CTR and organic clicks."
        ),
    },
    {
        "Scenario": "content_refresh",
        "label": "Content Refresh",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.08,
        "position_delta": -1.0,
        "impressions_mult": 1.08,
        "content_score_delta": 18,
        "geo_score_delta": 8,
        "conversion_intent_mult": 1.04,
        "explanation": (
            "Refreshing content and strengthening search-intent "
            "alignment may improve ranking and visibility."
        ),
    },
    {
        "Scenario": "internal_linking_boost",
        "label": "Internal Linking Improvement",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.05,
        "position_delta": -0.7,
        "impressions_mult": 1.10,
        "content_score_delta": 8,
        "geo_score_delta": 4,
        "conversion_intent_mult": 1.03,
        "explanation": (
            "Strengthening internal links between related pages "
            "may improve discoverability and authority distribution."
        ),
    },
    {
        "Scenario": "category_expansion",
        "label": "Category SEO Expansion",
        "applicable_page_types": [
            "category",
        ],
        "ctr_mult": 1.10,
        "position_delta": -1.3,
        "impressions_mult": 1.15,
        "content_score_delta": 24,
        "geo_score_delta": 10,
        "conversion_intent_mult": 1.06,
        "explanation": (
            "Expanding category copy, headings, FAQs and semantic "
            "coverage may increase category visibility."
        ),
    },
    {
        "Scenario": "product_content_enrichment",
        "label": "Product SEO Enrichment",
        "applicable_page_types": [
            "product",
        ],
        "ctr_mult": 1.12,
        "position_delta": -0.9,
        "impressions_mult": 1.11,
        "content_score_delta": 22,
        "geo_score_delta": 9,
        "conversion_intent_mult": 1.08,
        "explanation": (
            "Improving product descriptions, benefits, attributes, "
            "image text and FAQs may increase product performance."
        ),
    },
    {
        "Scenario": "structured_data_upgrade",
        "label": "Structured Data Upgrade",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.06,
        "position_delta": -0.4,
        "impressions_mult": 1.05,
        "content_score_delta": 5,
        "geo_score_delta": 12,
        "conversion_intent_mult": 1.03,
        "explanation": (
            "Completing page-type-aligned schema may improve "
            "machine readability and rich-result eligibility."
        ),
    },
    {
        "Scenario": "geo_answer_optimization",
        "label": "GEO Answer Optimization",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.07,
        "position_delta": -0.5,
        "impressions_mult": 1.08,
        "content_score_delta": 12,
        "geo_score_delta": 22,
        "conversion_intent_mult": 1.04,
        "explanation": (
            "Adding direct-answer blocks, summaries and FAQs may "
            "increase generative-search visibility potential."
        ),
    },
    {
        "Scenario": "entity_eet_upgrade",
        "label": "Entity and E-E-A-T Upgrade",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.05,
        "position_delta": -0.6,
        "impressions_mult": 1.07,
        "content_score_delta": 10,
        "geo_score_delta": 18,
        "conversion_intent_mult": 1.03,
        "explanation": (
            "Strengthening entity relationships, authorship, "
            "freshness and trust signals may improve readiness."
        ),
    },
    {
        "Scenario": "full_seo_geo_optimization",
        "label": "Full SEO and GEO Optimization",
        "applicable_page_types": [
            "category",
            "product",
            "blog",
            "informational",
            "other",
        ],
        "ctr_mult": 1.22,
        "position_delta": -1.8,
        "impressions_mult": 1.22,
        "content_score_delta": 32,
        "geo_score_delta": 30,
        "conversion_intent_mult": 1.10,
        "explanation": (
            "Combining metadata, content, internal links, schema, "
            "entity signals and GEO components may create the "
            "highest growth potential."
        ),
    },
]


def scenario_is_applicable(
    page_type: str,
    scenario: Dict[str, Any],
) -> bool:
    """
    Return whether a scenario applies to the supplied page type.
    """
    return page_type in scenario.get(
        "applicable_page_types",
        [],
    )


def estimate_current_content_score(
    row: pd.Series,
) -> float:
    """
    Estimate the current on-page content readiness score.
    """
    score = 20.0

    if clean_text(row.get("title", "")):
        score += 15

    if clean_text(row.get("meta_description", "")):
        score += 15

    if clean_text(row.get("h1", "")):
        score += 15

    word_count = len(
        clean_text(
            row.get("content", "")
        ).split()
    )

    if word_count >= 600:
        score += 25
    elif word_count >= 250:
        score += 18
    elif word_count >= 80:
        score += 10

    if clean_text(row.get("schema_type", "")):
        score += 10

    return min(
        100.0,
        score,
    )


def estimate_current_geo_score(
    row: pd.Series,
) -> float:
    """
    Estimate current generative-engine optimization readiness.
    """
    score = 15.0

    if clean_text(row.get("h1", "")):
        score += 10

    if clean_text(row.get("content", "")):
        score += 10

    if clean_text(row.get("schema_type", "")):
        score += 15

    if clean_text(row.get("brand", "")):
        score += 10

    content = clean_text(
        row.get("content", "")
    ).lower()

    if any(
        token in content
        for token in [
            "nedir",
            "nasıl",
            "kimler için",
            "sık sorulan",
            "what is",
            "how to",
            "who is it for",
            "frequently asked",
        ]
    ):
        score += 15

    if len(content.split()) >= 250:
        score += 15

    if clean_text(row.get("meta_description", "")):
        score += 10

    return min(
        100.0,
        score,
    )


def build_scenario_narrative(
    scenario_label: str,
    current_clicks: float,
    predicted_clicks: float,
    current_impressions: float,
    predicted_impressions: float,
    current_position: float,
    scenario_position: float,
    current_geo_score: float,
    scenario_geo_score: float,
) -> str:
    """
    Build a readable scenario-impact narrative.
    """
    click_change = (
        (
            predicted_clicks
            - current_clicks
        )
        / current_clicks
        * 100
        if current_clicks > 0
        else 0
    )

    impression_change = (
        (
            predicted_impressions
            - current_impressions
        )
        / current_impressions
        * 100
        if current_impressions > 0
        else 0
    )

    position_gain = max(
        0.0,
        current_position
        - scenario_position,
    )

    geo_gain = max(
        0.0,
        scenario_geo_score
        - current_geo_score,
    )

    return (
        f"If {scenario_label} is applied, estimated clicks may "
        f"change by {click_change:.1f}%, impressions by "
        f"{impression_change:.1f}%, average position may improve "
        f"by {position_gain:.1f}, and GEO readiness may increase "
        f"by {geo_gain:.0f} points."
    )


def simulate_seo_scenarios(
    latest_df: pd.DataFrame,
    model_clicks: RandomForestRegressor,
    model_impressions: RandomForestRegressor,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Simulate all applicable SEO and GEO scenarios for every page.
    """
    rows: List[Dict[str, Any]] = []

    for _, source_row in latest_df.iterrows():
        page_type = (
            clean_text(
                source_row.get(
                    "page_type",
                    "other",
                )
            ).lower()
            or "other"
        )

        current_content_score = (
            estimate_current_content_score(
                source_row
            )
        )

        current_geo_score = (
            estimate_current_geo_score(
                source_row
            )
        )

        for scenario in SEO_SCENARIOS:
            if not scenario_is_applicable(
                page_type,
                scenario,
            ):
                continue

            simulated = source_row.copy()

            new_position = max(
                1.0,
                float(source_row["position"])
                + scenario["position_delta"],
            )

            new_ctr = min(
                1.0,
                max(
                    0.0,
                    float(source_row["CTR"])
                    * scenario["ctr_mult"],
                ),
            )

            new_impressions = (
                float(source_row["impressions"])
                * scenario["impressions_mult"]
            )

            scenario_content_score = min(
                100.0,
                current_content_score
                + scenario["content_score_delta"],
            )

            scenario_geo_score = min(
                100.0,
                current_geo_score
                + scenario["geo_score_delta"],
            )

            simulated["position"] = new_position
            simulated["CTR"] = new_ctr
            simulated["impressions"] = new_impressions
            simulated["RankStrength"] = 1 / new_position
            simulated["VisibilityScore"] = (
                new_impressions
                * new_ctr
            )
            simulated["Top3Flag"] = int(
                new_position <= 3
            )
            simulated["Top10Flag"] = int(
                new_position <= 10
            )
            simulated["Page2Flag"] = int(
                10 < new_position <= 20
            )

            x_input = pd.DataFrame(
                [simulated]
            )[feature_columns]

            predicted_clicks = safe_prediction(
                model_clicks.predict(
                    x_input
                )[0]
            )

            predicted_impressions = safe_prediction(
                model_impressions.predict(
                    x_input
                )[0]
            )

            predicted_clicks *= scenario[
                "ctr_mult"
            ]

            predicted_clicks *= scenario[
                "conversion_intent_mult"
            ]

            predicted_clicks *= (
                1
                + max(
                    0,
                    float(source_row["position"])
                    - new_position,
                )
                * 0.05
            )

            predicted_impressions *= scenario[
                "impressions_mult"
            ]

            current_clicks = float(
                source_row["clicks"]
            )

            current_impressions = float(
                source_row["impressions"]
            )

            predicted_traffic_value = (
                predicted_clicks
                * SETTINGS.value_per_click
            )

            current_traffic_value = (
                current_clicks
                * SETTINGS.value_per_click
            )

            rows.append(
                {
                    "page": source_row["page"],
                    "page_type": page_type,
                    "date": source_row["date"],
                    "keyword_intent": source_row.get(
                        "keyword_intent",
                        "Uncategorized",
                    ),
                    "Scenario": scenario["Scenario"],
                    "ScenarioLabel": scenario["label"],
                    "ScenarioExplanation": (
                        scenario["explanation"]
                    ),
                    "CurrentClicks": round(
                        current_clicks,
                        2,
                    ),
                    "CurrentImpressions": round(
                        current_impressions,
                        2,
                    ),
                    "CurrentCTR": round(
                        float(source_row["CTR"]),
                        4,
                    ),
                    "CurrentPosition": round(
                        float(source_row["position"]),
                        2,
                    ),
                    "CurrentTrafficValue": round(
                        current_traffic_value,
                        2,
                    ),
                    "CurrentContentScore": round(
                        current_content_score,
                        2,
                    ),
                    "CurrentGeoReadinessScore": round(
                        current_geo_score,
                        2,
                    ),
                    "ScenarioCTR": round(
                        new_ctr,
                        4,
                    ),
                    "ScenarioPosition": round(
                        new_position,
                        2,
                    ),
                    "ScenarioImpressions": round(
                        new_impressions,
                        2,
                    ),
                    "ScenarioContentScore": round(
                        scenario_content_score,
                        2,
                    ),
                    "ScenarioGeoReadinessScore": round(
                        scenario_geo_score,
                        2,
                    ),
                    "PredictedNextClicks": round(
                        predicted_clicks,
                        2,
                    ),
                    "PredictedNextImpressions": round(
                        predicted_impressions,
                        2,
                    ),
                    "PredictedTrafficValue": round(
                        predicted_traffic_value,
                        2,
                    ),
                    "EstimatedClickChangePct": round(
                        (
                            (
                                predicted_clicks
                                - current_clicks
                            )
                            / current_clicks
                            * 100
                        )
                        if current_clicks > 0
                        else 0,
                        2,
                    ),
                    "EstimatedImpressionChangePct": round(
                        (
                            (
                                predicted_impressions
                                - current_impressions
                            )
                            / current_impressions
                            * 100
                        )
                        if current_impressions > 0
                        else 0,
                        2,
                    ),
                    "EstimatedPositionGain": round(
                        max(
                            0.0,
                            float(
                                source_row[
                                    "position"
                                ]
                            )
                            - new_position,
                        ),
                        2,
                    ),
                    "EstimatedGeoScoreGain": round(
                        scenario_geo_score
                        - current_geo_score,
                        2,
                    ),
                    "EstimatedContentScoreGain": round(
                        scenario_content_score
                        - current_content_score,
                        2,
                    ),
                    "ScenarioNarrative": (
                        build_scenario_narrative(
                            scenario["label"],
                            current_clicks,
                            predicted_clicks,
                            current_impressions,
                            predicted_impressions,
                            float(
                                source_row[
                                    "position"
                                ]
                            ),
                            new_position,
                            current_geo_score,
                            scenario_geo_score,
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def choose_best_scenario(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the highest-opportunity scenario for every page.
    """
    if simulation_df.empty:
        return simulation_df

    result = simulation_df.copy()

    result["OpportunityScore"] = (
        result["PredictedNextClicks"]
        * 0.42
        + result["PredictedNextImpressions"]
        * 0.0004
        + result["EstimatedPositionGain"]
        * 5.0
        + result["EstimatedGeoScoreGain"]
        * 0.80
        + result["EstimatedContentScoreGain"]
        * 0.45
        + result["ScenarioCTR"]
        * 100
        * 0.18
    )

    return (
        result.sort_values(
            [
                "page",
                "OpportunityScore",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "page",
            as_index=False,
        )
        .head(1)
        .reset_index(
            drop=True
        )
    )


def add_baseline_uplift(
    best_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the selected scenario against the maintain baseline.
    """
    baseline = simulation_df[
        simulation_df["Scenario"]
        == "maintain"
    ][
        [
            "page",
            "PredictedNextClicks",
            "PredictedNextImpressions",
        ]
    ].rename(
        columns={
            "PredictedNextClicks": (
                "BaselinePredictedClicks"
            ),
            "PredictedNextImpressions": (
                "BaselinePredictedImpressions"
            ),
        }
    )

    result = best_df.merge(
        baseline,
        on="page",
        how="left",
    )

    result["ClicksUplift"] = (
        result["PredictedNextClicks"]
        - result["BaselinePredictedClicks"]
    )

    result["ImpressionsUplift"] = (
        result["PredictedNextImpressions"]
        - result[
            "BaselinePredictedImpressions"
        ]
    )

    result["ClicksUpliftPct"] = np.where(
        result["BaselinePredictedClicks"] > 0,
        result["ClicksUplift"]
        / result["BaselinePredictedClicks"]
        * 100,
        0,
    )

    return result


def scenario_cost(
    scenario: str,
) -> Tuple[int, float]:
    """
    Return effort and estimated implementation cost.
    """
    mapping = {
        "maintain": (
            1,
            SETTINGS.cost_maintain,
        ),
        "title_meta_optimization": (
            2,
            SETTINGS.cost_title_meta,
        ),
        "content_refresh": (
            5,
            SETTINGS.cost_content_refresh,
        ),
        "internal_linking_boost": (
            3,
            SETTINGS.cost_internal_linking,
        ),
        "category_expansion": (
            6,
            SETTINGS.cost_content_refresh
            * 1.5,
        ),
        "product_content_enrichment": (
            5,
            SETTINGS.cost_content_refresh
            * 1.25,
        ),
        "structured_data_upgrade": (
            3,
            SETTINGS.cost_internal_linking,
        ),
        "geo_answer_optimization": (
            4,
            SETTINGS.cost_content_refresh,
        ),
        "entity_eet_upgrade": (
            4,
            SETTINGS.cost_content_refresh,
        ),
        "full_seo_geo_optimization": (
            9,
            SETTINGS.cost_content_refresh
            * 3.0,
        ),
    }

    return mapping.get(
        scenario,
        (
            4,
            SETTINGS.cost_default,
        ),
    )


def add_business_value_layers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add implementation effort, cost, value, ROI and payback metrics.
    """
    result = dataframe.copy()

    cost_values = result[
        "Scenario"
    ].map(
        scenario_cost
    )

    result["EffortScore"] = (
        cost_values.map(
            lambda value: value[0]
        )
    )

    result["EstimatedImplementationCost"] = (
        cost_values.map(
            lambda value: value[1]
        )
    )

    result["ExpectedIncrementalTrafficValue"] = (
        result["ClicksUplift"]
        * SETTINGS.value_per_click
    )

    result["ExpectedNetValue"] = (
        result[
            "ExpectedIncrementalTrafficValue"
        ]
        - result[
            "EstimatedImplementationCost"
        ]
    )

    result["EstimatedROI"] = np.where(
        result[
            "EstimatedImplementationCost"
        ] > 0,
        result["ExpectedNetValue"]
        / result[
            "EstimatedImplementationCost"
        ],
        0,
    )

    result["PaybackPeriod"] = np.where(
        result[
            "ExpectedIncrementalTrafficValue"
        ] > 0,
        result[
            "EstimatedImplementationCost"
        ]
        / result[
            "ExpectedIncrementalTrafficValue"
        ],
        0,
    )

    return result