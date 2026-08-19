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

    Performance optimization:
    - Prepare all scenario rows first.
    - Build one feature matrix.
    - Run click prediction once in batch.
    - Run impression prediction once in batch.
    - Preserve the existing scenario/business logic.
    """
    if latest_df.empty:
        return pd.DataFrame()

    feature_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # 1. PREPARE SCENARIO FEATURE ROWS
    # --------------------------------------------------------

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

        current_clicks = float(
            source_row["clicks"]
        )

        current_impressions = float(
            source_row["impressions"]
        )

        current_ctr = float(
            source_row["CTR"]
        )

        current_position = float(
            source_row["position"]
        )

        current_traffic_value = (
            current_clicks
            * SETTINGS.value_per_click
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
                current_position
                + scenario["position_delta"],
            )

            new_ctr = min(
                1.0,
                max(
                    0.0,
                    current_ctr
                    * scenario["ctr_mult"],
                ),
            )

            new_impressions = (
                current_impressions
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

            simulated["RankStrength"] = (
                1 / new_position
            )

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

            feature_rows.append(
                simulated.reindex(
                    feature_columns
                ).to_dict()
            )

            scenario_rows.append(
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

                    "ctr_mult": float(
                        scenario["ctr_mult"]
                    ),
                    "impressions_mult": float(
                        scenario["impressions_mult"]
                    ),
                    "conversion_intent_mult": float(
                        scenario[
                            "conversion_intent_mult"
                        ]
                    ),

                    "CurrentClicks": current_clicks,
                    "CurrentImpressions": (
                        current_impressions
                    ),
                    "CurrentCTR": current_ctr,
                    "CurrentPosition": (
                        current_position
                    ),
                    "CurrentTrafficValue": (
                        current_traffic_value
                    ),

                    "CurrentContentScore": (
                        current_content_score
                    ),
                    "CurrentGeoReadinessScore": (
                        current_geo_score
                    ),

                    "ScenarioCTR": new_ctr,
                    "ScenarioPosition": (
                        new_position
                    ),
                    "ScenarioImpressions": (
                        new_impressions
                    ),
                    "ScenarioContentScore": (
                        scenario_content_score
                    ),
                    "ScenarioGeoReadinessScore": (
                        scenario_geo_score
                    ),
                }
            )

    if not feature_rows:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 2. BUILD ONE BATCH FEATURE MATRIX
    # --------------------------------------------------------

    x_input = pd.DataFrame(
        feature_rows,
        columns=feature_columns,
    )

    # --------------------------------------------------------
    # 3. BATCH MODEL PREDICTIONS
    # --------------------------------------------------------

    raw_click_predictions = (
        model_clicks.predict(
            x_input
        )
    )

    raw_impression_predictions = (
        model_impressions.predict(
            x_input
        )
    )

    rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # 4. APPLY SCENARIO BUSINESS LOGIC
    # --------------------------------------------------------

    for (
        metadata,
        raw_clicks,
        raw_impressions,
    ) in zip(
        scenario_rows,
        raw_click_predictions,
        raw_impression_predictions,
    ):
        predicted_clicks = safe_prediction(
            raw_clicks
        )

        predicted_impressions = (
            safe_prediction(
                raw_impressions
            )
        )

        # The model already receives scenario-adjusted CTR, position
        # and impression features in x_input. Applying the same
        # multipliers again here would double-count the SEO effect.

        current_clicks = (
            metadata["CurrentClicks"]
        )

        current_impressions = (
            metadata[
                "CurrentImpressions"
            ]
        )

        current_position = (
            metadata["CurrentPosition"]
        )

        scenario_position = (
            metadata["ScenarioPosition"]
        )

        current_content_score = (
            metadata[
                "CurrentContentScore"
            ]
        )

        scenario_content_score = (
            metadata[
                "ScenarioContentScore"
            ]
        )

        current_geo_score = (
            metadata[
                "CurrentGeoReadinessScore"
            ]
        )

        scenario_geo_score = (
            metadata[
                "ScenarioGeoReadinessScore"
            ]
        )

        # Conversion intent affects the business-value layer rather
        # than artificially multiplying the model's click forecast.
        predicted_traffic_value = (
            predicted_clicks
            * SETTINGS.value_per_click
            * metadata[
                "conversion_intent_mult"
            ]
        )

        click_change_pct = (
            (
                predicted_clicks
                - current_clicks
            )
            / current_clicks
            * 100
            if current_clicks > 0
            else 0
        )

        impression_change_pct = (
            (
                predicted_impressions
                - current_impressions
            )
            / current_impressions
            * 100
            if current_impressions > 0
            else 0
        )

        estimated_position_gain = max(
            0.0,
            current_position
            - scenario_position,
        )

        estimated_geo_gain = (
            scenario_geo_score
            - current_geo_score
        )

        estimated_content_gain = (
            scenario_content_score
            - current_content_score
        )

        rows.append(
            {
                "page": metadata["page"],
                "page_type": (
                    metadata["page_type"]
                ),
                "date": metadata["date"],
                "keyword_intent": (
                    metadata[
                        "keyword_intent"
                    ]
                ),

                "Scenario": (
                    metadata["Scenario"]
                ),
                "ScenarioLabel": (
                    metadata["ScenarioLabel"]
                ),
                "ScenarioExplanation": (
                    metadata[
                        "ScenarioExplanation"
                    ]
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
                    metadata["CurrentCTR"],
                    4,
                ),
                "CurrentPosition": round(
                    current_position,
                    2,
                ),
                "CurrentTrafficValue": round(
                    metadata[
                        "CurrentTrafficValue"
                    ],
                    2,
                ),

                "CurrentContentScore": round(
                    current_content_score,
                    2,
                ),
                "CurrentGeoReadinessScore": (
                    round(
                        current_geo_score,
                        2,
                    )
                ),

                "ScenarioCTR": round(
                    metadata["ScenarioCTR"],
                    4,
                ),
                "ScenarioPosition": round(
                    scenario_position,
                    2,
                ),
                "ScenarioImpressions": round(
                    metadata[
                        "ScenarioImpressions"
                    ],
                    2,
                ),
                "ScenarioContentScore": round(
                    scenario_content_score,
                    2,
                ),
                "ScenarioGeoReadinessScore": (
                    round(
                        scenario_geo_score,
                        2,
                    )
                ),

                "PredictedNextClicks": round(
                    predicted_clicks,
                    2,
                ),
                "PredictedNextImpressions": (
                    round(
                        predicted_impressions,
                        2,
                    )
                ),
                "PredictedTrafficValue": round(
                    predicted_traffic_value,
                    2,
                ),

                "EstimatedClickChangePct": round(
                    click_change_pct,
                    2,
                ),
                "EstimatedImpressionChangePct": (
                    round(
                        impression_change_pct,
                        2,
                    )
                ),
                "EstimatedPositionGain": round(
                    estimated_position_gain,
                    2,
                ),
                "EstimatedGeoScoreGain": round(
                    estimated_geo_gain,
                    2,
                ),
                "EstimatedContentScoreGain": (
                    round(
                        estimated_content_gain,
                        2,
                    )
                ),

                "ScenarioNarrative": (
                    build_scenario_narrative(
                        metadata[
                            "ScenarioLabel"
                        ],
                        current_clicks,
                        predicted_clicks,
                        current_impressions,
                        predicted_impressions,
                        current_position,
                        scenario_position,
                        current_geo_score,
                        scenario_geo_score,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def add_opportunity_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the organic-growth opportunity score to every scenario row.

    This score measures the SEO/GEO upside only. Business value,
    implementation cost, ROI and confidence are added in later layers.
    """
    result = dataframe.copy()

    if result.empty:
        return result

    required_columns = [
        "PredictedNextClicks",
        "PredictedNextImpressions",
        "EstimatedPositionGain",
        "EstimatedGeoScoreGain",
        "EstimatedContentScoreGain",
        "ScenarioCTR",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing opportunity-score columns: "
            f"{missing_columns}"
        )

    result["OpportunityScore"] = (
        result["PredictedNextClicks"] * 0.42
        + result["PredictedNextImpressions"] * 0.0004
        + result["EstimatedPositionGain"] * 5.0
        + result["EstimatedGeoScoreGain"] * 0.80
        + result["EstimatedContentScoreGain"] * 0.45
        + result["ScenarioCTR"] * 100 * 0.18
    )

    return result


def _normalize_within_page(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a 0-1 min-max score within each page."""

    def normalize_group(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        minimum = float(numeric.min())
        maximum = float(numeric.max())

        if maximum == minimum:
            return pd.Series(
                0.5,
                index=series.index,
                dtype=float,
            )

        return (numeric - minimum) / (maximum - minimum)

    return dataframe.groupby(
        "page",
        group_keys=False,
    )[column].transform(normalize_group)


def add_business_decision_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add an Ads-style business-aware decision score to every SEO/GEO scenario.

    The final score balances organic opportunity with net value, ROI,
    implementation efficiency and model/data confidence. Scenarios with
    negative economics or low confidence receive guardrail penalties.
    """
    result = dataframe.copy()

    if result.empty:
        return result

    required_columns = [
        "page",
        "Scenario",
        "OpportunityScore",
        "ExpectedNetValue",
        "EstimatedROI",
        "EstimatedImplementationCost",
        "ConfidenceLevel",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing business-decision columns: "
            f"{missing_columns}"
        )

    confidence_map = {
        "High": 1.00,
        "Medium": 0.70,
        "Low": 0.35,
    }

    result["OpportunityScoreNormalized"] = (
        _normalize_within_page(
            result,
            "OpportunityScore",
        )
    )

    result["NetValueScoreNormalized"] = (
        _normalize_within_page(
            result,
            "ExpectedNetValue",
        )
    )

    result["ROIScoreNormalized"] = (
        _normalize_within_page(
            result,
            "EstimatedROI",
        )
    )

    cost_normalized = _normalize_within_page(
        result,
        "EstimatedImplementationCost",
    )
    result["CostEfficiencyScore"] = 1.0 - cost_normalized

    result["ConfidenceDecisionScore"] = (
        result["ConfidenceLevel"]
        .map(confidence_map)
        .fillna(0.50)
    )

    result["BusinessDecisionScore"] = 100 * (
        result["OpportunityScoreNormalized"] * 0.35
        + result["NetValueScoreNormalized"] * 0.25
        + result["ROIScoreNormalized"] * 0.20
        + result["ConfidenceDecisionScore"] * 0.10
        + result["CostEfficiencyScore"] * 0.10
    )

    negative_value_mask = (
        pd.to_numeric(
            result["ExpectedNetValue"],
            errors="coerce",
        ).fillna(0.0) <= 0
    )
    non_positive_roi_mask = (
        pd.to_numeric(
            result["EstimatedROI"],
            errors="coerce",
        ).fillna(0.0) <= 0
    )
    low_confidence_mask = (
        result["ConfidenceLevel"] == "Low"
    )

    result.loc[
        negative_value_mask,
        "BusinessDecisionScore",
    ] -= 20.0

    result.loc[
        non_positive_roi_mask,
        "BusinessDecisionScore",
    ] -= 10.0

    result.loc[
        low_confidence_mask,
        "BusinessDecisionScore",
    ] -= 8.0

    result["DecisionEligible"] = (
        result["Scenario"].eq("maintain")
        | (
            (~negative_value_mask)
            & (~non_positive_roi_mask)
        )
    )

    result["BusinessDecisionScore"] = result[
        "BusinessDecisionScore"
    ].round(4)

    return result


def choose_best_scenario(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the best scenario for every page.

    When BusinessDecisionScore is available, the selector uses the
    business-aware score. Otherwise it falls back to the legacy organic
    OpportunityScore behavior for backwards compatibility.
    """
    if simulation_df.empty:
        return simulation_df

    result = simulation_df.copy()

    if "OpportunityScore" not in result.columns:
        result = add_opportunity_score(
            result
        )

    if "BusinessDecisionScore" in result.columns:
        if "DecisionEligible" in result.columns:
            result["_decision_eligible_order"] = (
                result["DecisionEligible"]
                .fillna(False)
                .astype(int)
            )
        else:
            result["_decision_eligible_order"] = 1

        result = result.sort_values(
            [
                "page",
                "_decision_eligible_order",
                "BusinessDecisionScore",
                "ExpectedNetValue",
                "OpportunityScore",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                False,
            ],
        )
    else:
        result = result.sort_values(
            [
                "page",
                "OpportunityScore",
            ],
            ascending=[
                True,
                False,
            ],
        )

    return (
        result
        .groupby(
            "page",
            as_index=False,
        )
        .head(1)
        .drop(
            columns=["_decision_eligible_order"],
            errors="ignore",
        )
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
    Add implementation effort, cost, traffic value,
    projected business value, ROI and payback metrics.

    ExpectedIncrementalTrafficValue preserves the value of the
    model's next-observation uplift.

    ProjectedIncrementalTrafficValue evaluates that uplift across
    a configurable business horizon before comparing it with the
    one-time SEO implementation cost.
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

    result[
        "EstimatedImplementationCost"
    ] = (
        cost_values.map(
            lambda value: value[1]
        )
    )

    clicks_uplift = (
        pd.to_numeric(
            result["ClicksUplift"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # Preserve the original next-observation metric.
    result[
        "ExpectedIncrementalTrafficValue"
    ] = (
        clicks_uplift
        * SETTINGS.value_per_click
    )

    # The forecast target is the next observation, while SEO
    # implementation costs are one-time costs. Evaluate the
    # expected uplift over a configurable business horizon.
    business_horizon = max(
        1,
        int(
            getattr(
                SETTINGS,
                "business_value_horizon_days",
                30,
            )
        ),
    )

    result[
        "BusinessValueHorizonDays"
    ] = business_horizon

    result[
        "ProjectedIncrementalClicks"
    ] = (
        clicks_uplift
        * business_horizon
    )

    result[
        "ProjectedIncrementalTrafficValue"
    ] = (
        result[
            "ProjectedIncrementalClicks"
        ]
        * SETTINGS.value_per_click
    )

    result["ExpectedNetValue"] = (
        result[
            "ProjectedIncrementalTrafficValue"
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
        0.0,
    )

    # Payback is expressed in forecast periods.
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
        np.nan,
    )

    return result