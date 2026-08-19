
"""
Demo Store SEO Organic Growth Intelligence Pipeline

This project is a Python-based predictive decision-support system designed to
optimize organic growth performance using machine learning, scenario simulation,
and business-driven decision logic.

The system analyzes historical SEO data, predicts future performance, simulates
multiple optimization scenarios, and recommends the most effective actions based
on expected traffic impact, implementation effort, cost, and return on investment (ROI).

Core Capabilities:
- SEO data processing and KPI engineering (CTR, visibility, traffic value, rank strength)
- Predictive modeling for next-period clicks and impressions
- Model validation (MAE, RMSE, R²)
- Feature importance analysis for explainability
- SEO action scenario simulation (content, metadata, internal linking)
- Opportunity scoring and optimization logic
- Effort scoring and implementation cost estimation
- ROI and expected net value calculation
- Payback period estimation
- Priority-based decision framework
- Confidence scoring and guardrail mechanisms
- Rule-based fallback logic for low-data scenarios

Business Purpose:
This pipeline transforms SEO data into a structured decision intelligence system.
Instead of only reporting performance, it predicts outcomes, evaluates alternative
optimization strategies, and recommends actions that maximize organic growth impact
and resource efficiency.

Author: Ozlem Tonbul
"""
import os
import sys
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("seo_organic_growth_intelligence")


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def load_seo_data(file_path: str) -> pd.DataFrame:
    """
    Expected columns:
    - page
    - date
    - clicks
    - impressions
    - ctr (optional)
    - position
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SEO input file not found: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = ["page", "date", "clicks", "impressions", "position"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["page"] = df["page"].astype(str).str.strip()

    numeric_cols = ["clicks", "impressions", "position"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "ctr" in df.columns:
        df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce").fillna(0)
    else:
        df["ctr"] = 0.0

    df = df.dropna(subset=["date"])
    df = df[df["page"] != ""].copy()

    return df


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CTR"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)

    if "ctr" in df.columns:
        df["CTR"] = np.where(df["ctr"] > 0, df["ctr"], df["CTR"])

    value_per_click = env_float("SEO_VALUE_PER_CLICK", 0.50)

    df["TrafficValue"] = df["clicks"] * value_per_click
    df["RankStrength"] = np.where(df["position"] > 0, 1 / df["position"], 0)
    df["VisibilityScore"] = df["impressions"] * df["CTR"]

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month_num"] = df["date"].dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["page", "date"]).copy()
    group = df.groupby("page")

    df["clicks_lag_1"] = group["clicks"].shift(1)
    df["clicks_lag_7_avg"] = (
        group["clicks"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["impressions_lag_1"] = group["impressions"].shift(1)
    df["impressions_lag_7_avg"] = (
        group["impressions"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["position_lag_1"] = group["position"].shift(1)
    df["position_lag_7_avg"] = (
        group["position"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["ctr_lag_1"] = group["CTR"].shift(1)
    df["traffic_value_lag_1"] = group["TrafficValue"].shift(1)

    df["clicks_change"] = group["clicks"].diff()
    df["impressions_change"] = group["impressions"].diff()
    df["position_change"] = group["position"].diff()

    return df.fillna(0)


def prepare_training_data(seo_raw: pd.DataFrame) -> pd.DataFrame:
    df = seo_raw.copy()
    df = compute_kpis(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    df = df.sort_values(["page", "date"]).copy()
    group = df.groupby("page")

    df["target_clicks_next"] = group["clicks"].shift(-1)
    df["target_impressions_next"] = group["impressions"].shift(-1)

    df = df.dropna(subset=["target_clicks_next", "target_impressions_next"]).copy()
    return df


def get_feature_columns() -> List[str]:
    return [
        "clicks",
        "impressions",
        "position",
        "CTR",
        "TrafficValue",
        "RankStrength",
        "VisibilityScore",
        "day_of_week",
        "day_of_month",
        "month_num",
        "clicks_lag_1",
        "clicks_lag_7_avg",
        "impressions_lag_1",
        "impressions_lag_7_avg",
        "position_lag_1",
        "position_lag_7_avg",
        "ctr_lag_1",
        "traffic_value_lag_1",
        "clicks_change",
        "impressions_change",
        "position_change",
    ]


def train_and_validate_models(train_df: pd.DataFrame):
    feature_cols = get_feature_columns()
    X = train_df[feature_cols]

    y_clicks = train_df["target_clicks_next"]
    y_impr = train_df["target_impressions_next"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clicks, test_size=0.2, random_state=42
    )
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, y_impr, test_size=0.2, random_state=42
    )

    model_clicks = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )
    model_impr = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )

    model_clicks.fit(X_train_c, y_train_c)
    model_impr.fit(X_train_i, y_train_i)

    pred_clicks = model_clicks.predict(X_test_c)
    pred_impr = model_impr.predict(X_test_i)

    metrics_df = pd.DataFrame([
        {
            "Model": "Next_Clicks",
            "MAE": float(mean_absolute_error(y_test_c, pred_clicks)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_c, pred_clicks))),
            "R2": float(r2_score(y_test_c, pred_clicks)),
            "TrainRows": len(X_train_c),
            "TestRows": len(X_test_c),
        },
        {
            "Model": "Next_Impressions",
            "MAE": float(mean_absolute_error(y_test_i, pred_impr)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_i, pred_impr))),
            "R2": float(r2_score(y_test_i, pred_impr)),
            "TrainRows": len(X_train_i),
            "TestRows": len(X_test_i),
        },
    ])

    importance_clicks = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_clicks.feature_importances_,
        "Model": "Next_Clicks",
    }).sort_values("Importance", ascending=False)

    importance_impr = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_impr.feature_importances_,
        "Model": "Next_Impressions",
    }).sort_values("Importance", ascending=False)

    feature_importance_df = pd.concat(
        [importance_clicks, importance_impr],
        ignore_index=True
    )

    return model_clicks, model_impr, feature_cols, metrics_df, feature_importance_df


def get_latest_page_state(seo_raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare_training_data(seo_raw)
    if df.empty:
        return df

    latest_df = (
        df.sort_values(["page", "date"])
        .groupby("page", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_df


def safe_prediction(value: float) -> float:
    if value is None or np.isnan(value):
        return 0.0
    return max(0.0, float(value))


def simulate_seo_scenarios(
    latest_df: pd.DataFrame,
    model_clicks,
    model_impr,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    SEO action scenarios:
    - maintain
    - title_meta_optimization
    - content_refresh
    - internal_linking_boost
    """
    scenarios = [
        {"Scenario": "maintain", "ctr_mult": 1.00, "position_delta": 0.0, "impr_mult": 1.00},
        {"Scenario": "title_meta_optimization", "ctr_mult": 1.15, "position_delta": -0.3, "impr_mult": 1.03},
        {"Scenario": "content_refresh", "ctr_mult": 1.08, "position_delta": -1.0, "impr_mult": 1.08},
        {"Scenario": "internal_linking_boost", "ctr_mult": 1.05, "position_delta": -0.7, "impr_mult": 1.10},
    ]

    results = []

    for _, row in latest_df.iterrows():
        for sc in scenarios:
            sim_row = row.copy()

            new_position = max(1.0, float(row["position"]) + sc["position_delta"])
            new_ctr = min(1.0, max(0.0, float(row["CTR"]) * sc["ctr_mult"]))
            new_impressions = float(row["impressions"]) * sc["impr_mult"]

            sim_row["position"] = new_position
            sim_row["CTR"] = new_ctr
            sim_row["impressions"] = new_impressions
            sim_row["TrafficValue"] = float(row["clicks"]) * env_float("SEO_VALUE_PER_CLICK", 0.50)
            sim_row["RankStrength"] = 1 / new_position if new_position > 0 else 0
            sim_row["VisibilityScore"] = new_impressions * new_ctr

            X_input = pd.DataFrame([sim_row])[feature_cols]

            pred_clicks = safe_prediction(model_clicks.predict(X_input)[0])
            pred_impr = safe_prediction(model_impr.predict(X_input)[0])

            pred_clicks = pred_clicks * sc["ctr_mult"] * (
                1 + max(0, (float(row["position"]) - new_position)) * 0.05
            )

            results.append({
                "page": row["page"],
                "date": row["date"],
                "CurrentClicks": round(float(row["clicks"]), 2),
                "CurrentImpressions": round(float(row["impressions"]), 2),
                "CurrentCTR": round(float(row["CTR"]), 4),
                "CurrentPosition": round(float(row["position"]), 2),
                "Scenario": sc["Scenario"],
                "ScenarioCTR": round(float(new_ctr), 4),
                "ScenarioPosition": round(float(new_position), 2),
                "ScenarioImpressions": round(float(new_impressions), 2),
                "PredictedNextClicks": round(float(pred_clicks), 2),
                "PredictedNextImpressions": round(float(pred_impr), 2),
            })

    return pd.DataFrame(results)


def choose_best_scenario(sim_df: pd.DataFrame) -> pd.DataFrame:
    if sim_df.empty:
        return sim_df

    sim_df = sim_df.copy()

    sim_df["OpportunityScore"] = (
        (sim_df["PredictedNextClicks"] * 0.50) +
        (sim_df["PredictedNextImpressions"] * 0.0005) +
        ((20 - sim_df["ScenarioPosition"]) * 0.30) +
        (sim_df["ScenarioCTR"] * 100 * 0.20)
    )

    best_df = (
        sim_df.sort_values(["page", "OpportunityScore"], ascending=[True, False])
        .groupby("page", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best_df


def add_baseline_uplift(best_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
    df = best_df.copy()

    baseline_df = sim_df[sim_df["Scenario"] == "maintain"][
        ["page", "PredictedNextClicks", "PredictedNextImpressions"]
    ].rename(columns={
        "PredictedNextClicks": "BaselinePredictedClicks",
        "PredictedNextImpressions": "BaselinePredictedImpressions",
    })

    df = df.merge(baseline_df, on="page", how="left")

    df["ClicksUplift"] = df["PredictedNextClicks"] - df["BaselinePredictedClicks"]
    df["ImpressionsUplift"] = df["PredictedNextImpressions"] - df["BaselinePredictedImpressions"]

    df["ClicksUpliftPct"] = np.where(
        df["BaselinePredictedClicks"] > 0,
        (df["ClicksUplift"] / df["BaselinePredictedClicks"]) * 100,
        0
    )

    return df


def add_business_value_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds effort scoring, estimated implementation cost,
    expected traffic value, net value, ROI, and payback period.
    """
    df = df.copy()

    value_per_click = env_float("SEO_VALUE_PER_CLICK", 0.50)

    effort_map = {
        "maintain": {"EffortScore": 1, "EstimatedImplementationCost": env_float("SEO_COST_MAINTAIN", 0)},
        "title_meta_optimization": {"EffortScore": 2, "EstimatedImplementationCost": env_float("SEO_COST_TITLE_META", 40)},
        "content_refresh": {"EffortScore": 5, "EstimatedImplementationCost": env_float("SEO_COST_CONTENT_REFRESH", 120)},
        "internal_linking_boost": {"EffortScore": 3, "EstimatedImplementationCost": env_float("SEO_COST_INTERNAL_LINKING", 60)},
    }

    df["EffortScore"] = df["Scenario"].map(lambda x: effort_map.get(x, {}).get("EffortScore", 4))
    df["EstimatedImplementationCost"] = df["Scenario"].map(
        lambda x: effort_map.get(x, {}).get("EstimatedImplementationCost", 80)
    )

    df["ExpectedIncrementalTrafficValue"] = df["ClicksUplift"] * value_per_click
    df["ExpectedNetValue"] = df["ExpectedIncrementalTrafficValue"] - df["EstimatedImplementationCost"]

    df["EstimatedROI"] = np.where(
        df["EstimatedImplementationCost"] > 0,
        df["ExpectedNetValue"] / df["EstimatedImplementationCost"],
        0
    )

    df["PaybackPeriod"] = np.where(
        df["ExpectedIncrementalTrafficValue"] > 0,
        df["EstimatedImplementationCost"] / df["ExpectedIncrementalTrafficValue"],
        0
    )

    return df


def build_recommendation_reason(row: pd.Series) -> str:
    if row["Scenario"] == "title_meta_optimization":
        return "Low CTR opportunity. Title and meta optimization may improve click-through rate."
    if row["Scenario"] == "content_refresh":
        return "Ranking and relevance improvement opportunity. Content refresh may unlock additional organic growth."
    if row["Scenario"] == "internal_linking_boost":
        return "Internal linking may improve discoverability, relevance signals, and supporting impressions."
    return "Current page performance supports maintaining the existing SEO setup."


def build_recommendations(best_df: pd.DataFrame) -> pd.DataFrame:
    df = best_df.copy()

    def map_action(scenario: str) -> str:
        mapping = {
            "maintain": "Maintain",
            "title_meta_optimization": "Optimize Title/Meta",
            "content_refresh": "Refresh Content",
            "internal_linking_boost": "Improve Internal Linking",
        }
        return mapping.get(scenario, "Review")

    df["RecommendedAction"] = df["Scenario"].map(map_action)
    df["RecommendationReason"] = df.apply(build_recommendation_reason, axis=1)

    return df


def build_confidence_scores(
    recommendation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    df = recommendation_df.copy()

    page_history = (
        train_df.groupby("page", as_index=False)
        .size()
        .rename(columns={"size": "HistoryRows"})
    )
    df = df.merge(page_history, on="page", how="left")
    df["HistoryRows"] = df["HistoryRows"].fillna(0)

    clicks_r2 = float(metrics_df.loc[metrics_df["Model"] == "Next_Clicks", "R2"].iloc[0])
    impr_r2 = float(metrics_df.loc[metrics_df["Model"] == "Next_Impressions", "R2"].iloc[0])
    avg_r2 = (clicks_r2 + impr_r2) / 2

    def confidence_label(row):
        if row["HistoryRows"] >= 20 and avg_r2 >= 0.60:
            return "High"
        if row["HistoryRows"] >= 10 and avg_r2 >= 0.30:
            return "Medium"
        return "Low"

    df["ConfidenceLevel"] = df.apply(confidence_label, axis=1)
    return df


def apply_confidence_guardrail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    low_conf_mask = df["ConfidenceLevel"] == "Low"

    df.loc[low_conf_mask, "RecommendedAction"] = "Review"
    df.loc[low_conf_mask, "RecommendationReason"] = (
        "Low-confidence SEO recommendation. Manual validation is recommended before action."
    )

    return df


def add_priority_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    confidence_multiplier = {
        "High": 1.00,
        "Medium": 0.85,
        "Low": 0.60,
    }

    df["ConfidenceMultiplier"] = df["ConfidenceLevel"].map(confidence_multiplier).fillna(0.75)
    df["AdjustedNetValue"] = df["ExpectedNetValue"] * df["ConfidenceMultiplier"]

    def classify_priority(row):
        if row["AdjustedNetValue"] > 50 and row["EstimatedROI"] > 0.50:
            return "High Priority"
        if row["AdjustedNetValue"] > 0:
            return "Medium Priority"
        return "Low Priority"

    df["PriorityTier"] = df.apply(classify_priority, axis=1)
    return df


def build_rule_based_fallback(seo_raw: pd.DataFrame) -> pd.DataFrame:
    page_df = seo_raw.groupby("page", as_index=False).agg({
        "clicks": "sum",
        "impressions": "sum",
        "position": "mean",
    })

    page_df = compute_kpis(page_df)

    def fallback_action(row):
        if row["impressions"] > 1000 and row["CTR"] < 0.02:
            return "Optimize Title/Meta", "High visibility but weak CTR suggests snippet optimization opportunity."
        if row["position"] > 10 and row["impressions"] > 500:
            return "Refresh Content", "Strong visibility but weak ranking suggests content improvement opportunity."
        if row["clicks"] > 0 and row["position"] <= 5:
            return "Maintain", "Current page performance is relatively strong."
        return "Review", "Fallback rule-based SEO recommendation."

    decisions = page_df.apply(fallback_action, axis=1)
    page_df["RecommendedAction"] = decisions.apply(lambda x: x[0])
    page_df["RecommendationReason"] = decisions.apply(lambda x: x[1])
    page_df["ConfidenceLevel"] = "Low"

    page_df["EffortScore"] = 3
    page_df["EstimatedImplementationCost"] = env_float("SEO_COST_DEFAULT_FALLBACK", 80)
    page_df["ExpectedIncrementalTrafficValue"] = 0
    page_df["ExpectedNetValue"] = -page_df["EstimatedImplementationCost"]
    page_df["EstimatedROI"] = 0
    page_df["PaybackPeriod"] = 0
    page_df["PriorityTier"] = "Review"

    return page_df


def build_daily_weekly_monthly_outputs(seo_raw: pd.DataFrame):
    daily_df = seo_raw.groupby(["date", "page"], as_index=False).agg({
        "clicks": "sum",
        "impressions": "sum",
        "position": "mean",
    })
    daily_df = compute_kpis(daily_df)

    temp = seo_raw.copy()
    temp["week"] = temp["date"].dt.to_period("W").astype(str)
    temp["month"] = temp["date"].dt.to_period("M").astype(str)

    weekly_df = temp.groupby(["week", "page"], as_index=False).agg({
        "clicks": "sum",
        "impressions": "sum",
        "position": "mean",
    })
    weekly_df = compute_kpis(weekly_df)

    monthly_df = temp.groupby(["month", "page"], as_index=False).agg({
        "clicks": "sum",
        "impressions": "sum",
        "position": "mean",
    })
    monthly_df = compute_kpis(monthly_df)

    return daily_df, weekly_df, monthly_df


def build_recommendation_summary(recommendation_df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "page",
        "CurrentClicks",
        "CurrentImpressions",
        "CurrentCTR",
        "CurrentPosition",
        "Scenario",
        "RecommendedAction",
        "ConfidenceLevel",
        "PriorityTier",
        "EffortScore",
        "EstimatedImplementationCost",
        "PredictedNextClicks",
        "PredictedNextImpressions",
        "BaselinePredictedClicks",
        "ClicksUplift",
        "ClicksUpliftPct",
        "ExpectedIncrementalTrafficValue",
        "ExpectedNetValue",
        "EstimatedROI",
        "PaybackPeriod",
        "AdjustedNetValue",
        "RecommendationReason",
        "OpportunityScore",
    ]

    existing_cols = [col for col in summary_cols if col in recommendation_df.columns]
    return recommendation_df[existing_cols].copy()


def main() -> None:
    input_file = env("SEO_INPUT_FILE", "./data/seo_data.csv")
    output_dir = env("SEO_OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Reading SEO data from: %s", input_file)
    seo_raw = load_seo_data(input_file)

    if seo_raw.empty:
        logger.warning("SEO dataset is empty.")
        return

    train_df = prepare_training_data(seo_raw)
    daily_df, weekly_df, monthly_df = build_daily_weekly_monthly_outputs(seo_raw)

    if train_df.empty or len(train_df) < 20:
        logger.warning("Not enough historical data for robust ML. Using rule-based fallback.")
        fallback_df = build_rule_based_fallback(seo_raw)

        fallback_df.to_csv(
            os.path.join(output_dir, "seo_rule_based_recommendations.csv"),
            index=False
        )
        daily_df.to_csv(os.path.join(output_dir, "seo_daily_page_summary.csv"), index=False)
        weekly_df.to_csv(os.path.join(output_dir, "seo_weekly_page_summary.csv"), index=False)
        monthly_df.to_csv(os.path.join(output_dir, "seo_monthly_page_summary.csv"), index=False)
        return

    model_clicks, model_impr, feature_cols, metrics_df, feature_importance_df = train_and_validate_models(train_df)
    latest_df = get_latest_page_state(seo_raw)

    sim_df = simulate_seo_scenarios(
        latest_df=latest_df,
        model_clicks=model_clicks,
        model_impr=model_impr,
        feature_cols=feature_cols
    )

    best_df = choose_best_scenario(sim_df)
    best_df = add_baseline_uplift(best_df, sim_df)
    best_df = add_business_value_layers(best_df)

    recommendation_df = build_recommendations(best_df)
    recommendation_df = build_confidence_scores(recommendation_df, metrics_df, train_df)
    recommendation_df = apply_confidence_guardrail(recommendation_df)
    recommendation_df = add_priority_tier(recommendation_df)

    summary_df = build_recommendation_summary(recommendation_df)

    sim_df.to_csv(os.path.join(output_dir, "seo_action_scenarios.csv"), index=False)
    recommendation_df.to_csv(os.path.join(output_dir, "seo_recommendations.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "seo_recommendation_summary.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "seo_model_validation_metrics.csv"), index=False)
    feature_importance_df.to_csv(os.path.join(output_dir, "seo_feature_importance.csv"), index=False)
    daily_df.to_csv(os.path.join(output_dir, "seo_daily_page_summary.csv"), index=False)
    weekly_df.to_csv(os.path.join(output_dir, "seo_weekly_page_summary.csv"), index=False)
    monthly_df.to_csv(os.path.join(output_dir, "seo_monthly_page_summary.csv"), index=False)

    logger.info("SEO Organic Growth Intelligence pipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)