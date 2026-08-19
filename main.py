
from __future__ import annotations

import sys
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

import pandas as pd

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.extract import (
    crawl_website,
    build_url_inventory,
    fetch_sitemap_urls,
    mark_crawl_batch_completed,
    select_crawl_rotation_batch,
    enrich_crawl_with_pagespeed,
    fetch_ga4_landing_page_data,
    fetch_pagespeed_data,
    fetch_search_console_data,
    select_pagespeed_urls,
    load_optional_csv,
    load_seo_csv,
    standardize_seo_dataframe,
)
from src.extract.pagespeed_extractor import mark_pagespeed_batch_completed
from src.memory import save_recommendations_to_memory
from src.rag import ingest_pipeline_outputs
from src.features import (
    build_technical_seo_intelligence,
    add_page_classifications,
    build_holiday_map,
    build_geo_ai_visibility_intelligence,
    build_blog_content_to_commerce_intelligence,
    build_blog_keyword_content_gaps,
    build_keyword_opportunity_intelligence,
    build_page_opportunity_intelligence,
    build_product_category_opportunities,
    merge_gsc_and_ga4,
    prepare_training_data,
)
from src.models import (
    add_baseline_uplift,
    add_business_decision_score,
    add_business_value_layers,
    add_opportunity_score,
    choose_best_scenario,
    combine_shap_explanations,
    get_last_model_benchmark,
    get_latest_page_state,
    simulate_seo_scenarios,
    train_and_validate_models,
)
from src.models.multi_horizon_forecasting import (
    run_multi_horizon_forecasting,
)
from src.recommendations import (
    add_priority_tier,
    apply_confidence_guardrail,
    build_confidence_scores,
    build_recommendations,
    generate_page_commentaries,
    generate_seo_portfolio_commentary,
)
from src.reporting import (
    build_daily_weekly_monthly_outputs,
    build_keyword_intent_summary,
    build_page_type_summary,
    build_recommendation_summary,
    build_seo_holiday_impact,
    export_outputs,
    export_run_manifest,
    export_text_report,
)
from src.utils.date_utils import resolve_date_range
from src.warehouse import write_outputs_to_postgres


logger = get_logger(__name__)


PRIORITY_ORDER = {
    "High Priority": 1,
    "Medium Priority": 2,
    "Low Priority": 3,
}


def load_pipeline_sources() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the data sources configured for the pipeline.

    Supported modes
    ---------------
    csv:
        Load the SEO dataset from a local CSV file.
        GA4 metrics are initialized with zero values.

    api:
        Extract Google Search Console and GA4 data through APIs.

    hybrid:
        Load the SEO dataset from CSV and extract GA4 data
        through the API.

    Returns
    -------
    tuple
        SEO data, GA4 data, crawl data and product-feed data.
    """
    data_source_mode = SETTINGS.data_source_mode

    logger.info(
        "Loading pipeline sources using mode: %s",
        data_source_mode,
    )

    if data_source_mode == "csv":
        seo_dataframe = load_seo_csv(
            SETTINGS.input_file
        )

        ga4_dataframe = pd.DataFrame()

    elif data_source_mode == "api":
        seo_dataframe = fetch_search_console_data()

        seo_dataframe = standardize_seo_dataframe(
            seo_dataframe
        )

        ga4_dataframe = (
            fetch_ga4_landing_page_data()
        )

    elif data_source_mode == "hybrid":
        seo_dataframe = load_seo_csv(
            SETTINGS.input_file
        )

        ga4_dataframe = (
            fetch_ga4_landing_page_data()
        )

    else:
        raise ValueError(
            "Unsupported DATA_SOURCE_MODE. "
            f"Received: {data_source_mode!r}"
        )

    url_inventory_dataframe = pd.DataFrame()

    if SETTINGS.crawl_input_file:
        crawl_dataframe = load_optional_csv(
            file_path=SETTINGS.crawl_input_file,
            dataset_name="Crawl dataset",
        )
    elif SETTINGS.technical_crawl_enabled:
        if SETTINGS.full_site_inventory_enabled:
            sitemap_urls = fetch_sitemap_urls(
                sitemap_url=SETTINGS.sitemap_url,
                max_urls=SETTINGS.sitemap_max_urls,
                request_timeout=SETTINGS.crawl_request_timeout,
            )
            url_inventory_dataframe = build_url_inventory(
                sitemap_urls=sitemap_urls,
                seo_dataframe=seo_dataframe,
                allowed_host=urlsplit(
                    SETTINGS.crawl_start_url
                    or SETTINGS.gsc_site_url
                ).netloc.lower(),
            )
            crawl_batch = select_crawl_rotation_batch(
                inventory=url_inventory_dataframe,
                batch_size=SETTINGS.crawl_max_pages,
                state_file=SETTINGS.crawl_rotation_state_file,
            )
            seed_urls = (
                crawl_batch["url"].tolist()
                if not crawl_batch.empty
                else None
            )
            logger.info(
                "Full-site inventory prepared: %d URLs | crawl batch: %d URLs.",
                len(url_inventory_dataframe),
                len(crawl_batch),
            )
            crawl_dataframe = crawl_website(
                seed_urls=seed_urls,
            )
            if not crawl_dataframe.empty and "url" in crawl_dataframe.columns:
                mark_crawl_batch_completed(
                    crawled_urls=crawl_dataframe["url"].astype(str).tolist(),
                    state_file=SETTINGS.crawl_rotation_state_file,
                )
        else:
            crawl_dataframe = crawl_website()
    else:
        logger.info(
            "Technical crawl is disabled and CRAWL_INPUT_FILE is empty. Skipping."
        )
        crawl_dataframe = pd.DataFrame()

    product_feed_dataframe = load_optional_csv(
        file_path=SETTINGS.product_feed_file,
        dataset_name="Product feed",
    )

    if seo_dataframe.empty:
        raise ValueError(
            "The main SEO dataset is empty."
        )

    logger.info(
        "Source loading completed | "
        "SEO rows: %d | "
        "GA4 rows: %d | "
        "Crawl rows: %d | "
        "Product rows: %d",
        len(seo_dataframe),
        len(ga4_dataframe),
        len(crawl_dataframe),
        len(product_feed_dataframe),
    )

    return (
        seo_dataframe,
        ga4_dataframe,
        crawl_dataframe,
        product_feed_dataframe,
        url_inventory_dataframe,
    )


def prepare_integrated_dataset(
    seo_dataframe: pd.DataFrame,
    ga4_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SEO and GA4 data and ensure page classifications exist.
    """
    integrated_dataframe = merge_gsc_and_ga4(
        gsc_dataframe=seo_dataframe,
        ga4_dataframe=ga4_dataframe,
    )

    integrated_dataframe = add_page_classifications(
        integrated_dataframe
    )

    if integrated_dataframe.empty:
        raise ValueError(
            "The integrated SEO and GA4 dataset is empty."
        )

    logger.info(
        "Integrated dataset prepared: %d rows.",
        len(integrated_dataframe),
    )

    return integrated_dataframe


def train_pipeline_models(
    integrated_dataframe: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> Tuple[
    pd.DataFrame,
    object,
    object,
    List[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Prepare the training dataset, benchmark forecasting
    algorithms and train the selected production models.

    Returns
    -------
    training_dataframe
        Feature-engineered training observations.

    clicks_model
        Selected production model for next-period clicks.

    impressions_model
        Selected production model for next-period impressions.

    feature_columns
        Feature columns used by the forecasting models.

    model_metrics
        Validation metrics for the selected production models.

    feature_importance
        Native feature importance for the selected models.

    model_benchmark
        Complete Random Forest vs XGBoost vs LightGBM
        benchmark for both forecasting targets.
    """
    training_dataframe = prepare_training_data(
        seo_raw=integrated_dataframe,
        holiday_map=holiday_map,
    )

    if training_dataframe.empty:
        raise ValueError(
            "No model-training rows were generated. "
            "The dataset must contain multiple dated observations "
            "for each page."
        )

    logger.info(
        "Training dataset prepared: %d rows.",
        len(training_dataframe),
    )

    (
        clicks_model,
        impressions_model,
        feature_columns,
        model_metrics,
        feature_importance,
    ) = train_and_validate_models(
        train_df=training_dataframe,
        with_holiday=True,
    )

    model_benchmark = (
        get_last_model_benchmark()
    )

    if model_benchmark.empty:
        raise ValueError(
            "Model benchmark was not generated."
        )

    logger.info(
        "Forecasting models trained successfully "
        "| Benchmark rows: %d.",
        len(model_benchmark),
    )

    return (
        training_dataframe,
        clicks_model,
        impressions_model,
        feature_columns,
        model_metrics,
        feature_importance,
        model_benchmark,
    )


def get_selected_model_algorithm(
    model_metrics: pd.DataFrame,
    model_name: str,
) -> str:
    """
    Return the selected production algorithm for one forecast target.
    """
    if model_metrics is None or model_metrics.empty:
        raise ValueError(
            "Model metrics are empty. "
            "Cannot resolve the selected production algorithm."
        )

    required_columns = {
        "Model",
        "Algorithm",
    }

    missing_columns = (
        required_columns
        - set(
            model_metrics.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Model metrics are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    selected_rows = model_metrics[
        model_metrics["Model"].astype(str)
        == str(model_name)
    ]

    if "Selected" in selected_rows.columns:
        selected_rows = selected_rows[
            selected_rows[
                "Selected"
            ].astype(bool)
        ]

    if selected_rows.empty:
        raise ValueError(
            "No selected production model metric was found "
            f"for {model_name}."
        )

    algorithm = str(
        selected_rows.iloc[0][
            "Algorithm"
        ]
    ).strip()

    if not algorithm:
        raise ValueError(
            "Selected production algorithm cannot be empty "
            f"for {model_name}."
        )

    return algorithm


def enrich_shap_detail_with_page_context(
    shap_detail: pd.DataFrame,
    latest_page_state: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add page/date context to row-level SHAP explanations.

    SHAP stores the original dataframe index as RowIndex. This helper
    converts that technical identifier into page-level evidence that can
    be used by reporting, PostgreSQL, RAG and the dashboard.
    """
    if shap_detail is None:
        raise ValueError(
            "SHAP detail dataframe cannot be None."
        )

    if shap_detail.empty:
        return shap_detail.copy()

    if latest_page_state is None or latest_page_state.empty:
        return shap_detail.copy()

    context_columns = [
        column
        for column in [
            "page",
            "date",
            "page_type",
            "keyword_intent",
        ]
        if column in latest_page_state.columns
    ]

    if not context_columns:
        return shap_detail.copy()

    page_context = (
        latest_page_state[
            context_columns
        ]
        .copy()
    )

    page_context[
        "RowIndex"
    ] = (
        page_context.index
        .astype(str)
    )

    rename_map = {
        "page": "Page",
        "date": "ObservationDate",
        "page_type": "PageType",
        "keyword_intent": "KeywordIntent",
    }

    page_context = (
        page_context
        .rename(
            columns=rename_map
        )
    )

    enriched = shap_detail.merge(
        page_context,
        on="RowIndex",
        how="left",
        validate="many_to_one",
    )

    preferred_columns = [
        "Model",
        "Algorithm",
        "RowIndex",
        "Page",
        "ObservationDate",
        "PageType",
        "KeywordIntent",
        "Feature",
        "FeatureValue",
        "SHAPValue",
        "AbsSHAPValue",
        "Direction",
        "BaseValue",
        "Prediction",
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in enriched.columns
    ]

    remaining_columns = [
        column
        for column in enriched.columns
        if column not in ordered_columns
    ]

    return enriched[
        ordered_columns
        + remaining_columns
    ]


def build_pipeline_explainability(
    latest_page_state: pd.DataFrame,
    clicks_model: object,
    impressions_model: object,
    feature_columns: List[str],
    model_metrics: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build SHAP explainability outputs for both production models.

    The selected clicks and impressions algorithms are resolved from the
    real benchmark winner metrics. SHAP then explains those exact models,
    not a hard-coded algorithm.
    """
    if latest_page_state is None or latest_page_state.empty:
        logger.warning(
            "Latest page state is empty. "
            "Skipping SHAP explainability."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    clicks_algorithm = (
        get_selected_model_algorithm(
            model_metrics=model_metrics,
            model_name="Next_Clicks",
        )
    )

    impressions_algorithm = (
        get_selected_model_algorithm(
            model_metrics=model_metrics,
            model_name="Next_Impressions",
        )
    )

    shap_max_rows = int(
        getattr(
            SETTINGS,
            "shap_max_rows",
            200,
        )
    )

    shap_top_features = int(
        getattr(
            SETTINGS,
            "shap_top_features_per_row",
            10,
        )
    )

    (
        shap_detail,
        shap_summary,
    ) = combine_shap_explanations(
        clicks_model=clicks_model,
        impressions_model=impressions_model,
        dataframe=latest_page_state,
        feature_columns=feature_columns,
        clicks_algorithm=clicks_algorithm,
        impressions_algorithm=impressions_algorithm,
        max_rows=shap_max_rows,
        top_features_per_row=shap_top_features,
    )

    shap_detail = (
        enrich_shap_detail_with_page_context(
            shap_detail=shap_detail,
            latest_page_state=latest_page_state,
        )
    )

    logger.info(
        "Pipeline SHAP explainability completed "
        "| Detail rows: %d "
        "| Summary rows: %d "
        "| Clicks algorithm: %s "
        "| Impressions algorithm: %s.",
        len(shap_detail),
        len(shap_summary),
        clicks_algorithm,
        impressions_algorithm,
    )

    return (
        shap_detail,
        shap_summary,
    )

def build_pipeline_recommendations(
    integrated_dataframe: pd.DataFrame,
    training_dataframe: pd.DataFrame,
    holiday_map: Dict[str, str],
    clicks_model: object,
    impressions_model: object,
    feature_columns: List[str],
    model_metrics: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Simulate SEO scenarios and build final recommendations.
    """
    latest_page_state = get_latest_page_state(
        seo_raw=integrated_dataframe,
        holiday_map=holiday_map,
    )

    if latest_page_state.empty:
        raise ValueError(
            "No latest page state could be generated."
        )

    logger.info(
        "Latest page state prepared: %d pages.",
        len(latest_page_state),
    )

    scenario_simulation = simulate_seo_scenarios(
        latest_df=latest_page_state,
        model_clicks=clicks_model,
        model_impressions=impressions_model,
        feature_columns=feature_columns,
    )

    if scenario_simulation.empty:
        raise ValueError(
            "SEO scenario simulation returned no rows."
        )

    scenario_simulation = add_opportunity_score(
        scenario_simulation
    )

    scenario_simulation = add_baseline_uplift(
        best_df=scenario_simulation,
        simulation_df=scenario_simulation,
    )

    scenario_simulation = add_business_value_layers(
        scenario_simulation
    )

    scenario_simulation = build_confidence_scores(
        recommendation_df=scenario_simulation,
        metrics_df=model_metrics,
        train_df=training_dataframe,
    )

    scenario_simulation = add_business_decision_score(
        scenario_simulation
    )

    best_scenarios = choose_best_scenario(
        scenario_simulation
    )

    recommendations = build_recommendations(
        best_scenarios
    )

    recommendations = apply_confidence_guardrail(
        recommendations
    )

    recommendations = add_priority_tier(
        recommendations
    )

    recommendations["PriorityOrder"] = (
        recommendations["PriorityTier"]
        .map(PRIORITY_ORDER)
        .fillna(4)
        .astype(int)
    )

    recommendations = (
        recommendations.sort_values(
            [
                "PriorityOrder",
                "AdjustedNetValue",
                "OpportunityScore",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop(
            columns=["PriorityOrder"],
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Recommendation engine completed: %d recommendations.",
        len(recommendations),
    )

    return (
        latest_page_state,
        scenario_simulation,
        recommendations,
    )


def build_reporting_outputs(
    seo_dataframe: pd.DataFrame,
    integrated_dataframe: pd.DataFrame,
    training_dataframe: pd.DataFrame,
    latest_page_state: pd.DataFrame,
    scenario_simulation: pd.DataFrame,
    recommendations: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_benchmark: pd.DataFrame,
    feature_importance: pd.DataFrame,
    holiday_map: Dict[str, str],
    crawl_dataframe: pd.DataFrame,
    product_feed_dataframe: pd.DataFrame,
    url_inventory_dataframe: pd.DataFrame,
) -> Tuple[
    Dict[str, pd.DataFrame],
    str,
]:
    """
    Build reporting tables and optional LLM commentary.
    """
    keyword_intent_summary = (
        build_keyword_intent_summary(
            integrated_dataframe
        )
    )

    page_type_summary = (
        build_page_type_summary(
            integrated_dataframe
        )
    )

    holiday_impact = build_seo_holiday_impact(
        seo_raw=integrated_dataframe,
        holiday_map=holiday_map,
    )

    (
        daily_performance,
        weekly_performance,
        monthly_performance,
    ) = build_daily_weekly_monthly_outputs(
        integrated_dataframe
    )

    recommendation_summary = (
        build_recommendation_summary(
            recommendations
        )
    )

    page_opportunity_intelligence = (
        build_page_opportunity_intelligence(
            integrated_dataframe=integrated_dataframe,
            recommendations=recommendations,
        )
    )

    keyword_opportunity_intelligence = (
        build_keyword_opportunity_intelligence(
            seo_dataframe=seo_dataframe,
            integrated_dataframe=integrated_dataframe,
        )
    )

    product_category_opportunities = (
        build_product_category_opportunities(
            page_opportunity_intelligence
        )
    )

    blog_content_to_commerce = (
        build_blog_content_to_commerce_intelligence(
            page_intelligence=page_opportunity_intelligence,
            keyword_intelligence=keyword_opportunity_intelligence,
        )
    )

    blog_keyword_content_gaps = (
        build_blog_keyword_content_gaps(
            keyword_intelligence=keyword_opportunity_intelligence,
            page_intelligence=page_opportunity_intelligence,
        )
    )

    geo_ai_visibility_intelligence = (
        build_geo_ai_visibility_intelligence(
            page_intelligence=page_opportunity_intelligence,
            latest_page_state=latest_page_state,
        )
    )

    pagespeed_dataframe = pd.DataFrame()
    technical_crawl_dataframe = crawl_dataframe

    if SETTINGS.pagespeed_enabled:
        pagespeed_urls = select_pagespeed_urls(
            page_intelligence=page_opportunity_intelligence,
            url_inventory=url_inventory_dataframe,
            max_urls=SETTINGS.pagespeed_max_urls,
            state_file=SETTINGS.pagespeed_rotation_state_file,
            freshness_days=SETTINGS.pagespeed_freshness_days,
        )

        pagespeed_dataframe = fetch_pagespeed_data(
            pagespeed_urls
        )

        if (
            not pagespeed_dataframe.empty
            and "url" in pagespeed_dataframe.columns
        ):
            mark_pagespeed_batch_completed(
                pagespeed_dataframe=pagespeed_dataframe,
                state_file=SETTINGS.pagespeed_rotation_state_file,
            )

        technical_crawl_dataframe = enrich_crawl_with_pagespeed(
            crawl_dataframe=crawl_dataframe,
            pagespeed_dataframe=pagespeed_dataframe,
        )

    technical_seo_intelligence = (
        build_technical_seo_intelligence(
            crawl_dataframe=technical_crawl_dataframe,
            page_intelligence=page_opportunity_intelligence,
        )
    )

    recommendation_summary = (
        generate_page_commentaries(
            recommendation_summary
        )
    )

    portfolio_commentary = (
        generate_seo_portfolio_commentary(
            summary_df=recommendation_summary,
            intent_df=keyword_intent_summary,
        )
    )

    outputs: Dict[str, pd.DataFrame] = {
        "seo_integrated_data": integrated_dataframe,
        "seo_training_data": training_dataframe,
        "seo_latest_page_state": latest_page_state,
        "seo_scenario_simulation": scenario_simulation,
        "seo_recommendations": recommendation_summary,
        "seo_page_opportunity_intelligence": (
            page_opportunity_intelligence
        ),
        "seo_keyword_opportunity_intelligence": (
            keyword_opportunity_intelligence
        ),
        "seo_product_category_opportunities": (
            product_category_opportunities
        ),
        "seo_blog_content_to_commerce": (
            blog_content_to_commerce
        ),
        "seo_blog_keyword_content_gaps": (
            blog_keyword_content_gaps
        ),
        "seo_geo_ai_visibility_intelligence": (
            geo_ai_visibility_intelligence
        ),
        "seo_technical_seo_intelligence": (
            technical_seo_intelligence
        ),
        "seo_keyword_intent_summary": (
            keyword_intent_summary
        ),
        "seo_page_type_summary": (
            page_type_summary
        ),
        "seo_holiday_impact": (
            holiday_impact
        ),
        "seo_daily_performance": (
            daily_performance
        ),
        "seo_weekly_performance": (
            weekly_performance
        ),
        "seo_monthly_performance": (
            monthly_performance
        ),
        "seo_model_metrics": (
            model_metrics
        ),
        "seo_model_benchmark": (
            model_benchmark
        ),
        "seo_feature_importance": (
            feature_importance
        ),
    }

    if not url_inventory_dataframe.empty:
        outputs["seo_url_inventory"] = (
            url_inventory_dataframe
        )

    if not technical_crawl_dataframe.empty:
        outputs["seo_crawl_data"] = (
            technical_crawl_dataframe
        )

    if not pagespeed_dataframe.empty:
        outputs["seo_pagespeed_data"] = (
            pagespeed_dataframe
        )

    if not product_feed_dataframe.empty:
        outputs["seo_product_feed"] = (
            product_feed_dataframe
        )

    logger.info(
        "Reporting tables prepared: %d outputs.",
        len(outputs),
    )

    return (
        outputs,
        portfolio_commentary,
    )


def export_pipeline_outputs(
    outputs: Dict[str, pd.DataFrame],
    portfolio_commentary: str,
) -> None:
    """
    Export reporting outputs to CSV, JSON, text and PostgreSQL.
    """
    exported_files = export_outputs(
        outputs=outputs,
        output_dir=SETTINGS.output_dir,
    )

    portfolio_report_path = export_text_report(
        content=portfolio_commentary,
        output_dir=SETTINGS.output_dir,
        filename="seo_portfolio_commentary",
    )

    manifest_path = export_run_manifest(
        output_dir=SETTINGS.output_dir,
        outputs=outputs,
        input_file=SETTINGS.input_file,
    )

    postgres_tables = write_outputs_to_postgres(
        outputs
    )

    logger.info(
        "CSV files written: %d",
        len(exported_files),
    )

    logger.info(
        "Portfolio commentary written: %s",
        portfolio_report_path,
    )

    logger.info(
        "Run manifest written: %s",
        manifest_path,
    )

    logger.info(
        "PostgreSQL tables written: %d",
        len(postgres_tables),
    )


def run_pipeline() -> Dict[str, pd.DataFrame]:
    """
    Run the complete SEO Organic Growth Intelligence pipeline.
    """
    logger.info(
        "Starting %s",
        SETTINGS.project_name,
    )

    SETTINGS.validate()

    date_from, date_to = resolve_date_range()

    logger.info(
        "Reporting date range: %s to %s",
        date_from,
        date_to,
    )

    holiday_map = build_holiday_map(
        date_from=date_from,
        date_to=date_to,
    )

    (
        seo_dataframe,
        ga4_dataframe,
        crawl_dataframe,
        product_feed_dataframe,
        url_inventory_dataframe,
    ) = load_pipeline_sources()

    integrated_dataframe = (
        prepare_integrated_dataset(
            seo_dataframe=seo_dataframe,
            ga4_dataframe=ga4_dataframe,
        )
    )

    (
        training_dataframe,
        clicks_model,
        impressions_model,
        feature_columns,
        model_metrics,
        feature_importance,
        model_benchmark,
    ) = train_pipeline_models(
        integrated_dataframe=integrated_dataframe,
        holiday_map=holiday_map,
    )

    # --------------------------------------------------------
    # TRUE MULTI-HORIZON DAILY ML FORECASTING
    # --------------------------------------------------------

    multi_horizon_forecast = run_multi_horizon_forecasting(
        seo_raw=integrated_dataframe,
        horizons=(7, 14, 30, 90, 180, 365),
    )

    (
        latest_page_state,
        scenario_simulation,
        recommendations,
    ) = build_pipeline_recommendations(
        integrated_dataframe=integrated_dataframe,
        training_dataframe=training_dataframe,
        holiday_map=holiday_map,
        clicks_model=clicks_model,
        impressions_model=impressions_model,
        feature_columns=feature_columns,
        model_metrics=model_metrics,
    )

    (
        shap_detail,
        shap_summary,
    ) = build_pipeline_explainability(
        latest_page_state=latest_page_state,
        clicks_model=clicks_model,
        impressions_model=impressions_model,
        feature_columns=feature_columns,
        model_metrics=model_metrics,
    )

    decision_memory_rows = (
        save_recommendations_to_memory(
            recommendations=recommendations,
        )
    )

    logger.info(
        "Persistent agent decision memory completed: %d new decisions.",
        decision_memory_rows,
    )

    (
        outputs,
        portfolio_commentary,
    ) = build_reporting_outputs(
        seo_dataframe=seo_dataframe,
        integrated_dataframe=integrated_dataframe,
        training_dataframe=training_dataframe,
        latest_page_state=latest_page_state,
        scenario_simulation=scenario_simulation,
        recommendations=recommendations,
        model_metrics=model_metrics,
        model_benchmark=model_benchmark,
        feature_importance=feature_importance,
        holiday_map=holiday_map,
        crawl_dataframe=crawl_dataframe,
        product_feed_dataframe=product_feed_dataframe,
        url_inventory_dataframe=url_inventory_dataframe,
    )

    # Persist the genuine recursive daily ML forecast separately from
    # one-step SEO/GEO scenario simulation outputs.
    outputs[
        "seo_ml_daily_training"
    ] = multi_horizon_forecast.training
    outputs[
        "seo_ml_forecast_daily"
    ] = multi_horizon_forecast.daily_forecast
    outputs[
        "seo_ml_forecast_horizons"
    ] = multi_horizon_forecast.horizon_forecast
    outputs[
        "seo_ml_forecast_portfolio"
    ] = multi_horizon_forecast.portfolio_forecast
    outputs[
        "seo_ml_forecast_metrics"
    ] = multi_horizon_forecast.metrics
    outputs[
        "seo_ml_forecast_benchmark"
    ] = multi_horizon_forecast.benchmark
    outputs[
        "seo_ml_forecast_feature_importance"
    ] = multi_horizon_forecast.feature_importance

    if not shap_detail.empty:
        outputs[
            "seo_shap_detail"
        ] = shap_detail

    if not shap_summary.empty:
        outputs[
            "seo_shap_summary"
        ] = shap_summary

    if SETTINGS.rag_enabled:
        rag_ingestion_summary = (
            ingest_pipeline_outputs(
                outputs=outputs,
            )
        )

        if not rag_ingestion_summary.empty:
            outputs[
                "seo_rag_ingestion_summary"
            ] = rag_ingestion_summary

        total_rag_chunks = (
            int(
                rag_ingestion_summary[
                    "chunks"
                ]
                .fillna(0)
                .sum()
            )
            if (
                not rag_ingestion_summary.empty
                and "chunks"
                in rag_ingestion_summary.columns
            )
            else 0
        )

        logger.info(
            "RAG pipeline ingestion completed "
            "| Sources: %d | Chunks: %d.",
            len(rag_ingestion_summary),
            total_rag_chunks,
        )

    else:
        logger.info(
            "RAG pipeline ingestion is disabled. "
            "Skipping."
        )

    export_pipeline_outputs(
        outputs=outputs,
        portfolio_commentary=portfolio_commentary,
    )

    logger.info(
        "Pipeline completed successfully."
    )

    return outputs

def main() -> int:
    """
    Application entry point.
    """
    try:
        run_pipeline()
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Pipeline execution was cancelled by the user."
        )

        return 130

    except Exception:
        logger.exception(
            "Pipeline execution failed."
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
