from __future__ import annotations

from dashboard.services.analysis_service import (
    AnalysisData,
    aggregate_seo_kpis,
    fast_filter_period,
    filter_analysis_data,
    get_available_date_bounds,
    get_priority_recommendations,
    get_top_pages,
    load_analysis_data,
    resolve_date_column,
    resolve_page_column,
)

from dashboard.services.executive_service import (
    ExecutiveData,
    build_executive_kpis,
    build_executive_decision_intelligence,
    build_executive_opportunities,
    build_executive_top_pages,
    build_model_summary,
    build_recommendation_summary,
    load_executive_data,
)

from dashboard.services.decision_engine import (
    DecisionEngineResult,
    PeriodComparison,
    aggregate_period_kpis,
    build_decision_intelligence,
    build_decision_table,
    build_page_change_table,
    build_period_comparison,
    percent_change,
    resolve_forecast_status,
)


__all__ = [
    # ========================================================
    # DECISION ENGINE
    # ========================================================

    "DecisionEngineResult",
    "PeriodComparison",
    "aggregate_period_kpis",
    "build_decision_intelligence",
    "build_decision_table",
    "build_page_change_table",
    "build_period_comparison",
    "percent_change",
    "resolve_forecast_status",

    # ========================================================
    # ANALYSIS SERVICE
    # ========================================================

    "AnalysisData",
    "load_analysis_data",
    "fast_filter_period",
    "aggregate_seo_kpis",
    "filter_analysis_data",
    "get_available_date_bounds",
    "get_priority_recommendations",
    "get_top_pages",
    "resolve_date_column",
    "resolve_page_column",

    # ========================================================
    # EXECUTIVE SERVICE
    # ========================================================

    "ExecutiveData",
    "load_executive_data",
    "build_executive_kpis",
    "build_executive_decision_intelligence",
    "build_executive_opportunities",
    "build_executive_top_pages",
    "build_model_summary",
    "build_recommendation_summary",
]