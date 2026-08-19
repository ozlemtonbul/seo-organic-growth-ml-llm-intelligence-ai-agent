from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


# ============================================================
# DATA MODELS
# ============================================================


@dataclass(frozen=True)
class PeriodComparison:
    """Canonical comparison between an analysis and comparison period."""

    current: dict[str, float]
    previous: dict[str, float]
    deltas: dict[str, float | None]
    current_available: bool = True
    comparison_available: bool = True


@dataclass(frozen=True)
class DecisionEngineResult:
    """Shared decision-intelligence output consumed by dashboard pages."""

    comparison: PeriodComparison
    page_changes: pd.DataFrame
    decisions: pd.DataFrame
    forecast_horizon_days: int
    forecast_status: str


# ============================================================
# COLUMN HELPERS
# ============================================================


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _numeric_series(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return pd.Series(
            dtype="float64"
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def _sum_metric(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    values = _numeric_series(
        dataframe,
        candidates,
    )

    if values.empty:
        return 0.0

    return float(
        values.fillna(0).sum()
    )


def _mean_metric(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    values = _numeric_series(
        dataframe,
        candidates,
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.mean()
    )


def _weighted_position_metric(
    dataframe: pd.DataFrame,
) -> float:
    """
    Return impression-weighted average search position.

    Falls back to a simple mean only when no usable impression
    weights are available.
    """

    position_values = _numeric_series(
        dataframe,
        [
            "position",
            "Position",
            "CurrentPosition",
        ],
    )

    if position_values.empty:
        return 0.0

    impression_values = _numeric_series(
        dataframe,
        [
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ],
    )

    if not impression_values.empty:
        valid = (
            position_values.notna()
            & impression_values.notna()
            & impression_values.gt(0)
        )

        if valid.any():
            weights = impression_values.loc[
                valid
            ]

            return float(
                (
                    position_values.loc[
                        valid
                    ]
                    * weights
                ).sum()
                / weights.sum()
            )

    clean_position = (
        position_values
        .dropna()
    )

    if clean_position.empty:
        return 0.0

    return float(
        clean_position.mean()
    )


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(
        numerator / denominator
    )


# ============================================================
# PERIOD KPI ENGINE
# ============================================================


def aggregate_period_kpis(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Aggregate the canonical SEO + GA4 KPIs used by dashboard pages.

    CTR is calculated from total clicks / total impressions.
    Search position is impression-weighted.
    """

    if dataframe.empty:
        return {
            "clicks": 0.0,
            "impressions": 0.0,
            "ctr": 0.0,
            "position": 0.0,
            "sessions": 0.0,
            "users": 0.0,
            "conversions": 0.0,
            "revenue": 0.0,
            "purchases": 0.0,
        }

    clicks = _sum_metric(
        dataframe,
        [
            "clicks",
            "Clicks",
            "CurrentClicks",
        ],
    )

    impressions = _sum_metric(
        dataframe,
        [
            "impressions",
            "Impressions",
            "CurrentImpressions",
        ],
    )

    return {
        "clicks": clicks,
        "impressions": impressions,
        "ctr": _safe_divide(
            clicks,
            impressions,
        ),
        "position": _weighted_position_metric(
            dataframe
        ),
        "sessions": _sum_metric(
            dataframe,
            [
                "sessions",
                "Sessions",
            ],
        ),
        "users": _sum_metric(
            dataframe,
            [
                "users",
                "Users",
            ],
        ),
        "conversions": _sum_metric(
            dataframe,
            [
                "conversions",
                "Conversions",
            ],
        ),
        "revenue": _sum_metric(
            dataframe,
            [
                "revenue",
                "Revenue",
            ],
        ),
        "purchases": _sum_metric(
            dataframe,
            [
                "purchases",
                "Purchases",
            ],
        ),
    }


def percent_change(
    current: float,
    previous: float,
) -> float | None:
    """Return percentage change, or None when the previous value is zero."""
    if previous == 0:
        return None

    return float(
        ((current - previous) / abs(previous))
        * 100.0
    )


def build_period_comparison(
    current_period: pd.DataFrame,
    comparison_period: pd.DataFrame,
) -> PeriodComparison:
    """
    Build the shared period comparison used by Executive Overview,
    Page Analysis, Opportunity Optimizer, AI Insights, and Ask AI.

    Position is special: lower is better, therefore position_delta stores
    improvement in ranking positions (positive = improvement).
    """
    current = aggregate_period_kpis(
        current_period
    )
    previous = aggregate_period_kpis(
        comparison_period
    )

    deltas: dict[str, float | None] = {
        "clicks_pct": percent_change(
            current["clicks"],
            previous["clicks"],
        ),
        "impressions_pct": percent_change(
            current["impressions"],
            previous["impressions"],
        ),
        "ctr_pct": percent_change(
            current["ctr"],
            previous["ctr"],
        ),
        "sessions_pct": percent_change(
            current["sessions"],
            previous["sessions"],
        ),
        "conversions_pct": percent_change(
            current["conversions"],
            previous["conversions"],
        ),
        "revenue_pct": percent_change(
            current["revenue"],
            previous["revenue"],
        ),
        "position_improvement": (
            float(
                previous["position"]
                - current["position"]
            )
            if previous["position"] != 0
            and current["position"] != 0
            else None
        ),
    }

    comparison_available = not comparison_period.empty
    current_available = not current_period.empty

    if not comparison_available:
        deltas = {key: None for key in deltas}

    return PeriodComparison(
        current=current,
        previous=previous,
        deltas=deltas,
        current_available=current_available,
        comparison_available=comparison_available,
    )


# ============================================================
# PAGE CHANGE / ATTRIBUTION ENGINE
# ============================================================


def _aggregate_pages(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate actual performance to page level.

    CTR is recomputed from totals and average position is weighted
    by impressions so high-volume observations have the correct
    influence on the page-level position.
    """

    if dataframe.empty:
        return pd.DataFrame()

    page_column = _first_existing_column(
        dataframe,
        [
            "page",
            "Page",
            "url",
            "URL",
        ],
    )

    if page_column is None:
        return pd.DataFrame()

    result = dataframe.copy()

    result["_page"] = (
        result[
            page_column
        ]
        .astype(str)
        .str.strip()
    )

    source_map = {
        "_clicks": [
            "clicks",
            "Clicks",
        ],
        "_impressions": [
            "impressions",
            "Impressions",
        ],
        "_position": [
            "position",
            "Position",
        ],
    }

    for target, candidates in source_map.items():
        source = _first_existing_column(
            result,
            candidates,
        )

        if source is None:
            result[target] = 0.0
        else:
            result[target] = pd.to_numeric(
                result[source],
                errors="coerce",
            )

    result["_position_weighted_sum"] = (
        result["_position"]
        * result["_impressions"]
    )

    grouped = (
        result
        .groupby(
            "_page",
            as_index=False,
            dropna=False,
        )
        .agg(
            clicks=(
                "_clicks",
                "sum",
            ),
            impressions=(
                "_impressions",
                "sum",
            ),
            _position_weighted_sum=(
                "_position_weighted_sum",
                "sum",
            ),
            _simple_position=(
                "_position",
                "mean",
            ),
        )
        .rename(
            columns={
                "_page": "page"
            }
        )
    )

    grouped[
        "ctr"
    ] = (
        grouped[
            "clicks"
        ]
        / grouped[
            "impressions"
        ].replace(
            0,
            pd.NA,
        )
    ).fillna(
        0.0
    )

    grouped[
        "position"
    ] = (
        grouped[
            "_position_weighted_sum"
        ]
        / grouped[
            "impressions"
        ].replace(
            0,
            pd.NA,
        )
    )

    grouped[
        "position"
    ] = grouped[
        "position"
    ].fillna(
        grouped[
            "_simple_position"
        ]
    ).fillna(
        0.0
    )

    return (
        grouped.drop(
            columns=[
                "_position_weighted_sum",
                "_simple_position",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )


def build_page_change_table(
    current_period: pd.DataFrame,
    comparison_period: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare page-level performance and expose the main attribution signals.

    Positive PositionImprovement means ranking improved.
    """
    current = _aggregate_pages(
        current_period
    )
    previous = _aggregate_pages(
        comparison_period
    )

    if current.empty:
        return pd.DataFrame()

    current = current.rename(
        columns={
            "clicks": "CurrentClicks",
            "impressions": "CurrentImpressions",
            "ctr": "CurrentCTR",
            "position": "CurrentPosition",
        }
    )

    if previous.empty:
        result = current.copy()
        result["PreviousClicks"] = pd.NA
        result["PreviousImpressions"] = pd.NA
        result["PreviousCTR"] = pd.NA
        result["PreviousPosition"] = pd.NA
    else:
        previous = previous.rename(
            columns={
                "clicks": "PreviousClicks",
                "impressions": "PreviousImpressions",
                "ctr": "PreviousCTR",
                "position": "PreviousPosition",
            }
        )

        result = current.merge(
            previous,
            on="page",
            how="outer",
        )

    for column in [
        "CurrentClicks",
        "CurrentImpressions",
        "CurrentCTR",
        "CurrentPosition",
        "PreviousClicks",
        "PreviousImpressions",
        "PreviousCTR",
        "PreviousPosition",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    if previous.empty:
        result["ClicksDelta"] = pd.NA
        result["ImpressionsDelta"] = pd.NA
        result["CTRDelta"] = pd.NA
    else:
        result["ClicksDelta"] = (
            result["CurrentClicks"].fillna(0)
            - result["PreviousClicks"].fillna(0)
        )
        result["ImpressionsDelta"] = (
            result["CurrentImpressions"].fillna(0)
            - result["PreviousImpressions"].fillna(0)
        )
        result["CTRDelta"] = (
            result["CurrentCTR"].fillna(0)
            - result["PreviousCTR"].fillna(0)
        )

    result["PositionImprovement"] = (
        result["PreviousPosition"]
        - result["CurrentPosition"]
    )

    previous_clicks = result[
        "PreviousClicks"
    ].abs()

    result["ClicksChangePct"] = (
        result["ClicksDelta"]
        / previous_clicks.replace(0, pd.NA)
        * 100.0
    )

    previous_impressions = result[
        "PreviousImpressions"
    ].abs()

    result["ImpressionsChangePct"] = (
        result["ImpressionsDelta"]
        / previous_impressions.replace(0, pd.NA)
        * 100.0
    )

    return (
        result
        .sort_values(
            "ClicksDelta",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PROBLEM / OPPORTUNITY CLASSIFICATION
# ============================================================


def _classify_page_signal(
    row: pd.Series,
) -> str:
    clicks_change = row.get(
        "ClicksChangePct"
    )
    ctr_delta = row.get(
        "CTRDelta"
    )
    position_improvement = row.get(
        "PositionImprovement"
    )
    impressions_change = row.get(
        "ImpressionsChangePct"
    )

    clicks_change = (
        float(clicks_change)
        if pd.notna(clicks_change)
        else 0.0
    )
    ctr_delta = (
        float(ctr_delta)
        if pd.notna(ctr_delta)
        else 0.0
    )
    position_improvement = (
        float(position_improvement)
        if pd.notna(position_improvement)
        else 0.0
    )
    impressions_change = (
        float(impressions_change)
        if pd.notna(impressions_change)
        else 0.0
    )

    if clicks_change <= -10:
        if (
            impressions_change >= 0
            and ctr_delta < 0
        ):
            return "ctr_loss"

        if position_improvement <= -0.5:
            return "ranking_loss"

        return "traffic_loss"

    if (
        impressions_change >= 10
        and ctr_delta < 0
    ):
        return "ctr_opportunity"

    if clicks_change >= 10:
        return "growth_opportunity"

    return "stable_optimization"


def _signal_text(
    signal: str,
    language: str,
) -> tuple[str, str]:
    mapping = {
        "ctr_loss": (
            (
                "CTR Kaybı",
                "Gösterim korunurken veya artarken CTR geriledi; sayfa SERP görünürlüğünü tıklamaya yeterince çeviremiyor.",
            ),
            (
                "CTR Loss",
                "CTR declined while impressions were stable or growing; the page is not converting SERP visibility into clicks efficiently.",
            ),
        ),
        "ranking_loss": (
            (
                "Sıralama Kaybı",
                "Tıklama kaybına ortalama pozisyondaki kötüleşme eşlik ediyor.",
            ),
            (
                "Ranking Loss",
                "The click decline is accompanied by a deterioration in average ranking position.",
            ),
        ),
        "traffic_loss": (
            (
                "Organik Trafik Kaybı",
                "Sayfanın organik tıklamaları karşılaştırma dönemine göre anlamlı biçimde geriledi.",
            ),
            (
                "Organic Traffic Loss",
                "The page lost a meaningful share of organic clicks versus the comparison period.",
            ),
        ),
        "ctr_opportunity": (
            (
                "CTR Fırsatı",
                "Gösterimler büyüyor ancak CTR aynı hızda gelişmiyor; snippet optimizasyonu fırsatı var.",
            ),
            (
                "CTR Opportunity",
                "Impressions are growing but CTR is not keeping pace, indicating a snippet-optimization opportunity.",
            ),
        ),
        "growth_opportunity": (
            (
                "Büyüme Fırsatı",
                "Sayfa organik tıklamalarını güçlü biçimde artırdı; büyümeyi koruyup ölçekleme fırsatı var.",
            ),
            (
                "Growth Opportunity",
                "The page achieved strong click growth and may offer an opportunity to preserve and scale the gain.",
            ),
        ),
        "stable_optimization": (
            (
                "Optimizasyon Fırsatı",
                "Büyük bir performans kırılması yok; model ve iş kuralları sonraki en değerli optimizasyonu belirleyebilir.",
            ),
            (
                "Optimization Opportunity",
                "No major performance break is visible; model and business-rule signals can determine the next highest-value optimization.",
            ),
        ),
    }

    tr_value, en_value = mapping[
        signal
    ]

    return (
        tr_value
        if language == "tr"
        else en_value
    )


def _format_evidence(
    row: pd.Series,
    language: str,
) -> str:
    clicks = row.get(
        "ClicksChangePct"
    )
    impressions = row.get(
        "ImpressionsChangePct"
    )
    ctr_delta = row.get(
        "CTRDelta"
    )
    position = row.get(
        "PositionImprovement"
    )

    def pct(value: Any) -> str:
        return (
            f"{float(value):+.1f}%"
            if pd.notna(value)
            else "-"
        )

    ctr_points = (
        f"{float(ctr_delta) * 100:+.2f} pp"
        if pd.notna(ctr_delta)
        else "-"
    )

    position_text = (
        f"{float(position):+.2f}"
        if pd.notna(position)
        else "-"
    )

    if language == "tr":
        return (
            f"Tıklama {pct(clicks)} | "
            f"Gösterim {pct(impressions)} | "
            f"CTR {ctr_points} | "
            f"Pozisyon iyileşmesi {position_text}"
        )

    return (
        f"Clicks {pct(clicks)} | "
        f"Impressions {pct(impressions)} | "
        f"CTR {ctr_points} | "
        f"Position improvement {position_text}"
    )


def _localize_action(
    value: Any,
    language: str,
) -> str:
    raw = str(
        value or ""
    ).strip()

    if not raw:
        return "-"

    mapping = {
        "Apply Full SEO and GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Full SEO + GEO Optimization",
        ),
        "Apply Full SEO + GEO Optimization": (
            "Tam SEO + GEO Optimizasyonu",
            "Full SEO + GEO Optimization",
        ),
        "Optimize Title and Meta": (
            "Başlık ve Meta Optimizasyonu",
            "Title and Meta Optimization",
        ),
        "Refresh Content": (
            "İçerik Güncellemesi",
            "Content Refresh",
        ),
        "Improve Internal Linking": (
            "İç Link Optimizasyonu",
            "Internal Linking Optimization",
        ),
        "Fix Technical SEO Issues": (
            "Teknik SEO Düzeltmesi",
            "Technical SEO Fix",
        ),
        "Maintain Current Setup": (
            "Mevcut Durumu Koru",
            "Maintain Current Setup",
        ),
        "Maintain": (
            "Mevcut Durumu Koru",
            "Maintain",
        ),
        "Review": (
            "İncele",
            "Review",
        ),
    }

    pair = mapping.get(raw)

    if pair is None:
        return raw

    return (
        pair[0]
        if language == "tr"
        else pair[1]
    )


def _localize_confidence(
    value: Any,
    language: str,
) -> str:
    raw = str(value or "").strip()

    mapping = {
        "High": ("Yüksek", "High"),
        "Medium": ("Orta", "Medium"),
        "Low": ("Düşük", "Low"),
    }

    pair = mapping.get(raw)

    if pair is None:
        return raw or "-"

    return pair[0] if language == "tr" else pair[1]


def _localize_priority(
    value: Any,
    language: str,
) -> str:
    raw = str(value or "").strip()

    mapping = {
        "High Priority": (
            "Yüksek Öncelik",
            "High Priority",
        ),
        "Medium Priority": (
            "Orta Öncelik",
            "Medium Priority",
        ),
        "Low Priority": (
            "Düşük Öncelik",
            "Low Priority",
        ),
    }

    pair = mapping.get(raw)

    if pair is None:
        return raw or "-"

    return pair[0] if language == "tr" else pair[1]


# ============================================================
# DECISION TABLE
# ============================================================


def build_decision_table(
    page_changes: pd.DataFrame,
    recommendations: pd.DataFrame,
    language: str = "tr",
    limit: int = 30,
) -> pd.DataFrame:
    """
    Turn raw page changes + recommendation output into the shared business
    decision contract:

    Page -> Problem/Opportunity -> Evidence -> Why -> Action -> Expected Impact.
    """
    if page_changes.empty:
        return pd.DataFrame()

    result = page_changes.copy()

    recommendation_view = pd.DataFrame()

    if not recommendations.empty:
        recommendation_page = _first_existing_column(
            recommendations,
            ["page", "Page", "url", "URL"],
        )

        if recommendation_page is not None:
            columns = [
                recommendation_page,
                "RecommendedAction",
                "RecommendationReason",
                "ConfidenceLevel",
                "PriorityTier",
                "ExpectedIncrementalTrafficValue",
                "ExpectedNetValue",
                "EstimatedROI",
                "ClicksUplift",
                "ClicksUpliftPct",
                "PredictedNextClicks",
                "PredictedNextImpressions",
                "OpportunityScore",
            ]

            columns = [
                column
                for column in columns
                if column in recommendations.columns
            ]

            recommendation_view = (
                recommendations[
                    columns
                ]
                .rename(
                    columns={
                        recommendation_page: "page"
                    }
                )
                .drop_duplicates(
                    subset=["page"],
                    keep="first",
                )
            )

    if not recommendation_view.empty:
        result = result.merge(
            recommendation_view,
            on="page",
            how="left",
        )

    result["Signal"] = result.apply(
        _classify_page_signal,
        axis=1,
    )

    labels = result["Signal"].map(
        lambda value: _signal_text(
            value,
            language,
        )[0]
    )

    reasons = result["Signal"].map(
        lambda value: _signal_text(
            value,
            language,
        )[1]
    )

    result["ProblemOpportunity"] = labels
    result["Why"] = reasons
    result["Evidence"] = result.apply(
        lambda row: _format_evidence(
            row,
            language,
        ),
        axis=1,
    )

    if "RecommendationReason" in result.columns:
        recommendation_reason = (
            result["RecommendationReason"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if language == "tr":
            reason_mapping = {
                "Combining metadata, content, internal linking, structured data, entity signals and GEO components offers the broadest growth opportunity.":
                    "Metadata, içerik, iç linkleme, yapılandırılmış veri, varlık sinyalleri ve GEO bileşenlerini birlikte optimize etmek en geniş büyüme fırsatını sunar.",
            }
            localized_reason = recommendation_reason.map(
                lambda value: reason_mapping.get(value, "")
            )
        else:
            localized_reason = recommendation_reason

        result["Why"] = result["Why"].where(
            localized_reason.eq(""),
            result["Why"] + " " + localized_reason,
        )

    if "RecommendedAction" in result.columns:
        result["Action"] = result[
            "RecommendedAction"
        ].map(
            lambda value: _localize_action(
                value,
                language,
            )
        )
    else:
        result["Action"] = "-"

    if "ConfidenceLevel" in result.columns:
        result["Confidence"] = result[
            "ConfidenceLevel"
        ].map(
            lambda value: _localize_confidence(
                value,
                language,
            )
        )
    else:
        result["Confidence"] = "-"

    if "PriorityTier" in result.columns:
        result["Priority"] = result[
            "PriorityTier"
        ].map(
            lambda value: _localize_priority(
                value,
                language,
            )
        )
    else:
        result["Priority"] = "-"

    def expected_impact(row: pd.Series) -> str:
        uplift = row.get("ClicksUplift")
        uplift_pct = row.get("ClicksUpliftPct")
        traffic_value = row.get(
            "ExpectedIncrementalTrafficValue"
        )
        roi = row.get("EstimatedROI")

        parts: list[str] = []

        if pd.notna(uplift):
            parts.append(
                (
                    f"+{float(uplift):.1f} tahmini tıklama"
                    if language == "tr"
                    else f"+{float(uplift):.1f} estimated clicks"
                )
            )

        if pd.notna(uplift_pct):
            parts.append(
                (
                    f"%{float(uplift_pct):.1f} tahmini uplift"
                    if language == "tr"
                    else f"{float(uplift_pct):.1f}% estimated uplift"
                )
            )

        if pd.notna(traffic_value):
            parts.append(
                (
                    f"{float(traffic_value):.2f} ek trafik değeri"
                    if language == "tr"
                    else f"{float(traffic_value):.2f} incremental traffic value"
                )
            )

        if pd.notna(roi):
            parts.append(
                f"ROI {float(roi):.2f}"
            )

        return " | ".join(parts) or "-"

    result["ExpectedImpact"] = result.apply(
        expected_impact,
        axis=1,
    )

    priority_rank = {
        "Yüksek Öncelik": 0,
        "High Priority": 0,
        "Orta Öncelik": 1,
        "Medium Priority": 1,
        "Düşük Öncelik": 2,
        "Low Priority": 2,
    }

    result["_PriorityRank"] = (
        result["Priority"]
        .map(priority_rank)
        .fillna(9)
    )

    if "OpportunityScore" in result.columns:
        result["_OpportunityScore"] = pd.to_numeric(
            result["OpportunityScore"],
            errors="coerce",
        ).fillna(0)
    else:
        result["_OpportunityScore"] = 0.0

    result = result.sort_values(
        [
            "_PriorityRank",
            "_OpportunityScore",
            "ClicksDelta",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    columns = [
        "page",
        "ProblemOpportunity",
        "Evidence",
        "Why",
        "Action",
        "Priority",
        "Confidence",
        "ExpectedImpact",
        "CurrentClicks",
        "ClicksChangePct",
        "CurrentImpressions",
        "ImpressionsChangePct",
        "CurrentCTR",
        "CTRDelta",
        "CurrentPosition",
        "PositionImprovement",
    ]

    optional_columns = [
        "ExpectedNetValue",
        "EstimatedROI",
        "PredictedNextClicks",
        "PredictedNextImpressions",
        "OpportunityScore",
    ]

    columns.extend(
        column
        for column in optional_columns
        if column in result.columns
    )

    return (
        result[
            [
                column
                for column in columns
                if column in result.columns
            ]
        ]
        .head(max(1, int(limit)))
        .reset_index(drop=True)
    )


# ============================================================
# FORECAST CONTRACT
# ============================================================


def resolve_forecast_status(
    recommendations: pd.DataFrame,
    forecast_horizon_days: int,
) -> str:
    """
    Report what the current model can truthfully forecast.

    The existing production models output Next_Clicks and Next_Impressions,
    which are one-step targets. We intentionally do NOT multiply those values
    to fake 7/14/30-day forecasts. Multi-horizon forecasting will be connected
    when the forecasting layer emits horizon-aware predictions.
    """
    has_one_step_forecast = (
        not recommendations.empty
        and "PredictedNextClicks"
        in recommendations.columns
        and "PredictedNextImpressions"
        in recommendations.columns
    )

    if not has_one_step_forecast:
        return "unavailable"

    if int(forecast_horizon_days) == 1:
        return "ready"

    return "one_step_model_available"


# ============================================================
# MAIN ENGINE
# ============================================================


def build_decision_intelligence(
    current_period: pd.DataFrame,
    comparison_period: pd.DataFrame,
    recommendations: pd.DataFrame,
    language: str = "tr",
    forecast_horizon_days: int = 7,
    limit: int = 30,
) -> DecisionEngineResult:
    """Build the single shared decision-intelligence contract."""
    comparison = build_period_comparison(
        current_period=current_period,
        comparison_period=comparison_period,
    )

    page_changes = build_page_change_table(
        current_period=current_period,
        comparison_period=comparison_period,
    )

    decisions = build_decision_table(
        page_changes=page_changes,
        recommendations=recommendations,
        language=language,
        limit=limit,
    )

    return DecisionEngineResult(
        comparison=comparison,
        page_changes=page_changes,
        decisions=decisions,
        forecast_horizon_days=int(
            forecast_horizon_days
        ),
        forecast_status=resolve_forecast_status(
            recommendations=recommendations,
            forecast_horizon_days=(
                forecast_horizon_days
            ),
        ),
    )
