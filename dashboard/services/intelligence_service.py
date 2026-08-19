from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from dashboard.filters import DateRange, filter_dataframe_by_date
from dashboard.utils import load_csv


TECHNICAL_FILES = (
    "seo_technical_seo_intelligence.csv",
    "technical_seo_intelligence.csv",
)
CONTENT_GAP_FILES = (
    "seo_blog_keyword_content_gaps.csv",
    "blog_keyword_content_gaps.csv",
)
CONTENT_COMMERCE_FILES = (
    "seo_blog_content_to_commerce_intelligence.csv",
    "blog_content_to_commerce_intelligence.csv",
)
GEO_FILES = (
    "seo_geo_ai_visibility_intelligence.csv",
    "geo_ai_visibility_intelligence.csv",
)


@dataclass(frozen=True)
class AdvancedIntelligenceData:
    technical: pd.DataFrame
    content_gaps: pd.DataFrame
    content_commerce: pd.DataFrame
    geo: pd.DataFrame


def _load_first(candidates: Iterable[str]) -> pd.DataFrame:
    for filename in candidates:
        frame = load_csv(filename, parse_dates=["date", "Date", "ObservationDate"])
        if frame is not None and not frame.empty:
            return frame
    return pd.DataFrame()


def load_advanced_intelligence() -> AdvancedIntelligenceData:
    """Load persisted advanced-intelligence outputs without triggering the pipeline."""
    return AdvancedIntelligenceData(
        technical=_load_first(TECHNICAL_FILES),
        content_gaps=_load_first(CONTENT_GAP_FILES),
        content_commerce=_load_first(CONTENT_COMMERCE_FILES),
        geo=_load_first(GEO_FILES),
    )


def filter_period(frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    date_columns = [c for c in ("date", "Date", "ObservationDate") if c in frame.columns]
    if not date_columns:
        # Snapshot outputs legitimately have no date column. They remain the latest audited state.
        return frame.copy().reset_index(drop=True)
    return filter_dataframe_by_date(
        dataframe=frame,
        date_range=DateRange(start_date=start_date, end_date=end_date),
        date_column=date_columns[0],
    )


def _numeric(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(index=frame.index, dtype="float64")


def technical_summary(frame: pd.DataFrame) -> dict[str, int | float | str]:
    if frame.empty:
        return {"issues": 0, "critical_high": 0, "affected_urls": 0, "audited": 0, "status": "no_data"}
    severity = frame.get("Severity", pd.Series("", index=frame.index)).astype(str)
    status = frame.get("AuditStatus", pd.Series("Audited", index=frame.index)).astype(str)
    page_col = "Page" if "Page" in frame.columns else "page" if "page" in frame.columns else None
    audited_mask = ~status.str.contains("Not Audited", case=False, na=False)
    return {
        "issues": int(audited_mask.sum()),
        "critical_high": int((audited_mask & severity.isin(["Critical", "High"])).sum()),
        "affected_urls": int(frame.loc[audited_mask, page_col].nunique()) if page_col else 0,
        "audited": int(audited_mask.sum()),
        "status": "ready" if audited_mask.any() else "not_audited",
    }


def content_summary(gaps: pd.DataFrame, commerce: pd.DataFrame) -> dict[str, int | float]:
    high = 0
    if not gaps.empty:
        priority_col = next((c for c in ("ContentGapPriority", "Priority") if c in gaps.columns), None)
        if priority_col:
            high = int(gaps[priority_col].astype(str).str.lower().eq("high").sum())
    score = _numeric(gaps, ("ContentGapScore", "OpportunityScore"))
    commerce_score = _numeric(commerce, ("ContentToCommerceScore", "CommerceScore"))
    return {
        "content_gaps": int(len(gaps)),
        "high_priority": high,
        "avg_gap_score": float(score.mean()) if not score.dropna().empty else 0.0,
        "avg_commerce_score": float(commerce_score.mean()) if not commerce_score.dropna().empty else 0.0,
    }


def geo_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    if frame.empty:
        return {"pages": 0, "high_priority": 0, "avg_readiness": 0.0, "avg_opportunity": 0.0}
    priority = frame.get("GEOPriority", pd.Series("", index=frame.index)).astype(str)
    readiness = _numeric(frame, ("GEOReadinessScore",))
    opportunity = _numeric(frame, ("GEOOpportunityScore",))
    return {
        "pages": int(len(frame)),
        "high_priority": int(priority.str.lower().eq("high").sum()),
        "avg_readiness": float(readiness.mean()) if not readiness.dropna().empty else 0.0,
        "avg_opportunity": float(opportunity.mean()) if not opportunity.dropna().empty else 0.0,
    }
