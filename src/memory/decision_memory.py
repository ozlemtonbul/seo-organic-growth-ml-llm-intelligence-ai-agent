from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import uuid4

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.warehouse.postgres_loader import build_postgres_engine


logger = get_logger(__name__)


DECISION_MEMORY_TABLE = "seo_agent_decision_memory"


VALID_STATUSES = {
    "proposed",
    "approved",
    "applied",
    "rejected",
    "cancelled",
}


def _utc_now() -> datetime:
    """
    Return the current UTC timestamp.
    """
    return datetime.now(timezone.utc)


def _normalize_text(value: object) -> str:
    """
    Normalize optional text values.
    """
    if value is None:
        return ""

    return str(value).strip()


def _normalize_status(value: str) -> str:
    """
    Validate and normalize a decision status.
    """
    status = _normalize_text(
        value
    ).lower()

    if status not in VALID_STATUSES:
        raise ValueError(
            "Decision status must be one of: "
            f"{sorted(VALID_STATUSES)}"
        )

    return status


def build_decision_memory_rows(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert recommendation output into persistent
    decision-memory records.

    One recommendation becomes one decision-memory row.
    """
    if recommendations is None:
        raise ValueError(
            "Recommendations DataFrame cannot be None."
        )

    if recommendations.empty:
        return pd.DataFrame()

    required_columns = [
        "page",
        "RecommendedAction",
        "RecommendationReason",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in recommendations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required recommendation columns: "
            f"{missing_columns}"
        )

    now = _utc_now()

    rows = []

    for _, row in recommendations.iterrows():
        page = _normalize_text(
            row.get(
                "page",
                "",
            )
        )

        action = _normalize_text(
            row.get(
                "RecommendedAction",
                "",
            )
        )

        reason = _normalize_text(
            row.get(
                "RecommendationReason",
                "",
            )
        )

        if not page or not action:
            continue

        rows.append(
            {
                "decision_id": str(
                    uuid4()
                ),
                "page": page,
                "recommended_action": action,
                "recommendation_reason": reason,
                "status": "proposed",
                "priority_tier": _normalize_text(
                    row.get(
                        "PriorityTier",
                        "",
                    )
                ),
                "confidence_level": _normalize_text(
                    row.get(
                        "ConfidenceLevel",
                        "",
                    )
                ),
                "scenario": _normalize_text(
                    row.get(
                        "Scenario",
                        "",
                    )
                ),
                "opportunity_score": row.get(
                    "OpportunityScore",
                    None,
                ),
                "adjusted_net_value": row.get(
                    "AdjustedNetValue",
                    None,
                ),
                "estimated_roi": row.get(
                    "EstimatedROI",
                    None,
                ),
                "created_at": now,
                "updated_at": now,
                "actioned_at": None,
                "outcome_recorded_at": None,
                "before_clicks": None,
                "after_clicks": None,
                "before_impressions": None,
                "after_impressions": None,
                "before_ctr": None,
                "after_ctr": None,
                "before_position": None,
                "after_position": None,
                "before_revenue": None,
                "after_revenue": None,
                "outcome_label": "",
                "outcome_notes": "",
            }
        )

    return pd.DataFrame(
        rows
    )


def ensure_decision_memory_table(
    engine: Optional[Engine] = None,
) -> None:
    """
    Create the decision-memory table if it does not exist.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    sql = f"""
    CREATE TABLE IF NOT EXISTS {DECISION_MEMORY_TABLE} (
        decision_id TEXT PRIMARY KEY,
        page TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        recommendation_reason TEXT,
        status TEXT NOT NULL,
        priority_tier TEXT,
        confidence_level TEXT,
        scenario TEXT,
        opportunity_score DOUBLE PRECISION,
        adjusted_net_value DOUBLE PRECISION,
        estimated_roi DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        actioned_at TIMESTAMPTZ,
        outcome_recorded_at TIMESTAMPTZ,
        before_clicks DOUBLE PRECISION,
        after_clicks DOUBLE PRECISION,
        before_impressions DOUBLE PRECISION,
        after_impressions DOUBLE PRECISION,
        before_ctr DOUBLE PRECISION,
        after_ctr DOUBLE PRECISION,
        before_position DOUBLE PRECISION,
        after_position DOUBLE PRECISION,
        before_revenue DOUBLE PRECISION,
        after_revenue DOUBLE PRECISION,
        outcome_label TEXT,
        outcome_notes TEXT
    )
    """

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            )
        )


def decision_exists(
    page: str,
    recommended_action: str,
    status_filter: Optional[
        Iterable[str]
    ] = None,
    engine: Optional[Engine] = None,
) -> bool:
    """
    Check whether the same page + action already exists
    in decision memory.

    By default, all statuses are considered.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_decision_memory_table(
        resolved_engine
    )

    params = {
        "page": _normalize_text(
            page
        ),
        "recommended_action": _normalize_text(
            recommended_action
        ),
    }

    where_status = ""

    if status_filter:
        normalized_statuses = [
            _normalize_status(
                status
            )
            for status in status_filter
        ]

        placeholders = []

        for index, status in enumerate(
            normalized_statuses
        ):
            key = f"status_{index}"
            params[
                key
            ] = status
            placeholders.append(
                f":{key}"
            )

        where_status = (
            " AND status IN ("
            + ", ".join(
                placeholders
            )
            + ")"
        )

    sql = f"""
    SELECT 1
    FROM {DECISION_MEMORY_TABLE}
    WHERE page = :page
      AND recommended_action = :recommended_action
      {where_status}
    LIMIT 1
    """

    with resolved_engine.connect() as connection:
        result = connection.execute(
            text(
                sql
            ),
            params,
        ).first()

    return result is not None


def save_recommendations_to_memory(
    recommendations: pd.DataFrame,
    engine: Optional[Engine] = None,
    skip_existing_open_decisions: bool = True,
) -> int:
    """
    Persist recommendations into decision memory.

    Existing open decisions can be skipped to prevent
    the agent from repeatedly storing the same proposal.
    """
    if not SETTINGS.postgres_enabled:
        logger.info(
            "Decision memory is disabled because "
            "PostgreSQL integration is disabled."
        )
        return 0

    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_decision_memory_table(
        resolved_engine
    )

    memory_rows = build_decision_memory_rows(
        recommendations
    )

    if memory_rows.empty:
        return 0

    if skip_existing_open_decisions:
        keep_mask = []

        for _, row in memory_rows.iterrows():
            exists = decision_exists(
                page=str(
                    row["page"]
                ),
                recommended_action=str(
                    row[
                        "recommended_action"
                    ]
                ),
                status_filter=[
                    "proposed",
                    "approved",
                    "applied",
                ],
                engine=resolved_engine,
            )

            keep_mask.append(
                not exists
            )

        memory_rows = memory_rows.loc[
            keep_mask
        ].reset_index(
            drop=True
        )

    if memory_rows.empty:
        logger.info(
            "No new decision-memory rows were written."
        )
        return 0

    memory_rows.to_sql(
        name=DECISION_MEMORY_TABLE,
        con=resolved_engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    logger.info(
        "Decision memory written: %d rows.",
        len(
            memory_rows
        ),
    )

    return int(
        len(
            memory_rows
        )
    )


def get_decision_history(
    page: Optional[str] = None,
    limit: int = 100,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """
    Read recent decision-memory records.

    Optionally filter by page.
    """
    if not SETTINGS.postgres_enabled:
        return pd.DataFrame()

    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_decision_memory_table(
        resolved_engine
    )

    params = {
        "limit": int(
            limit
        ),
    }

    where_clause = ""

    if page:
        where_clause = (
            "WHERE page = :page"
        )

        params[
            "page"
        ] = _normalize_text(
            page
        )

    sql = f"""
    SELECT *
    FROM {DECISION_MEMORY_TABLE}
    {where_clause}
    ORDER BY created_at DESC
    LIMIT :limit
    """

    return pd.read_sql(
        text(
            sql
        ),
        con=resolved_engine,
        params=params,
    )


def update_decision_status(
    decision_id: str,
    status: str,
    engine: Optional[Engine] = None,
) -> None:
    """
    Update decision lifecycle status.
    """
    normalized_status = _normalize_status(
        status
    )

    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_decision_memory_table(
        resolved_engine
    )

    now = _utc_now()

    actioned_at = (
        now
        if normalized_status
        in {
            "approved",
            "applied",
            "rejected",
            "cancelled",
        }
        else None
    )

    sql = f"""
    UPDATE {DECISION_MEMORY_TABLE}
    SET
        status = :status,
        updated_at = :updated_at,
        actioned_at = COALESCE(
            :actioned_at,
            actioned_at
        )
    WHERE decision_id = :decision_id
    """

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            ),
            {
                "status": normalized_status,
                "updated_at": now,
                "actioned_at": actioned_at,
                "decision_id": _normalize_text(
                    decision_id
                ),
            },
        )


def record_decision_outcome(
    decision_id: str,
    outcome_label: str,
    outcome_notes: str = "",
    before_clicks: float | None = None,
    after_clicks: float | None = None,
    before_impressions: float | None = None,
    after_impressions: float | None = None,
    before_ctr: float | None = None,
    after_ctr: float | None = None,
    before_position: float | None = None,
    after_position: float | None = None,
    before_revenue: float | None = None,
    after_revenue: float | None = None,
    engine: Optional[Engine] = None,
) -> None:
    """
    Record before/after business outcomes for a decision.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_decision_memory_table(
        resolved_engine
    )

    now = _utc_now()

    sql = f"""
    UPDATE {DECISION_MEMORY_TABLE}
    SET
        outcome_label = :outcome_label,
        outcome_notes = :outcome_notes,
        before_clicks = :before_clicks,
        after_clicks = :after_clicks,
        before_impressions = :before_impressions,
        after_impressions = :after_impressions,
        before_ctr = :before_ctr,
        after_ctr = :after_ctr,
        before_position = :before_position,
        after_position = :after_position,
        before_revenue = :before_revenue,
        after_revenue = :after_revenue,
        outcome_recorded_at = :outcome_recorded_at,
        updated_at = :updated_at
    WHERE decision_id = :decision_id
    """

    params = {
        "decision_id": _normalize_text(
            decision_id
        ),
        "outcome_label": _normalize_text(
            outcome_label
        ),
        "outcome_notes": _normalize_text(
            outcome_notes
        ),
        "before_clicks": before_clicks,
        "after_clicks": after_clicks,
        "before_impressions": before_impressions,
        "after_impressions": after_impressions,
        "before_ctr": before_ctr,
        "after_ctr": after_ctr,
        "before_position": before_position,
        "after_position": after_position,
        "before_revenue": before_revenue,
        "after_revenue": after_revenue,
        "outcome_recorded_at": now,
        "updated_at": now,
    }

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            ),
            params,
        )


def summarize_decision_memory(
    page: Optional[str] = None,
    limit: int = 20,
    engine: Optional[Engine] = None,
) -> str:
    """
    Build a compact text summary that can later be used
    by the agent / RAG context layer.
    """
    history = get_decision_history(
        page=page,
        limit=limit,
        engine=engine,
    )

    if history.empty:
        return ""

    lines = []

    for _, row in history.iterrows():
        lines.append(
            " | ".join(
                [
                    f"Page: {row.get('page', '')}",
                    (
                        "Action: "
                        f"{row.get('recommended_action', '')}"
                    ),
                    (
                        "Status: "
                        f"{row.get('status', '')}"
                    ),
                    (
                        "Outcome: "
                        f"{row.get('outcome_label', '')}"
                    ),
                    (
                        "Created: "
                        f"{row.get('created_at', '')}"
                    ),
                ]
            )
        )

    return "\n".join(
        lines
    )