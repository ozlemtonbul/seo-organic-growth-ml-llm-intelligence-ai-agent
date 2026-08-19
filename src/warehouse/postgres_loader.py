from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_timedelta64_dtype,
)
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.utils.date_utils import resolve_date_range


logger = get_logger(__name__)


def build_postgres_engine() -> Engine:
    """
    Build a SQLAlchemy PostgreSQL engine.
    """
    if not SETTINGS.postgres_enabled:
        raise RuntimeError(
            "PostgreSQL integration is disabled."
        )

    if not SETTINGS.postgres_password:
        raise ValueError(
            "POSTGRES_PASSWORD is required when PostgreSQL is enabled."
        )

    return create_engine(
        SETTINGS.postgres_url,
        pool_pre_ping=True,
        future=True,
    )


def normalize_table_name(
    table_name: str,
) -> str:
    """
    Normalize an output name into a PostgreSQL-safe table name.
    """
    normalized = (
        str(table_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not normalized:
        raise ValueError(
            "PostgreSQL table names cannot be empty."
        )

    return normalized


def build_run_metadata() -> Dict[str, object]:
    """
    Build metadata describing one complete pipeline execution.
    """
    date_from, date_to = resolve_date_range()

    run_id = str(
        uuid4()
    )

    run_timestamp = datetime.now(
        timezone.utc
    )

    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "source_date_from": str(
            date_from
        ),
        "source_date_to": str(
            date_to
        ),
        "project_name": SETTINGS.project_name,
        "app_environment": SETTINGS.app_environment,
    }


def add_run_metadata(
    dataframe: pd.DataFrame,
    run_metadata: Dict[str, object],
) -> pd.DataFrame:
    """
    Add pipeline-run metadata columns to a DataFrame.

    These fields allow historical comparison between different
    executions without overwriting previous data.
    """
    if dataframe is None:
        raise ValueError(
            "The DataFrame cannot be None."
        )

    result = dataframe.copy()

    result[
        "pipeline_run_id"
    ] = str(
        run_metadata[
            "run_id"
        ]
    )

    result[
        "pipeline_run_timestamp"
    ] = run_metadata[
        "run_timestamp"
    ]

    result[
        "pipeline_source_date_from"
    ] = str(
        run_metadata[
            "source_date_from"
        ]
    )

    result[
        "pipeline_source_date_to"
    ] = str(
        run_metadata[
            "source_date_to"
        ]
    )

    return result


def infer_postgres_column_type(
    series: pd.Series,
) -> str:
    """
    Infer a safe PostgreSQL type for a newly introduced DataFrame column.

    This is used only for schema evolution when a historical table
    already exists but the new pipeline output contains extra columns.
    """
    if is_bool_dtype(series.dtype):
        return "BOOLEAN"

    if is_integer_dtype(series.dtype):
        return "BIGINT"

    if is_float_dtype(series.dtype):
        return "DOUBLE PRECISION"

    if is_datetime64_any_dtype(series.dtype):
        timezone_value = getattr(
            series.dtype,
            "tz",
            None,
        )

        if timezone_value is not None:
            return "TIMESTAMPTZ"

        return "TIMESTAMP"

    if is_timedelta64_dtype(series.dtype):
        return "INTERVAL"

    non_null = series.dropna()

    if not non_null.empty:
        sample_value = non_null.iloc[0]

        if isinstance(
            sample_value,
            bool,
        ):
            return "BOOLEAN"

        if isinstance(
            sample_value,
            int,
        ) and not isinstance(
            sample_value,
            bool,
        ):
            return "BIGINT"

        if isinstance(
            sample_value,
            float,
        ):
            return "DOUBLE PRECISION"

        if isinstance(
            sample_value,
            datetime,
        ):
            if sample_value.tzinfo is not None:
                return "TIMESTAMPTZ"

            return "TIMESTAMP"

        if isinstance(
            sample_value,
            date,
        ):
            return "DATE"

    # Safest fallback for dynamically introduced pipeline fields
    # such as error messages, labels, URLs and commentary.
    return "TEXT"


def get_existing_table_columns(
    table_name: str,
    engine: Engine,
) -> set[str]:
    """
    Return the existing PostgreSQL columns for one table.
    """
    safe_table_name = normalize_table_name(
        table_name
    )

    inspector = inspect(
        engine
    )

    if not inspector.has_table(
        safe_table_name
    ):
        return set()

    return {
        str(
            column[
                "name"
            ]
        )
        for column in inspector.get_columns(
            safe_table_name
        )
    }


def ensure_table_schema(
    dataframe: pd.DataFrame,
    table_name: str,
    engine: Engine,
) -> list[str]:
    """
    Add missing DataFrame columns to an existing PostgreSQL table.

    Historical rows are preserved.

    Returns
    -------
    list[str]
        Names of columns added during this migration.
    """
    if dataframe is None:
        raise ValueError(
            "The DataFrame cannot be None."
        )

    safe_table_name = normalize_table_name(
        table_name
    )

    inspector = inspect(
        engine
    )

    # New tables are created naturally by pandas.to_sql.
    if not inspector.has_table(
        safe_table_name
    ):
        return []

    existing_columns = get_existing_table_columns(
        table_name=safe_table_name,
        engine=engine,
    )

    missing_columns = [
        str(column)
        for column in dataframe.columns
        if str(column) not in existing_columns
    ]

    if not missing_columns:
        return []

    preparer = engine.dialect.identifier_preparer

    quoted_table_name = preparer.quote(
        safe_table_name
    )

    added_columns: list[str] = []

    with engine.begin() as connection:
        for column_name in missing_columns:
            postgres_type = infer_postgres_column_type(
                dataframe[
                    column_name
                ]
            )

            quoted_column_name = preparer.quote(
                column_name
            )

            connection.execute(
                text(
                    f"""
                    ALTER TABLE {quoted_table_name}
                    ADD COLUMN IF NOT EXISTS
                    {quoted_column_name} {postgres_type};
                    """
                )
            )

            added_columns.append(
                column_name
            )

            logger.info(
                "PostgreSQL schema evolved | "
                "Table: %s | Column: %s | Type: %s",
                safe_table_name,
                column_name,
                postgres_type,
            )

    return added_columns


def write_dataframe_to_postgres(
    dataframe: pd.DataFrame,
    table_name: str,
    engine: Optional[Engine] = None,
    if_exists: Optional[str] = None,
) -> None:
    """
    Write a DataFrame into a PostgreSQL table.

    When appending to an existing historical table, newly introduced
    DataFrame columns are automatically added before insertion.
    """
    if dataframe is None:
        raise ValueError(
            "The DataFrame cannot be None."
        )

    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    resolved_if_exists = (
        if_exists
        if if_exists is not None
        else SETTINGS.postgres_if_exists
    )

    if resolved_if_exists not in {
        "fail",
        "replace",
        "append",
    }:
        raise ValueError(
            "if_exists must be fail, replace, or append."
        )

    safe_table_name = normalize_table_name(
        table_name
    )

    if resolved_if_exists == "append":
        ensure_table_schema(
            dataframe=dataframe,
            table_name=safe_table_name,
            engine=resolved_engine,
        )

    dataframe.to_sql(
        name=safe_table_name,
        con=resolved_engine,
        if_exists=resolved_if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )

    logger.info(
        "PostgreSQL table written: %s | Rows: %d",
        safe_table_name,
        len(dataframe),
    )


def write_pipeline_run_manifest(
    engine: Engine,
    run_metadata: Dict[str, object],
    written_tables: Dict[str, int],
) -> None:
    """
    Persist one record describing the completed pipeline run.
    """
    total_rows = int(
        sum(
            written_tables.values()
        )
    )

    manifest = pd.DataFrame(
        [
            {
                "run_id": run_metadata[
                    "run_id"
                ],
                "run_timestamp": run_metadata[
                    "run_timestamp"
                ],
                "source_date_from": run_metadata[
                    "source_date_from"
                ],
                "source_date_to": run_metadata[
                    "source_date_to"
                ],
                "project_name": run_metadata[
                    "project_name"
                ],
                "app_environment": run_metadata[
                    "app_environment"
                ],
                "table_count": len(
                    written_tables
                ),
                "total_rows_written": total_rows,
            }
        ]
    )

    write_dataframe_to_postgres(
        dataframe=manifest,
        table_name="seo_pipeline_runs",
        engine=engine,
        if_exists="append",
    )

    logger.info(
        "PostgreSQL pipeline run manifest written: %s",
        run_metadata[
            "run_id"
        ],
    )


def write_outputs_to_postgres(
    outputs: Dict[
        str,
        Optional[pd.DataFrame],
    ],
) -> Dict[str, int]:
    """
    Persist all pipeline outputs historically in PostgreSQL.

    Each pipeline run receives a unique run_id and timestamp.

    Existing historical rows are preserved because outputs
    are appended instead of replaced.

    PostgreSQL schemas evolve automatically when future pipeline
    versions introduce new columns.

    Returns
    -------
    dict
        Mapping of PostgreSQL table names to inserted row counts.
    """
    if not SETTINGS.postgres_enabled:
        logger.info(
            "PostgreSQL integration is disabled. "
            "Skipping warehouse export."
        )

        return {}

    engine = build_postgres_engine()

    run_metadata = build_run_metadata()

    written_tables: Dict[
        str,
        int,
    ] = {}

    try:
        for (
            table_name,
            dataframe,
        ) in outputs.items():

            if dataframe is None:
                logger.warning(
                    "Skipping undefined PostgreSQL output: %s",
                    table_name,
                )
                continue

            if dataframe.empty:
                logger.info(
                    "Skipping empty PostgreSQL output: %s",
                    table_name,
                )
                continue

            safe_table_name = normalize_table_name(
                table_name
            )

            historical_dataframe = add_run_metadata(
                dataframe=dataframe,
                run_metadata=run_metadata,
            )

            write_dataframe_to_postgres(
                dataframe=historical_dataframe,
                table_name=safe_table_name,
                engine=engine,
                if_exists="append",
            )

            written_tables[
                safe_table_name
            ] = int(
                len(
                    historical_dataframe
                )
            )

        write_pipeline_run_manifest(
            engine=engine,
            run_metadata=run_metadata,
            written_tables=written_tables,
        )

    finally:
        engine.dispose()

    logger.info(
        "Historical PostgreSQL persistence completed "
        "| Run ID: %s | Tables: %d.",
        run_metadata[
            "run_id"
        ],
        len(
            written_tables
        ),
    )

    return written_tables