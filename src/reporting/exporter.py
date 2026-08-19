from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.logging_config import get_logger
from config.settings import SETTINGS


logger = get_logger(__name__)


PIPELINE_NAME = (
    "AI-Powered SEO, Technical SEO, Content, GEO "
    "and Organic Revenue Intelligence Platform"
)


# ============================================================
# HISTORICAL OUTPUT CONFIGURATION
# ============================================================

HISTORICAL_OUTPUT_NAMES = {
    "seo_shap_detail",
    "seo_shap_summary",
}


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_output_name(
    name: str,
) -> str:
    """
    Validate and normalize an output table name.
    """
    normalized = str(
        name
    ).strip()

    if not normalized:
        raise ValueError(
            "Output table names cannot be empty."
        )

    invalid_characters = {
        "/",
        "\\",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
    }

    if any(
        character in normalized
        for character in invalid_characters
    ):
        raise ValueError(
            "Output table names cannot contain file-system "
            f"reserved characters. Received: {normalized!r}"
        )

    return normalized


def ensure_output_directory(
    output_dir: str,
) -> Path:
    """
    Create and return the output directory.
    """
    if not output_dir:
        raise ValueError(
            "An output directory is required."
        )

    target = Path(
        output_dir
    )

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    return target


# ============================================================
# RUN METADATA
# ============================================================

def _build_run_metadata() -> tuple[str, str]:
    """
    Build one stable identifier and timestamp for the
    current export operation.

    SEO_RUN_ID may optionally be supplied by the pipeline
    environment. Otherwise a unique run id is generated.
    """
    now = datetime.now().astimezone()

    model_run_timestamp = (
        now.isoformat()
    )

    configured_run_id = str(
        os.getenv(
            "SEO_RUN_ID",
            "",
        )
    ).strip()

    if configured_run_id:
        run_id = configured_run_id

    else:
        timestamp_part = now.strftime(
            "%Y%m%dT%H%M%S"
        )

        unique_part = uuid.uuid4().hex[
            :8
        ]

        run_id = (
            f"seo_{timestamp_part}_{unique_part}"
        )

    return (
        run_id,
        model_run_timestamp,
    )


# ============================================================
# HISTORICAL SNAPSHOT HELPERS
# ============================================================

def _prepare_historical_dataframe(
    dataframe: pd.DataFrame,
    run_id: str,
    model_run_timestamp: str,
) -> pd.DataFrame:
    """
    Add pipeline-run metadata to a historical snapshot.

    The current/live CSV is intentionally not modified.
    Metadata is added only to the historical copy.
    """
    historical = dataframe.copy()

    historical.insert(
        0,
        "ModelRunTimestamp",
        model_run_timestamp,
    )

    historical.insert(
        0,
        "RunID",
        run_id,
    )

    return historical


def _append_historical_csv(
    dataframe: pd.DataFrame,
    output_name: str,
    target: Path,
    run_id: str,
    model_run_timestamp: str,
) -> Path:
    """
    Append a pipeline output to its historical CSV.

    Historical artifacts are stored under:

        outputs/history/

    Existing current-output behavior remains unchanged.
    """
    history_directory = (
        target
        / "history"
    )

    history_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        history_directory
        / f"{output_name}_history.csv"
    )

    historical = (
        _prepare_historical_dataframe(
            dataframe=dataframe,
            run_id=run_id,
            model_run_timestamp=model_run_timestamp,
        )
    )

    if history_path.exists():
        try:
            existing = pd.read_csv(
                history_path,
                encoding="utf-8-sig",
                low_memory=False,
            )

        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

        if not existing.empty:
            combined = pd.concat(
                [
                    existing,
                    historical,
                ],
                ignore_index=True,
                sort=False,
            )

        else:
            combined = historical.copy()

    else:
        combined = historical.copy()

    # --------------------------------------------------------
    # ATOMIC WRITE
    # --------------------------------------------------------

    temporary_path = (
        history_directory
        / f".{output_name}_history.tmp.csv"
    )

    combined.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(
        history_path
    )

    logger.info(
        (
            "Historical CSV updated: %s "
            "| Current run rows: %d "
            "| Historical rows: %d "
            "| RunID: %s"
        ),
        history_path,
        len(historical),
        len(combined),
        run_id,
    )

    return history_path


# ============================================================
# DATAFRAME EXPORT
# ============================================================

def export_outputs(
    outputs: Dict[
        str,
        Optional[pd.DataFrame],
    ],
    output_dir: str,
) -> Dict[str, Path]:
    """
    Export all DataFrame outputs as UTF-8 CSV files.

    Current/live outputs are written using their existing
    filenames.

    Selected model explainability outputs are additionally
    persisted as historical snapshots.

    Parameters
    ----------
    outputs:
        Mapping of output table names to DataFrames.

    output_dir:
        Target output directory.

    Returns
    -------
    dict
        Mapping of exported current table names to file paths.
    """
    target = ensure_output_directory(
        output_dir
    )

    written_files: Dict[
        str,
        Path,
    ] = {}

    run_id, model_run_timestamp = (
        _build_run_metadata()
    )

    logger.info(
        (
            "Export operation started "
            "| RunID: %s "
            "| Timestamp: %s"
        ),
        run_id,
        model_run_timestamp,
    )

    for name, dataframe in outputs.items():

        if dataframe is None:
            logger.warning(
                "Skipping undefined CSV output: %s",
                name,
            )
            continue

        safe_name = validate_output_name(
            name
        )

        # ----------------------------------------------------
        # CURRENT / LIVE OUTPUT
        # ----------------------------------------------------

        path = (
            target
            / f"{safe_name}.csv"
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        written_files[
            safe_name
        ] = path

        logger.info(
            "CSV written: %s | Rows: %d",
            path,
            len(dataframe),
        )

        # ----------------------------------------------------
        # HISTORICAL MODEL SNAPSHOTS
        # ----------------------------------------------------

        if (
            safe_name
            in HISTORICAL_OUTPUT_NAMES
        ):
            _append_historical_csv(
                dataframe=dataframe,
                output_name=safe_name,
                target=target,
                run_id=run_id,
                model_run_timestamp=model_run_timestamp,
            )

    return written_files


# ============================================================
# RUN MANIFEST
# ============================================================

def build_run_manifest(
    output_dir: str,
    outputs: Dict[
        str,
        Optional[pd.DataFrame],
    ],
    input_file: str,
) -> Dict[str, object]:
    """
    Build pipeline execution metadata.
    """
    target = ensure_output_directory(
        output_dir
    )

    table_counts = {
        name: int(
            len(dataframe)
        )
        for name, dataframe
        in outputs.items()
        if dataframe is not None
    }

    return {
        "pipeline": PIPELINE_NAME,

        "run_timestamp": (
            datetime
            .now()
            .astimezone()
            .isoformat()
        ),

        "input_file": (
            str(
                Path(
                    input_file
                ).resolve()
            )
            if input_file
            else ""
        ),

        "output_dir": str(
            target.resolve()
        ),

        "tables": table_counts,

        "llm_enabled": bool(
            SETTINGS.llm_enabled
            and SETTINGS.anthropic_api_key
        ),

        "postgres_enabled": (
            SETTINGS.postgres_enabled
        ),

        "data_source_mode": (
            SETTINGS.data_source_mode
        ),

        "date_mode": (
            SETTINGS.date_mode
        ),

        "gsc_enabled": bool(
            SETTINGS.gsc_site_url
        ),

        "ga4_enabled": bool(
            SETTINGS.ga4_property_id
        ),

        "crawl_enabled": bool(
            SETTINGS.crawl_input_file
        ),

        "product_feed_enabled": bool(
            SETTINGS.product_feed_file
        ),
    }


def export_run_manifest(
    output_dir: str,
    outputs: Dict[
        str,
        Optional[pd.DataFrame],
    ],
    input_file: str,
) -> Path:
    """
    Export the run manifest as formatted JSON.
    """
    target = ensure_output_directory(
        output_dir
    )

    manifest = build_run_manifest(
        output_dir=output_dir,
        outputs=outputs,
        input_file=input_file,
    )

    manifest_path = (
        target
        / "run_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Run manifest written: %s",
        manifest_path,
    )

    return manifest_path


# ============================================================
# TEXT REPORT EXPORT
# ============================================================

def export_text_report(
    content: str,
    output_dir: str,
    filename: str,
) -> Path:
    """
    Export a plain-text reporting artifact.
    """
    target = ensure_output_directory(
        output_dir
    )

    safe_filename = (
        validate_output_name(
            filename
        )
    )

    if not safe_filename.lower().endswith(
        ".txt"
    ):
        safe_filename = (
            f"{safe_filename}.txt"
        )

    path = (
        target
        / safe_filename
    )

    path.write_text(
        str(
            content
        ),
        encoding="utf-8",
    )

    logger.info(
        "Text report written: %s",
        path,
    )

    return path