from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from sqlalchemy.engine import Engine

from config.logging_config import get_logger
from src.rag.chunking import build_chunk_records, normalize_text
from src.rag.embeddings import embed_texts
from src.rag.knowledge_store import (
    create_or_update_document,
    delete_document_chunks,
    ensure_vector_index,
    insert_knowledge_chunks,
)


logger = get_logger(__name__)


def _content_hash(text: str) -> str:
    """
    Generate a stable SHA-256 hash for source content.
    """
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def _stable_document_id(
    source_type: str,
    source_name: str,
    source_uri: str = "",
) -> str:
    """
    Generate a deterministic document ID.

    Re-ingesting the same logical source updates the same document
    instead of creating duplicate documents.
    """
    identity = "|".join(
        [
            str(source_type).strip().lower(),
            str(source_name).strip().lower(),
            str(source_uri).strip().lower(),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def ingest_text(
    text: str,
    source_type: str,
    source_name: str,
    title: str = "",
    source_uri: str = "",
    metadata: object = None,
    engine: Engine | None = None,
) -> dict[str, object]:
    """
    Ingest one text source into the RAG knowledge store.

    Flow:
        text
        -> normalize
        -> chunk
        -> embeddings
        -> PostgreSQL / pgvector
    """
    normalized_text = normalize_text(text)

    if not normalized_text:
        return {
            "source_type": source_type,
            "source_name": source_name,
            "document_id": "",
            "chunks": 0,
            "status": "skipped_empty",
        }

    ensure_vector_index(engine)

    document_id = _stable_document_id(
        source_type=source_type,
        source_name=source_name,
        source_uri=source_uri,
    )

    content_hash = _content_hash(
        normalized_text
    )

    document_id = create_or_update_document(
        source_type=source_type,
        source_name=source_name,
        title=title,
        source_uri=source_uri,
        metadata=metadata,
        content_hash=content_hash,
        document_id=document_id,
        engine=engine,
    )

    chunk_records = build_chunk_records(
        normalized_text
    )

    if not chunk_records:
        return {
            "source_type": source_type,
            "source_name": source_name,
            "document_id": document_id,
            "chunks": 0,
            "status": "skipped_empty",
        }

    embeddings = embed_texts(
        [
            str(record["content"])
            for record in chunk_records
        ]
    )

    if len(embeddings) != len(chunk_records):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    prepared_chunks: list[dict[str, object]] = []

    for record, embedding in zip(
        chunk_records,
        embeddings,
    ):
        prepared_chunks.append(
            {
                "chunk_index": int(
                    record["chunk_index"]
                ),
                "content": str(
                    record["content"]
                ),
                "character_count": int(
                    record["character_count"]
                ),
                "embedding": embedding,
                "metadata": metadata or {},
            }
        )

    # Re-ingestion replaces old chunks belonging to this source.
    delete_document_chunks(
        document_id=document_id,
        engine=engine,
    )

    written = insert_knowledge_chunks(
        document_id=document_id,
        source_type=source_type,
        source_name=source_name,
        chunks=prepared_chunks,
        engine=engine,
    )

    logger.info(
        "RAG source ingested | "
        "Source: %s | Type: %s | Chunks: %d",
        source_name,
        source_type,
        written,
    )

    return {
        "source_type": source_type,
        "source_name": source_name,
        "document_id": document_id,
        "chunks": written,
        "status": "ingested",
    }


def _row_to_text(
    row: Mapping[str, object],
    columns: Iterable[str],
) -> str:
    """
    Convert selected DataFrame fields into semantic text.
    """
    sections: list[str] = []

    for column in columns:
        value = row.get(column)

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        text_value = str(value).strip()

        if not text_value:
            continue

        sections.append(
            f"{column}: {text_value}"
        )

    return "\n".join(sections)


def ingest_dataframe(
    dataframe: pd.DataFrame,
    source_type: str,
    source_name: str,
    text_columns: Iterable[str],
    title: str = "",
    source_uri: str = "",
    max_rows: int | None = None,
    engine: Engine | None = None,
) -> dict[str, object]:
    """
    Convert a structured SEO DataFrame into RAG knowledge.

    Each row becomes a semantic text block before the complete source
    is chunked and embedded.
    """
    if dataframe is None or dataframe.empty:
        return {
            "source_type": source_type,
            "source_name": source_name,
            "document_id": "",
            "rows": 0,
            "chunks": 0,
            "status": "skipped_empty",
        }

    available_columns = [
        column
        for column in text_columns
        if column in dataframe.columns
    ]

    if not available_columns:
        raise ValueError(
            f"No requested RAG columns exist in {source_name}. "
            f"Requested: {list(text_columns)}"
        )

    data = dataframe.copy()

    if max_rows is not None:
        data = data.head(
            int(max_rows)
        )

    text_blocks: list[str] = []

    for _, row in data.iterrows():
        block = _row_to_text(
            row.to_dict(),
            available_columns,
        )

        if block:
            text_blocks.append(block)

    combined_text = "\n\n---\n\n".join(
        text_blocks
    )

    result = ingest_text(
        text=combined_text,
        source_type=source_type,
        source_name=source_name,
        title=title,
        source_uri=source_uri,
        metadata={
            "rows": len(data),
            "columns": available_columns,
        },
        engine=engine,
    )

    result["rows"] = len(data)

    return result


def ingest_text_file(
    file_path: str | Path,
    source_type: str = "knowledge_file",
    source_name: str | None = None,
    title: str = "",
    engine: Engine | None = None,
) -> dict[str, object]:
    """
    Ingest a UTF-8 TXT or Markdown knowledge document.
    """
    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {path}"
        )

    if path.suffix.lower() not in {
        ".txt",
        ".md",
    }:
        raise ValueError(
            "RAG text-file ingestion currently supports "
            ".txt and .md files."
        )

    content = path.read_text(
        encoding="utf-8"
    )

    return ingest_text(
        text=content,
        source_type=source_type,
        source_name=(
            source_name
            or path.name
        ),
        title=(
            title
            or path.stem
        ),
        source_uri=str(
            path.resolve()
        ),
        metadata={
            "file_name": path.name,
            "suffix": path.suffix.lower(),
        },
        engine=engine,
    )


def ingest_pipeline_outputs(
    outputs: Mapping[
        str,
        pd.DataFrame | None,
    ],
    engine: Engine | None = None,
) -> pd.DataFrame:
    """
    Ingest selected high-value SEO pipeline outputs into RAG.

    Large raw datasets are deliberately excluded. The RAG knowledge base
    should contain decision-useful evidence instead of hundreds of
    thousands of raw API rows.

    Knowledge domains
    -----------------
    - SEO recommendations and business priorities
    - Page opportunity intelligence
    - Technical SEO
    - GEO / AI visibility
    - Content-commerce opportunities
    - Content gaps
    - Selected production-model metrics
    - Full Random Forest / XGBoost / LightGBM benchmark
    - Global SHAP explainability
    - Page-level SHAP forecast explanations
    """
    source_definitions = {
        "seo_recommendations": {
            "source_type": "recommendation",
            "columns": [
                "page",
                "Scenario",
                "RecommendedAction",
                "RecommendationReason",
                "PriorityTier",
                "ConfidenceLevel",
                "EstimatedROI",
                "AdjustedNetValue",
            ],
            "max_rows": None,
        },
        "seo_page_opportunity_intelligence": {
            "source_type": "page_opportunity",
            "columns": [
                "page",
                "PageType",
                "PageOpportunityScore",
                "CommerceScore",
                "Clicks",
                "Impressions",
                "CTR",
                "Position",
                "Revenue",
            ],
            "max_rows": None,
        },
        "seo_technical_seo_intelligence": {
            "source_type": "technical_seo",
            "columns": [
                "page",
                "url",
                "status_code",
                "title",
                "meta_description",
                "canonical",
                "robots",
                "h1",
                "performance_score",
                "seo_score",
                "lcp",
                "cls",
                "inp",
            ],
            "max_rows": None,
        },
        "seo_geo_ai_visibility_intelligence": {
            "source_type": "geo_visibility",
            "columns": [
                "page",
                "PageType",
                "GeoReadinessScore",
                "AnswerReadinessScore",
                "EntityScore",
                "EEATScore",
            ],
            "max_rows": None,
        },
        "seo_blog_content_to_commerce": {
            "source_type": "content_commerce",
            "columns": [
                "page",
                "query",
                "Clicks",
                "Impressions",
                "Revenue",
                "RecommendedAction",
            ],
            "max_rows": None,
        },
        "seo_blog_keyword_content_gaps": {
            "source_type": "content_gap",
            "columns": [
                "query",
                "page",
                "Clicks",
                "Impressions",
                "CTR",
                "Position",
                "Intent",
            ],
            "max_rows": None,
        },
        "seo_model_metrics": {
            "source_type": "model_performance",
            "columns": [
                "Model",
                "Algorithm",
                "MAE",
                "RMSE",
                "R2",
                "TrainRows",
                "TestRows",
                "ValidationMethod",
                "Selected",
            ],
            "max_rows": None,
        },
        "seo_model_benchmark": {
            "source_type": "model_benchmark",
            "columns": [
                "Model",
                "Algorithm",
                "MAE",
                "RMSE",
                "R2",
                "TrainRows",
                "TestRows",
                "ValidationMethod",
                "Selected",
                "Status",
                "FirstTestDate",
            ],
            "max_rows": None,
        },
        "seo_shap_summary": {
            "source_type": "model_explainability",
            "columns": [
                "Model",
                "Algorithm",
                "Feature",
                "MeanAbsSHAP",
                "MeanSHAP",
                "PositiveImpactRows",
                "NegativeImpactRows",
                "ZeroImpactRows",
                "ImportanceRank",
            ],
            "max_rows": None,
        },
        "seo_shap_detail": {
            "source_type": "forecast_explanation",
            "columns": [
                "Model",
                "Algorithm",
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
            ],
            "max_rows": 4000,
        },
    }

    results: list[
        dict[str, object]
    ] = []

    for (
        output_name,
        definition,
    ) in source_definitions.items():
        dataframe = outputs.get(
            output_name
        )

        if (
            dataframe is None
            or dataframe.empty
        ):
            continue

        try:
            result = ingest_dataframe(
                dataframe=dataframe,
                source_type=str(
                    definition[
                        "source_type"
                    ]
                ),
                source_name=output_name,
                text_columns=definition[
                    "columns"
                ],
                title=output_name.replace(
                    "_",
                    " ",
                ).title(),
                max_rows=definition.get(
                    "max_rows"
                ),
                engine=engine,
            )

            results.append(
                result
            )

        except Exception as exc:
            logger.exception(
                "RAG ingestion failed for %s: %s",
                output_name,
                exc,
            )

            results.append(
                {
                    "source_type": (
                        definition[
                            "source_type"
                        ]
                    ),
                    "source_name": output_name,
                    "document_id": "",
                    "rows": len(
                        dataframe
                    ),
                    "chunks": 0,
                    "status": "failed",
                    "error": str(
                        exc
                    ),
                }
            )

    return pd.DataFrame(
        results
    )

