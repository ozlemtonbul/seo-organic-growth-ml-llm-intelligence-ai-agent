from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.warehouse.postgres_loader import build_postgres_engine


logger = get_logger(__name__)


KNOWLEDGE_DOCUMENTS_TABLE = "seo_knowledge_documents"
KNOWLEDGE_CHUNKS_TABLE = "seo_knowledge_chunks"


def _utc_now() -> datetime:
    """
    Return current UTC timestamp.
    """
    return datetime.now(
        timezone.utc
    )


def _normalize_text(
    value: object,
) -> str:
    """
    Normalize optional text values.
    """
    if value is None:
        return ""

    return str(value).strip()


def _json_text(
    value: object,
) -> str:
    """
    Convert metadata into JSON text safely.
    """
    if value is None:
        return "{}"

    if isinstance(
        value,
        str,
    ):
        try:
            parsed = json.loads(
                value
            )

            return json.dumps(
                parsed,
                ensure_ascii=False,
            )

        except json.JSONDecodeError:
            return json.dumps(
                {
                    "value": value,
                },
                ensure_ascii=False,
            )

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _vector_literal(
    embedding: Iterable[float],
) -> str:
    """
    Convert a Python embedding into pgvector literal format.

    Example:
        [0.1, 0.2, 0.3]
    becomes:
        "[0.1,0.2,0.3]"
    """
    values = [
        float(
            value
        )
        for value in embedding
    ]

    expected_dimensions = int(
        SETTINGS.rag_embedding_dimensions
    )

    if (
        len(
            values
        )
        != expected_dimensions
    ):
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Expected {expected_dimensions}, "
            f"received {len(values)}."
        )

    return (
        "["
        + ",".join(
            str(
                value
            )
            for value in values
        )
        + "]"
    )


def ensure_pgvector_extension(
    engine: Optional[Engine] = None,
) -> None:
    """
    Ensure pgvector extension exists in PostgreSQL.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS vector;"
            )
        )

    logger.info(
        "pgvector extension verified."
    )


def ensure_knowledge_tables(
    engine: Optional[Engine] = None,
) -> None:
    """
    Create RAG document and chunk tables if missing.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_pgvector_extension(
        resolved_engine
    )

    dimensions = int(
        SETTINGS.rag_embedding_dimensions
    )

    document_sql = f"""
    CREATE TABLE IF NOT EXISTS {KNOWLEDGE_DOCUMENTS_TABLE} (
        document_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        source_name TEXT NOT NULL,
        source_uri TEXT,
        title TEXT,
        content_hash TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    );
    """

    chunk_sql = f"""
    CREATE TABLE IF NOT EXISTS {KNOWLEDGE_CHUNKS_TABLE} (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL
            REFERENCES {KNOWLEDGE_DOCUMENTS_TABLE}(document_id)
            ON DELETE CASCADE,

        chunk_index INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        source_name TEXT NOT NULL,

        content TEXT NOT NULL,
        character_count INTEGER NOT NULL DEFAULT 0,

        metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,

        embedding VECTOR({dimensions}) NOT NULL,

        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,

        UNIQUE(document_id, chunk_index)
    );
    """

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                document_sql
            )
        )

        connection.execute(
            text(
                chunk_sql
            )
        )

        connection.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS
                idx_{KNOWLEDGE_CHUNKS_TABLE}_document_id
                ON {KNOWLEDGE_CHUNKS_TABLE}(document_id);
                """
            )
        )

        connection.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS
                idx_{KNOWLEDGE_CHUNKS_TABLE}_source_type
                ON {KNOWLEDGE_CHUNKS_TABLE}(source_type);
                """
            )
        )

    logger.info(
        "RAG knowledge tables verified | "
        "Embedding dimensions: %d.",
        dimensions,
    )


def ensure_vector_index(
    engine: Optional[Engine] = None,
) -> None:
    """
    Create an HNSW cosine-distance index for semantic retrieval.

    The index is created separately so table creation works even
    when the table is still empty.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    sql = f"""
    CREATE INDEX IF NOT EXISTS
    idx_{KNOWLEDGE_CHUNKS_TABLE}_embedding_hnsw
    ON {KNOWLEDGE_CHUNKS_TABLE}
    USING hnsw (embedding vector_cosine_ops);
    """

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            )
        )

    logger.info(
        "RAG HNSW vector index verified."
    )


def create_or_update_document(
    source_type: str,
    source_name: str,
    title: str = "",
    source_uri: str = "",
    metadata: object = None,
    content_hash: str = "",
    document_id: str | None = None,
    engine: Optional[Engine] = None,
) -> str:
    """
    Create or update one knowledge document.

    Returns
    -------
    str
        document_id
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    resolved_document_id = (
        _normalize_text(
            document_id
        )
        or str(
            uuid4()
        )
    )

    now = _utc_now()

    sql = f"""
    INSERT INTO {KNOWLEDGE_DOCUMENTS_TABLE} (
        document_id,
        source_type,
        source_name,
        source_uri,
        title,
        content_hash,
        metadata_json,
        created_at,
        updated_at
    )
    VALUES (
        :document_id,
        :source_type,
        :source_name,
        :source_uri,
        :title,
        :content_hash,
        CAST(:metadata_json AS JSONB),
        :created_at,
        :updated_at
    )

    ON CONFLICT (document_id)
    DO UPDATE SET
        source_type = EXCLUDED.source_type,
        source_name = EXCLUDED.source_name,
        source_uri = EXCLUDED.source_uri,
        title = EXCLUDED.title,
        content_hash = EXCLUDED.content_hash,
        metadata_json = EXCLUDED.metadata_json,
        updated_at = EXCLUDED.updated_at;
    """

    params = {
        "document_id": resolved_document_id,
        "source_type": _normalize_text(
            source_type
        ),
        "source_name": _normalize_text(
            source_name
        ),
        "source_uri": _normalize_text(
            source_uri
        ),
        "title": _normalize_text(
            title
        ),
        "content_hash": _normalize_text(
            content_hash
        ),
        "metadata_json": _json_text(
            metadata
        ),
        "created_at": now,
        "updated_at": now,
    }

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            ),
            params,
        )

    logger.info(
        "Knowledge document stored: %s | %s",
        resolved_document_id,
        source_name,
    )

    return resolved_document_id


def delete_document_chunks(
    document_id: str,
    engine: Optional[Engine] = None,
) -> int:
    """
    Delete all chunks for one knowledge document.

    Useful when a document is re-ingested.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    with resolved_engine.begin() as connection:
        result = connection.execute(
            text(
                f"""
                DELETE FROM {KNOWLEDGE_CHUNKS_TABLE}
                WHERE document_id = :document_id;
                """
            ),
            {
                "document_id": _normalize_text(
                    document_id
                ),
            },
        )

    deleted = int(
        result.rowcount
        or 0
    )

    logger.info(
        "Knowledge chunks deleted: %d | Document: %s",
        deleted,
        document_id,
    )

    return deleted


def insert_knowledge_chunk(
    document_id: str,
    chunk_index: int,
    source_type: str,
    source_name: str,
    content: str,
    embedding: Iterable[float],
    metadata: object = None,
    character_count: int | None = None,
    chunk_id: str | None = None,
    engine: Optional[Engine] = None,
) -> str:
    """
    Store one embedded knowledge chunk.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    normalized_content = _normalize_text(
        content
    )

    if not normalized_content:
        raise ValueError(
            "Knowledge chunk content cannot be empty."
        )

    resolved_chunk_id = (
        _normalize_text(
            chunk_id
        )
        or str(
            uuid4()
        )
    )

    vector_value = _vector_literal(
        embedding
    )

    now = _utc_now()

    sql = f"""
    INSERT INTO {KNOWLEDGE_CHUNKS_TABLE} (
        chunk_id,
        document_id,
        chunk_index,
        source_type,
        source_name,
        content,
        character_count,
        metadata_json,
        embedding,
        created_at,
        updated_at
    )
    VALUES (
        :chunk_id,
        :document_id,
        :chunk_index,
        :source_type,
        :source_name,
        :content,
        :character_count,
        CAST(:metadata_json AS JSONB),
        CAST(:embedding AS VECTOR),
        :created_at,
        :updated_at
    )

    ON CONFLICT (document_id, chunk_index)
    DO UPDATE SET
        source_type = EXCLUDED.source_type,
        source_name = EXCLUDED.source_name,
        content = EXCLUDED.content,
        character_count = EXCLUDED.character_count,
        metadata_json = EXCLUDED.metadata_json,
        embedding = EXCLUDED.embedding,
        updated_at = EXCLUDED.updated_at;
    """

    params = {
        "chunk_id": resolved_chunk_id,
        "document_id": _normalize_text(
            document_id
        ),
        "chunk_index": int(
            chunk_index
        ),
        "source_type": _normalize_text(
            source_type
        ),
        "source_name": _normalize_text(
            source_name
        ),
        "content": normalized_content,
        "character_count": (
            int(
                character_count
            )
            if character_count is not None
            else len(
                normalized_content
            )
        ),
        "metadata_json": _json_text(
            metadata
        ),
        "embedding": vector_value,
        "created_at": now,
        "updated_at": now,
    }

    with resolved_engine.begin() as connection:
        connection.execute(
            text(
                sql
            ),
            params,
        )

    return resolved_chunk_id


def insert_knowledge_chunks(
    document_id: str,
    source_type: str,
    source_name: str,
    chunks: Iterable[dict[str, object]],
    engine: Optional[Engine] = None,
) -> int:
    """
    Store multiple knowledge chunks.

    Each chunk dict must contain:
    - chunk_index
    - content
    - embedding

    Optional:
    - character_count
    - metadata
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    written = 0

    for chunk in chunks:
        insert_knowledge_chunk(
            document_id=document_id,
            chunk_index=int(
                chunk[
                    "chunk_index"
                ]
            ),
            source_type=source_type,
            source_name=source_name,
            content=str(
                chunk[
                    "content"
                ]
            ),
            embedding=chunk[
                "embedding"
            ],
            metadata=chunk.get(
                "metadata"
            ),
            character_count=(
                int(
                    chunk[
                        "character_count"
                    ]
                )
                if chunk.get(
                    "character_count"
                )
                is not None
                else None
            ),
            engine=resolved_engine,
        )

        written += 1

    logger.info(
        "Knowledge chunks stored: %d | Document: %s",
        written,
        document_id,
    )

    return written


def get_knowledge_counts(
    engine: Optional[Engine] = None,
) -> dict[str, int]:
    """
    Return current RAG storage counts.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    with resolved_engine.connect() as connection:
        documents = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {KNOWLEDGE_DOCUMENTS_TABLE};
                """
            )
        ).scalar()

        chunks = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {KNOWLEDGE_CHUNKS_TABLE};
                """
            )
        ).scalar()

    return {
        "documents": int(
            documents
            or 0
        ),
        "chunks": int(
            chunks
            or 0
        ),
    }


def semantic_search(
    query_embedding: Iterable[float],
    top_k: int | None = None,
    min_similarity: float | None = None,
    source_type: str | None = None,
    engine: Optional[Engine] = None,
) -> list[dict[str, object]]:
    """
    Search knowledge chunks using pgvector cosine similarity.

    Returns highest-similarity chunks first.
    """
    resolved_engine = (
        engine
        if engine is not None
        else build_postgres_engine()
    )

    ensure_knowledge_tables(
        resolved_engine
    )

    resolved_top_k = int(
        top_k
        if top_k is not None
        else SETTINGS.rag_top_k
    )

    resolved_min_similarity = float(
        min_similarity
        if min_similarity is not None
        else SETTINGS.rag_min_similarity
    )

    vector_value = _vector_literal(
        query_embedding
    )

    source_filter = ""

    params = {
        "embedding": vector_value,
        "top_k": resolved_top_k,
        "min_similarity": (
            resolved_min_similarity
        ),
    }

    if source_type:
        source_filter = (
            "AND c.source_type = :source_type"
        )

        params[
            "source_type"
        ] = _normalize_text(
            source_type
        )

    sql = f"""
    SELECT
        c.chunk_id,
        c.document_id,
        c.chunk_index,
        c.source_type,
        c.source_name,
        c.content,
        c.character_count,
        c.metadata_json,
        d.title,
        d.source_uri,

        (
            1 - (
                c.embedding
                <=> CAST(:embedding AS VECTOR)
            )
        ) AS similarity

    FROM {KNOWLEDGE_CHUNKS_TABLE} c

    JOIN {KNOWLEDGE_DOCUMENTS_TABLE} d
        ON d.document_id = c.document_id

    WHERE
        (
            1 - (
                c.embedding
                <=> CAST(:embedding AS VECTOR)
            )
        ) >= :min_similarity

        {source_filter}

    ORDER BY
        c.embedding
        <=> CAST(:embedding AS VECTOR)

    LIMIT :top_k;
    """

    with resolved_engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    sql
                ),
                params,
            )
            .mappings()
            .all()
        )

    return [
        dict(
            row
        )
        for row in rows
    ]