from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger
from config.settings import SETTINGS


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the configured local embedding model.

    The model is loaded only once per Python process.
    """
    provider = (
        SETTINGS.rag_embedding_provider
        .strip()
        .lower()
    )

    if provider != "local":
        raise ValueError(
            "Only the local RAG embedding provider is "
            f"currently supported. Received: {provider!r}"
        )

    logger.info(
        "Loading local embedding model: %s",
        SETTINGS.rag_embedding_model,
    )

    model = SentenceTransformer(
        SETTINGS.rag_embedding_model
    )

    dimension = int(
        model.get_sentence_embedding_dimension()
    )

    if dimension != SETTINGS.rag_embedding_dimensions:
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Model produces {dimension} dimensions but "
            "RAG_EMBEDDING_DIMENSIONS is configured as "
            f"{SETTINGS.rag_embedding_dimensions}."
        )

    logger.info(
        "Local embedding model loaded successfully | "
        "Model: %s | Dimensions: %d",
        SETTINGS.rag_embedding_model,
        dimension,
    )

    return model


def embed_texts(
    texts: Iterable[str],
) -> List[List[float]]:
    """
    Generate normalized embeddings for multiple texts.
    """
    cleaned_texts = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not cleaned_texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        batch_size=SETTINGS.rag_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Embedding model returned an unexpected shape."
        )

    if (
        embeddings.shape[1]
        != SETTINGS.rag_embedding_dimensions
    ):
        raise RuntimeError(
            "Generated embedding dimension does not match "
            "RAG_EMBEDDING_DIMENSIONS."
        )

    return embeddings.tolist()


def embed_text(
    text: str,
) -> List[float]:
    """
    Generate one normalized embedding vector.
    """
    value = str(text or "").strip()

    if not value:
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    embeddings = embed_texts([value])

    if not embeddings:
        raise RuntimeError(
            "Embedding generation returned no result."
        )

    return embeddings[0]


def embedding_runtime_info() -> dict[str, object]:
    """
    Return safe runtime information about the embedding layer.
    """
    return {
        "provider": SETTINGS.rag_embedding_provider,
        "model": SETTINGS.rag_embedding_model,
        "dimensions": SETTINGS.rag_embedding_dimensions,
        "batch_size": SETTINGS.rag_batch_size,
    }