from __future__ import annotations

import re
from typing import List

from config.settings import SETTINGS


def normalize_text(text: str) -> str:
    """
    Normalize text before chunking.

    Removes unnecessary whitespace while preserving
    paragraph structure and semantic content.
    """
    value = str(text or "").strip()

    if not value:
        return ""

    # Normalize Windows / old Mac line endings.
    value = re.sub(r"\r\n?", "\n", value)

    # Collapse repeated spaces and tabs.
    value = re.sub(r"[ \t]+", " ", value)

    # Avoid excessive blank lines.
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """
    Split text into overlapping semantic chunks.

    Chunk size is character based. When possible, chunks end
    at a natural sentence or paragraph boundary.
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    resolved_chunk_size = int(
        chunk_size
        if chunk_size is not None
        else SETTINGS.rag_chunk_size
    )

    resolved_overlap = int(
        chunk_overlap
        if chunk_overlap is not None
        else SETTINGS.rag_chunk_overlap
    )

    if resolved_chunk_size < 1:
        raise ValueError(
            "chunk_size must be at least 1."
        )

    if resolved_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if resolved_overlap >= resolved_chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if len(normalized) <= resolved_chunk_size:
        return [normalized]

    chunks: List[str] = []

    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(
            start + resolved_chunk_size,
            text_length,
        )

        candidate = normalized[
            start:end
        ]

        # Prefer a natural boundary rather than cutting
        # directly in the middle of a sentence.
        if end < text_length:
            boundary_candidates = [
                candidate.rfind("\n\n"),
                candidate.rfind(". "),
                candidate.rfind("? "),
                candidate.rfind("! "),
                candidate.rfind("; "),
            ]

            best_boundary = max(
                boundary_candidates
            )

            minimum_boundary = int(
                resolved_chunk_size * 0.60
            )

            if best_boundary >= minimum_boundary:
                end = (
                    start
                    + best_boundary
                    + 1
                )

                candidate = normalized[
                    start:end
                ]

        candidate = candidate.strip()

        if candidate:
            chunks.append(
                candidate
            )

        if end >= text_length:
            break

        next_start = (
            end
            - resolved_overlap
        )

        # Defensive protection against an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def build_chunk_records(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[dict[str, object]]:
    """
    Convert text into indexed RAG chunk records.

    Example output:

    [
        {
            "chunk_index": 0,
            "content": "...",
            "character_count": 742
        }
    ]
    """
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    records: List[
        dict[str, object]
    ] = []

    for index, chunk in enumerate(
        chunks
    ):
        records.append(
            {
                "chunk_index": index,
                "content": chunk,
                "character_count": len(
                    chunk
                ),
            }
        )

    return records