from __future__ import annotations

from config.logging_config import get_logger
from config.settings import SETTINGS
from src.llm.manager import generate_text
from src.rag.retrieval import prepare_rag_prompt


logger = get_logger(__name__)


def _safe_float(
    value: object,
    default: float = 0.0,
) -> float:
    """
    Convert optional numeric values safely.
    """
    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _extract_field(
    content: str,
    field_name: str,
) -> str:
    """
    Extract a simple `Field: value` line from RAG chunk content.
    """
    normalized_content = str(
        content or ""
    )

    prefix = (
        f"{field_name}:"
    ).lower()

    for line in normalized_content.splitlines():
        stripped = line.strip()

        if (
            stripped.lower()
            .startswith(prefix)
        ):
            return (
                stripped.split(
                    ":",
                    1,
                )[1]
                .strip()
            )

    return ""


def _split_recommendation_records(
    content: str,
) -> list[str]:
    """
    Split one recommendation chunk into individual records.

    Recommendation chunks may contain multiple page records
    separated by `---`.
    """
    normalized_content = str(
        content or ""
    ).strip()

    if not normalized_content:
        return []

    return [
        record.strip()
        for record in normalized_content.split(
            "---"
        )
        if record.strip()
    ]


def _clean_page_value(
    page: str,
) -> str:
    """
    Normalize page text for display and deduplication.

    Markdown-style links are preserved when present.
    """
    return str(
        page or ""
    ).strip()


def _build_recommendation_fallback(
    results: list[
        dict[str, object]
    ],
    max_items: int = 5,
) -> str:
    """
    Build a deterministic multi-result SEO recommendation answer.

    High/Medium Priority business candidates are preferred.

    Fragmented chunk sections are ignored when they do not
    contain both a page and an actionable recommendation.

    Duplicate page recommendations are removed.
    """
    recommendations: list[
        dict[str, object]
    ] = []

    seen_pages: set[str] = set()

    for result in results:
        content = str(
            result.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        source_name = str(
            result.get(
                "source_name",
                "",
            )
            or ""
        ).strip()

        candidate_type = str(
            result.get(
                "candidate_type",
                "semantic",
            )
            or "semantic"
        ).strip()

        final_score = _safe_float(
            result.get(
                "final_retrieval_score",
                result.get(
                    "similarity",
                    0.0,
                ),
            )
        )

        records = (
            _split_recommendation_records(
                content
            )
        )

        for record in records:
            page = _clean_page_value(
                _extract_field(
                    record,
                    "page",
                )
            )

            scenario = _extract_field(
                record,
                "Scenario",
            )

            action = _extract_field(
                record,
                "RecommendedAction",
            )

            reason = _extract_field(
                record,
                "RecommendationReason",
            )

            priority = _extract_field(
                record,
                "PriorityTier",
            )

            confidence = _extract_field(
                record,
                "ConfidenceLevel",
            )

            # Ignore fragmented chunk sections.
            #
            # Some RAG chunks begin or end in the middle of a
            # recommendation record. Those fragments may still
            # contain PriorityTier / ROI values, but they are not
            # complete recommendations.
            if (
                not page
                or not action
            ):
                continue

            roi = _extract_field(
                record,
                "EstimatedROI",
            )

            net_value = _extract_field(
                record,
                "AdjustedNetValue",
            )

            # Priority-focused fallback should contain only
            # actionable High or Medium Priority records.
            if priority not in {
                "High Priority",
                "Medium Priority",
            }:
                continue

            # Do not repeat the same page when overlapping chunks
            # contain the same recommendation.
            if page in seen_pages:
                continue

            seen_pages.add(
                page
            )

            priority_rank = (
                2
                if priority
                == "High Priority"
                else 1
            )

            recommendations.append(
                {
                    "page": page,
                    "scenario": scenario,
                    "action": action,
                    "reason": reason,
                    "priority": priority,
                    "confidence": confidence,
                    "roi": roi,
                    "net_value": net_value,
                    "source_name": source_name,
                    "candidate_type": (
                        candidate_type
                    ),
                    "final_score": (
                        final_score
                    ),
                    "priority_rank": (
                        priority_rank
                    ),
                }
            )

    recommendations.sort(
        key=lambda item: (
            int(
                item[
                    "priority_rank"
                ]
            ),
            _safe_float(
                item[
                    "net_value"
                ]
            ),
            _safe_float(
                item[
                    "roi"
                ]
            ),
            _safe_float(
                item[
                    "final_score"
                ]
            ),
        ),
        reverse=True,
    )

    selected = recommendations[
        :max(
            1,
            int(
                max_items
            ),
        )
    ]

    if not selected:
        return ""

    lines = [
        (
            "LLM kullanımı şu anda devre dışı veya "
            "kullanılamıyor. Retrieved proje verisine "
            "göre en öncelikli SEO fırsatları:"
        ),
        "",
    ]

    for index, item in enumerate(
        selected,
        start=1,
    ):
        action_text = str(
            item[
                "action"
            ]
            or item[
                "scenario"
            ]
            or "SEO optimization"
        )

        lines.append(
            f"{index}. "
            f"{item['priority']} — "
            f"{action_text}"
        )

        if item[
            "page"
        ]:
            lines.append(
                "   Sayfa: "
                f"{item['page']}"
            )

        if item[
            "reason"
        ]:
            lines.append(
                "   Neden: "
                f"{item['reason']}"
            )

        metrics: list[str] = []

        if item[
            "confidence"
        ]:
            metrics.append(
                "Confidence: "
                f"{item['confidence']}"
            )

        if item[
            "roi"
        ]:
            metrics.append(
                "Estimated ROI: "
                f"{item['roi']}"
            )

        if item[
            "net_value"
        ]:
            metrics.append(
                "Adjusted Net Value: "
                f"{item['net_value']}"
            )

        if metrics:
            lines.append(
                "   "
                + " | ".join(
                    metrics
                )
            )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    ).strip()


def _build_generic_fallback(
    results: list[
        dict[str, object]
    ],
    max_items: int = 3,
) -> str:
    """
    Build a deterministic fallback for non-recommendation queries.
    """
    if not results:
        return (
            "Bu soru için knowledge base içinde "
            "yeterli proje kanıtı bulunamadı."
        )

    selected = results[
        :max(
            1,
            int(
                max_items
            ),
        )
    ]

    lines = [
        (
            "LLM kullanımı şu anda devre dışı veya "
            "kullanılamıyor. En ilgili proje kayıtları:"
        ),
        "",
    ]

    display_index = 1

    for result in selected:
        content = str(
            result.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        source_name = str(
            result.get(
                "source_name",
                "",
            )
            or ""
        ).strip()

        score = _safe_float(
            result.get(
                "final_retrieval_score",
                result.get(
                    "similarity",
                    0.0,
                ),
            )
        )

        lines.append(
            f"{display_index}. "
            f"Kaynak: "
            f"{source_name or 'unknown'}"
        )

        lines.append(
            "   Retrieval Score: "
            f"{score:.4f}"
        )

        lines.append(
            ""
        )

        lines.append(
            content
        )

        lines.append(
            ""
        )

        display_index += 1

    return "\n".join(
        lines
    ).strip()


def _build_deterministic_fallback(
    query: str,
    results: list[
        dict[str, object]
    ],
) -> str:
    """
    Select the most useful deterministic fallback mode.

    Recommendation and opportunity questions receive a
    structured business-aware multi-item summary.

    Other questions receive generic retrieved evidence.
    """
    normalized_query = (
        str(
            query or ""
        )
        .strip()
        .lower()
    )

    recommendation_terms = {
        "öncelik",
        "öncelikli",
        "fırsat",
        "opportunity",
        "recommend",
        "öneri",
        "aksiyon",
        "action",
    }

    wants_recommendations = any(
        term in normalized_query
        for term in recommendation_terms
    )

    if wants_recommendations:
        recommendation_answer = (
            _build_recommendation_fallback(
                results=results,
                max_items=5,
            )
        )

        if recommendation_answer:
            return recommendation_answer

    return _build_generic_fallback(
        results=results,
        max_items=3,
    )


def answer_with_rag(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
    source_type: str | None = None,
    max_characters: int = 12000,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, object]:
    """
    Answer a user question using project knowledge retrieved
    from PostgreSQL + pgvector.

    The retrieved context is provider-independent.

    Final generation uses the existing multi-provider LLM
    manager when an LLM is enabled and available.

    When the LLM is disabled, unavailable or usage-limited,
    a deterministic business-aware answer is produced from
    retrieved evidence.
    """
    normalized_query = str(
        query or ""
    ).strip()

    if not normalized_query:
        raise ValueError(
            "RAG query cannot be empty."
        )

    retrieval = prepare_rag_prompt(
        query=normalized_query,
        top_k=top_k,
        min_similarity=min_similarity,
        source_type=source_type,
        max_characters=max_characters,
    )

    prompt = str(
        retrieval.get(
            "prompt",
            "",
        )
        or ""
    ).strip()

    answer = None

    if SETTINGS.llm_enabled:
        answer = generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if answer:
        answer_mode = (
            "llm_grounded"
        )

        final_answer = answer

    else:
        answer_mode = (
            "deterministic_fallback"
        )

        results = retrieval.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            results = []

        final_answer = (
            _build_deterministic_fallback(
                query=normalized_query,
                results=results,
            )
        )

    logger.info(
        "RAG answer completed | "
        "Mode: %s | Results: %d",
        answer_mode,
        int(
            retrieval.get(
                "result_count",
                0,
            )
            or 0
        ),
    )

    return {
        "query": normalized_query,
        "answer": final_answer,
        "answer_mode": answer_mode,
        "result_count": retrieval.get(
            "result_count",
            0,
        ),
        "results": retrieval.get(
            "results",
            [],
        ),
        "context": retrieval.get(
            "context",
            "",
        ),
        "prompt": prompt,
    }