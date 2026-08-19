from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


GEO_SIGNAL_COLUMNS = {
    "h1": ["h1", "H1"],
    "content": ["content", "body", "text", "page_content"],
    "schema": ["schema_type", "schema", "structured_data_type"],
    "brand": ["brand", "Brand", "EntityName"],
    "meta": ["meta_description", "description", "MetaDescription"],
    "faq": ["faq", "faq_text", "questions", "question_answer_content"],
    "author": ["author", "author_name", "reviewed_by"],
    "updated": ["date_modified", "updated_at", "last_modified"],
}

CORPORATE_URL_TOKENS = (
    "/contact", "/iletisim", "/hakkimizda", "/about", "/privacy", "/gizlilik",
    "/terms", "/kvkk", "/sales/guest", "/login", "/register", "/account",
)
FAQ_URL_TOKENS = (
    "/faq", "/sss", "/sikca-sorulan", "/sik-sorulan", "/yardim", "/help",
)
GUIDE_URL_TOKENS = (
    "/beden-tablosu", "/size-guide", "/rehber", "/guide", "/blog",
)


def _numeric(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(0.0, index=dataframe.index, dtype=float)
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(0.0)


def _first_existing(
    row: pd.Series,
    candidates: Iterable[str],
) -> str:
    for column in candidates:
        if column in row.index:
            value = row.get(column, "")
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _has_signal(row: pd.Series, signal: str) -> bool:
    return bool(
        _first_existing(
            row,
            GEO_SIGNAL_COLUMNS[signal],
        )
    )


def _answer_signal(row: pd.Series) -> bool:
    text = " ".join([
        _first_existing(
            row,
            GEO_SIGNAL_COLUMNS["content"],
        ),
        _first_existing(
            row,
            GEO_SIGNAL_COLUMNS["faq"],
        ),
    ]).lower()

    return any(
        token in text
        for token in [
            "nedir",
            "nasıl",
            "nasil",
            "kimler için",
            "sık sorulan",
            "sik sorulan",
            "soru",
            "cevap",
            "what is",
            "how to",
            "who is it for",
            "frequently asked",
            "faq",
        ]
    )


def _content_depth(row: pd.Series) -> int:
    text = _first_existing(
        row,
        GEO_SIGNAL_COLUMNS["content"],
    )
    return len(text.split()) if text else 0


def _infer_page_type(
    page: object,
    declared: object = "",
) -> str:
    declared_text = str(declared or "").strip().lower()
    page_text = str(page or "").strip().lower()

    if any(
        token in page_text
        for token in CORPORATE_URL_TOKENS
    ):
        return "corporate"

    if any(
        token in page_text
        for token in FAQ_URL_TOKENS
    ):
        return "faq"

    if any(
        token in page_text
        for token in GUIDE_URL_TOKENS
    ):
        return "guide"

    if declared_text in {
        "product",
        "category",
        "blog",
        "informational",
        "guide",
        "faq",
        "corporate",
        "landing",
        "homepage",
    }:
        return declared_text

    if re.search(
        r"/p[-_/]?\d|/product|/urun",
        page_text,
    ):
        return "product"

    if any(
        token in page_text
        for token in (
            "/kategori",
            "/category",
            "/koleksiyon",
            "/collection",
        )
    ):
        return "category"

    return declared_text or "other"


def _coalesce_detail_signals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge logic may create *_detail columns when page_intelligence already contains
    an empty signal column. Coalesce those values back into the canonical column
    so real crawl/page-state signals are not silently ignored.
    """
    result = data.copy()

    canonical_columns = {
        column
        for candidates in GEO_SIGNAL_COLUMNS.values()
        for column in candidates
    }

    for canonical in canonical_columns:
        detail_column = f"{canonical}_detail"
        if detail_column not in result.columns:
            continue

        if canonical not in result.columns:
            result[canonical] = result[detail_column]
        else:
            current = result[canonical]
            empty_mask = (
                current.isna()
                | current.astype(str).str.strip().eq("")
            )
            result.loc[
                empty_mask,
                canonical,
            ] = result.loc[
                empty_mask,
                detail_column,
            ]

    return result


def _base_geo_readiness(row: pd.Series) -> float:
    """
    Build a transparent GEO-readiness proxy from observable page signals.

    Page-type baseline only reflects structural expectations. Observable content,
    schema, FAQ, entity, expertise and freshness signals remain the main drivers.
    """
    page_type = _infer_page_type(
        row.get("page", ""),
        row.get("page_type", ""),
    )

    baseline = {
        "product": 10.0,
        "category": 9.0,
        "blog": 12.0,
        "informational": 12.0,
        "guide": 13.0,
        "faq": 14.0,
        "corporate": 7.0,
        "landing": 9.0,
        "homepage": 10.0,
        "other": 8.0,
    }.get(
        page_type,
        8.0,
    )

    score = baseline

    if _has_signal(row, "h1"):
        score += 10
    if _has_signal(row, "meta"):
        score += 8
    if _has_signal(row, "schema"):
        score += 14
    if _has_signal(row, "brand"):
        score += 8
    if (
        _has_signal(row, "faq")
        or _answer_signal(row)
    ):
        score += 16
    if _has_signal(row, "author"):
        score += 8
    if _has_signal(row, "updated"):
        score += 6

    word_count = _content_depth(row)

    if page_type == "product":
        if word_count >= 180:
            score += 15
        elif word_count >= 80:
            score += 9
        elif word_count >= 30:
            score += 4
    elif page_type == "category":
        if word_count >= 350:
            score += 18
        elif word_count >= 150:
            score += 11
        elif word_count >= 60:
            score += 5
    else:
        if word_count >= 600:
            score += 20
        elif word_count >= 250:
            score += 14
        elif word_count >= 80:
            score += 7

    return float(
        min(
            100.0,
            score,
        )
    )


def _missing_signals(
    row: pd.Series,
) -> list[str]:
    missing: list[str] = []

    page_type = _infer_page_type(
        row.get("page", ""),
        row.get("page_type", ""),
    )

    if not _has_signal(row, "h1"):
        missing.append(
            "Clear H1 / topic definition"
        )
    if not _has_signal(row, "meta"):
        missing.append(
            "Concise meta summary"
        )
    if not _has_signal(row, "schema"):
        missing.append(
            "Relevant structured data / schema"
        )
    if not (
        _has_signal(row, "faq")
        or _answer_signal(row)
    ):
        missing.append(
            "Answer-first Q&A / FAQ coverage"
        )

    word_count = _content_depth(row)
    min_depth = {
        "product": 80,
        "category": 150,
        "faq": 100,
        "guide": 250,
        "blog": 250,
        "informational": 250,
        "corporate": 80,
    }.get(
        page_type,
        150,
    )

    if word_count < min_depth:
        missing.append(
            "Deeper factual and semantically complete content"
        )

    if not _has_signal(row, "brand"):
        missing.append(
            "Brand/entity clarity"
        )

    # Author/freshness are more relevant to editorial/informational assets.
    if page_type in {
        "blog",
        "informational",
        "guide",
        "faq",
    }:
        if not _has_signal(row, "author"):
            missing.append(
                "Author / expertise signal"
            )
        if not _has_signal(row, "updated"):
            missing.append(
                "Freshness / last-updated signal"
            )

    return missing


def _page_type_actions(
    page_type: str,
) -> list[str]:
    page_type = str(
        page_type or "other"
    ).lower()

    if page_type == "product":
        return [
            "Add complete product attributes such as material, fit, use case and sizing",
            "Use Product structured data and keep price/availability/brand facts consistent",
            "Add concise purchase-oriented Q&A that can be extracted by answer engines",
        ]

    if page_type == "category":
        return [
            "Add a clear category definition and buying-selection criteria",
            "Answer common comparison, sizing, material and use-case questions",
            "Link to supporting guides and high-value product entities",
        ]

    if page_type in {
        "blog",
        "informational",
        "guide",
    }:
        return [
            "Lead with a direct answer, then expand with evidence and practical guidance",
            "Strengthen author/expertise, freshness and source-worthy factual sections",
            "Link naturally to the most relevant commercial category/product destination",
        ]

    if page_type == "faq":
        return [
            "Group related questions by user intent and lead each answer with a concise direct response",
            "Add FAQPage structured data only when the visible content and eligibility rules support it",
            "Link answers to authoritative category, product and policy pages",
        ]

    if page_type == "corporate":
        return [
            "Clarify the organisation/entity purpose and the exact user task answered by the page",
            "Use appropriate Organization/WebPage structured data where relevant",
            "Keep policy, contact and trust information explicit and easy to extract",
        ]

    return [
        "Clarify the primary entity and user question answered by the page",
        "Add structured, extractable factual sections and relevant schema",
        "Strengthen internal links to authoritative supporting pages",
    ]


def _priority(
    score: float,
) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def build_geo_ai_visibility_intelligence(
    page_intelligence: pd.DataFrame,
    latest_page_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build page-level GEO / AI-readiness decision intelligence.

    This is explicitly a readiness proxy. It does not claim that a page is
    currently cited by ChatGPT, Gemini or AI Overviews unless a dedicated
    citation/answer-engine tracking source is connected later.
    """
    if page_intelligence.empty:
        return pd.DataFrame()

    data = page_intelligence.copy()

    if "page" not in data.columns:
        raise ValueError(
            "page_intelligence must include a page column."
        )

    if "page_type" not in data.columns:
        data["page_type"] = ""

    data["page_type"] = [
        _infer_page_type(
            page,
            declared,
        )
        for page, declared in zip(
            data["page"],
            data["page_type"],
        )
    ]

    if (
        latest_page_state is not None
        and not latest_page_state.empty
        and "page" in latest_page_state.columns
    ):
        detail = latest_page_state.drop_duplicates(
            "page",
            keep="last",
        ).copy()

        merge_columns = ["page"]

        for candidates in GEO_SIGNAL_COLUMNS.values():
            for column in candidates:
                if (
                    column in detail.columns
                    and column not in merge_columns
                ):
                    merge_columns.append(column)

        if (
            "CurrentGeoReadinessScore" in detail.columns
            and "CurrentGeoReadinessScore"
            not in merge_columns
        ):
            merge_columns.append(
                "CurrentGeoReadinessScore"
            )

        data = data.merge(
            detail[merge_columns],
            on="page",
            how="left",
            suffixes=(
                "",
                "_detail",
            ),
        )

        data = _coalesce_detail_signals(
            data
        )

    readiness: list[float] = []

    for _, row in data.iterrows():
        existing = pd.to_numeric(
            row.get(
                "CurrentGeoReadinessScore",
                np.nan,
            ),
            errors="coerce",
        )

        signal_score = _base_geo_readiness(
            row
        )

        if (
            pd.notna(existing)
            and float(existing) > 0
        ):
            score = (
                float(existing) * 0.55
                + signal_score * 0.45
            )
        else:
            score = signal_score

        readiness.append(
            round(
                min(
                    100.0,
                    max(
                        0.0,
                        score,
                    ),
                ),
                1,
            )
        )

    data["GEOReadinessScore"] = readiness
    data["GEOReadinessGap"] = (
        100.0
        - data["GEOReadinessScore"]
    ).clip(
        0,
        100,
    )

    page_opportunity = _numeric(
        data,
        "PageOpportunityScore",
    )
    demand = _numeric(
        data,
        "DemandScore",
    )
    commerce = _numeric(
        data,
        "CommerceScore",
    )

    data["GEOOpportunityScore"] = (
        data["GEOReadinessGap"] * 0.45
        + page_opportunity * 0.25
        + demand * 0.15
        + commerce * 0.15
    ).clip(
        0,
        100,
    ).round(
        1
    )

    data["GEOPriority"] = (
        data["GEOOpportunityScore"].map(
            _priority
        )
    )

    data["GEOMissingSignals"] = data.apply(
        lambda row: (
            " | ".join(
                _missing_signals(
                    row
                )
            )
            or "No major readiness gap detected"
        ),
        axis=1,
    )

    data["GEORecommendedActions"] = data.apply(
        lambda row: " | ".join(
            _page_type_actions(
                _infer_page_type(
                    row.get(
                        "page",
                        "",
                    ),
                    row.get(
                        "page_type",
                        "other",
                    ),
                )
            )
        ),
        axis=1,
    )

    data["GEOWhyNow"] = data.apply(
        lambda row: (
            f"Readiness {float(row['GEOReadinessScore']):.1f}/100; "
            f"SEO opportunity "
            f"{float(pd.to_numeric(row.get('PageOpportunityScore', 0), errors='coerce') or 0):.1f}/100; "
            f"commercial score "
            f"{float(pd.to_numeric(row.get('CommerceScore', 0), errors='coerce') or 0):.1f}/100; "
            f"page type {row.get('page_type', 'other')}."
        ),
        axis=1,
    )

    data["AIMeasurementStatus"] = (
        "GEO readiness proxy only; direct ChatGPT/Gemini/AI Overviews "
        "citation tracking is not connected."
    )

    preferred = [
        "page",
        "EntityName",
        "page_type",
        "keyword_intent",
        "GEOReadinessScore",
        "GEOReadinessGap",
        "GEOOpportunityScore",
        "GEOPriority",
        "GEOMissingSignals",
        "GEORecommendedActions",
        "GEOWhyNow",
        "PageOpportunityScore",
        "DemandScore",
        "CommerceScore",
        "Revenue",
        "Purchases",
        "AddToCarts",
        "Impressions",
        "Clicks",
        "CurrentPosition",
        "AIMeasurementStatus",
    ]

    columns = [
        column
        for column in preferred
        if column in data.columns
    ]

    sort_columns = [
        column
        for column in [
            "GEOOpportunityScore",
            "PageOpportunityScore",
        ]
        if column in data.columns
    ]

    return (
        data[columns]
        .sort_values(
            sort_columns,
            ascending=[False] * len(
                sort_columns
            ),
        )
        .reset_index(
            drop=True
        )
    )
