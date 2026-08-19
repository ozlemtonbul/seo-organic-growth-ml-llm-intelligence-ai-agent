from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from config.logging_config import get_logger
from config.settings import SETTINGS


logger = get_logger(__name__)


PAGESPEED_ENDPOINT = (
    "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
)


# ============================================================
# URL HELPERS
# ============================================================


def _normalize_url(value: str) -> str:
    """
    Normalize URLs used by PageSpeed inventory and state tracking.
    """
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        parts = urlsplit(text)
    except ValueError:
        return ""

    if not parts.scheme or not parts.netloc:
        return ""

    path = parts.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            "",
            "",
        )
    )


# ============================================================
# PAGESPEED ROTATION STATE
# ============================================================


def _load_pagespeed_state(
    state_file: str,
) -> dict[str, dict[str, object]]:
    """
    Load persisted PageSpeed rotation state.
    """
    path = Path(
        state_file
    )

    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return (
            payload
            if isinstance(
                payload,
                dict,
            )
            else {}
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


def mark_pagespeed_batch_completed(
    pagespeed_dataframe: pd.DataFrame,
    state_file: str,
) -> None:
    """
    Persist PageSpeed measurement history.

    Stores:
    - last checked timestamp
    - check count
    - latest performance score
    - latest SEO score
    - latest LCP / CLS / INP
    - latest error
    """
    if (
        pagespeed_dataframe is None
        or pagespeed_dataframe.empty
        or "url" not in pagespeed_dataframe.columns
    ):
        return

    state = _load_pagespeed_state(
        state_file
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    for _, row in pagespeed_dataframe.iterrows():
        url = _normalize_url(
            row.get(
                "url",
                "",
            )
        )

        if not url:
            continue

        previous = state.get(
            url,
            {},
        )

        state[url] = {
            "last_checked_at": now,
            "check_count": int(
                previous.get(
                    "check_count",
                    0,
                )
                or 0
            )
            + 1,
            "performance_score": (
                row.get(
                    "performance_score"
                )
            ),
            "seo_score": (
                row.get(
                    "seo_score"
                )
            ),
            "lcp": row.get(
                "lcp"
            ),
            "cls": row.get(
                "cls"
            ),
            "inp": row.get(
                "inp"
            ),
            "pagespeed_error": str(
                row.get(
                    "pagespeed_error",
                    "",
                )
                or row.get(
                    "pagespeed_runtime_error",
                    "",
                )
                or ""
            ),
        }

    path = Path(
        state_file
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# PRIORITY SCORING
# ============================================================


def _prepare_priority_intelligence(
    page_intelligence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare PageSpeed business/SEO priority scores.
    """
    if (
        page_intelligence is None
        or page_intelligence.empty
        or "page" not in page_intelligence.columns
    ):
        return pd.DataFrame(
            columns=[
                "url",
                "pagespeed_priority_score",
            ]
        )

    data = page_intelligence.copy()

    numeric_columns = [
        "PageOpportunityScore",
        "CommerceScore",
        "Revenue",
        "Sessions",
        "Clicks",
        "Impressions",
    ]

    for column in numeric_columns:
        if column not in data.columns:
            data[column] = 0.0

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(
            0.0
        )

    data["url"] = data[
        "page"
    ].map(
        _normalize_url
    )

    data = data[
        data["url"].ne("")
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "url",
                "pagespeed_priority_score",
            ]
        )

    revenue_rank = (
        data["Revenue"]
        .rank(
            pct=True
        )
        .fillna(
            0.0
        )
    )

    sessions_rank = (
        data["Sessions"]
        .rank(
            pct=True
        )
        .fillna(
            0.0
        )
    )

    clicks_rank = (
        data["Clicks"]
        .rank(
            pct=True
        )
        .fillna(
            0.0
        )
    )

    impressions_rank = (
        data["Impressions"]
        .rank(
            pct=True
        )
        .fillna(
            0.0
        )
    )

    data[
        "pagespeed_priority_score"
    ] = (
        data[
            "PageOpportunityScore"
        ]
        * 0.35
        + data[
            "CommerceScore"
        ]
        * 0.25
        + revenue_rank
        * 15.0
        + sessions_rank
        * 10.0
        + clicks_rank
        * 10.0
        + impressions_rank
        * 5.0
    )

    return (
        data[
            [
                "url",
                "pagespeed_priority_score",
            ]
        ]
        .sort_values(
            "pagespeed_priority_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "url",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SMART ROTATION SELECTION
# ============================================================


def select_pagespeed_urls(
    page_intelligence: pd.DataFrame,
    url_inventory: pd.DataFrame | None = None,
    max_urls: int | None = None,
    state_file: str | None = None,
    freshness_days: int | None = None,
) -> List[str]:
    """
    Select PageSpeed URLs using full-site smart rotation.

    Priority order
    --------------
    1. URLs never measured before.
    2. URLs whose PageSpeed result is stale.
    3. Oldest PageSpeed measurements.
    4. Highest SEO / commercial / traffic priority.
    5. Lowest historical check count.

    This ensures that PageSpeed eventually covers the
    whole site instead of repeatedly measuring the same
    top-performing URLs.
    """

    max_urls = int(
        max_urls
        or SETTINGS.pagespeed_max_urls
    )

    state_file = str(
        state_file
        or getattr(
            SETTINGS,
            "pagespeed_rotation_state_file",
            "./outputs/pagespeed_rotation_state.json",
        )
    )

    freshness_days = int(
        freshness_days
        if freshness_days is not None
        else getattr(
            SETTINGS,
            "pagespeed_freshness_days",
            14,
        )
    )

    priority = (
        _prepare_priority_intelligence(
            page_intelligence
        )
    )

    # ========================================================
    # FULL SITE INVENTORY
    # ========================================================

    candidate_urls: set[str] = set()

    if (
        url_inventory is not None
        and not url_inventory.empty
        and "url" in url_inventory.columns
    ):
        for value in (
            url_inventory[
                "url"
            ]
            .dropna()
            .astype(str)
        ):
            normalized = _normalize_url(
                value
            )

            if normalized:
                candidate_urls.add(
                    normalized
                )

    # Fallback / additional URLs from intelligence layer.
    if not priority.empty:
        candidate_urls.update(
            priority[
                "url"
            ].dropna().astype(str)
        )

    if not candidate_urls:
        return []

    candidates = pd.DataFrame(
        {
            "url": sorted(
                candidate_urls
            )
        }
    )

    candidates = candidates.merge(
        priority,
        on="url",
        how="left",
    )

    candidates[
        "pagespeed_priority_score"
    ] = pd.to_numeric(
        candidates[
            "pagespeed_priority_score"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    # ========================================================
    # HISTORICAL STATE
    # ========================================================

    state = _load_pagespeed_state(
        state_file
    )

    candidates[
        "last_checked_at"
    ] = candidates[
        "url"
    ].map(
        lambda url: str(
            state.get(
                url,
                {},
            ).get(
                "last_checked_at",
                "",
            )
        )
    )

    candidates[
        "check_count"
    ] = candidates[
        "url"
    ].map(
        lambda url: int(
            state.get(
                url,
                {},
            ).get(
                "check_count",
                0,
            )
            or 0
        )
    )

    candidates[
        "never_checked"
    ] = candidates[
        "last_checked_at"
    ].eq(
        ""
    )

    parsed_last_checked = pd.to_datetime(
        candidates[
            "last_checked_at"
        ],
        errors="coerce",
        utc=True,
    )

    freshness_cutoff = (
        datetime.now(
            timezone.utc
        )
        - timedelta(
            days=freshness_days
        )
    )

    candidates[
        "is_stale"
    ] = (
        parsed_last_checked.isna()
        | (
            parsed_last_checked
            < freshness_cutoff
        )
    )

    # Timestamp used only for deterministic sorting.
    candidates[
        "_last_checked_sort"
    ] = parsed_last_checked

    # Never-checked timestamps become oldest.
    candidates[
        "_last_checked_sort"
    ] = candidates[
        "_last_checked_sort"
    ].fillna(
        pd.Timestamp(
            "1970-01-01",
            tz="UTC",
        )
    )

    # ========================================================
    # SMART PRIORITY SORT
    # ========================================================

    candidates = candidates.sort_values(
        [
            "never_checked",
            "is_stale",
            "_last_checked_sort",
            "pagespeed_priority_score",
            "check_count",
            "url",
        ],
        ascending=[
            False,
            False,
            True,
            False,
            True,
            True,
        ],
    )

    selected = (
        candidates
        .head(
            max_urls
        )[
            "url"
        ]
        .tolist()
    )

    logger.info(
        "Smart PageSpeed batch selected: %d URLs "
        "| total candidate inventory: %d "
        "| never checked: %d "
        "| stale: %d.",
        len(selected),
        len(candidates),
        int(
            candidates[
                "never_checked"
            ].sum()
        ),
        int(
            candidates[
                "is_stale"
            ].sum()
        ),
    )

    return selected


# ============================================================
# PAGESPEED RESPONSE HELPERS
# ============================================================


def _category_score(
    payload: dict,
    category: str,
) -> float | None:
    score = (
        payload
        .get(
            "lighthouseResult",
            {},
        )
        .get(
            "categories",
            {},
        )
        .get(
            category,
            {},
        )
        .get(
            "score"
        )
    )

    if score is None:
        return None

    try:
        return round(
            float(
                score
            )
            * 100.0,
            2,
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _audit_numeric(
    payload: dict,
    audit_id: str,
    divisor: float = 1.0,
) -> float | None:
    value = (
        payload
        .get(
            "lighthouseResult",
            {},
        )
        .get(
            "audits",
            {},
        )
        .get(
            audit_id,
            {},
        )
        .get(
            "numericValue"
        )
    )

    if value is None:
        return None

    try:
        return round(
            float(
                value
            )
            / divisor,
            4,
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _field_percentile(
    payload: dict,
    metric: str,
    divisor: float = 1.0,
) -> float | None:
    value = (
        payload
        .get(
            "loadingExperience",
            {},
        )
        .get(
            "metrics",
            {},
        )
        .get(
            metric,
            {},
        )
        .get(
            "percentile"
        )
    )

    if value is None:
        return None

    try:
        return round(
            float(
                value
            )
            / divisor,
            4,
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# SINGLE URL EXTRACTION
# ============================================================


def fetch_pagespeed_for_url(
    url: str,
    strategy: str | None = None,
) -> Dict[str, object]:
    """
    Fetch PageSpeed / Lighthouse metrics for one URL.
    """
    strategy = (
        strategy
        or SETTINGS.pagespeed_strategy
    ).lower()

    params: List[
        tuple[
            str,
            str,
        ]
    ] = [
        (
            "url",
            url,
        ),
        (
            "strategy",
            strategy,
        ),
        (
            "category",
            "performance",
        ),
        (
            "category",
            "seo",
        ),
    ]

    if SETTINGS.pagespeed_api_key:
        params.append(
            (
                "key",
                SETTINGS.pagespeed_api_key,
            )
        )

    response = requests.get(
        PAGESPEED_ENDPOINT,
        params=params,
        timeout=SETTINGS.pagespeed_request_timeout,
    )

    response.raise_for_status()

    payload = response.json()

    lcp = _field_percentile(
        payload,
        "LARGEST_CONTENTFUL_PAINT_MS",
        1000.0,
    )

    if lcp is None:
        lcp = _audit_numeric(
            payload,
            "largest-contentful-paint",
            1000.0,
        )

    cls = _field_percentile(
        payload,
        "CUMULATIVE_LAYOUT_SHIFT_SCORE",
        100.0,
    )

    if cls is None:
        cls = _audit_numeric(
            payload,
            "cumulative-layout-shift",
        )

    inp = _field_percentile(
        payload,
        "INTERACTION_TO_NEXT_PAINT",
        1.0,
    )

    if inp is None:
        inp = _audit_numeric(
            payload,
            "interaction-to-next-paint",
        )

    return {
        "url": _normalize_url(
            url
        ),
        "pagespeed_strategy": strategy,
        "performance_score": _category_score(
            payload,
            "performance",
        ),
        "seo_score": _category_score(
            payload,
            "seo",
        ),
        "lcp": lcp,
        "cls": cls,
        "inp": inp,
        "fcp": _audit_numeric(
            payload,
            "first-contentful-paint",
            1000.0,
        ),
        "speed_index": _audit_numeric(
            payload,
            "speed-index",
            1000.0,
        ),
        "total_blocking_time": _audit_numeric(
            payload,
            "total-blocking-time",
            1.0,
        ),
        "pagespeed_overall_category": (
            payload
            .get(
                "loadingExperience",
                {},
            )
            .get(
                "overall_category",
                "",
            )
        ),
        "pagespeed_runtime_error": (
            payload
            .get(
                "lighthouseResult",
                {},
            )
            .get(
                "runtimeError",
                {},
            )
            .get(
                "message",
                "",
            )
        ),
    }


# ============================================================
# BATCH EXTRACTION
# ============================================================


def fetch_pagespeed_data(
    urls: Iterable[str],
) -> pd.DataFrame:
    """
    Fetch PageSpeed data for the current selected batch.
    """
    if not SETTINGS.pagespeed_enabled:
        logger.info(
            "PageSpeed integration is disabled. Skipping."
        )

        return pd.DataFrame()

    unique_urls = list(
        dict.fromkeys(
            _normalize_url(
                url
            )
            for url in urls
            if _normalize_url(
                url
            )
        )
    )[
        : SETTINGS.pagespeed_max_urls
    ]

    if not unique_urls:
        return pd.DataFrame()

    if not SETTINGS.pagespeed_api_key:
        logger.warning(
            "PAGESPEED_API_KEY is empty. "
            "API calls may still work, "
            "but a key is recommended "
            "for automated usage."
        )

    logger.info(
        "Starting PageSpeed extraction for %d URLs (%s).",
        len(
            unique_urls
        ),
        SETTINGS.pagespeed_strategy,
    )

    rows: List[
        Dict[
            str,
            object,
        ]
    ] = []

    for index, url in enumerate(
        unique_urls,
        start=1,
    ):
        try:
            rows.append(
                fetch_pagespeed_for_url(
                    url
                )
            )

        except requests.RequestException as exc:
            logger.warning(
                "PageSpeed request failed for %s: %s",
                url,
                exc,
            )

            rows.append(
                {
                    "url": url,
                    "pagespeed_strategy": (
                        SETTINGS.pagespeed_strategy
                    ),
                    "pagespeed_error": str(
                        exc
                    ),
                }
            )

        if (
            SETTINGS.pagespeed_delay_seconds > 0
            and index < len(
                unique_urls
            )
        ):
            time.sleep(
                SETTINGS.pagespeed_delay_seconds
            )

    result = pd.DataFrame(
        rows
    )

    logger.info(
        "PageSpeed extraction completed: %d rows.",
        len(
            result
        ),
    )

    return result


# ============================================================
# CRAWL + PAGESPEED ENRICHMENT
# ============================================================


def enrich_crawl_with_pagespeed(
    crawl_dataframe: pd.DataFrame,
    pagespeed_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge PageSpeed results into technical crawl data.
    """
    if (
        pagespeed_dataframe is None
        or pagespeed_dataframe.empty
    ):
        return (
            crawl_dataframe.copy()
            if crawl_dataframe is not None
            else pd.DataFrame()
        )

    psi = pagespeed_dataframe.copy()

    psi[
        "_url_key"
    ] = psi[
        "url"
    ].map(
        _normalize_url
    )

    if (
        crawl_dataframe is None
        or crawl_dataframe.empty
    ):
        return psi.drop(
            columns=[
                "_url_key",
            ],
            errors="ignore",
        ).copy()

    crawl = crawl_dataframe.copy()

    url_column = next(
        (
            column
            for column in [
                "url",
                "page",
                "address",
                "final_url",
            ]
            if column
            in crawl.columns
        ),
        None,
    )

    if url_column is None:
        return crawl

    crawl[
        "_url_key"
    ] = crawl[
        url_column
    ].map(
        _normalize_url
    )

    merged = crawl.merge(
        psi.drop(
            columns=[
                "url",
            ],
            errors="ignore",
        ),
        on="_url_key",
        how="left",
    )

    return merged.drop(
        columns=[
            "_url_key",
        ],
        errors="ignore",
    )