from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract.onpage_content_extractor import (
    OnPageCrawlConfig,
    crawl_onpage_urls,
)
from src.features.blog_intelligence import (
    build_blog_content_to_commerce_intelligence,
    build_blog_keyword_content_gaps,
)
from src.features.geo_intelligence import (
    build_geo_ai_visibility_intelligence,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs"

PAGE_OPPORTUNITY_FILE = OUTPUT_DIR / "seo_page_opportunity_intelligence.csv"
KEYWORD_OPPORTUNITY_FILE = OUTPUT_DIR / "seo_keyword_opportunity_intelligence.csv"
LATEST_PAGE_STATE_FILE = OUTPUT_DIR / "seo_latest_page_state.csv"
ONPAGE_FILE = OUTPUT_DIR / "seo_onpage_content_signals.csv"

BLOG_COMMERCE_OUTPUT = OUTPUT_DIR / "seo_blog_content_to_commerce_intelligence.csv"
BLOG_GAPS_OUTPUT = OUTPUT_DIR / "seo_blog_keyword_content_gaps.csv"
GEO_OUTPUT = OUTPUT_DIR / "seo_geo_ai_visibility_intelligence.csv"


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _normalize_page(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "page" not in frame.columns:
        return frame
    result = frame.copy()
    result["page"] = result["page"].astype(str).str.strip().str.rstrip("/")
    return result


def _fresh_pages(cache: pd.DataFrame, freshness_days: int) -> set[str]:
    if cache.empty or "page" not in cache.columns or "onpage_crawled_at" not in cache.columns:
        return set()

    timestamps = pd.to_datetime(
        cache["onpage_crawled_at"],
        errors="coerce",
        utc=True,
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=freshness_days)

    return set(
        cache.loc[
            timestamps.ge(cutoff),
            "page",
        ].astype(str)
    )


def _merge_latest_with_onpage(
    latest_page_state: pd.DataFrame,
    onpage: pd.DataFrame,
) -> pd.DataFrame:
    latest = _normalize_page(latest_page_state)
    signals = _normalize_page(onpage)

    if signals.empty:
        return latest

    signal_columns = [
        "page",
        "title",
        "h1",
        "meta_description",
        "content",
        "content_word_count",
        "schema_type",
        "faq",
        "brand",
        "author",
        "date_modified",
        "status_code",
        "final_url",
        "crawl_error",
        "onpage_crawled_at",
    ]
    keep = [column for column in signal_columns if column in signals.columns]
    signals = signals[keep].drop_duplicates("page", keep="last")

    merged = latest.merge(
        signals,
        on="page",
        how="left",
        suffixes=("", "_onpage"),
    )

    for column in signal_columns:
        if column == "page":
            continue

        source = f"{column}_onpage"
        if source not in merged.columns:
            continue

        if column not in merged.columns:
            merged[column] = merged[source]
        else:
            current = merged[column]
            empty = current.isna() | current.astype(str).str.strip().eq("")
            merged.loc[empty, column] = merged.loc[empty, source]

        merged = merged.drop(columns=[source])

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh on-page content signals and rebuild Content + GEO Intelligence."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent HTTP workers. Default: 4",
    )
    parser.add_argument(
        "--freshness-days",
        type=int,
        default=7,
        help="Do not re-crawl URLs refreshed within this many days. Default: 7",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of URLs to crawl in this run. 0 = all stale/missing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-crawl all target URLs regardless of cache freshness.",
    )
    args = parser.parse_args()

    print("Loading existing intelligence outputs...")

    page_opportunity = _normalize_page(_read_csv(PAGE_OPPORTUNITY_FILE))
    keyword_opportunity = _normalize_page(_read_csv(KEYWORD_OPPORTUNITY_FILE))
    latest_page_state = _normalize_page(_read_csv(LATEST_PAGE_STATE_FILE))
    cache = _normalize_page(_read_csv(ONPAGE_FILE, required=False))

    target_urls = (
        page_opportunity["page"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.rstrip("/")
        .drop_duplicates()
        .tolist()
    )

    fresh = set() if args.force else _fresh_pages(cache, args.freshness_days)
    crawl_urls = [url for url in target_urls if url not in fresh]

    if args.limit > 0:
        crawl_urls = crawl_urls[: args.limit]

    print(f"- Target pages: {len(target_urls):,}")
    print(f"- Cached/fresh pages: {len(fresh):,}")
    print(f"- Pages to crawl now: {len(crawl_urls):,}")

    if crawl_urls:
        print("\nExtracting on-page content signals...")
        crawled = crawl_onpage_urls(
            crawl_urls,
            config=OnPageCrawlConfig(
                workers=max(1, args.workers),
            ),
        )

        if not cache.empty:
            cache = cache[~cache["page"].isin(crawled["page"])].copy()
            onpage = pd.concat(
                [cache, crawled],
                ignore_index=True,
                sort=False,
            )
        else:
            onpage = crawled

        onpage.to_csv(
            ONPAGE_FILE,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        onpage = cache
        print("\nNo crawl required; using fresh on-page cache.")

    print("\nRebuilding Content Intelligence...")
    blog_content_to_commerce = build_blog_content_to_commerce_intelligence(
        page_intelligence=page_opportunity,
        keyword_intelligence=keyword_opportunity,
    )
    blog_keyword_content_gaps = build_blog_keyword_content_gaps(
        keyword_intelligence=keyword_opportunity,
        page_intelligence=page_opportunity,
    )

    print("Rebuilding GEO Intelligence with real on-page signals...")
    enriched_latest = _merge_latest_with_onpage(
        latest_page_state,
        onpage,
    )

    geo_ai_visibility = build_geo_ai_visibility_intelligence(
        page_intelligence=page_opportunity,
        latest_page_state=enriched_latest,
    )

    blog_content_to_commerce.to_csv(
        BLOG_COMMERCE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )
    blog_keyword_content_gaps.to_csv(
        BLOG_GAPS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )
    geo_ai_visibility.to_csv(
        GEO_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRefresh complete.")
    print(f"- On-page signals: {len(onpage):,} rows")
    print(f"- Content-to-Commerce: {len(blog_content_to_commerce):,} rows")
    print(f"- Content gaps: {len(blog_keyword_content_gaps):,} rows")
    print(f"- GEO Intelligence: {len(geo_ai_visibility):,} rows")

    if not geo_ai_visibility.empty and "GEOReadinessScore" in geo_ai_visibility.columns:
        scores = pd.to_numeric(
            geo_ai_visibility["GEOReadinessScore"],
            errors="coerce",
        ).dropna()
        if not scores.empty:
            print(
                "- GEO readiness: "
                f"min={scores.min():.1f}, "
                f"avg={scores.mean():.1f}, "
                f"max={scores.max():.1f}, "
                f"unique={scores.nunique()}"
            )


if __name__ == "__main__":
    main()
