from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    keep = [c for c in signal_columns if c in signals.columns]
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
    print("Loading existing intelligence outputs...")

    page_opportunity = _normalize_page(_read_csv(PAGE_OPPORTUNITY_FILE))
    keyword_opportunity = _normalize_page(_read_csv(KEYWORD_OPPORTUNITY_FILE))
    latest_page_state = _normalize_page(_read_csv(LATEST_PAGE_STATE_FILE))
    onpage = _normalize_page(_read_csv(ONPAGE_FILE, required=False))

    if onpage.empty:
        print(
            "WARNING: seo_onpage_content_signals.csv is not available yet. "
            "Run scripts/refresh_onpage_content_geo.py first for real GEO readiness signals."
        )

    enriched_latest = _merge_latest_with_onpage(
        latest_page_state,
        onpage,
    )

    print("\nRebuilding Content Intelligence...")
    blog_content_to_commerce = build_blog_content_to_commerce_intelligence(
        page_intelligence=page_opportunity,
        keyword_intelligence=keyword_opportunity,
    )
    blog_keyword_content_gaps = build_blog_keyword_content_gaps(
        keyword_intelligence=keyword_opportunity,
        page_intelligence=page_opportunity,
    )

    print("Rebuilding GEO Intelligence...")
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
    print(f"- Content-to-Commerce: {len(blog_content_to_commerce):,} rows")
    print(f"- Content gaps: {len(blog_keyword_content_gaps):,} rows")
    print(f"- GEO Intelligence: {len(geo_ai_visibility):,} rows")


if __name__ == "__main__":
    main()
