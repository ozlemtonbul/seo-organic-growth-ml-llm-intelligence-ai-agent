from __future__ import annotations

import re

import numpy as np
import pandas as pd


BLOG_PAGE_TYPES = {"blog", "informational", "guide", "faq"}
COMMERCIAL_PAGE_TYPES = {"product", "category"}

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


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _pct(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if values.empty:
        return values
    if values.nunique(dropna=False) <= 1:
        return pd.Series(50.0, index=values.index)
    return values.rank(method="average", pct=True) * 100.0


def _priority(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _infer_page_type(page: object, declared: object = "") -> str:
    """Normalize page type using both declared type and URL semantics."""
    declared_text = str(declared or "").strip().lower()
    page_text = str(page or "").strip().lower()

    if any(token in page_text for token in CORPORATE_URL_TOKENS):
        return "corporate"
    if any(token in page_text for token in FAQ_URL_TOKENS):
        return "faq"
    if any(token in page_text for token in GUIDE_URL_TOKENS):
        return "guide"

    if declared_text in {
        "product", "category", "blog", "informational", "guide", "faq",
        "corporate", "landing", "homepage",
    }:
        return declared_text

    if re.search(r"/p[-_/]?\d|/product|/urun", page_text):
        return "product"
    if any(token in page_text for token in ("/kategori", "/category", "/koleksiyon", "/collection")):
        return "category"

    return declared_text or "other"


def _blog_action(row: pd.Series) -> str:
    position = float(row.get("CurrentPosition", 0) or 0)
    impressions = float(row.get("Impressions", row.get("KeywordImpressions", 0)) or 0)
    clicks = float(row.get("Clicks", row.get("KeywordClicks", 0)) or 0)

    if 4 <= position <= 15:
        return (
            "Refresh the article around the winning query cluster, strengthen "
            "answer-first sections and add contextual internal links to the recommended "
            "commercial landing page."
        )
    if 0 < position <= 3:
        return (
            "Defend the ranking, improve SERP CTR and conversion paths, and add "
            "stronger contextual product/category links without changing intent."
        )
    if impressions > 0 and clicks == 0:
        return (
            "Rewrite title/meta and the opening answer to improve CTR, then connect "
            "the article to the most relevant commercial landing page."
        )
    return (
        "Expand the article around the recommended keyword cluster and build a "
        "clear informational-to-commercial internal-link journey."
    )


def _gap_reason(row: pd.Series) -> str:
    page_type = str(row.get("CurrentLandingPageType", "other")).lower()
    impressions = float(row.get("Impressions", 0) or 0)
    clicks = float(row.get("Clicks", 0) or 0)
    position = float(
        row.get(
            "CurrentPosition",
            row.get("AveragePosition", 0),
        ) or 0
    )
    query = str(row.get("query", row.get("Keyword", "")) or "").strip()

    if page_type == "corporate":
        return (
            f"The informational query “{query}” is landing on a corporate/utility page. "
            "That page is not the right content asset for this search intent."
        )
    if page_type == "product":
        return (
            f"The informational query “{query}” is currently associated with a product page. "
            "A supporting guide can answer the informational intent without weakening the product page."
        )
    if page_type == "category":
        return (
            f"The category page is receiving informational demand for “{query}”. "
            "Supporting content can capture the informational intent and pass authority to the category."
        )
    if page_type in {"faq", "guide"}:
        return (
            f"An existing informational asset is present for “{query}”, but the opportunity score "
            "indicates that coverage, CTR or ranking depth can still be improved."
        )
    if impressions >= 1000 and clicks / max(impressions, 1) < 0.02:
        return (
            f"The query “{query}” already has strong visibility ({impressions:.0f} impressions) "
            "but weak click capture, indicating an under-served content/CTR opportunity."
        )
    if 4 <= position <= 15:
        return (
            f"The query “{query}” is within striking distance at position {position:.1f}; "
            "better intent coverage and supporting content may move it higher."
        )
    return (
        f"The query “{query}” shows informational demand that is not fully served by the current landing experience."
    )


def _gap_action(row: pd.Series) -> str:
    page_type = str(row.get("CurrentLandingPageType", "other")).lower()
    query = str(row.get("query", row.get("Keyword", "")) or "").strip()

    if page_type == "corporate":
        return (
            "Create a dedicated guide, FAQ or landing page for this informational intent; "
            "do not force the corporate/utility page to rank for it. Link the new content to the relevant commercial page."
        )
    if page_type == "product":
        return (
            "Create a supporting guide/FAQ that answers the informational intent, then add contextual links "
            "to the product and its parent category."
        )
    if page_type == "category":
        return (
            "Expand the category with concise buying guidance and FAQs, and create a supporting article "
            "for deeper informational questions. Link both assets contextually."
        )
    if page_type in {"faq", "guide"}:
        return (
            "Expand the existing informational asset around related subtopics, improve title/meta for CTR, "
            "and strengthen links to the most relevant category/product destination."
        )
    return (
        f"Create or expand supporting content around “{query}”, answer the intent directly, "
        "and connect it to the most relevant commercial landing page with contextual internal links."
    )


def build_blog_content_to_commerce_intelligence(
    page_intelligence: pd.DataFrame,
    keyword_intelligence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build blog/content-to-commerce decisions.

    Each informational URL is connected to its strongest current keyword opportunity
    and to a relevant commercial product/category target. The score combines
    organic demand/ranking potential with the commerce strength of the destination.
    """
    if page_intelligence.empty:
        return pd.DataFrame()

    pages = page_intelligence.copy()
    if "page_type" not in pages.columns:
        return pd.DataFrame()

    pages["page_type"] = [
        _infer_page_type(page, declared)
        for page, declared in zip(pages["page"], pages["page_type"])
    ]

    blogs = pages[
        pages["page_type"].astype(str).str.lower().isin(BLOG_PAGE_TYPES)
    ].copy()
    if blogs.empty:
        return pd.DataFrame()

    commercial = pages[
        pages["page_type"].astype(str).str.lower().isin(COMMERCIAL_PAGE_TYPES)
    ].copy()

    if (
        not keyword_intelligence.empty
        and {"page", "query"}.issubset(keyword_intelligence.columns)
    ):
        kw = keyword_intelligence.copy()
        sort_col = (
            "KeywordOpportunityScore"
            if "KeywordOpportunityScore" in kw.columns
            else "Impressions"
        )
        if sort_col in kw.columns:
            kw = kw.sort_values(sort_col, ascending=False)
        kw = kw.drop_duplicates("page", keep="first")
        keep = [
            c for c in [
                "page", "query", "CurrentPosition", "AveragePosition", "Clicks",
                "Impressions", "CTR", "keyword_intent",
                "KeywordOpportunityScore", "KeywordPriority",
                "RecommendedKeywordAction",
            ] if c in kw.columns
        ]
        kw = kw[keep].rename(columns={
            "query": "PrimaryKeyword",
            "Clicks": "KeywordClicks",
            "Impressions": "KeywordImpressions",
            "CTR": "KeywordCTR",
            "keyword_intent": "KeywordIntent",
        })
        blogs = blogs.merge(kw, on="page", how="left")

    for c in ["PrimaryKeyword", "KeywordIntent", "RecommendedKeywordAction"]:
        if c not in blogs.columns:
            blogs[c] = ""
    for c in [
        "CurrentPosition", "KeywordClicks", "KeywordImpressions",
        "KeywordCTR", "KeywordOpportunityScore",
    ]:
        if c not in blogs.columns:
            blogs[c] = 0.0

    target_rows = []
    if not commercial.empty:
        commerce_sort = [
            c for c in ["CommerceScore", "PageOpportunityScore", "Revenue"]
            if c in commercial.columns
        ]
        if commerce_sort:
            commercial = commercial.sort_values(
                commerce_sort,
                ascending=[False] * len(commerce_sort),
            )

        def tokens(value: str) -> set[str]:
            stop = {
                "https", "http", "www", "com", "tr", "blog", "kategori",
                "category", "urun", "product", "ve", "ile", "icin",
            }
            return {
                x
                for x in re.findall(
                    r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+",
                    str(value).lower(),
                )
                if len(x) >= 3 and x not in stop
            }

        commercial_tokens = {
            idx: tokens(str(row.get("page", "")))
            for idx, row in commercial.iterrows()
        }

        for _, row in blogs.iterrows():
            source_tokens = tokens(
                f"{row.get('page', '')} {row.get('PrimaryKeyword', '')}"
            )
            best_idx = None
            best_overlap = -1
            for idx, tks in commercial_tokens.items():
                overlap = len(source_tokens & tks)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            target = (
                commercial.loc[best_idx]
                if best_idx is not None
                else None
            )
            target_rows.append({
                "RecommendedCommercialPage": (
                    "" if target is None else target.get("page", "")
                ),
                "RecommendedCommercialPageType": (
                    "" if target is None else target.get("page_type", "")
                ),
                "TargetCommerceScore": (
                    0.0
                    if target is None
                    else float(target.get("CommerceScore", 0) or 0)
                ),
                "TargetRevenue": (
                    0.0
                    if target is None
                    else float(target.get("Revenue", 0) or 0)
                ),
                "TargetPurchases": (
                    0.0
                    if target is None
                    else float(target.get("Purchases", 0) or 0)
                ),
                "TargetAddToCarts": (
                    0.0
                    if target is None
                    else float(target.get("AddToCarts", 0) or 0)
                ),
            })
    else:
        target_rows = [{
            "RecommendedCommercialPage": "",
            "RecommendedCommercialPageType": "",
            "TargetCommerceScore": 0.0,
            "TargetRevenue": 0.0,
            "TargetPurchases": 0.0,
            "TargetAddToCarts": 0.0,
        } for _ in range(len(blogs))]

    target_df = pd.DataFrame(target_rows, index=blogs.index)
    for c in target_df.columns:
        blogs[c] = target_df[c]

    blog_demand = _pct(_num(blogs, "Impressions"))
    blog_clicks = _pct(_num(blogs, "Clicks"))
    keyword_opp = _num(blogs, "KeywordOpportunityScore")
    target_commerce = _num(blogs, "TargetCommerceScore")

    blogs["ContentToCommerceScore"] = (
        keyword_opp * 0.40
        + blog_demand * 0.20
        + blog_clicks * 0.10
        + target_commerce * 0.30
    ).clip(0, 100)
    blogs["ContentPriority"] = (
        blogs["ContentToCommerceScore"].map(_priority)
    )
    blogs["RecommendedBlogAction"] = blogs.apply(
        _blog_action,
        axis=1,
    )
    blogs["InternalLinkRecommendation"] = np.where(
        blogs["RecommendedCommercialPage"].astype(str).ne(""),
        "Add contextual internal links and a relevant CTA to "
        + blogs["RecommendedCommercialPage"].astype(str),
        "No commercial target is available yet; map this article to a product/category page.",
    )
    blogs["CommerceObjective"] = (
        "Move qualified informational organic traffic toward a relevant "
        "product/category page and measure add-to-cart, purchase and revenue."
    )

    preferred = [
        "page", "page_type", "PrimaryKeyword", "CurrentPosition",
        "KeywordImpressions", "KeywordClicks", "KeywordCTR", "KeywordIntent",
        "KeywordOpportunityScore", "RecommendedCommercialPage",
        "RecommendedCommercialPageType", "TargetCommerceScore", "TargetRevenue",
        "TargetPurchases", "TargetAddToCarts", "ContentToCommerceScore",
        "ContentPriority", "RecommendedBlogAction",
        "InternalLinkRecommendation", "CommerceObjective",
    ]
    remaining = [c for c in blogs.columns if c not in preferred]
    return blogs[
        [c for c in preferred if c in blogs.columns] + remaining
    ].sort_values(
        ["ContentToCommerceScore", "KeywordImpressions"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_blog_keyword_content_gaps(
    keyword_intelligence: pd.DataFrame,
    page_intelligence: pd.DataFrame,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Find informational queries that deserve supporting-content coverage.

    The output keeps corporate/product/category landing pages visible when they are
    receiving informational demand, but the recommendation is adapted to the actual
    landing-page type instead of giving every row the same generic instruction.
    """
    if keyword_intelligence.empty:
        return pd.DataFrame()

    data = keyword_intelligence.copy()

    if "KeywordIntent" in data.columns and "keyword_intent" not in data.columns:
        data["keyword_intent"] = data["KeywordIntent"]

    intent = data.get(
        "keyword_intent",
        pd.Series("", index=data.index),
    ).astype(str)

    data = data[intent.str.lower().eq("informational")].copy()
    if data.empty:
        return pd.DataFrame()

    page_types = {}
    if (
        not page_intelligence.empty
        and {"page", "page_type"}.issubset(page_intelligence.columns)
    ):
        page_type_frame = page_intelligence.drop_duplicates(
            "page",
            keep="last",
        )[["page", "page_type"]].copy()
        page_type_frame["ResolvedPageType"] = [
            _infer_page_type(page, declared)
            for page, declared in zip(
                page_type_frame["page"],
                page_type_frame["page_type"],
            )
        ]
        page_types = page_type_frame.set_index("page")[
            "ResolvedPageType"
        ].to_dict()

    data["CurrentLandingPageType"] = [
        _infer_page_type(
            page,
            page_types.get(page, ""),
        )
        for page in data["page"]
    ]

    # Already-served blog/informational pages are still excluded, while FAQ/guide
    # pages remain actionable because they may need expansion rather than creation.
    data = data[
        ~data["CurrentLandingPageType"].isin({"blog", "informational"})
    ].copy()
    if data.empty:
        return pd.DataFrame()

    base_opp = _num(
        data,
        "KeywordOpportunityScore",
    )
    demand = _pct(_num(data, "Impressions"))

    type_adjustment = data["CurrentLandingPageType"].map({
        "corporate": 15.0,
        "product": 8.0,
        "category": 5.0,
        "faq": 0.0,
        "guide": 0.0,
        "other": 3.0,
    }).fillna(3.0)

    data["ContentGapScore"] = (
        base_opp * 0.60
        + demand * 0.30
        + type_adjustment
    ).clip(0, 100)

    data["ContentGapPriority"] = (
        data["ContentGapScore"].map(_priority)
    )
    data["ContentGapReason"] = data.apply(
        _gap_reason,
        axis=1,
    )
    data["RecommendedContentAction"] = data.apply(
        _gap_action,
        axis=1,
    )
    data["SuggestedCommercialDestination"] = data["page"]

    return data.sort_values(
        ["ContentGapScore", "Impressions"],
        ascending=[False, False],
    ).head(limit).reset_index(drop=True)
